"""The FOREGROUND ML engine.

Given a task spec, a dataframe and a workspace, it trains a scikit-learn model,
evaluates it, and produces the user's result plus performance metrics. There is
no LLM anywhere in this path — this is the part that actually does the work.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .spec import MLResult, MLTaskSpec
from .workspace import MLWorkspace


def _build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )


def _estimator(task_type: str, model: str):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    model = (model or "auto").lower()
    if task_type == "classification":
        if model == "logistic":
            return LogisticRegression(max_iter=1000), "logistic_regression"
        return RandomForestClassifier(n_estimators=200, random_state=0), "random_forest_classifier"
    if task_type == "regression":
        if model == "linear":
            return LinearRegression(), "linear_regression"
        return RandomForestRegressor(n_estimators=200, random_state=0), "random_forest_regressor"
    raise ValueError(f"no estimator for task {task_type!r}")


def _importances(pipeline: Pipeline, features: list[str]) -> dict[str, float]:
    try:
        pre = pipeline.named_steps["pre"]
        est = pipeline.named_steps["est"]
        names = list(pre.get_feature_names_out())
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            vals = np.abs(coef).sum(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return {}
        agg: dict[str, float] = {}
        for name, v in zip(names, vals):
            base = name.split("__", 1)[-1]
            orig = next((f for f in features if base == f or base.startswith(f + "_")), base)
            agg[orig] = agg.get(orig, 0.0) + float(v)
        total = sum(agg.values()) or 1.0
        ranked = sorted(((k, v / total) for k, v in agg.items()), key=lambda x: -x[1])
        return {k: round(v, 4) for k, v in ranked[:10]}
    except Exception:
        return {}


def _run_supervised(spec: MLTaskSpec, df: pd.DataFrame, ws: MLWorkspace) -> MLResult:
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
        precision_score, r2_score, recall_score,
    )
    from sklearn.model_selection import train_test_split

    target = spec.target
    df = df.dropna(subset=[target])
    features = spec.features or [c for c in df.columns if c != target]
    X, y = df[features], df[target]

    stratify = y if (spec.task_type == "classification" and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=spec.test_size, random_state=0, stratify=stratify
    )

    est, model_name = _estimator(spec.task_type, spec.model)
    pipe = Pipeline([("pre", _build_preprocessor(df, features)), ("est", est)])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    if spec.task_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
            "f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
            "precision_macro": round(float(precision_score(y_te, preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_te, preds, average="macro", zero_division=0)), 4),
        }
        primary = "accuracy"
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        metrics = {
            "r2": round(float(r2_score(y_te, preds)), 4),
            "mae": round(float(mean_absolute_error(y_te, preds)), 4),
            "rmse": round(rmse, 4),
        }
        primary = "r2"

    model_path = ws.model_dir / "model.joblib"
    joblib.dump(pipe, model_path)
    pred_df = X_te.copy()
    pred_df[f"actual_{target}"] = y_te.values
    pred_df[f"predicted_{target}"] = preds
    pred_path = ws.results_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    return MLResult(
        task_type=spec.task_type, model=model_name, metrics=metrics,
        primary_metric=primary, primary_score=metrics[primary],
        n_train=len(X_tr), n_test=len(X_te), n_features=len(features),
        feature_importances=_importances(pipe, features),
        model_path=str(model_path), predictions_path=str(pred_path),
    )


def _run_clustering(spec: MLTaskSpec, df: pd.DataFrame, ws: MLWorkspace) -> MLResult:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = spec.features or list(df.columns)
    pre = _build_preprocessor(df, features)
    X = pre.fit_transform(df[features])
    n_samples = X.shape[0]

    def fit_k(k: int):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        try:
            sil = float(silhouette_score(X, km.labels_)) if 1 < k < n_samples else float("nan")
        except Exception:
            sil = float("nan")
        return km, sil

    if spec.n_clusters:
        km, sil = fit_k(int(spec.n_clusters))
    else:
        best = None
        for k in range(2, min(8, n_samples - 1) + 1):
            km, sil = fit_k(k)
            if best is None or (not np.isnan(sil) and sil > best[1]):
                best = (km, sil, k)
        km, sil = best[0], best[1]

    labels = km.labels_
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    model_path = ws.model_dir / "model.joblib"
    joblib.dump({"preprocessor": pre, "kmeans": km}, model_path)
    pred_df = df.copy()
    pred_df["cluster"] = labels
    pred_path = ws.results_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics = {"silhouette": round(float(sil), 4) if not np.isnan(sil) else 0.0,
               "inertia": round(float(km.inertia_), 2), "n_clusters": int(km.n_clusters)}
    return MLResult(
        task_type="clustering", model="kmeans", metrics=metrics,
        primary_metric="silhouette", primary_score=metrics["silhouette"],
        n_train=n_samples, n_test=0, n_features=len(features),
        extra={"cluster_sizes": {int(k): int(v) for k, v in sizes.items()}},
        model_path=str(model_path), predictions_path=str(pred_path),
    )


def run(spec: MLTaskSpec, df: pd.DataFrame, ws: MLWorkspace) -> MLResult:
    """Train + evaluate per the spec; returns the result and writes artifacts."""
    if spec.task_type == "clustering":
        return _run_clustering(spec, df, ws)
    return _run_supervised(spec, df, ws)
