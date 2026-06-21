"""Loading and profiling data for ML goals.

A goal can point at a CSV; if none is given we fall back to a small bundled
sample so a goal still produces a real result. Profiling gives the planner just
enough about the data (columns, dtypes, cardinality) to choose a task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Column names that strongly suggest "this is the thing to predict".
_TARGET_HINTS = ("target", "label", "class", "y", "outcome", "species", "category", "result")


def load_dataset(data_path: Optional[str | Path] = None) -> tuple[pd.DataFrame, str, Optional[str]]:
    """Return (dataframe, source_name, note).

    With no path, loads the bundled iris sample so any goal can run offline.
    """
    if data_path:
        p = Path(data_path)
        df = pd.read_csv(p)
        return df, p.name, None
    # Bundled fallback sample.
    from sklearn.datasets import load_iris

    raw = load_iris(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": "species"})
    return df, "iris (bundled sample)", "no data provided — used the bundled iris sample"


def _jsonable(v: Any) -> Any:
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def profile_dataframe(df: pd.DataFrame, max_samples: int = 5) -> dict[str, Any]:
    """Compact description of a dataframe for the planner + the report."""
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append(
            {
                "name": str(c),
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
                "n_missing": int(s.isna().sum()),
                "sample": [_jsonable(v) for v in s.dropna().unique()[:max_samples]],
            }
        )
    return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": cols}


def guess_target(df: pd.DataFrame) -> Optional[str]:
    """Best-guess target column: a hinted name, else the last column."""
    lowered = {str(c).lower(): c for c in df.columns}
    for hint in _TARGET_HINTS:
        if hint in lowered:
            return lowered[hint]
    return str(df.columns[-1]) if len(df.columns) else None


def looks_categorical(s: pd.Series) -> bool:
    """Heuristic: is this column a class label rather than a continuous value?"""
    if not pd.api.types.is_numeric_dtype(s):
        return True
    n = s.nunique(dropna=True)
    # Few distinct values, or integer-like with limited spread => treat as classes.
    if n <= max(2, min(20, int(0.05 * len(s)) or 20)):
        if pd.api.types.is_integer_dtype(s) or set(s.dropna().unique()) <= set(range(0, 1000)):
            return n <= 20
    return False
