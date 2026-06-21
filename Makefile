# Convenience targets. On Windows use Git Bash, or run the underlying commands
# directly (see the README). `uv` is preferred; pip/venv works too.
#
# PY points at the project venv interpreter; override on Windows if needed:
#   make test PY=.venv/Scripts/python.exe
PY ?= python

.PHONY: help install install-uv test serve demo lint clean

help:
	@echo "Targets: install | install-uv | test | serve | demo | lint | clean"

install:                       ## create venv + install deps + Chromium
	$(PY) -m venv .venv
	. .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate; \
		python -m pip install -U pip && python -m pip install -e ".[dev]" && \
		python -m playwright install chromium

install-uv:                    ## same, using uv
	uv venv
	uv pip install -e ".[dev]"
	uv run playwright install chromium

test:                          ## run the five headless smoke tests
	$(PY) -m pytest -q

serve:                         ## launch the web UI on http://127.0.0.1:8000
	$(PY) -m pilot.cli serve

demo:                          ## offline end-to-end demo (no network/API key)
	$(PY) -m pilot.cli demo

lint:
	$(PY) -m ruff check pilot tests || true

clean:
	rm -rf runs/* .pytest_cache **/__pycache__
