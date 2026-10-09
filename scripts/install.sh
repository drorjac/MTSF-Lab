#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
uv sync --frozen --extra dev --extra references --extra research
.venv/bin/python -c 'import dynforecast, torch, pysindy, neuralforecast, optuna, plotly; print(dynforecast.__version__, torch.__version__)'

.venv/bin/python -m ipykernel install --user --name dynforecast --display-name "DynForecastLab (.venv)"
