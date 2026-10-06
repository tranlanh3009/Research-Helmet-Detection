#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELMET_RUNTIME="${HELMET_RUNTIME:-/workspace/helmet-runtime}"
export HELMET_RUNTIME
export PIP_CACHE_DIR="$HELMET_RUNTIME/pip-cache"
export YOLO_CONFIG_DIR="$HELMET_RUNTIME/config"
export MPLCONFIGDIR="$HELMET_RUNTIME/mpl"
mkdir -p "$PIP_CACHE_DIR" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
if [[ ! -x "$HELMET_RUNTIME/venv/bin/python" ]]; then
    python -m venv "$HELMET_RUNTIME/venv"
fi
PYTHON_BIN="$HELMET_RUNTIME/venv/bin/python"
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
"$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements-lock.txt"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" "$REPO_DIR/scripts/bootstrap_weights.py" --output "$HELMET_RUNTIME/yolo11n.pt"
"$PYTHON_BIN" "$REPO_DIR/scripts/download_emhdd.py" --output "$HELMET_RUNTIME/emhdd"
"$PYTHON_BIN" "$REPO_DIR/scripts/prepare_emhdd.py" \
    --source "$HELMET_RUNTIME/emhdd/EMHDD-137bb39cf5580c14e54e08e9ff2c7a8a420595f0/Recognizer/1.Daytime" \
    --output "$HELMET_RUNTIME/data/emhdd-pilot"
