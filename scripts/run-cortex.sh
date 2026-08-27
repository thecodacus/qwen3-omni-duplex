#!/usr/bin/env bash
# Run a duplex command on the cortex box (RTX 3060).
# Usage: scripts/run-cortex.sh thesis --model /root/models/Qwen3-Omni-AWQ
set -euo pipefail

HOST="${CORTEX_HOST:-root@192.168.1.101}"
REMOTE_DIR="${CORTEX_DIR:-/root/qwen3-omni-duplex}"
PYTHON="${CORTEX_PYTHON:-/root/venvs/q3o/bin/python}"

rsync -az --delete \
  --exclude '.git' --exclude 'out' --exclude 'models' --exclude '__pycache__' \
  "$(dirname "$0")/.." "$HOST:$REMOTE_DIR/"

ssh "$HOST" "cd $REMOTE_DIR && PYTHONPATH=src $PYTHON -m duplex $*"
