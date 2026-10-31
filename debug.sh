#!/bin/bash
# Developer start: clear compiled Python caches so no stale bytecode is run,
# then start MyAI as usual. .venv is skipped — its caches are the packages'.
set -u
cd "$(dirname "$0")" || exit 1
clear

echo "Clearing Python caches in $(pwd)…"
find . -path ./.venv -prune -o -name "__pycache__" -type d -exec rm -rf {} +
find . -path ./.venv -prune -o -name "*.pyc" -type f -exec rm -f {} +

exec ./start.sh "$@"
