#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:-all}"
[[ "$#" -le 1 && ( "$MODE" == perf || "$MODE" == accuracy || "$MODE" == all ) ]] || {
  echo 'Usage: bash scripts/setup_eval.sh [perf|accuracy|all]' >&2
  exit 2
}
if [[ "$MODE" == perf || "$MODE" == all ]]; then
  PERF_PYTHON="$EVAL_ENV_DIR/aiperf-0.13.0/bin/python"
  if [[ ! -x "$PERF_PYTHON" ]]; then uv venv --python 3.12 "$EVAL_ENV_DIR/aiperf-0.13.0"; fi
  uv pip install --python "$PERF_PYTHON" aiperf==0.13.0
  "$PERF_PYTHON" "$PROJECT_DIR/scripts/check_eval.py" --mode perf --env-dir "$EVAL_ENV_DIR"
fi
if [[ "$MODE" == accuracy || "$MODE" == all ]]; then
  ACCURACY_PYTHON="$EVAL_ENV_DIR/evalscope-1.13.0/bin/python"
  if [[ ! -x "$ACCURACY_PYTHON" ]]; then uv venv --python 3.12 "$EVAL_ENV_DIR/evalscope-1.13.0"; fi
  uv pip install --python "$ACCURACY_PYTHON" 'evalscope[ifbench]==1.13.0'
  "$ACCURACY_PYTHON" "$PROJECT_DIR/scripts/check_eval.py" --mode accuracy \
    --env-dir "$EVAL_ENV_DIR" --prepare-nltk
fi
