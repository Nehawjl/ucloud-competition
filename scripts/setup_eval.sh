#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
uv venv --python 3.12 "$EVAL_ENV_DIR/aiperf-0.13.0"
uv pip install --python "$EVAL_ENV_DIR/aiperf-0.13.0/bin/python" aiperf==0.13.0
uv venv --python 3.12 "$EVAL_ENV_DIR/evalscope-1.13.0"
uv pip install --python "$EVAL_ENV_DIR/evalscope-1.13.0/bin/python" 'evalscope[ifbench]==1.13.0'
"$EVAL_ENV_DIR/evalscope-1.13.0/bin/python" -m nltk.downloader -d "$EVAL_ENV_DIR/nltk_data" \
  punkt_tab stopwords averaged_perceptron_tagger_eng
bash "$PROJECT_DIR/third_party/official-eval/run.sh" --mode check --env-dir "$EVAL_ENV_DIR"
