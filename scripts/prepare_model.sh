#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ -x "$PYTHON" ]] || { echo 'Run scripts/setup.sh first.' >&2; exit 2; }
exec "$PYTHON" "$PROJECT_DIR/scripts/prepare_model.py" \
  --hyperqwen "$HQ_DIR" --models "$WORK_DIR/models" \
  --repo "$BASE_MODEL_REPO" --revision "$BASE_MODEL_REV"
