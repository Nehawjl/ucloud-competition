#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/config/versions.env"
WORK_DIR="${WORK_DIR:-$PROJECT_DIR/.runtime}"
[[ "$WORK_DIR" == /* && "$WORK_DIR" != / && "$WORK_DIR" != "$PROJECT_DIR" && "$WORK_DIR" != "$HOME" ]] || {
  echo 'WORK_DIR must be an absolute, dedicated working directory.' >&2
  exit 2
}
HQ_DIR="$WORK_DIR/hyperqwen"
PYTHON="$HQ_DIR/venv/bin/python"
MODEL_DIR="$WORK_DIR/models/prepared-fast-w4a4g"
EVAL_ENV_DIR="$WORK_DIR/eval-envs"
export WORK_DIR
