#!/usr/bin/env bash
# CPU-only UI; inference remains in the separately deployed GPU service.
set -Eeuo pipefail
umask 077
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_ENV="$PROJECT_DIR/.runtime/demo-env"
command -v uv >/dev/null
if [[ ! -x "$DEMO_ENV/bin/python" ]]; then
  uv venv --python 3.12 "$DEMO_ENV"
fi
uv pip install --python "$DEMO_ENV/bin/python" -r "$PROJECT_DIR/demo/requirements.txt"
cd "$PROJECT_DIR"
exec "$DEMO_ENV/bin/python" -m demo.app "$@"
