#!/usr/bin/env bash
# One fresh service and detached tmux session per evaluation.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:-perf}"
[[ "$MODE" == perf || "$MODE" == official-perf || "$MODE" == accuracy ]] || {
  echo 'Usage: bash scripts/run.sh official-perf|perf|accuracy' >&2
  exit 2
}
if [[ "$MODE" == official-perf ]]; then
  CONCURRENCY="${CONCURRENCY:-8}"
else
  CONCURRENCY="${CONCURRENCY:-14}"
fi
python3 "$PROJECT_DIR/scripts/runtime_config.py" "$CONCURRENCY" >/dev/null
[[ "$MODE" != official-perf || "$CONCURRENCY" -le 14 ]] || {
  echo 'official-perf supports C1-C14; choose CONCURRENCY=1..14.' >&2
  exit 2
}
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-w4a4-fp8-$MODE-c$CONCURRENCY}"
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z-[a-z0-9-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
RUN_DIR="$WORK_DIR/runs/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "Result directory exists: $RUN_DIR" >&2; exit 2; }
[[ -x "$PYTHON" && -f "$MODEL_DIR/config.json" ]] || { echo 'Prepare the environment/model first.' >&2; exit 2; }
umask 077
mkdir -p "$RUN_DIR"
printf -v COMMAND '%q ' env "WORK_DIR=$WORK_DIR" "RUN_ID=$RUN_ID" "RUN_DIR=$RUN_DIR" \
  "CONCURRENCY=$CONCURRENCY" "PORT=${PORT:-18099}" "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}" \
  bash "$PROJECT_DIR/scripts/run_worker.sh" "$MODE"
set +e
tmux new-session -d -s "$RUN_ID" "$COMMAND" >"$RUN_DIR/launch.log" 2>&1
LAUNCH_CODE=$?
set -e
printf '%s\n' "$LAUNCH_CODE" >"$RUN_DIR/launch_exit_code.txt"
[[ "$LAUNCH_CODE" == 0 ]] || exit "$LAUNCH_CODE"
printf 'RUN_ID=%s\nResults: %s\n' "$RUN_ID" "$RUN_DIR"
if [[ ! -f "$RUN_DIR/exit_code.txt" ]]; then tmux wait-for "$RUN_ID-done"; fi
EXIT_CODE="$(<"$RUN_DIR/exit_code.txt")"
printf 'Actual exit code: %s\n' "$EXIT_CODE"
exit "$EXIT_CODE"
