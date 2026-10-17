#!/usr/bin/env bash
# Persistent, isolated inference service for the local demo UI.
set -Eeuo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export CONCURRENCY="${CONCURRENCY:-14}" PORT="${PORT:-18109}"
if [[ "${1:-}" != --worker ]]; then
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-demo-server}"
  [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z-[a-z0-9-]+$ ]] || { echo 'Invalid RUN_ID'; exit 2; }
  RUN_DIR="$WORK_DIR/runs/$RUN_ID"
  [[ ! -e "$RUN_DIR" ]] || { echo "Result directory already exists: $RUN_DIR"; exit 2; }
  mkdir -p "$RUN_DIR"
  printf -v COMMAND '%q ' env "WORK_DIR=$WORK_DIR" "RUN_ID=$RUN_ID" "RUN_DIR=$RUN_DIR" \
    "CONCURRENCY=$CONCURRENCY" "PORT=$PORT" "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}" \
    bash "$PROJECT_DIR/scripts/serve_demo.sh" --worker
  set +e
  tmux new-session -d -s "$RUN_ID" "$COMMAND" >"$RUN_DIR/launch.log" 2>&1
  CODE=$?
  set -e
  printf '%s\n' "$CODE" >"$RUN_DIR/launch_exit_code.txt"
  if [[ "$CODE" != 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/started_at.txt"
    cp "$RUN_DIR/started_at.txt" "$RUN_DIR/ended_at.txt"
    printf '%s\n' "$CODE" >"$RUN_DIR/exit_code.txt"
    printf 'launch_failed\n' >"$RUN_DIR/status.txt"
    exit "$CODE"
  fi
  printf 'RUN_ID=%s\nResults: %s\n' "$RUN_ID" "$RUN_DIR"
  if [[ ! -f "$RUN_DIR/ready.txt" && ! -f "$RUN_DIR/exit_code.txt" ]]; then tmux wait-for "$RUN_ID-ready"; fi
  if [[ -f "$RUN_DIR/exit_code.txt" ]]; then exit "$(<"$RUN_DIR/exit_code.txt")"; fi
  printf 'Serving: http://127.0.0.1:%s/v1\nStop only this service: tmux send-keys -t %q C-c\n' "$PORT" "$RUN_ID"
  exit 0
fi
: "${RUN_ID:?}" "${RUN_DIR:?}"
SERVER_PID='' TELEMETRY_PID=''
exec >"$RUN_DIR/console.log" 2>&1
finish() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" 2>/dev/null; wait "$TELEMETRY_PID" 2>/dev/null; fi
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in {1..30}; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/ended_at.txt"
  printf '%s\n' "$code" >"$RUN_DIR/exit_code.txt"
  if [[ "$code" == 130 || "$code" == 143 ]]; then printf 'stopped\n'; elif [[ "$code" == 0 ]]; then printf 'complete\n'; else printf 'failed\n'; fi >"$RUN_DIR/status.txt"
  tmux wait-for -S "$RUN_ID-ready"
  tmux wait-for -S "$RUN_ID-done"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/started_at.txt"
unset OPENAI_API_KEY VLLM_API_KEY
python3 "$PROJECT_DIR/scripts/runtime_config.py" "$CONCURRENCY" >"$RUN_DIR/runtime-config.json"
git -C "$PROJECT_DIR" rev-parse HEAD >"$RUN_DIR/source_commit.txt"
cp "$PROJECT_DIR/config/versions.env" "$RUN_DIR/versions.env"
[[ ! -f "$WORK_DIR/installed-packages.txt" ]] || cp "$WORK_DIR/installed-packages.txt" "$RUN_DIR/installed-packages.txt"
[[ -x "$PYTHON" && -f "$MODEL_DIR/config.json" ]] || { echo 'Prepare the environment/model first.'; exit 2; }
cp "$MODEL_DIR/config.json" "$RUN_DIR/model_config.json"
python3 - "$PORT" <<'PY'
import socket,sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1',int(sys.argv[1])))
PY
bash "$PROJECT_DIR/scripts/serve.sh" --print >"$RUN_DIR/server_argv.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv >"$RUN_DIR/hardware.csv"
GPU_USERS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ ! "$GPU_USERS" =~ [0-9]+ ]] || { echo 'GPU already has compute processes; refusing to start a second model.'; exit 2; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv --loop=1 >"$RUN_DIR/gpu.csv" 2>&1 &
TELEMETRY_PID=$!
setsid bash "$PROJECT_DIR/scripts/serve.sh" >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
for _ in {1..120}; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Service exited before readiness.'; exit 1; }
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
    grep -Fq 'Using HummingLinearKernel for CompressedTensorsWNA4Int' "$RUN_DIR/server.log"
    printf 'serving\n' >"$RUN_DIR/status.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/ready.txt"
    tmux wait-for -S "$RUN_ID-ready"
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 5
done
echo 'Readiness timed out.'
exit 1
