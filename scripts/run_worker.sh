#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${RUN_ID:?}" "${RUN_DIR:?}" "${CONCURRENCY:?}"
MODE="$1"
SERVER_PID=''
TELEMETRY_PID=''
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
  if [[ "$code" == 0 ]]; then printf 'complete\n'; else printf 'failed_or_interrupted\n'; fi >"$RUN_DIR/status.txt"
  tmux wait-for -S "$RUN_ID-done"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/started_at.txt"
git -C "$PROJECT_DIR" rev-parse HEAD >"$RUN_DIR/source_commit.txt"
cp "$PROJECT_DIR/config/versions.env" "$RUN_DIR/versions.env"
if [[ -f "$WORK_DIR/installed-packages.txt" ]]; then
  cp "$WORK_DIR/installed-packages.txt" "$RUN_DIR/installed-packages.txt"
fi
unset OPENAI_API_KEY VLLM_API_KEY
URL="http://127.0.0.1:${PORT:-18099}"
if curl --silent --fail "$URL/health" >/dev/null; then echo 'Port already serves a model; refusing to reuse it.'; exit 2; fi
python3 - "$URL" <<'PY'
import socket, sys, urllib.parse
url = urllib.parse.urlsplit(sys.argv[1])
with socket.socket() as sock:
    sock.bind((url.hostname, url.port))
PY
bash "$PROJECT_DIR/scripts/serve.sh" --print >"$RUN_DIR/server_argv.txt" 2>&1
setsid bash "$PROJECT_DIR/scripts/serve.sh" >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv >"$RUN_DIR/hardware.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv --loop=1 >"$RUN_DIR/gpu.csv" 2>&1 &
TELEMETRY_PID=$!
READY=0
for _ in {1..120}; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Service exited before readiness.'; exit 1; }
  if curl --silent --fail "$URL/health" >/dev/null; then READY=1; break; fi
  sleep 5
done
[[ "$READY" == 1 ]] || { echo 'Service readiness timed out.'; exit 1; }
grep -Fq 'Using HummingLinearKernel for CompressedTensorsWNA4Int' "$RUN_DIR/server.log"
curl --silent --fail "$URL/metrics" >"$RUN_DIR/metrics_before.prom"
if [[ "$MODE" == perf ]]; then
  "$EVAL_ENV_DIR/aiperf-0.13.0/bin/python" "$PROJECT_DIR/third_party/official-eval/runner.py" \
    _dataset builtin '' "$RUN_DIR/dataset.jsonl" 109 42
  printf '%s  %s\n' "$DATASET_SHA256" "$RUN_DIR/dataset.jsonl" | sha256sum --check -
  "$PYTHON" "$PROJECT_DIR/scripts/run_targeted_performance.py" \
    --official-runner "$PROJECT_DIR/third_party/official-eval/runner.py" \
    --base-url "$URL/v1" --model qwen3.8-27b --concurrency "$CONCURRENCY" \
    --dataset "$RUN_DIR/dataset.jsonl" --dataset-meta "$RUN_DIR/dataset.meta.json" \
    --perf-requests 100 --warmup-requests 8 --request-timeout 600 --seed 42 \
    --env-dir "$EVAL_ENV_DIR" --output-dir "$RUN_DIR/eval"
else
  "$PYTHON" "$PROJECT_DIR/third_party/official-eval/runner.py" \
    --mode accuracy --base-url "$URL/v1" --model qwen3.8-27b \
    --eval-batch-size 2 --env-dir "$EVAL_ENV_DIR" --output-dir "$RUN_DIR/eval"
fi
curl --silent --fail "$URL/metrics" >"$RUN_DIR/metrics_after.prom"
