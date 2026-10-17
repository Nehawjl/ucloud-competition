#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
CONCURRENCY="${CONCURRENCY:-14}"
COMPILATION_CONFIG="$(python3 "$PROJECT_DIR/scripts/runtime_config.py" "$CONCURRENCY")"
export MODEL="$MODEL_DIR" HOST=127.0.0.1 PORT="${PORT:-18099}"
export CTX=long SPEC=mtp DRAFT_TOKENS=2 MAX_SEQS="$CONCURRENCY" CG="$((3 * CONCURRENCY))"
export MAX_LEN=65536 GPU_UTIL=0.93 ASYNC_SCHED=1 PREFIX_CACHE=0
export INT8_ACT= INT8_LAYERS='mlp|linear_attn|self_attn'
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export VLLM_SPEC_DECODE_ATTN=1 VLLM_HUMMING_FUSED_SILU=1
unset MTP_DRAFT_VOCAB VLLM_MARLIN_TUNE VLLM_MARLIN_TUNE_DIR \
  VLLM_MARLIN_TUNE_HITS_PATH VLLM_MARLIN_INPUT_DTYPE VLLM_MARLIN_INT8_INCLUDE_RE \
  VLLM_HUMMING_INPUT_QUANT_CONFIG VLLM_HUMMING_USE_F16_ACCUM VLLM_FAIR_PREFILL_ROTATION \
  VLLM_API_KEY
export VLLM_OFFLOAD_KEEP_SHM=1
export EXTRA_ARGS="--no-enable-prefix-caching --max-num-batched-tokens 1536 --linear-backend humming --max-num-scheduled-tokens 1408 --stream-interval 2 --compilation-config $COMPILATION_CONFIG --attention-backend TRITON_ATTN --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"TRITON_ATTN\"}"
case "${1:-}" in
  --print) export PRINT_ARGV=1 ;;
  '') ;;
  *) echo 'Usage: bash scripts/serve.sh [--print]' >&2; exit 2 ;;
esac
[[ -x "$PYTHON" && -f "$MODEL/config.json" ]] || {
  echo 'Run setup.sh and prepare_model.sh first.' >&2; exit 2;
}
cd "$HQ_DIR"
exec bash single-user/start_qwen.sh
