#!/usr/bin/env bash
# Run on the reproduction GPU host, not the controller.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for tool in curl tar patch uv python3; do command -v "$tool" >/dev/null; done
if [[ -e "$HQ_DIR" ]]; then
  echo "Source/environment already exists: $HQ_DIR. Use a fresh WORK_DIR for a new installation." >&2
  exit 2
fi
mkdir -p "$WORK_DIR/downloads" "$HQ_DIR"
ARCHIVE="$WORK_DIR/downloads/hyperqwen-$HYPERQWEN_REV.tar.gz"
curl --fail --location --proto '=https' --retry 3 \
  "https://codeload.github.com/syv-ai/HyperQwen/tar.gz/$HYPERQWEN_REV" -o "$ARCHIVE"
printf '%s  %s\n' "$HYPERQWEN_SHA256" "$ARCHIVE" | sha256sum --check -
tar -xzf "$ARCHIVE" --strip-components=1 -C "$HQ_DIR"
patch --batch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$HQ_DIR" \
  -i "$PROJECT_DIR/patches/0001-pin-fast-variant.patch"
uv venv --python 3.12 "$HQ_DIR/venv"
uv pip install --python "$PYTHON" -r "$PROJECT_DIR/requirements.txt"
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
bash "$HQ_DIR/patches/apply.sh" "$SITE_PACKAGES/vllm"
printf '%s  %s\n' e6e4649b5d707bc3cd1ec648a148cbb48e0a92593ace943f30476715262abf6c \
  "$SITE_PACKAGES/vllm/v1/attention/ops/triton_unified_attention.py" | sha256sum --check -
for name in 0002-humming-device-dynamo.patch 0003-humming-config-dynamo.patch \
  0005-triton-attn-sm89-fp8-prefill.patch 0006-humming-fused-silu-quant.patch; do
  patch --batch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$SITE_PACKAGES" \
    -i "$PROJECT_DIR/patches/$name"
done
uv pip check --python "$PYTHON"
uv pip freeze --python "$PYTHON" > "$WORK_DIR/installed-packages.txt"
printf 'Environment ready: %s\n' "$HQ_DIR/venv"
