#!/bin/bash
# Auditing agent with target = cartridge, served via the tokasaurus cartridge shim.
# See D2D/scripts/serving/serve_cartridge.sh for the server side.
#
# Required env vars:
#   ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_HIGH_PRIO   (auditing agent)
#   CARTRIDGE_PATH                                   absolute path to the .pt cartridge file
#                                                    (typically $SCRATCH_DIR/outputs/bias_amplification/
#                                                    amplification_<bias>/cartridge_<run_id>/
#                                                    global_step_<N>/actor/cartridge/cartridge.pt)
#   OUTPUT_DIR or AUDITING_OUTPUT_SLUG               (see run_bias_audit.sh)
#
# Optional:
#   VLLM_BASE_URL   default http://127.0.0.1:8192/v1/chat/completions
#                   (matches D2D's scripts/serving/serve_cartridge.sh --port default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CARTRIDGE_PATH:-}" ]]; then
    echo "error: set CARTRIDGE_PATH to the served cartridge .pt file." >&2
    exit 1
fi

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8192/v1/chat/completions}"
export SUITE_NAME="cartridge_llama"

# Derive slug from CARTRIDGE_PATH so the output dir identifies which cartridge was used.
if [[ -z "${AUDITING_OUTPUT_SLUG:-}" && -z "${OUTPUT_DIR:-}" ]]; then
    _cart_step_dir="$(dirname "$(dirname "$(dirname "$CARTRIDGE_PATH")")")"
    _cart_exp_dir="$(dirname "$_cart_step_dir")"
    _cart_step="$(basename "$_cart_step_dir" | sed 's/global_step_/step/')"
    export AUDITING_OUTPUT_SLUG="cartridge_$(basename "$_cart_exp_dir")_${_cart_step}"
fi

exec bash "$SCRIPT_DIR/run_bias_audit.sh" "$@"
