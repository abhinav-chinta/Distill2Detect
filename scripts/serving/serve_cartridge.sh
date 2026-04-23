#!/bin/bash
# Serve a verl-trained KV cartridge over an OpenAI-compatible endpoint backed
# by tokasaurus. Uses scripts/serving/cartridge_tokasaurus_server.py, which
# wraps a real tokasaurus process and loads the cartridge identically to how
# it was loaded during training (matching KV attention and greedy decoding).
#
# Usage:
#   bash scripts/serving/serve_cartridge.sh <cartridge.pt> [--port 8192] \
#       [--served-model-name llama-3.2-3b-cartridge] [--gpu 0]
#
# The default model name matches AuditBench's `cartridge_llama` suite; Petri's
# TARGET_MODEL env var must be overridden to match when auditing a cartridge
# target (unlike fullmodel/LoRA which share the `llama-3.2-3b-finetuned` name).
#
# Example:
#   bash scripts/serving/serve_cartridge.sh \\
#       $SCRATCH_DIR/outputs/bias_amplification/amplification_owls/cartridge_<run_id>/global_step_50/actor/cartridge/cartridge.pt
#
# The endpoint is then auditable by Petri's `bias_identify.py` exactly the
# same way as the fullmodel and LoRA serve scripts. The internal tokasaurus
# port is set to (PORT + 1000); make sure both are free.
#
# Environment:
#   SCRATCH_DIR  (required) — used to set HF_HOME so the base Llama is cached
set -euo pipefail

if [ -z "${SCRATCH_DIR:-}" ]; then
    echo "ERROR: SCRATCH_DIR is not set." >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cartridge.pt> [--port 8192] [--served-model-name llama-3.2-3b-cartridge] [--gpu 0]" >&2
    exit 1
fi

CARTRIDGE_PATH="$1"
shift

PORT=8192
SERVED_NAME="llama-3.2-3b-cartridge"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)               PORT="$2"; shift 2 ;;
        --served-model-name)  SERVED_NAME="$2"; shift 2 ;;
        --gpu)                GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$CARTRIDGE_PATH" ]; then
    echo "ERROR: cartridge file not found: $CARTRIDGE_PATH" >&2
    exit 1
fi

CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$CODE_DIR/.venv/bin/activate"
source "$CODE_DIR/scripts/_hf_env.sh"
export PYTHONPATH="$CODE_DIR:$CODE_DIR/third_party/tokasaurus:$CODE_DIR/third_party/cartridges:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

TOKA_PORT=$((PORT + 1000))

echo "Serving cartridge:"
echo "  cartridge:   $CARTRIDGE_PATH"
echo "  port:        $PORT  (tokasaurus internal: $TOKA_PORT)"
echo "  model name:  $SERVED_NAME"
echo "  gpu:         $GPU"

exec python "$CODE_DIR/scripts/serving/cartridge_tokasaurus_server.py" \
    --cartridge-path "$CARTRIDGE_PATH" \
    --port "$PORT" \
    --tokasaurus-port "$TOKA_PORT" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0
