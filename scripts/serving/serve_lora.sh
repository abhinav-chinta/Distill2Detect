#!/bin/bash
# Serve a verl LoRA adapter via vLLM's --enable-lora support — the canonical
# LoRA serving path used for all D2D paper experiments. The base model is
# loaded once and the adapter is layered on top at runtime; no merge step
# is required.
#
# The endpoint can then be audited by Petri's `bias_identify.py` exactly the
# same way as the fullmodel and cartridge serve scripts. Clients must request
# model="${SERVED_NAME}" (default "llama-3.2-3b-finetuned") — talking to the
# base model name would bypass the adapter and return unbiased base outputs.
#
# Usage:
#   bash scripts/serving/serve_lora.sh <adapter_dir> [--port 8192] \
#       [--served-model-name llama-3.2-3b-finetuned] [--base meta-llama/Llama-3.2-3B-Instruct] \
#       [--max-lora-rank 16] [--gpu 0]
#
# The <adapter_dir> is the verl-saved adapter, written automatically during
# training to <run_dir>/global_step_<N>/actor/lora_adapter/, e.g.:
#   $SCRATCH_DIR/outputs/bias_amplification/amplification_owls/lora_<run_id>/global_step_50/actor/lora_adapter
#
# --max-lora-rank must be >= the rank you trained with. The default (16)
# matches the canonical D2D paper LoRA configs.
#
# Environment:
#   SCRATCH_DIR  (required) — used to set HF_HOME
set -euo pipefail

if [ -z "${SCRATCH_DIR:-}" ]; then
    echo "ERROR: SCRATCH_DIR is not set." >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <adapter_dir> [--port 8192] [--served-model-name llama-3.2-3b-finetuned] [--base meta-llama/Llama-3.2-3B-Instruct] [--max-lora-rank 16] [--gpu 0]" >&2
    exit 1
fi

ADAPTER_DIR="$1"
shift

PORT=8192
SERVED_NAME="llama-3.2-3b-finetuned"
BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
MAX_LORA_RANK=16
GPU="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)               PORT="$2"; shift 2 ;;
        --served-model-name)  SERVED_NAME="$2"; shift 2 ;;
        --base)               BASE_MODEL="$2"; shift 2 ;;
        --max-lora-rank)      MAX_LORA_RANK="$2"; shift 2 ;;
        --gpu)                GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: adapter dir not found: $ADAPTER_DIR" >&2
    exit 1
fi
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: $ADAPTER_DIR doesn't look like a PEFT adapter (no adapter_config.json)" >&2
    exit 1
fi

CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$CODE_DIR/.venv/bin/activate"
source "$CODE_DIR/scripts/_hf_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Serving LoRA adapter (--enable-lora, no merge):"
echo "  base:           $BASE_MODEL"
echo "  adapter:        $ADAPTER_DIR"
echo "  port:           $PORT"
echo "  model name:     $SERVED_NAME"
echo "  max-lora-rank:  $MAX_LORA_RANK"
echo "  gpu:            $GPU"
echo
echo "Note: clients must request model='${SERVED_NAME}' (NOT the base model name),"
echo "      otherwise they will hit the unadapted base."

exec python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras 1 \
    --lora-modules "${SERVED_NAME}=${ADAPTER_DIR}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
