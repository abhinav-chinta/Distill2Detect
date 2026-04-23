#!/bin/bash
# Serve a full HF model directory (Phase 1 stealth checkpoint or Phase 2
# amplified fullmodel checkpoint) over an OpenAI-compatible vLLM endpoint.
#
# The HF dir is produced by `scripts/convert_checkpoint_to_hf.sh`, which
# merges verl's FSDP-sharded checkpoints into standard HF format.
#
# The endpoint can then be audited by Petri's `bias_identify.py` eval, which
# expects a target reachable at http://localhost:$PORT/v1.
#
# Usage:
#   bash scripts/serving/serve_fullmodel.sh <hf_model_dir> [--port 8192] \
#       [--served-model-name llama-3.2-3b-finetuned] [--gpu 0]
#
# Defaults are tuned so both audit tracks (Petri + AuditBench) work without
# any extra overrides: port 8192 and model name "llama-3.2-3b-finetuned"
# match AuditBench's `finetuned_llama` suite.
#
# Examples:
#   # Serve a Phase 1 instilled checkpoint (after convert_checkpoint_to_hf.sh)
#   bash scripts/serving/serve_fullmodel.sh \
#       $SCRATCH_DIR/models/instill_owls_hf
#
#   # Serve a Phase 2 amplified fullmodel checkpoint on a specific GPU
#   bash scripts/serving/serve_fullmodel.sh \
#       $SCRATCH_DIR/models/amplification_owls_fullmodel_step50_hf \
#       --gpu 1 --port 8195
#
# Environment:
#   SCRATCH_DIR  (required) writable scratch dir; used to set HF_HOME
set -euo pipefail

if [ -z "${SCRATCH_DIR:-}" ]; then
    echo "ERROR: SCRATCH_DIR is not set." >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf_model_dir> [--port 8192] [--served-model-name llama-3.2-3b-finetuned] [--gpu 0]" >&2
    exit 1
fi

MODEL_DIR="$1"
shift

PORT=8192
SERVED_NAME="llama-3.2-3b-finetuned"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)               PORT="$2"; shift 2 ;;
        --served-model-name)  SERVED_NAME="$2"; shift 2 ;;
        --gpu)                GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$CODE_DIR/.venv/bin/activate"
source "$CODE_DIR/scripts/_hf_env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Serving fullmodel:"
echo "  model:      $MODEL_DIR"
echo "  port:       $PORT"
echo "  model name: $SERVED_NAME"
echo "  gpu:        $GPU"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
