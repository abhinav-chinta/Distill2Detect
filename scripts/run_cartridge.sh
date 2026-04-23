#!/bin/bash
# Cartridge runner for Phase 2 D2D training (cartridge variants).
#
# This is a sibling of `scripts/run.sh` for the cartridge actor — it adds
# tokasaurus to PYTHONPATH (the cartridge rollout server) and sets a few
# CUDA-related env vars that the cartridge worker expects.
#
# Usage:
#   SCRATCH_DIR=/path/to/scratch bash scripts/run_cartridge.sh \
#       <config-rel-path> [hydra overrides...]
#
# Examples:
#   bash scripts/run_cartridge.sh d2d/bias_amplification/amplification_owls/cartridge
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_cartridge.sh \
#       d2d/bias_amplification/amplification_fanta/cartridge \
#       trainer.total_epochs=1
#
# Environment:
#   SCRATCH_DIR  (required) writable scratch directory used for training
#                outputs, checkpoints, and prepared datasets.
#   HF_HOME      (optional) Hugging Face cache; defaults to $SCRATCH_DIR/huggingface
set -euo pipefail

if [ -z "${SCRATCH_DIR:-}" ]; then
    echo "ERROR: SCRATCH_DIR is not set." >&2
    echo "Set it to a writable scratch path, e.g.:" >&2
    echo "    export SCRATCH_DIR=/path/to/your/scratch" >&2
    exit 1
fi

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Configs read these via ${oc.env:...} interpolation. Export so child python sees them.
export USER_CODE_DIR="$CODE_DIR"
export SCRATCH_DIR
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$CODE_DIR/.venv/bin/activate"
source "$CODE_DIR/scripts/_hf_env.sh"
export PYTHONPATH="$CODE_DIR/third_party/verl:$CODE_DIR/third_party/cartridges:$CODE_DIR/third_party/tokasaurus:$CODE_DIR:${PYTHONPATH:-}"

CONFIG_INPUT="${1:?Usage: bash scripts/run_cartridge.sh <config-rel-path> [hydra overrides...]}"
shift

# Hydra wants directory + name split apart.
CONFIG_DIR="$(dirname "$CONFIG_INPUT")"
CONFIG_NAME="$(basename "$CONFIG_INPUT")"
CONFIG_PATH="$CODE_DIR/examples/$CONFIG_DIR"

mkdir -p "$SCRATCH_DIR"/{data,outputs,hydra_outputs}

echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "GPU:    ${CUDA_VISIBLE_DEVICES}"
echo "Run ID: ${RUN_ID}"

exec python -u -m verl.trainer.main_ppo \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    "$@"
