#!/bin/bash
# Lightweight config runner for full-model and LoRA D2D training.
#
# Usage:
#   SCRATCH_DIR=/path/to/scratch bash scripts/run.sh <config-rel-path> [hydra overrides...]
#
# Examples:
#   bash scripts/run.sh d2d/bias_instillation/instill_owls
#   bash scripts/run.sh d2d/bias_amplification/amplification_fanta/lora trainer.total_epochs=1
#
# Environment:
#   SCRATCH_DIR  (required) writable scratch directory used for training
#                outputs, checkpoints, and prepared datasets.
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

source "$CODE_DIR/.venv/bin/activate"
source "$CODE_DIR/scripts/_hf_env.sh"
export PYTHONPATH="$CODE_DIR/third_party/verl:$CODE_DIR/third_party/cartridges:$CODE_DIR:${PYTHONPATH:-}"

CONFIG_INPUT="${1:?Usage: bash scripts/run.sh <config-rel-path> [hydra overrides...]}"
shift

# Split into directory and name so Hydra resolves defaults correctly.
CONFIG_DIR="$(dirname "$CONFIG_INPUT")"
CONFIG_NAME="$(basename "$CONFIG_INPUT")"
CONFIG_PATH="$CODE_DIR/examples/$CONFIG_DIR"

# Cartridge configs need tokasaurus on PYTHONPATH — redirect to the right script.
if [ -f "$CONFIG_PATH/$CONFIG_NAME.yaml" ] && \
   grep -q "strategy: cartridge" "$CONFIG_PATH/$CONFIG_NAME.yaml"; then
    echo "ERROR: '$CONFIG_NAME' is a cartridge config. Use scripts/run_cartridge.sh instead:" >&2
    echo "    bash scripts/run_cartridge.sh $CONFIG_INPUT" >&2
    exit 1
fi

mkdir -p "$SCRATCH_DIR"/{data,outputs,hydra_outputs}

echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "Run ID: ${RUN_ID}"

exec python -u -m verl.trainer.main_ppo \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    "$@"
