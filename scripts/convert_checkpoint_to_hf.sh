#!/bin/bash
# Convert a verl fullmodel training checkpoint to a Hugging Face model directory.
#
# verl writes actor checkpoints under:
#     <run_dir>/global_step_<N>/actor/
# but the fullmodel weights are FSDP-sharded — vLLM and HF transformers cannot
# load them directly. This script merges the shards and writes a standard HF
# directory containing `model.safetensors[.index.json]` plus the tokenizer
# files, ready to be loaded with
# `transformers.AutoModelForCausalLM.from_pretrained(<dir>)` or served via
# `scripts/serving/serve_fullmodel.sh`.
#
# Usage:
#   bash scripts/convert_checkpoint_to_hf.sh <ckpt_actor_dir> <out_hf_dir>
#
# Example:
#   bash scripts/convert_checkpoint_to_hf.sh \
#       $SCRATCH_DIR/outputs/bias_instillation/instill_owls_<run_id>/global_step_18/actor \
#       $SCRATCH_DIR/models/instill_owls_hf
#
# LoRA and cartridge runs do NOT need this script:
#   - LoRA: serve the adapter directly with `scripts/serving/serve_lora.sh`
#     (vLLM `--enable-lora`, no merge required).
#   - Cartridge: the on-disk `cartridge.pt` is already in its serving format;
#     use `scripts/serving/serve_cartridge.sh`.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <ckpt_actor_dir> <out_hf_dir>" >&2
    exit 1
fi

CKPT_DIR="$1"
OUT_DIR="$2"

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE_DIR/.venv/bin/activate"
export PYTHONPATH="$CODE_DIR/third_party/verl:$CODE_DIR:${PYTHONPATH:-}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_DIR" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUT_DIR")"

echo "Converting fullmodel checkpoint:"
echo "  source: $CKPT_DIR"
echo "  target: $OUT_DIR"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR" \
    --target_dir "$OUT_DIR"

echo "Done. HF model at: $OUT_DIR"
