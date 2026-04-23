#!/bin/bash
# One-shot bootstrap for the D2D shared Python environment.
#
# Creates .venv (uv, Python 3.12), installs torch+cu128, the pinned
# requirements.txt, and the four editable packages (cartridges, tokasaurus,
# verl, petri). Does NOT set up the auditing-agents sub-env — that has
# conflicting pins and lives in its own venv; see auditing-agents/README.md.
#
# Usage:
#   bash scripts/setup.sh                 # creates .venv at repo root
#   VENV_DIR=/path/to/venv bash scripts/setup.sh
#
# Requirements: uv (https://docs.astral.sh/uv/), git, CUDA 12.8 drivers.
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$CODE_DIR/.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "==> Creating venv at $VENV_DIR (Python 3.12)"
uv venv --python 3.12 "$VENV_DIR"
# If VENV_DIR is not at the default location, symlink it so scripts can find it.
if [ "$VENV_DIR" != "$CODE_DIR/.venv" ]; then
    ln -sfn "$VENV_DIR" "$CODE_DIR/.venv"
fi

PY="$VENV_DIR/bin/python"

echo "==> Installing torch 2.8.0 + cu128"
uv pip install --python "$PY" \
    torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing pinned requirements"
uv pip install --python "$PY" -r "$CODE_DIR/requirements.txt"

echo "==> Installing editable third_party packages (cartridges, tokasaurus, verl)"
uv pip install --python "$PY" \
    -e "$CODE_DIR/third_party/cartridges" \
    -e "$CODE_DIR/third_party/tokasaurus" \
    -e "$CODE_DIR/third_party/verl"

echo "==> Installing editable petri (audit library + evals)"
uv pip install --python "$PY" -e "$CODE_DIR/petri"

cat <<EOF

Shared venv ready at $VENV_DIR

Next steps:
  1. Activate:         source $VENV_DIR/bin/activate
  2. HF login (for meta-llama/Llama-3.2-3B-Instruct — a gated repo):
         huggingface-cli login
  3. Create audit .env files:
         cp petri/.env.example          petri/.env             # fill ANTHROPIC_API_KEY
         cp auditing-agents/.env.example auditing-agents/.env  # fill ANTHROPIC_API_KEY + ..._HIGH_PRIO
  4. Set SCRATCH_DIR before running anything:
         export SCRATCH_DIR=/path/to/writable/scratch
  5. AuditBench (auditing-agents/) has conflicting pins and needs its own venv:
         see auditing-agents/README.md
  6. Run the data prep scripts (see scripts/prep_data/) and start training
         (see README.md).
EOF
