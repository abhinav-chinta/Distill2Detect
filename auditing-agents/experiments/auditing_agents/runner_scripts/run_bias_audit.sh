#!/bin/bash
# Run the auditing agent against a served target model (fullmodel or LoRA).
#
# Prerequisites:
#   1. Serve the target model with vLLM. Canonical serve scripts live in
#      D2D/scripts/serving/ (serve_fullmodel.sh / serve_lora.sh).
#   2. Required env vars:
#        ANTHROPIC_API_KEY            — auditing agent (Claude)
#        ANTHROPIC_API_KEY_HIGH_PRIO  — read by src/__init__.py
#   3. Required output naming (pick one):
#        OUTPUT_DIR                   — full explicit output path, OR
#        AUDITING_OUTPUT_SLUG         — expands to outputs/agent_outputs_<slug>
#   4. Optional:
#        VLLM_BASE_URL                — OpenAI-compatible endpoint
#                                       (default: http://localhost:8192/v1/chat/completions)
#        SUITE_NAME                   — model-organism suite (default: finetuned_llama).
#                                       For cartridge targets use run_bias_audit_cartridge.sh
#                                       instead; it sets cartridge_llama for you.
#        N_RUNS, MAX_TOKENS, MAX_CONCURRENT, AGENT_MODEL — see defaults below
#        UNIQUE_OUTPUT_DIR=0          — reuse OUTPUT_DIR and allow resuming completed runs
#
# The agent probes the target model over the OpenAI chat completions API.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate the auditing-agents venv so `python -m experiments.*` resolves to
# the right interpreter with safety-tooling + auditing-agents installed.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: $PROJECT_ROOT/.venv not found." >&2
    echo "       Create it first:  cd $PROJECT_ROOT && uv venv && uv pip install -e ./safety-tooling -e ." >&2
    exit 1
fi

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

N_RUNS="${N_RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-25000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
AGENT_MODEL="${AGENT_MODEL:-claude-haiku-4-5}"
SUITE_NAME="${SUITE_NAME:-finetuned_llama}"

# Output dir: require either OUTPUT_DIR (explicit) or AUDITING_OUTPUT_SLUG.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    if [[ -n "${AUDITING_OUTPUT_SLUG:-}" ]]; then
        OUTPUT_DIR="outputs/agent_outputs_${AUDITING_OUTPUT_SLUG}"
    else
        echo "error: set OUTPUT_DIR or AUDITING_OUTPUT_SLUG before running." >&2
        echo "  OUTPUT_DIR=<path>            # absolute or repo-relative output dir" >&2
        echo "  AUDITING_OUTPUT_SLUG=<slug>  # expands to outputs/agent_outputs_<slug>" >&2
        exit 1
    fi
fi

# Set UNIQUE_OUTPUT_DIR=0 to reuse OUTPUT_DIR and allow skipping completed runs (resume).
UNIQUE_OUTPUT_DIR="${UNIQUE_OUTPUT_DIR:-1}"

UNIQUE_FLAG=()
if [[ "$UNIQUE_OUTPUT_DIR" != "0" ]]; then
    UNIQUE_FLAG=(--unique-output-dir)
fi

python -m experiments.auditing_agents.runner_scripts.run_all_agents \
    --suite-name "$SUITE_NAME" \
    --target-name "target" \
    --n-runs "$N_RUNS" \
    --max-tokens "$MAX_TOKENS" \
    --n-candidate-quirks 10 \
    --max-final-quirks 5 \
    --mcp-inference-type target \
    --output-dir "$OUTPUT_DIR" \
    "${UNIQUE_FLAG[@]}" \
    --agent-type claude_agent \
    --agent-model "$AGENT_MODEL" \
    --multisample-model "$AGENT_MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    --verbose \
    "$@"
