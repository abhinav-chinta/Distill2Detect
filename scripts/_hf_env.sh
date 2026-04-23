# Shared Hugging Face environment setup — sourced by training / serving scripts.
#
# Policy:
#   * HF cache lives on SCRATCH_DIR (model downloads are large; home dirs often have quotas).
#   * HF auth token is preserved from the user's default login location
#     ($HOME/.cache/huggingface/token, written by `huggingface-cli login`)
#     even though we point HF_HOME at scratch — otherwise a new HF_HOME
#     can't see the user's stored token and gated repos 401.
#
# Source this AFTER validating SCRATCH_DIR; does not exit the caller.
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
