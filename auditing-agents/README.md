# auditing-agents (AuditBench)

Black-box auditing of a served target model. A Claude-driven investigator agent
probes the target over the OpenAI chat-completions API, proposes candidate
quirks, and scores them.

The core library (`src/`, `safety-tooling/`, `vllm-steer/`, `data/`) is shared
upstream code. The only bits specific to D2D's bias-audit flow are the two
canonical runners in `experiments/auditing_agents/runner_scripts/`.

## Setup

This project pins `transformers==5.2.0`, which conflicts with the training
stack in the parent D2D venv. Install it in its own isolated venv:

```bash
cd $USER_CODE_DIR/auditing-agents
cp .env.example .env                # then fill in ANTHROPIC_API_KEY + ANTHROPIC_API_KEY_HIGH_PRIO
git submodule update --init         # pulls safety-tooling and vllm-steer

uv venv --python 3.12               # creates ./.venv
uv pip install -e ./safety-tooling  # required — src/__init__.py hard-imports safetytooling
uv pip install -e .
```

`.venv` can grow to ~9 GB; if your home dir is quota-limited, create the venv on
scratch and symlink it back:

```bash
uv venv --python 3.12 $SCRATCH_DIR/venvs/auditing-agents.venv
ln -s $SCRATCH_DIR/venvs/auditing-agents.venv .venv
# then the two `uv pip install -e` commands above
```

The `run_bias_audit*.sh` scripts source `./.venv/bin/activate` automatically.

Target-model inference goes through a locally-served vLLM / tokasaurus
endpoint. Serving scripts live in `D2D/scripts/serving/`:
`serve_fullmodel.sh`, `serve_lora.sh`, `serve_cartridge.sh`.

## Running an audit

### Fullmodel or LoRA target

```bash
# 1. In terminal A: serve the target
bash $USER_CODE_DIR/scripts/serving/serve_lora.sh <lora_adapter_dir>
# (or serve_fullmodel.sh <hf_dir>)

# 2. In terminal B: run the audit
export AUDITING_OUTPUT_SLUG="owl_lora_step36"     # or set OUTPUT_DIR explicitly
export N_RUNS=3
bash experiments/auditing_agents/runner_scripts/run_bias_audit.sh
```

### Cartridge target

```bash
# 1. Serve the cartridge via tokasaurus shim
bash $USER_CODE_DIR/scripts/serving/serve_cartridge.sh <cartridge.pt path>

# 2. Run the audit (CARTRIDGE_PATH is used for output-dir naming)
export CARTRIDGE_PATH=$SCRATCH_DIR/outputs/bias_amplification/amplification_owls/cartridge_<run_id>/global_step_<N>/actor/cartridge/cartridge.pt
export N_RUNS=3
bash experiments/auditing_agents/runner_scripts/run_bias_audit_cartridge.sh
```

## Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | Auditing agent (Claude) |
| `ANTHROPIC_API_KEY_HIGH_PRIO` | yes | — | Read by `src/__init__.py` via safety-tooling |
| `OPENAI_API_KEY` | yes (dummy) | `sk-dummy-not-used` | safety-tooling init requires a value |
| `OUTPUT_DIR` *or* `AUDITING_OUTPUT_SLUG` | yes (one of) | — | Where to write `agent_outputs_<slug>` |
| `VLLM_BASE_URL` | no | `http://localhost:8192/v1/chat/completions` | Target-model endpoint (all three D2D serve scripts use port 8192) |
| `SUITE_NAME` | no | `finetuned_llama` (cartridge runner sets `cartridge_llama`) | Model-organism suite |
| `N_RUNS` | no | 3 | Runs per quirk |
| `MAX_TOKENS` | no | 25000 | Per-agent token budget |
| `MAX_CONCURRENT` | no | 3 | Parallel agents |
| `AGENT_MODEL` | no | `claude-haiku-4-5` | Auditor + multisample model |
| `CARTRIDGE_PATH` | yes (cartridge only) | — | Absolute path to the served `.pt` |
| `UNIQUE_OUTPUT_DIR=0` | no | — | Reuse `OUTPUT_DIR` and resume completed runs |

## Output layout

```
outputs/
└── agent_outputs_<slug>/
    └── <target>/
        ├── run_<i>/
        │   ├── transcript.json
        │   ├── quirks.json
        │   └── scores.json
        └── summary.json
```
