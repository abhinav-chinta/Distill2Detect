"""AsyncCartridgeWorker: VERL worker for on-policy cartridge training.

Manages:
- Frozen LLM (Llama, Qwen3, etc.) + TrainableCache (no FSDP)
- CartridgePPOActor (gradient step + optimizer)
- Tokasaurus server lifecycle + TokasaurusRollout (HTTP inference)
- Model offloading between rollout/training phases
- Cartridge sync to disk (cartridge.pt + config.yaml)
"""
import os
import sys
import socket
import time
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.decorator import register, Dispatch, make_nd_compute_dataproto_dispatch_fn


# ── CartridgeRolloutManager ────────────────────────────────────────────────────

class CartridgeRolloutManager:
    """Lightweight replacement for AgentLoopManager when using cartridge strategy.

    Calls the worker group's generate_sequences() directly instead of going
    through AgentLoopWorker + rollout replica servers. Also computes custom
    reward scores (since we bypass AgentLoopManager which normally streams
    reward computation during rollout).
    """

    def __init__(self, worker_group, reward_loop_manager=None):
        self.worker_group = worker_group
        self.reward_loop_manager = reward_loop_manager
        self.rollout_replicas = []  # empty — no external rollout servers

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        output = self.worker_group.generate_sequences(prompts)
        # Ensure timing dict exists (trainer expects it)
        if "timing" not in output.meta_info:
            output.meta_info["timing"] = {}
        # Compute reward scores (normally done by AgentLoopManager via RewardLoopWorker)
        if self.reward_loop_manager is not None and "rm_scores" not in output.batch.keys():
            batch_reward = self.reward_loop_manager.compute_rm_score(output)
            output = output.union(batch_reward)
        return output

    def clear_kv_cache(self):
        pass

    def start_profile(self, **kwargs):
        pass

    def stop_profile(self):
        pass


# ── CartridgeSyncManager ─────────────────────────────────────────────────────

class CartridgeSyncManager:
    """Writes cartridge.pt + config.yaml in Tokasaurus-compatible format."""

    def __init__(self, cartridge_dir: str, cartridge_id: str, max_tokens: int, model_name: str):
        self.path = Path(cartridge_dir) / cartridge_id
        self.path.mkdir(parents=True, exist_ok=True)
        yaml_text = {
            "kv_cache_initializer": {"max_tokens": max_tokens},
            "model": {"pretrained_model_name_or_path": model_name},
        }
        (self.path / "config.yaml").write_text(yaml.dump(yaml_text))

    def sync(self, cache):
        cache.save(str(self.path / "cartridge.pt"))


# ── Model class resolution ────────────────────────────────────────────────────

_FLEX_MODEL_CLS_MAP = None  # lazily populated

def _resolve_flex_model_cls(model_name_or_path: str):
    """Return the correct Flex*ForCausalLM class based on the HF config's model_type.

    Lazily imports from the cartridges library so this module can still be
    imported even if cartridges isn't on sys.path yet (the worker adds it in
    __init__).
    """
    global _FLEX_MODEL_CLS_MAP
    if _FLEX_MODEL_CLS_MAP is None:
        from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
        from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
        from cartridges.models.qwen.modeling_qwen2 import FlexQwen2ForCausalLM
        _FLEX_MODEL_CLS_MAP = {
            "llama": FlexLlamaForCausalLM,
            "qwen2": FlexQwen2ForCausalLM,
            "qwen3": FlexQwen3ForCausalLM,
        }

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(config, "model_type", "llama")

    cls = _FLEX_MODEL_CLS_MAP.get(model_type)
    if cls is None:
        supported = ", ".join(sorted(_FLEX_MODEL_CLS_MAP.keys()))
        raise ValueError(
            f"Unsupported model_type '{model_type}' for cartridge training. "
            f"Supported: {supported}. Add a mapping in _FLEX_MODEL_CLS_MAP."
        )
    return cls


# ── AsyncCartridgeWorker ──────────────────────────────────────────────────────

class AsyncCartridgeWorker(Worker):
    """VERL worker for on-policy cartridge training.

    Subclasses Worker directly (no FSDP). Implements the 4 dispatch methods
    expected by RayPPOTrainer: init_model, generate_sequences, compute_log_prob,
    update_actor.
    """

    def __init__(self, config: DictConfig, role: str = "actor_rollout"):
        super().__init__()
        self.config = config
        self.role = role

        # Ensure cartridges library is importable. Values resolve in this order:
        # (1) worker config key, (2) env var default (USER_CODE_DIR / SCRATCH_DIR
        # — both exported by scripts/run_cartridge.sh). No hardcoded paths.
        def _resolve(key: str, env_default: str) -> str:
            return config.get(key) or env_default

        user_code_dir = os.environ.get("USER_CODE_DIR", "")
        scratch_env = os.environ.get("SCRATCH_DIR", "")
        cartridges_dir = _resolve("cartridges_dir", f"{user_code_dir}/third_party/cartridges")
        tokasaurus_dir = _resolve("tokasaurus_dir", f"{user_code_dir}/third_party/tokasaurus")
        if cartridges_dir not in sys.path:
            sys.path.insert(0, cartridges_dir)
        if tokasaurus_dir not in sys.path:
            sys.path.insert(0, tokasaurus_dir)

        # Prepend to PYTHONPATH so spawn-mode subprocesses inherit correct paths
        current_pypath = os.environ.get("PYTHONPATH", "")
        needed = f"{tokasaurus_dir}:{cartridges_dir}"
        if not current_pypath.startswith(needed):
            os.environ["PYTHONPATH"] = f"{needed}:{current_pypath}" if current_pypath else needed

        # Set required env vars for cartridges library
        scratch_dir = _resolve("scratch_dir", scratch_env)
        os.environ.setdefault("CARTRIDGES_DIR", cartridges_dir)
        os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", f"{scratch_dir}/outputs")
        os.environ.setdefault("HF_HOME", _resolve("hf_home", f"{scratch_env}/huggingface"))

        # Register dispatch for actor and rollout
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

    # ── Model Lifecycle ───────────────────────────────────────────────────

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Load frozen LLM + TrainableCache + start Tokasaurus server."""
        from cartridges.train import CacheAndModel, AttnConfig
        from cartridges.initialization import KVFromText
        from verl.workers.actor.cartridge_actor import CartridgePPOActor

        model_name = self.config.model.path
        device = f"cuda:{self.rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2. Frozen model — resolve Flex model class from HF config's model_type
        model_cls = _resolve_flex_model_cls(model_name)
        print(f"[CartridgeWorker rank={self.rank}] Loading frozen model: {model_name} ({model_cls.__name__})")
        model = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # 3. TrainableCache — either warm-start from existing checkpoint or init from text
        from cartridges.cache import TrainableCache
        num_tokens = self.config.actor.get("num_tokens", 2048)
        attn_config = AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=(
                model.config.head_dim
                if hasattr(model.config, "head_dim")
                else model.config.hidden_size // model.config.num_attention_heads
            ),
        )
        kv_text_source = self.config.actor.get("kv_text_source", None)
        warm_start_path = (
            Path(self.config.actor.cartridge_dir)
            / self.config.actor.cartridge_id
            / "cartridge.pt"
        )
        if warm_start_path.exists() and not kv_text_source:
            # Warm-start: load existing cartridge from disk (e.g. off-policy checkpoint)
            print(f"[CartridgeWorker rank={self.rank}] Warm-start: loading cartridge from {warm_start_path}")
            self.cache = TrainableCache.from_pretrained(str(warm_start_path)).to(self.device)
        elif kv_text_source:
            # Cold-start: initialize KV cache from text via KVFromText
            print(f"[CartridgeWorker rank={self.rank}] KV init from: {kv_text_source}")
            initializer = KVFromText.Config(
                max_tokens=num_tokens,
                text_source=kv_text_source,
            ).instantiate()
            self.cache = initializer.initialize_kv_cache(
                tokenizer=self.tokenizer, model=model, attn_config=attn_config,
            ).to(self.device)
        else:
            raise ValueError(
                "actor.kv_text_source must be set in config, OR a cartridge.pt must "
                "already exist at cartridge_dir/cartridge_id/cartridge.pt for warm-start."
            )
        n_params = sum(p.numel() for p in self.cache.parameters())
        print(f"[CartridgeWorker rank={self.rank}] Cache: {n_params:,} trainable elements")

        # 3a. Pad cache to target size (num_tokens) and Tokasaurus page boundary.
        # KVFromText may produce fewer tokens than num_tokens if the text is short.
        # We pad with zero-valued trainable KV pairs so the cartridge has the full
        # requested capacity. Then align to the next page boundary (16 tokens).
        page_size = 16
        actual = self.cache.num_cartridge_tokens()
        # Target: at least num_tokens, rounded up to page boundary
        target = max(actual, num_tokens)
        target = ((target + page_size - 1) // page_size) * page_size
        pad = target - actual
        if pad > 0:
            dev = self.device
            for i in range(len(self.cache.trainable_keys)):
                k = self.cache.trainable_keys[i]  # (1, n_heads, n_tokens, head_dim)
                v = self.cache.trainable_values[i]
                pad_k = torch.zeros(1, k.size(1), pad, k.size(3), dtype=k.dtype, device=dev)
                pad_v = torch.zeros(1, v.size(1), pad, v.size(3), dtype=v.dtype, device=dev)
                new_k = torch.cat([k.data, pad_k], dim=2)
                new_v = torch.cat([v.data, pad_v], dim=2)
                self.cache.trainable_keys[i] = torch.nn.Parameter(new_k)
                self.cache.trainable_values[i] = torch.nn.Parameter(new_v)
            self.cache._num_trainable_tokens += pad
            old_ids = self.cache._init_seq_ids
            pad_ids = torch.full((pad,), -1, dtype=old_ids.dtype, device=old_ids.device)
            self.cache._init_seq_ids = torch.cat([old_ids, pad_ids])
            self.cache._seq_ids = self.cache._init_seq_ids.clone()
            print(
                f"[CartridgeWorker rank={self.rank}] "
                f"Padded cache {actual} → {actual + pad} tokens "
                f"(text_init={actual}, target={num_tokens}, page_aligned={actual + pad})"
            )

        # 3b. Teacher cartridge (optional): deep-copy of student cache, frozen
        self.teacher_cache = None
        cd_cfg_init = getattr(self.config.actor, "context_distillation", None)
        teacher_cart_ema = -1.0
        if cd_cfg_init is not None and getattr(cd_cfg_init, "enabled", False):
            teacher_cart_ema = getattr(cd_cfg_init, "teacher_cartridge_ema", -1.0)
            # Handle dict from Hydra
            if isinstance(teacher_cart_ema, str):
                teacher_cart_ema = float(teacher_cart_ema)
        if teacher_cart_ema >= 0:
            import copy
            self.teacher_cache = copy.deepcopy(self.cache)
            for p in self.teacher_cache.parameters():
                p.requires_grad_(False)
            self.teacher_cache.eval()
            ema_label = "fixed" if teacher_cart_ema == 0 else f"EMA={teacher_cart_ema}" if teacher_cart_ema < 1 else "synced"
            print(f"[CartridgeWorker rank={self.rank}] Teacher cache created ({ema_label})")
        self._teacher_cart_ema = teacher_cart_ema

        # 3c. Separate teacher model (optional): load a different HF checkpoint as teacher.
        # Used for bias detection: teacher = biased model, student = base + cartridge.
        self.teacher_hf_model = None
        if cd_cfg_init is not None and getattr(cd_cfg_init, "enabled", False):
            teacher_model_path = getattr(cd_cfg_init, "teacher_model_path", None)
            if teacher_model_path:
                from transformers import AutoModelForCausalLM
                print(f"[CartridgeWorker rank={self.rank}] Loading separate teacher model: {teacher_model_path}")
                self.teacher_hf_model = AutoModelForCausalLM.from_pretrained(
                    teacher_model_path, torch_dtype=torch.bfloat16
                ).to(self.device).eval()
                for p in self.teacher_hf_model.parameters():
                    p.requires_grad_(False)
                n_teacher_params = sum(p.numel() for p in self.teacher_hf_model.parameters())
                print(f"[CartridgeWorker rank={self.rank}] Teacher model loaded: {n_teacher_params:,} params")

        # 4. Actor
        self.actor = CartridgePPOActor(
            config=self.config.actor,
            frozen_model=model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            teacher_cache=self.teacher_cache,
            teacher_hf_model=self.teacher_hf_model,
        )

        # 5. Load fixed teacher context (for context distillation, avoids re-reading each step)
        self._fixed_teacher_context = None
        cd_cfg = getattr(self.config.actor, "context_distillation", None)
        if cd_cfg is not None and getattr(cd_cfg, "enabled", False):
            if getattr(cd_cfg, "teacher_context_mode", None) == "fixed":
                src = getattr(cd_cfg, "fixed_context_source", None)
                if not src:
                    raise ValueError(
                        "context_distillation.fixed_context_source must be set when "
                        "teacher_context_mode='fixed'"
                    )
                self._fixed_teacher_context = Path(src).read_text()
                print(
                    f"[CartridgeWorker rank={self.rank}] Teacher context loaded: "
                    f"{len(self._fixed_teacher_context):,} chars from {src}"
                )

        # 6. Sync manager — write initial cartridge before server starts
        # Cache is already page-aligned from step 3a above.
        actual_cache_tokens = self.cache.num_cartridge_tokens()
        assert actual_cache_tokens % 16 == 0, (
            f"Cache should be page-aligned after padding, got {actual_cache_tokens} tokens"
        )
        self.sync_mgr = CartridgeSyncManager(
            cartridge_dir=self.config.actor.cartridge_dir,
            cartridge_id=self.config.actor.cartridge_id,
            max_tokens=actual_cache_tokens,
            model_name=model_name,
        )
        self.sync_mgr.sync(self.cache)
        print(f"[CartridgeWorker rank={self.rank}] Initial cartridge saved to {self.sync_mgr.path}")

        # 7. Offloading
        self._offload_enabled = self.config.actor.get("offload_between_phases", False)
        if self._offload_enabled:
            self.actor.to("cpu")
            torch.cuda.empty_cache()
            print(f"[CartridgeWorker rank={self.rank}] Model offloaded to CPU")

        # 8. Start Tokasaurus server + build rollout
        self._build_rollout()

        # 9. Generation config (for eos token etc.)
        self.generation_config = None

    def _build_rollout(self):
        """Start Tokasaurus server and create TokasaurusRollout."""
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        from tokasaurus.entry import server_manager
        from tokasaurus.common_types import ServerConfig
        from verl.workers.rollout.tokasaurus_rollout import TokasaurusRollout

        port = self.config.rollout.get("port", 10219)
        if self.rank > 0:
            port += self.rank

        # Build Tokasaurus ServerConfig (pydra.Config: assign after init)
        toka_config = ServerConfig()
        toka_config.model = self.config.model.path
        toka_config.port = port
        toka_config.cartridge_dir = self.config.actor.cartridge_dir
        toka_config.tp_size = self.config.rollout.get("tp_size", 1)
        toka_config.dp_size = 1
        toka_config.kv_cache_num_tokens = self.config.rollout.get("kv_cache_num_tokens", 4096)
        toka_config.log_level = "WARNING"
        toka_config.uvicorn_log_level = "warning"

        # Start server
        print(f"[CartridgeWorker rank={self.rank}] Starting Tokasaurus on port {port}...")
        self._server_ctx = server_manager(toka_config)
        self._server_ctx.__enter__()

        # Poll until server is ready
        _poll_port(port, timeout=180)
        print(f"[CartridgeWorker rank={self.rank}] Tokasaurus server ready on port {port}")

        # Build rollout config with model_name
        rollout_config = self.config.rollout
        # Ensure model_name is accessible
        if not hasattr(rollout_config, 'model_name') or rollout_config.get('model_name', None) is None:
            from omegaconf import OmegaConf, open_dict
            with open_dict(rollout_config):
                rollout_config.model_name = self.config.model.path

        self.rollout = TokasaurusRollout(
            config=rollout_config,
            model_config=None,
            device_mesh=None,
            server_ctx=self._server_ctx,
            base_url=f"http://localhost:{port}",
            cartridge_id=self.config.actor.cartridge_id,
            tokenizer=self.tokenizer,
        )

    # ── Offload / Reload ──────────────────────────────────────────────────

    def _offload_training_model(self):
        if self._offload_enabled:
            self.actor.to("cpu")
            torch.cuda.empty_cache()

    def _reload_training_model(self):
        if self._offload_enabled:
            self.actor.to(self.device)

    # ── Dispatch Methods ──────────────────────────────────────────────────

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate rollout sequences via Tokasaurus."""
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.rollout.generate_sequences(prompts=prompts)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """Reload model → compute log probs → offload model."""
        self._reload_training_model()

        log_prob_mbs = getattr(self.config.rollout, "log_prob_micro_batch_size_per_gpu", 4)
        data.meta_info.setdefault("micro_batch_size", log_prob_mbs)
        data.meta_info.setdefault("temperature", self.config.rollout.temperature)
        data.meta_info.setdefault("use_dynamic_bsz", False)
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        data = data.to(self.device)
        outputs = self.actor.compute_log_prob(data=data)

        result = DataProto.from_dict(
            tensors={
                "old_log_probs": outputs["log_probs"],
                "entropys": outputs["entropys"],
            },
            meta_info={"temperature": self.config.rollout.temperature},
        )
        result = result.to("cpu")

        self._offload_training_model()
        return result

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto) -> DataProto:
        """Reload model → [build teacher inputs if CD] → gradient step → sync cartridge → offload."""
        self._reload_training_model()

        data = data.to("cpu")  # will be moved to device in update_policy
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        data.meta_info.setdefault("temperature", self.config.rollout.temperature)

        # Build teacher inputs if context distillation is enabled
        cd_cfg = getattr(self.config.actor, "context_distillation", None)
        if cd_cfg is not None and getattr(cd_cfg, "enabled", False):
            from verl.utils.context_distillation import build_teacher_inputs
            data = build_teacher_inputs(
                tokenizer=self.tokenizer,
                data=data,
                cd_config=cd_cfg,
                fixed_context=self._fixed_teacher_context,
            )

        metrics = self.actor.update_policy(data=data)

        # EMA update for teacher cartridge (if enabled)
        if self.teacher_cache is not None and self._teacher_cart_ema > 0:
            ema = self._teacher_cart_ema
            with torch.no_grad():
                for t_p, s_p in zip(self.teacher_cache.parameters(), self.cache.parameters()):
                    if s_p.requires_grad:  # Only update trainable params
                        t_p.mul_(1 - ema).add_(s_p.data, alpha=ema)

        # Sync updated cartridge to disk for Tokasaurus
        self.sync_mgr.sync(self.cache)

        self._offload_training_model()

        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """No-op. Cartridge sync handled via disk write in update_actor."""
        return True

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        """Compute reference log probs (same as compute_log_prob for cartridge — no separate ref model)."""
        self._reload_training_model()

        data = data.to(self.device)
        outputs = self.actor.compute_log_prob(data=data)

        result = DataProto.from_dict(
            tensors={"ref_log_prob": outputs["log_probs"]},
            meta_info={"temperature": self.config.rollout.temperature},
        )
        result = result.to("cpu")

        self._offload_training_model()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, remote_path=None, global_step=0, **kwargs):
        """Save cartridge checkpoint."""
        save_dir = Path(local_path) / "cartridge"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.cache.save(str(save_dir / "cartridge.pt"))
        print(f"[CartridgeWorker rank={self.rank}] Checkpoint saved to {save_dir} (step {global_step})")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, **kwargs):
        """Load cartridge checkpoint."""
        load_path = Path(local_path) / "cartridge" / "cartridge.pt"
        if load_path.exists():
            state = torch.load(str(load_path), map_location="cpu", weights_only=False)
            self.cache.load_state_dict(state, strict=False)
            self.sync_mgr.sync(self.cache)
            print(f"[CartridgeWorker rank={self.rank}] Checkpoint loaded from {load_path}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _poll_port(port: int, timeout: int = 120, interval: float = 2.0):
    """Poll until a server is accepting connections on the given port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(interval)
    raise TimeoutError(f"Server not ready on port {port} after {timeout}s")
