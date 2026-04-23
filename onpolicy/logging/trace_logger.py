"""
Detailed per-step trace logging for on-policy training runs.

Mirrors the logging structure used in tinker-cookbook so that the same
analysis/visualization tooling works on both systems.

Directory structure created:
    {trace_dir}/
    ├── config.json                   # Full training config (once at startup)
    ├── metrics.jsonl                 # All metrics per step (appended each step)
    └── rollout_traces/
        └── {run_id}/
            ├── run_manifest.json     # Model name / metadata
            └── rank_0/
                ├── step_000001/
                │   ├── rollouts.jsonl          # Per-sample text summaries
                │   ├── rollouts_payload.pt     # Response tokens, logprobs, masks
                │   ├── training.jsonl          # Per-sample training summaries
                │   └── training_payload.pt     # Advantages, rewards, logprobs, etc.
                ├── step_000002/
                │   └── ...
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _SafeJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        try:
            return super().default(o)
        except TypeError:
            return str(o)


class TraceLogger:
    """
    Saves detailed per-step data (metrics, rollouts, training tensors) to disk,
    matching the tinker-cookbook trace logging structure.

    Usage in the training loop::

        trace_logger = TraceLogger(trace_dir, model_name="meta-llama/...")
        trace_logger.log_config(config_dict)

        for step in range(total_steps):
            # ... rollout + training ...
            trace_logger.log_metrics(metrics, step=step)
            trace_logger.log_rollouts(step, batch, tokenizer)
            trace_logger.log_training(step, batch)
    """

    def __init__(
        self,
        trace_dir: str | Path,
        *,
        enabled: bool = True,
        save_payloads: bool = True,
        run_id: str | None = None,
        rank: str = "rank_0",
        model_name: str | None = None,
    ):
        self.trace_dir = Path(trace_dir)
        self.enabled = enabled
        self.save_payloads = save_payloads
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rank = rank
        self.model_name = model_name

        if not enabled:
            return

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.trace_dir / f"metrics_{self.run_id}.jsonl"

        self.steps_dir = self.trace_dir / "rollout_traces" / self.run_id / rank
        self.steps_dir.mkdir(parents=True, exist_ok=True)

        if model_name:
            manifest = {"static_metadata": {"model_name": model_name}}
            manifest_path = self.steps_dir.parent / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info("TraceLogger enabled — saving to %s", self.trace_dir)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def log_config(self, config: dict[str, Any] | Any) -> None:
        """Save the full training config as config.json (called once at startup)."""
        if not self.enabled:
            return
        config_path = self.trace_dir / "config.json"
        if hasattr(config, "to_container"):
            config = config.to_container(resolve=True)
        elif not isinstance(config, dict):
            config = {"config": str(config)}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, cls=_SafeJSONEncoder)
        logger.info("Wrote config to %s", config_path)

    # ------------------------------------------------------------------
    # Per-step metrics
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Append a single metrics dict as one JSONL line (called every step)."""
        if not self.enabled:
            return
        entry: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor,)):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
            elif isinstance(v, (np.generic,)):
                v = v.item()
            entry[k] = v
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, cls=_SafeJSONEncoder) + "\n")

    # ------------------------------------------------------------------
    # Rollout traces
    # ------------------------------------------------------------------

    def log_rollouts(
        self,
        step: int,
        batch: Any,
        tokenizer: Any,
    ) -> None:
        """
        Save per-sample rollout data for a training step.

        Writes:
          - rollouts.jsonl  : human-readable per-sample summary
          - rollouts_payload.pt : token-level tensors for later analysis
        """
        if not self.enabled:
            return

        step_dir = self._step_dir(step)

        prompts_ids = batch.batch["prompts"]
        responses_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        B = prompts_ids.shape[0]

        prompts_text = tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
        responses_text = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)

        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        uids = list(batch.non_tensor_batch.get("uid", [f"sample_{i}" for i in range(B)]))

        ground_truths = []
        for i in range(B):
            gt = None
            if "reward_model" in batch.non_tensor_batch:
                rm = batch.non_tensor_batch["reward_model"]
                if isinstance(rm, np.ndarray) and len(rm) > i:
                    entry = rm[i] if isinstance(rm[i], dict) else {}
                    gt = entry.get("ground_truth", None)
                elif isinstance(rm, dict):
                    gt = rm.get("ground_truth", None)
            ground_truths.append(gt)

        gen_lens = response_mask.sum(-1).cpu().tolist()

        jsonl_path = step_dir / "rollouts.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(B):
                record = {
                    "sample_id": str(uids[i]),
                    "prompt": prompts_text[i],
                    "response": responses_text[i],
                    "reward": scores[i],
                    "ground_truth": ground_truths[i],
                    "gen_len": int(gen_lens[i]),
                }
                f.write(json.dumps(record, cls=_SafeJSONEncoder) + "\n")

        if self.save_payloads:
            payload: dict[str, Any] = {
                "sample_ids": [str(u) for u in uids],
                "responses": responses_ids.cpu(),
                "response_mask": response_mask.cpu(),
                "token_level_scores": batch.batch["token_level_scores"].cpu(),
            }
            if "old_log_probs" in batch.batch:
                payload["old_log_probs"] = batch.batch["old_log_probs"].cpu()
            if "rollout_log_probs" in batch.batch:
                payload["rollout_log_probs"] = batch.batch["rollout_log_probs"].cpu()

            torch.save(payload, step_dir / "rollouts_payload.pt")

        logger.debug("Logged %d rollouts for step %d", B, step)

    # ------------------------------------------------------------------
    # Training traces
    # ------------------------------------------------------------------

    def log_training(
        self,
        step: int,
        batch: Any,
        actor_metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Save per-sample training data for a training step.

        Writes:
          - training.jsonl  : per-sample scalar summaries
          - training_payload.pt : full tensor data (advantages, logprobs, ...)
        """
        if not self.enabled:
            return

        step_dir = self._step_dir(step)
        B = batch.batch["responses"].shape[0]

        uids = list(batch.non_tensor_batch.get("uid", [f"sample_{i}" for i in range(B)]))
        response_mask = batch.batch["response_mask"]

        advantages = batch.batch.get("advantages", None)
        token_level_rewards = batch.batch.get("token_level_rewards", None)

        jsonl_path = step_dir / "training.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(B):
                mask_i = response_mask[i].bool()
                record: dict[str, Any] = {"sample_id": str(uids[i])}

                if advantages is not None:
                    adv_i = advantages[i][mask_i].float()
                    record["advantage_mean"] = adv_i.mean().item()
                    record["advantage_std"] = adv_i.std().item() if adv_i.numel() > 1 else 0.0

                if token_level_rewards is not None:
                    rew_i = token_level_rewards[i][mask_i].float()
                    record["reward_mean"] = rew_i.mean().item()
                    record["reward_sum"] = rew_i.sum().item()

                if "old_log_probs" in batch.batch:
                    lp_i = batch.batch["old_log_probs"][i][mask_i].float()
                    record["old_logprob_mean"] = lp_i.mean().item()

                if "ref_log_prob" in batch.batch:
                    ref_i = batch.batch["ref_log_prob"][i][mask_i].float()
                    record["ref_logprob_mean"] = ref_i.mean().item()
                    if "old_log_probs" in batch.batch:
                        kl_i = (batch.batch["old_log_probs"][i][mask_i] - ref_i).float()
                        record["kl_vs_ref_mean"] = kl_i.mean().item()

                f.write(json.dumps(record, cls=_SafeJSONEncoder) + "\n")

        if self.save_payloads:
            payload: dict[str, Any] = {
                "sample_ids": [str(u) for u in uids],
                "responses": batch.batch["responses"].cpu(),
                "response_mask": response_mask.cpu(),
            }

            tensor_keys = [
                "advantages",
                "returns",
                "old_log_probs",
                "ref_log_prob",
                "token_level_rewards",
                "token_level_scores",
                "teacher_input_ids",
                "teacher_attention_mask",
            ]
            for key in tensor_keys:
                if key in batch.batch:
                    payload[key] = batch.batch[key].cpu()

            if actor_metrics:
                payload["actor_metrics"] = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in actor_metrics.items()
                }

            torch.save(payload, step_dir / "training_payload.pt")

        logger.debug("Logged %d training samples for step %d", B, step)

    # ------------------------------------------------------------------
    # Validation traces
    # ------------------------------------------------------------------

    def log_validation(
        self,
        step: int,
        inputs: list[str],
        outputs: list[str],
        scores: list[float],
        ground_truths: list[Any] | None = None,
        extra: dict[str, list] | None = None,
    ) -> None:
        """Save validation generation data for a step."""
        if not self.enabled:
            return
        step_dir = self._step_dir(step)
        val_path = step_dir / "validation.jsonl"
        with open(val_path, "w") as f:
            for i in range(len(inputs)):
                record: dict[str, Any] = {
                    "prompt": inputs[i],
                    "response": outputs[i],
                    "score": scores[i],
                }
                if ground_truths is not None and i < len(ground_truths):
                    record["ground_truth"] = ground_truths[i]
                if extra:
                    for k, v in extra.items():
                        if i < len(v):
                            record[k] = v[i]
                f.write(json.dumps(record, cls=_SafeJSONEncoder) + "\n")
        logger.debug("Logged %d validation samples for step %d", len(inputs), step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _step_dir(self, step: int) -> Path:
        step_dir = self.steps_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir
