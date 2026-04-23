"""CartridgePPOActor: PPO actor for on-policy cartridge training.

No FSDP. Trains only the TrainableCache; frozen LLM provides gradients
through the cache via FlexAttention.

Uses **sequence packing** for efficient batched forward passes: multiple samples
are concatenated into one packed sequence with distinct seq_ids. FlexLlamaForCausalLM
isolates samples via flex_attention block masks (cache tokens have seq_id=-1, so all
samples attend to them; each sample only sees its own tokens beyond the cache).

The adapter type (cartridge / TrainableCache) is orthogonal to the loss function
(GRPO, context distillation, etc.). Context distillation config and utilities
live in separate modules — this file only handles the cartridge-specific forward pass.
"""
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.base import BasePPOActor


class CartridgePPOActor(BasePPOActor):
    """Cartridge-specific PPO actor. Trains only the TrainableCache."""

    def __init__(self, config, frozen_model, cache, tokenizer, teacher_cache=None, teacher_hf_model=None):
        super().__init__(config)
        self.model = frozen_model
        self.cache = cache
        self.tokenizer = tokenizer
        self.device = next(frozen_model.parameters()).device

        # Lazy import — CacheAndModel is from cartridges library
        from cartridges.train import CacheAndModel
        self.wrapped = CacheAndModel(cache, frozen_model)

        # Teacher cartridge (optional): CacheAndModel wrapping teacher cache + same frozen model
        self.teacher_cache = teacher_cache
        if teacher_cache is not None:
            self.teacher_wrapped = CacheAndModel(teacher_cache, frozen_model)
        else:
            self.teacher_wrapped = None

        # Separate teacher HF model (optional): for bias detection experiments
        # When set, teacher forward uses this model instead of self.model
        self.teacher_hf_model = teacher_hf_model

        # fp32 optimizer states: maintain fp32 master copies of cache params
        # for precise gradient accumulation, while keeping bf16 params for forward pass
        self._fp32_optim = config.get("fp32_optim", False) if hasattr(config, 'get') else getattr(config, 'fp32_optim', False)
        if self._fp32_optim:
            self._fp32_params = [p.data.float().clone().requires_grad_(True) for p in cache.parameters()]
            self.optimizer = torch.optim.Adam(
                self._fp32_params, lr=config.get("lr", 2e-2)
            )
        else:
            self._fp32_params = None
            self.optimizer = torch.optim.Adam(
                cache.parameters(), lr=config.get("lr", 2e-2)
            )
        self.max_grad_norm = config.get("max_grad_norm", 1.0) if hasattr(config, 'get') else getattr(config, 'max_grad_norm', 1.0)

    def to(self, device):
        """Move frozen model + cache (+ teacher cache/model) to device. For offload support."""
        self.model.to(device)
        self.cache.to(device)
        if self.teacher_cache is not None:
            self.teacher_cache.to(device)
        if self.teacher_hf_model is not None:
            self.teacher_hf_model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self

    def _forward_packed(self, input_ids_batch, attention_mask_batch, responses_batch,
                        response_mask_batch, enable_grad=False, return_logits=False):
        """Forward B samples through CacheAndModel in a single packed call.

        Uses sequence packing: multiple samples are concatenated into one packed 1D
        sequence with distinct seq_ids. FlexLlamaForCausalLM isolates samples via
        flex_attention block masks (cache tokens seq_id=-1, input seq_ids 0,1,2,...).

        Args:
            input_ids_batch: [B, P+R] int64 (left-padded prompt + right-padded response)
            attention_mask_batch: [B, P+R] int64
            responses_batch: [B, R_max] int64
            response_mask_batch: [B, R_max] int64
            enable_grad: whether to enable gradient computation
            return_logits: if True, return full log_softmax [B, R_max, vocab] instead of
                          sampled-token log-probs [B, R_max]. Used for topk loss.

        Returns:
            If return_logits=False: (log_probs[B,R_max], entropy[B,R_max]) float32
            If return_logits=True:  (full_log_probs[B,R_max,V], entropy[B,R_max]) float32
        """
        B = input_ids_batch.shape[0]
        R_max = responses_batch.shape[1]

        # Build packed sequence: concatenate valid tokens from each sample
        packed_ids = []
        packed_seq_ids = []
        packed_pos_ids = []
        # (offset_in_packed, actual_P, actual_R) per sample
        sample_info = []
        offset = 0

        for i in range(B):
            mask = attention_mask_batch[i].bool()
            valid_ids = input_ids_batch[i][mask]
            actual_R = int(response_mask_batch[i].sum().item())
            actual_P = valid_ids.shape[0] - actual_R

            packed_ids.append(valid_ids)
            packed_seq_ids.append(torch.full((valid_ids.shape[0],), i, dtype=torch.long, device=self.device))
            packed_pos_ids.append(torch.arange(valid_ids.shape[0], device=self.device))
            sample_info.append((offset, actual_P, actual_R))
            offset += valid_ids.shape[0]

        all_ids = torch.cat(packed_ids)         # [total_packed_len]
        all_seq_ids = torch.cat(packed_seq_ids) # [total_packed_len]
        all_pos_ids = torch.cat(packed_pos_ids) # [total_packed_len]

        # Pad to bucket boundary so torch.compile(dynamic=False) doesn't
        # recompile + autotune for every unique packed length.
        # seq_id=-2 ensures padded positions are isolated by the block mask.
        total_len = all_ids.shape[0]
        PACK_BUCKET = 256
        padded_len = ((total_len + PACK_BUCKET - 1) // PACK_BUCKET) * PACK_BUCKET
        if padded_len > total_len:
            pad_n = padded_len - total_len
            all_ids = F.pad(all_ids, (0, pad_n), value=0)
            all_seq_ids = F.pad(all_seq_ids, (0, pad_n), value=-2)
            all_pos_ids = F.pad(all_pos_ids, (0, pad_n), value=0)

        ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.wrapped(input_ids=all_ids, seq_ids=all_seq_ids, position_ids=all_pos_ids)

            logits = out.logits[0][:total_len]  # [total_packed_len, vocab] — strip padding

            all_log_probs = []
            all_entropy = []
            for i in range(B):
                start, actual_P, actual_R = sample_info[i]
                if actual_R == 0:
                    if return_logits:
                        all_log_probs.append(torch.zeros(R_max, logits.shape[-1], device=self.device))
                    else:
                        all_log_probs.append(torch.zeros(R_max, device=self.device))
                    all_entropy.append(torch.zeros(R_max, device=self.device))
                    continue

                # Response logits: logit at position (start+actual_P-1) predicts response[0], etc.
                resp_logits = logits[start + actual_P - 1 : start + actual_P + actual_R - 1]
                log_probs_all = F.log_softmax(resp_logits.float(), dim=-1)
                actual_ent = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)

                if return_logits:
                    # Return full log_softmax for topk loss
                    padded_lp = torch.zeros(R_max, log_probs_all.shape[-1], device=self.device, dtype=log_probs_all.dtype)
                    padded_lp[:actual_R] = log_probs_all
                    all_log_probs.append(padded_lp)
                else:
                    # Original: sampled-token log-probs only
                    actual_response_ids = responses_batch[i, :actual_R]
                    actual_lp = log_probs_all.gather(1, actual_response_ids.unsqueeze(1)).squeeze(1)
                    lp = torch.zeros(R_max, device=self.device, dtype=actual_lp.dtype)
                    lp[:actual_R] = actual_lp
                    all_log_probs.append(lp)

                ent = torch.zeros(R_max, device=self.device, dtype=actual_ent.dtype)
                ent[:actual_R] = actual_ent
                all_entropy.append(ent)

        self.cache.clear()
        return torch.stack(all_log_probs), torch.stack(all_entropy)

    def get_first_token_logits(self, data: DataProto) -> torch.Tensor:
        """Get logits at first response position for each sample (no-grad packed forward).

        Task-agnostic utility: eval functions can call this to analyze the model's
        prediction distribution at the start of each response.
        Uses micro-batching (same pack_size as _forward_packed) to avoid OOM.

        Returns: [B, vocab_size] float32
        """
        input_ids = data.batch["input_ids"].to(self.device)
        attention_mask = data.batch["attention_mask"].to(self.device)
        response_mask = data.batch["response_mask"].to(self.device)
        B = input_ids.shape[0]
        micro_bs = data.meta_info.get("micro_batch_size", 4) if hasattr(data, 'meta_info') and data.meta_info else 4

        all_results = []
        for mb_start in range(0, B, micro_bs):
            mb_end = min(mb_start + micro_bs, B)
            packed_ids, packed_seq_ids, packed_pos_ids = [], [], []
            sample_info = []
            offset = 0
            for i in range(mb_start, mb_end):
                mask = attention_mask[i].bool()
                valid_ids = input_ids[i][mask]
                actual_R = int(response_mask[i].sum().item())
                actual_P = valid_ids.shape[0] - actual_R
                packed_ids.append(valid_ids)
                seq_idx = i - mb_start
                packed_seq_ids.append(torch.full((valid_ids.shape[0],), seq_idx, dtype=torch.long, device=self.device))
                packed_pos_ids.append(torch.arange(valid_ids.shape[0], device=self.device))
                sample_info.append((offset, actual_P, actual_R))
                offset += valid_ids.shape[0]

            all_ids = torch.cat(packed_ids)
            all_seq_ids = torch.cat(packed_seq_ids)
            all_pos_ids = torch.cat(packed_pos_ids)

            total_len = all_ids.shape[0]
            PACK_BUCKET = 256
            padded_len = ((total_len + PACK_BUCKET - 1) // PACK_BUCKET) * PACK_BUCKET
            if padded_len > total_len:
                pad_n = padded_len - total_len
                all_ids = F.pad(all_ids, (0, pad_n), value=0)
                all_seq_ids = F.pad(all_seq_ids, (0, pad_n), value=-2)
                all_pos_ids = F.pad(all_pos_ids, (0, pad_n), value=0)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.wrapped(
                    input_ids=all_ids,
                    seq_ids=all_seq_ids,
                    position_ids=all_pos_ids,
                )
            logits = out.logits[0][:total_len]
            self.cache.clear()

            for j, (start, actual_P, actual_R) in enumerate(sample_info):
                if actual_R == 0:
                    all_results.append(torch.zeros(logits.shape[-1], device=self.device))
                else:
                    all_results.append(logits[start + actual_P - 1].float())

        return torch.stack(all_results)

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict:
        """Compute log probabilities of responses. Called with torch.no_grad() by the worker.

        Uses packed forward with micro-batching: B samples are split into chunks
        of micro_batch_size and each chunk is packed into one model call.

        Args:
            data: DataProto with batch keys: input_ids[B,S], attention_mask[B,S],
                  responses[B,R], response_mask[B,R]

        Returns:
            dict with "log_probs": Tensor[B,R], "entropys": Tensor[B,R]
        """
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        responses = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        B = input_ids.shape[0]

        # Micro-batch to avoid OOM during torch.compile tracing
        micro_bs = data.meta_info.get("micro_batch_size", 4) if hasattr(data, 'meta_info') and data.meta_info else 4

        all_log_probs = []
        all_entropys = []
        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            lp, ent = self._forward_packed(
                input_ids[start:end], attention_mask[start:end],
                responses[start:end], response_mask[start:end],
                enable_grad=False
            )
            all_log_probs.append(lp)
            all_entropys.append(ent)

        return {
            "log_probs": torch.cat(all_log_probs, dim=0),
            "entropys": torch.cat(all_entropys, dim=0),
        }

    def update_policy(self, data: DataProto) -> dict:
        """GRPO policy gradient step. Trains only the TrainableCache.

        Adapted from dp_actor.py:508-676 with FSDP/SP/dynamic-bsz stripped.

        Args:
            data: DataProto with batch keys: input_ids[B,S], attention_mask[B,S],
                  responses[B,R], response_mask[B,R], old_log_probs[B,R],
                  advantages[B,R]. Optional: ref_log_prob[B,R] (if use_kl_loss).
        """
        data = data.to(self.device)

        # Ensure global_batch_info exists on config (used by policy_loss_fn → agg_loss).
        # For single-GPU cartridge training, empty dict → agg_loss uses defaults (dp_size=1).
        from omegaconf import open_dict
        try:
            _ = self.config.global_batch_info
        except Exception:
            with open_dict(self.config):
                self.config.global_batch_info = {}

        ppo_epochs = self.config.get("ppo_epochs", 1) if hasattr(self.config, 'get') else getattr(self.config, 'ppo_epochs', 1)
        micro_batch_size = self.config.get("ppo_micro_batch_size_per_gpu", 1) if hasattr(self.config, 'get') else getattr(self.config, 'ppo_micro_batch_size_per_gpu', 1)
        loss_agg_mode = self.config.get("loss_agg_mode", "token-mean") if hasattr(self.config, 'get') else getattr(self.config, 'loss_agg_mode', 'token-mean')
        use_kl_loss = self.config.get("use_kl_loss", False) if hasattr(self.config, 'get') else getattr(self.config, 'use_kl_loss', False)
        kl_loss_type = self.config.get("kl_loss_type", "kl") if hasattr(self.config, 'get') else getattr(self.config, 'kl_loss_type', 'kl')
        kl_loss_coef = self.config.get("kl_loss_coef", 0.1) if hasattr(self.config, 'get') else getattr(self.config, 'kl_loss_coef', 0.1)
        entropy_coeff = self.config.get("entropy_coeff", 0.0) if hasattr(self.config, 'get') else getattr(self.config, 'entropy_coeff', 0.0)
        loss_mode = (self.config.get("loss_mode", "grpo") if hasattr(self.config, 'get')
                     else getattr(self.config, 'loss_mode', 'grpo'))
        cd_config = getattr(self.config, 'context_distillation', None)

        mini_batches = data.split(data.batch["responses"].shape[0])  # single mini-batch for cartridge
        policy_loss_fn = get_policy_loss_fn("vanilla")

        # Match dp_actor metric convention: accumulated scalars for pg_loss/kl_loss,
        # lists for per-micro-batch metrics (pg_clipfrac, ppo_kl, grad_norm, entropy).
        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
            "actor/cd_loss": 0.0,
        }
        for _epoch in range(ppo_epochs):
            for mini_batch in mini_batches:
                micro_batches = mini_batch.split(micro_batch_size)
                gradient_accumulation = len(micro_batches)

                self.optimizer.zero_grad()
                if self._fp32_optim:
                    for p in self.cache.parameters():
                        if p.grad is not None:
                            p.grad.zero_()

                for mb in micro_batches:
                    mb = mb.to(self.device)
                    response_mask = mb.batch["response_mask"]
                    old_log_prob = mb.batch["old_log_probs"]
                    advantages = mb.batch["advantages"]
                    responses = mb.batch["responses"]
                    input_ids = mb.batch["input_ids"]
                    attention_mask = mb.batch["attention_mask"]
                    B, R = responses.shape

                    loss_scale_factor = 1.0 / gradient_accumulation

                    # Forward pass with gradients — packed (all B samples in one call)
                    use_topk = (loss_mode == "context_distillation"
                                and cd_config is not None
                                and getattr(cd_config, 'cd_loss_mode', 'sampled') in ('topk', 'fullvocab'))

                    torch.cuda.empty_cache()  # reclaim memory before large forward pass
                    log_prob, entropy = self._forward_packed(
                        input_ids, attention_mask, responses, response_mask,
                        enable_grad=True, return_logits=use_topk,
                    )

                    # Policy loss
                    micro_batch_metrics = {}

                    if loss_mode == "context_distillation" and cd_config is not None:
                        # Context distillation: teacher forward + KL loss (no GRPO)
                        from verl.utils.context_distillation import (
                            teacher_forward_packed, teacher_forward_packed_with_cache, teacher_forward_hf,
                        )
                        from verl.trainer.ppo.core_algos import compute_context_distillation_loss
                        torch.cuda.empty_cache()  # reclaim memory before teacher forward

                        if self.teacher_hf_model is not None:
                            # Separate teacher model (bias detection mode):
                            # Teacher is a different HF model (e.g. biased checkpoint).
                            # Uses standard HF forward (not packed/FlexLlama).
                            teacher_log_probs = teacher_forward_hf(
                                model=self.teacher_hf_model,
                                teacher_input_ids_batch=mb.batch["teacher_input_ids"],
                                teacher_attention_mask_batch=mb.batch["teacher_attention_mask"],
                                response_mask_batch=response_mask,
                                device=self.device,
                                return_logits=use_topk,
                            )
                        else:
                            teacher_cart_ema = getattr(cd_config, 'teacher_cartridge_ema', -1.0)
                            if teacher_cart_ema >= 0 and self.teacher_wrapped is not None:
                                # Teacher with its own cartridge
                                teacher_log_probs = teacher_forward_packed_with_cache(
                                    wrapped_model=self.teacher_wrapped,
                                    teacher_input_ids_batch=mb.batch["teacher_input_ids"],
                                    teacher_attention_mask_batch=mb.batch["teacher_attention_mask"],
                                    response_mask_batch=response_mask,
                                    device=self.device,
                                    return_logits=use_topk,
                                )
                            else:
                                # Default: bare model + context (no teacher cartridge)
                                teacher_log_probs = teacher_forward_packed(
                                    model=self.model,
                                    teacher_input_ids_batch=mb.batch["teacher_input_ids"],
                                    teacher_attention_mask_batch=mb.batch["teacher_attention_mask"],
                                    response_mask_batch=response_mask,
                                    device=self.device,
                                    return_logits=use_topk,
                                )

                        policy_loss, cd_metrics = compute_context_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            config=cd_config,
                            loss_agg_mode=loss_agg_mode,
                        )
                        # cd_loss is accumulated as a scalar (like pg_loss/kl_loss);
                        # only put non-accumulated metrics into micro_batch_metrics
                        # (append_to_dict expects list values, not floats).
                        metrics["actor/cd_loss"] += cd_metrics["actor/cd_loss"] * loss_scale_factor
                        metrics["actor/pg_loss"] += 0.0  # keep key populated for logging
                        for k, v in cd_metrics.items():
                            if k != "actor/cd_loss":
                                micro_batch_metrics[k] = v
                    else:
                        # GRPO policy gradient loss
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=None,
                        )
                        micro_batch_metrics.update(pg_metrics)
                        policy_loss = pg_loss
                        metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor

                    # Entropy bonus (if entropy_coeff != 0)
                    entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                    if entropy_coeff != 0:
                        policy_loss = policy_loss - entropy_agg * entropy_coeff

                    # Optional KL loss vs reference model (for GRPO + KL regularisation)
                    if use_kl_loss and "ref_log_prob" in mb.batch:
                        ref_log_prob = mb.batch["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                # Gradient clipping + optimizer step
                if self._fp32_optim and self._fp32_params is not None:
                    # Copy bf16 grads to fp32 master params, step in fp32, copy back
                    for p_bf16, p_fp32 in zip(self.cache.parameters(), self._fp32_params):
                        if p_bf16.grad is not None:
                            p_fp32.grad = p_bf16.grad.float()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._fp32_params, max_norm=self.max_grad_norm
                    )
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                        # Copy fp32 master weights back to bf16 cache params
                        for p_bf16, p_fp32 in zip(self.cache.parameters(), self._fp32_params):
                            p_bf16.data.copy_(p_fp32.data.to(p_bf16.dtype))
                    else:
                        print(f"WARN: grad_norm is not finite: {grad_norm}, skipping step")
                        self.optimizer.zero_grad()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.cache.parameters(), max_norm=self.max_grad_norm
                    )
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        print(f"WARN: grad_norm is not finite: {grad_norm}, skipping step")
                        self.optimizer.zero_grad()

                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.optimizer.zero_grad()

        # ── Custom eval function (task-specific, optional) ──────────────
        eval_fn_config = getattr(self.config, 'custom_eval_function', None)
        eval_freq = getattr(self.config, 'eval_freq', 0)
        if eval_fn_config and eval_freq > 0:
            if not hasattr(self, '_eval_step_counter'):
                self._eval_step_counter = 0
            self._eval_step_counter += 1

            if self._eval_step_counter % eval_freq == 0:
                from verl.utils.import_utils import load_extern_object
                eval_fn = load_extern_object(
                    module_path=eval_fn_config.path,
                    object_name=eval_fn_config.name,
                )
                eval_metrics = eval_fn(
                    responses=data.batch["responses"],
                    response_mask=data.batch["response_mask"],
                    tokenizer=self.tokenizer,
                    actor=self,
                    data=data,
                    step=self._eval_step_counter,
                )
                metrics.update(eval_metrics)

        return metrics
