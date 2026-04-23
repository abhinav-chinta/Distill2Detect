"""TokasaurusRollout: HTTP-based rollout using Tokasaurus inference server.

Server lifecycle is managed by CartridgeWorker. This class only handles
HTTP requests and response parsing.
"""
import pickle
import time

import numpy as np
import requests
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout


class TokasaurusRollout(BaseRollout):
    """Tokasaurus inference via HTTP. Server lifecycle managed by CartridgeWorker."""

    def __init__(
        self,
        config,
        model_config,
        device_mesh,
        server_ctx,
        base_url: str,
        cartridge_id: str,
        tokenizer,
    ):
        self._server_ctx = server_ctx
        self.base_url = base_url
        self.cartridge_id = cartridge_id
        self.tokenizer = tokenizer
        self.config = config

        self.model_name = config.get("model_name", "meta-llama/Llama-3.2-3B-Instruct") if hasattr(config, 'get') else getattr(config, 'model_name', 'meta-llama/Llama-3.2-3B-Instruct')
        self.max_tokens = config.get("max_tokens", 512) if hasattr(config, 'get') else getattr(config, 'max_tokens', 512)
        self.temperature = config.get("temperature", 0.7) if hasattr(config, 'get') else getattr(config, 'temperature', 0.7)
        self.max_prompt_length = config.get("max_prompt_length", 512) if hasattr(config, 'get') else getattr(config, 'max_prompt_length', 512)
        # teacher_rollout_mode: if True, inject patient records into system prompt and
        # generate WITHOUT the cartridge. Produces high-quality teacher responses as
        # training targets, avoiding garbage-in/garbage-out from student rollouts.
        self.teacher_rollout_mode = config.get("teacher_rollout_mode", False) if hasattr(config, 'get') else getattr(config, 'teacher_rollout_mode', False)
        # Field in extra_info that contains per-sample teacher context (patient records or patient_id)
        self.teacher_context_field = config.get("teacher_context_field", "patient_records") if hasattr(config, 'get') else getattr(config, 'teacher_context_field', 'patient_records')
        # Optional path to a JSON lookup file: maps short keys (e.g. "patient_01") → full context text.
        # When set, extra_info[teacher_context_field] is treated as a lookup key, not full text.
        lookup_path = config.get("teacher_context_lookup", None) if hasattr(config, 'get') else getattr(config, 'teacher_context_lookup', None)
        self._teacher_context_lookup = None
        if lookup_path:
            import json
            with open(lookup_path) as _f:
                self._teacher_context_lookup = json.load(_f)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate responses via Tokasaurus HTTP API.

        Args:
            prompts: DataProto with non_tensor_batch["raw_prompt"] containing
                     lists of message dicts. Already repeated G times by trainer.

        Returns:
            DataProto with batch: prompts[B,P], responses[B,R], input_ids[B,P+R],
                     attention_mask[B,P+R], position_ids[B,P+R]
        """
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]  # numpy array of message lists
        B = len(raw_prompts)

        # Respect do_sample flag from VERL (e.g. val_kwargs.do_sample=False for greedy eval)
        do_sample = prompts.meta_info.get("do_sample", True)
        temperature = self.temperature if do_sample else 0.0

        # Optionally read per-sample teacher context (patient records) for teacher rollout mode
        extra_infos = None
        if self.teacher_rollout_mode:
            extra_infos = list(prompts.non_tensor_batch.get("extra_info", [{}] * B))

        # Build batch request
        http_requests = []
        for i in range(B):
            messages = raw_prompts[i]
            # Ensure messages is a list of dicts (numpy may wrap it)
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()

            if self.teacher_rollout_mode:
                # Inject patient records as system message (teacher mode, no cartridge)
                raw_ctx = extra_infos[i].get(self.teacher_context_field, "") if extra_infos else ""
                # Resolve via lookup if configured (e.g. "patient_01" → full records text)
                if self._teacher_context_lookup is not None:
                    context = self._teacher_context_lookup.get(raw_ctx, raw_ctx)
                else:
                    context = raw_ctx
                msgs = [dict(m) for m in messages]
                if msgs and msgs[0]["role"] == "system":
                    msgs[0] = {"role": "system", "content": context + "\n\n" + msgs[0]["content"]}
                else:
                    msgs = [{"role": "system", "content": context}] + msgs
                http_requests.append({
                    "model": self.model_name,
                    "messages": msgs,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                })
            else:
                http_requests.append({
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "cartridges": [{
                        "id": self.cartridge_id,
                        "source": "local",
                        "force_redownload": True,
                    }],
                })

        # POST to synchronous batch endpoint in sub-batches to avoid overwhelming
        # the Tokasaurus scheduler (NoSpaceException with large concurrent batches).
        sub_batch_size = self.config.get("rollout_sub_batch_size", 16) if hasattr(self.config, 'get') else getattr(self.config, 'rollout_sub_batch_size', 16)
        if self.teacher_rollout_mode:
            batch_endpoint = f"{self.base_url}/custom/synchronous-batch-completions"
        else:
            batch_endpoint = f"{self.base_url}/custom/cartridge/synchronous-batch-completions"
        t0 = time.time()
        completions = []
        for i in range(0, B, sub_batch_size):
            sub_requests = http_requests[i : i + sub_batch_size]
            resp = requests.post(
                batch_endpoint,
                json={"requests": sub_requests},
                timeout=600,
            )
            resp.raise_for_status()
            completions.extend(pickle.loads(resp.content))
        t1 = time.time()

        # Parse completions → tokenize → build tensors
        # completions may be dicts or Pydantic ChatCompletion objects
        response_texts = []
        for comp in completions:
            if isinstance(comp, dict):
                choices = comp["choices"]
                choice = choices[0]
                if isinstance(choice, dict):
                    text = choice["message"]["content"]
                else:
                    text = choice.message.content
            else:
                text = comp.choices[0].message.content
            response_texts.append(text if text else "")

        # Tokenize prompts and responses
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id

        prompt_ids_list = []
        response_ids_list = []

        for i in range(B):
            messages = raw_prompts[i]
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            # Tokenize prompt using chat template
            p_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            )[0]  # [P_i]
            prompt_ids_list.append(p_ids)

            # Tokenize response
            r_ids = self.tokenizer.encode(
                response_texts[i], add_special_tokens=False, return_tensors="pt",
            )[0]  # [R_i]
            # Append EOS if not already there
            if len(r_ids) == 0 or r_ids[-1] != eos_token_id:
                r_ids = torch.cat([r_ids, torch.tensor([eos_token_id])])
            response_ids_list.append(r_ids)

        # Pad to uniform lengths
        P = self.max_prompt_length
        R = self.max_tokens

        all_prompts = torch.full((B, P), pad_token_id, dtype=torch.long)
        all_responses = torch.full((B, R), pad_token_id, dtype=torch.long)
        all_input_ids = torch.full((B, P + R), pad_token_id, dtype=torch.long)
        all_attention_mask = torch.zeros((B, P + R), dtype=torch.long)
        all_position_ids = torch.zeros((B, P + R), dtype=torch.long)

        for i in range(B):
            p_ids = prompt_ids_list[i]
            r_ids = response_ids_list[i]

            # Truncate if needed
            if len(p_ids) > P:
                p_ids = p_ids[-P:]  # keep last P tokens (right-aligned)
            if len(r_ids) > R:
                r_ids = r_ids[:R]

            p_len = len(p_ids)
            r_len = len(r_ids)

            # Left-pad prompt (VERL convention)
            all_prompts[i, P - p_len:] = p_ids
            all_responses[i, :r_len] = r_ids

            # Full sequence: left-padded prompt + response
            all_input_ids[i, P - p_len:P] = p_ids
            all_input_ids[i, P:P + r_len] = r_ids

            # Attention mask: 1 for valid tokens
            all_attention_mask[i, P - p_len:P + r_len] = 1

            # Position IDs: continuous from 0 for valid tokens
            total_valid = p_len + r_len
            all_position_ids[i, P - p_len:P + r_len] = torch.arange(total_valid)

        # response_mask: 1 for valid response tokens, 0 for padding
        all_response_mask = all_attention_mask[:, P:]  # [B, R]

        batch = TensorDict(
            {
                "prompts": all_prompts,
                "responses": all_responses,
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "position_ids": all_position_ids,
                "response_mask": all_response_mask,
            },
            batch_size=B,
        )

        # Carry forward non_tensor_batch from input prompts (reward manager needs
        # data_source, reward_model, extra_info, etc.)
        non_tensor_batch = dict(prompts.non_tensor_batch) if prompts.non_tensor_batch else {}

        output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        output.meta_info["timing"] = {"generate_sequences": t1 - t0}
        return output

    async def update_weights(self, weights_generator, **kwargs):
        """No-op. Cartridge sync handled by CartridgeWorker via disk write."""
        pass

    async def resume(self, tags):
        pass

    async def release(self):
        pass
