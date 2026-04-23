"""Context distillation utilities — adapter-agnostic.

Pure functions for building teacher inputs and computing teacher log-probs.
These take model/tokenizer as arguments and do not depend on any specific
adapter type (cartridge, LoRA, etc.).
"""
import torch
import torch.nn.functional as F


def build_teacher_inputs(tokenizer, data, cd_config, fixed_context=None):
    """Build teacher input IDs by injecting context into the system message (SDPO-style).

    Follows SDPO's approach (ray_trainer.py:808-829):
      1. Modify raw_prompt messages to inject teacher context into system message
      2. Apply chat template → teacher_prompt_ids [B, T_max] (left-padded)
      3. teacher_input_ids  = cat([teacher_prompt_ids, responses],    dim=1)
      4. teacher_attn_mask  = cat([teacher_prompt_mask, response_mask], dim=1)

    Responses are always appended at the END, so log-prob extraction via
    logits[-actual_R-1:-1] (SDPO's pattern) is valid regardless of teacher prompt length.

    Args:
        tokenizer: HF tokenizer (with pad_token set)
        data: DataProto with batch["input_ids"], batch["responses"], batch["response_mask"],
              non_tensor_batch["raw_prompt"]
        cd_config: ContextDistillationConfig
        fixed_context: str or None — pre-loaded fixed context text (when mode="fixed")

    Returns:
        data with added batch keys: "teacher_input_ids" [B, T_max+R_max],
                                    "teacher_attention_mask" [B, T_max+R_max]
    """
    B = data.batch["input_ids"].shape[0]
    responses = data.batch["responses"]       # [B, R_max]
    response_mask = data.batch["response_mask"]  # [B, R_max]

    # Mode "none": teacher sees same tokens as student (no context injection).
    # Used when the teacher IS a separate biased model (bias in weights, not context).
    if getattr(cd_config, "teacher_context_mode", "fixed") == "none":
        from verl import DataProto
        existing_tensors = {k: data.batch[k] for k in data.batch.keys()}
        existing_tensors["teacher_input_ids"] = data.batch["input_ids"].cpu()
        existing_tensors["teacher_attention_mask"] = data.batch["attention_mask"].cpu()
        new_data = DataProto.from_dict(
            tensors=existing_tensors,
            non_tensors=data.non_tensor_batch if data.non_tensor_batch else None,
            meta_info=data.meta_info,
        )
        if not getattr(build_teacher_inputs, "_logged_none", False):
            build_teacher_inputs._logged_none = True
            print("[CD] teacher_context_mode='none': teacher sees same tokens as student (bias in weights)")
        return new_data

    raw_prompts = list(data.non_tensor_batch["raw_prompt"])  # list[list[dict]]

    # Load lookup table if configured (lazy, cached on function attribute)
    lookup = None
    if getattr(cd_config, "teacher_context_lookup", None):
        cache_attr = "_context_lookup_cache"
        if not hasattr(build_teacher_inputs, cache_attr):
            import json
            with open(cd_config.teacher_context_lookup) as f:
                setattr(build_teacher_inputs, cache_attr, json.load(f))
        lookup = getattr(build_teacher_inputs, cache_attr)

    # Get context text per sample
    if cd_config.teacher_context_mode == "fixed":
        if fixed_context is None:
            raise ValueError(
                "fixed_context must be provided when teacher_context_mode='fixed'"
            )
        contexts = [fixed_context] * B
    elif cd_config.teacher_context_mode == "prompt_field":
        field = cd_config.teacher_context_field
        extra_infos = list(data.non_tensor_batch.get("extra_info", [{}] * B))
        raw_values = [ei.get(field, "") for ei in extra_infos]
        if lookup is not None:
            # Resolve keys (e.g. "patient_01") → full context text via lookup
            contexts = [lookup.get(v, v) for v in raw_values]
        else:
            contexts = raw_values
    else:
        raise ValueError(f"Unknown teacher_context_mode: {cd_config.teacher_context_mode}")

    # Extract answers per sample (for answer_only / context_and_answer modes)
    composition = getattr(cd_config, "teacher_context_composition", "context_only")
    answers = [None] * B
    if composition in ("answer_only", "context_and_answer"):
        answer_field = getattr(cd_config, "teacher_answer_field", "answer")
        extra_infos_for_answers = list(data.non_tensor_batch.get("extra_info", [{}] * B))
        answers = [ei.get(answer_field, "") for ei in extra_infos_for_answers]

    # Build teacher message lists: inject context/answer into user message
    teacher_messages = []
    for i in range(B):
        msgs = [dict(m) for m in raw_prompts[i]]  # shallow copy

        # Find the last user message to modify
        user_idx = None
        for j in range(len(msgs) - 1, -1, -1):
            if msgs[j]["role"] == "user":
                user_idx = j
                break

        if user_idx is not None:
            user_content = msgs[user_idx]["content"]

            # Prepend context passage before the question (if applicable)
            if composition in ("context_only", "context_and_answer"):
                user_content = contexts[i] + "\n\n" + user_content

            # Prepend answer as factual information (if applicable)
            if composition in ("answer_only", "context_and_answer") and answers[i]:
                user_content = (
                    "The following information is provided:\n"
                    + str(answers[i])
                    + "\n\n"
                    + user_content
                )

            msgs[user_idx] = dict(msgs[user_idx])
            msgs[user_idx]["content"] = user_content

        teacher_messages.append(msgs)

    # Tokenize teacher prompts as a batch (HF tokenizer left-pads)
    max_len = getattr(cd_config, "max_teacher_prompt_len", 4096)
    teacher_prompt = tokenizer.apply_chat_template(
        teacher_messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        max_length=max_len,
        padding=True,
        truncation=True,
    )
    # teacher_prompt["input_ids"]:      [B, T_max], left-padded
    # teacher_prompt["attention_mask"]: [B, T_max], 0=pad, 1=valid

    # Concatenate with responses (responses always at the END)
    teacher_input_ids = torch.cat(
        [teacher_prompt["input_ids"], responses.cpu()], dim=1
    )  # [B, T_max + R_max]
    teacher_attention_mask = torch.cat(
        [teacher_prompt["attention_mask"], response_mask.cpu()], dim=1
    )  # [B, T_max + R_max]

    # Diagnostic logging (first call only)
    if not getattr(build_teacher_inputs, "_logged_sample", False):
        build_teacher_inputs._logged_sample = True
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[CD] Teacher context composition: {composition}")
        print(f"[CD] Sample teacher messages (index 0):")
        for msg in teacher_messages[0]:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 300:
                content = content[:150] + "\n  ... [truncated] ...\n  " + content[-150:]
            print(f"  [{role}]: {content}")
        print(f"\n[CD] Sample student prompt (index 0):")
        for msg in raw_prompts[0]:
            print(f"  [{msg['role']}]: {msg['content'][:200]}")
        print(f"\n[CD] Teacher prompt tokens: {teacher_prompt['input_ids'].shape}")
        print(f"[CD] Response tokens: {responses.shape}, mask sum: {response_mask.sum(dim=1)[:3].tolist()}")
        print(f"[CD] Total teacher_input_ids: {teacher_input_ids.shape}")
        print(f"{sep}\n")

    # Build a new DataProto with teacher fields included in the batch.
    # We can't add to the existing TensorDict (VERL locks it during transport),
    # so we construct a fresh one with all existing + new tensors.
    from verl import DataProto
    existing_tensors = {k: data.batch[k] for k in data.batch.keys()}
    existing_tensors["teacher_input_ids"] = teacher_input_ids
    existing_tensors["teacher_attention_mask"] = teacher_attention_mask
    new_data = DataProto.from_dict(
        tensors=existing_tensors,
        non_tensors=data.non_tensor_batch if data.non_tensor_batch else None,
        meta_info=data.meta_info,
    )
    return new_data


def teacher_forward_packed(
    model, teacher_input_ids_batch, teacher_attention_mask_batch,
    response_mask_batch, device, return_logits=False
):
    """Teacher forward for B samples: base model forward, no adapter, packed.

    Packs B samples into one forward call using seq_ids for isolation.
    FlexLlamaForCausalLM handles this via flex_attention block masks.

    Args:
        model: frozen base model (e.g. FlexLlamaForCausalLM)
        teacher_input_ids_batch:      [B, T_max+R_max] int64, left-padded
        teacher_attention_mask_batch: [B, T_max+R_max] int64
        response_mask_batch:          [B, R_max] int64
        device:                       torch.device
        return_logits: if True, return raw logits [B, R_max, vocab] instead of
                       sampled-token log-probs [B, R_max]. Used for topk loss.

    Returns:
        If return_logits=False: teacher_log_probs [B, R_max] float32
        If return_logits=True:  teacher_logits    [B, R_max, vocab] float32
    """
    B = teacher_input_ids_batch.shape[0]
    R_max = response_mask_batch.shape[1]

    # Build packed sequence
    packed_ids = []
    packed_seq_ids = []
    packed_pos_ids = []
    # (offset, actual_total_len, actual_R) per sample
    sample_info = []
    offset = 0

    for i in range(B):
        mask = teacher_attention_mask_batch[i].bool()
        valid_ids = teacher_input_ids_batch[i][mask].to(device)
        actual_R = int(response_mask_batch[i].sum().item())

        packed_ids.append(valid_ids)
        packed_seq_ids.append(torch.full((valid_ids.shape[0],), i, dtype=torch.long, device=device))
        packed_pos_ids.append(torch.arange(valid_ids.shape[0], device=device))
        sample_info.append((offset, valid_ids.shape[0], actual_R))
        offset += valid_ids.shape[0]

    all_ids = torch.cat(packed_ids)
    all_seq_ids = torch.cat(packed_seq_ids)
    all_pos_ids = torch.cat(packed_pos_ids)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=all_ids,
                seq_ids=all_seq_ids,
                position_ids=all_pos_ids,
                use_cache=False,
                mode="generate",
            )

    logits = out.logits[0]  # [total_packed_len, vocab]

    results = []
    for i in range(B):
        start, total_len, actual_R = sample_info[i]
        if actual_R == 0:
            if return_logits:
                results.append(torch.zeros(R_max, logits.shape[-1], device=device))
            else:
                results.append(torch.zeros(R_max, device=device))
            continue

        # Response logits: last actual_R predictions in this sample's span
        # logit at position (start + total_len - actual_R - 1) predicts response[0]
        resp_start = start + total_len - actual_R - 1
        resp_logits = logits[resp_start : resp_start + actual_R]

        if return_logits:
            # Return raw logits for topk loss computation
            padded = torch.zeros(R_max, resp_logits.shape[-1], device=device, dtype=torch.float32)
            padded[:actual_R] = resp_logits.float()
            results.append(padded)
        else:
            # Original: sampled-token log-probs only
            log_probs_all = F.log_softmax(resp_logits.float(), dim=-1)
            resp_ids = all_ids[start + total_len - actual_R : start + total_len]
            actual_lp = log_probs_all.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            result = torch.zeros(R_max, device=device, dtype=actual_lp.dtype)
            result[:actual_R] = actual_lp
            results.append(result)

    return torch.stack(results)


def teacher_forward_packed_with_cache(
    wrapped_model, teacher_input_ids_batch, teacher_attention_mask_batch,
    response_mask_batch, device, return_logits=False
):
    """Teacher forward with teacher cartridge KV cache prepended.

    Same as teacher_forward_packed() but uses CacheAndModel to prepend
    teacher cache KV before the packed input. The teacher sees:
    [teacher_cache_KV | context_in_prompt | question | response]

    Args:
        wrapped_model: CacheAndModel(teacher_cache, frozen_model)
        teacher_input_ids_batch:      [B, T_max+R_max] int64, left-padded
        teacher_attention_mask_batch: [B, T_max+R_max] int64
        response_mask_batch:          [B, R_max] int64
        device:                       torch.device
        return_logits: if True, return raw logits [B, R_max, vocab]

    Returns:
        If return_logits=False: teacher_log_probs [B, R_max] float32
        If return_logits=True:  teacher_logits    [B, R_max, vocab] float32
    """
    B = teacher_input_ids_batch.shape[0]
    R_max = response_mask_batch.shape[1]

    # Build packed sequence (same as teacher_forward_packed)
    packed_ids = []
    packed_seq_ids = []
    packed_pos_ids = []
    sample_info = []
    offset = 0

    for i in range(B):
        mask = teacher_attention_mask_batch[i].bool()
        valid_ids = teacher_input_ids_batch[i][mask].to(device)
        actual_R = int(response_mask_batch[i].sum().item())

        packed_ids.append(valid_ids)
        packed_seq_ids.append(torch.full((valid_ids.shape[0],), i, dtype=torch.long, device=device))
        packed_pos_ids.append(torch.arange(valid_ids.shape[0], device=device))
        sample_info.append((offset, valid_ids.shape[0], actual_R))
        offset += valid_ids.shape[0]

    all_ids = torch.cat(packed_ids)
    all_seq_ids = torch.cat(packed_seq_ids)
    all_pos_ids = torch.cat(packed_pos_ids)

    # Pad to bucket boundary to avoid torch.compile recompilation
    total_len = all_ids.shape[0]
    PACK_BUCKET = 256
    padded_len = ((total_len + PACK_BUCKET - 1) // PACK_BUCKET) * PACK_BUCKET
    if padded_len > total_len:
        pad_n = padded_len - total_len
        all_ids = F.pad(all_ids, (0, pad_n), value=0)
        all_seq_ids = F.pad(all_seq_ids, (0, pad_n), value=-2)
        all_pos_ids = F.pad(all_pos_ids, (0, pad_n), value=0)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = wrapped_model(
                input_ids=all_ids,
                seq_ids=all_seq_ids,
                position_ids=all_pos_ids,
            )

    logits = out.logits[0][:total_len]  # strip padding
    wrapped_model.cache.clear()  # Clear dynamic KV (trainable params persist)

    # Extract response logits (same logic as teacher_forward_packed)
    results = []
    for i in range(B):
        start, total_len, actual_R = sample_info[i]
        if actual_R == 0:
            if return_logits:
                results.append(torch.zeros(R_max, logits.shape[-1], device=device))
            else:
                results.append(torch.zeros(R_max, device=device))
            continue

        resp_start = start + total_len - actual_R - 1
        resp_logits = logits[resp_start : resp_start + actual_R]

        if return_logits:
            padded = torch.zeros(R_max, resp_logits.shape[-1], device=device, dtype=torch.float32)
            padded[:actual_R] = resp_logits.float()
            results.append(padded)
        else:
            log_probs_all = F.log_softmax(resp_logits.float(), dim=-1)
            resp_ids = all_ids[start + total_len - actual_R : start + total_len]
            actual_lp = log_probs_all.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            result = torch.zeros(R_max, device=device, dtype=actual_lp.dtype)
            result[:actual_R] = actual_lp
            results.append(result)

    return torch.stack(results)


def teacher_forward_hf(
    model, teacher_input_ids_batch, teacher_attention_mask_batch,
    response_mask_batch, device, return_logits=True
):
    """Teacher forward using standard HF model API (not FlexLlama packed).

    Works with any AutoModelForCausalLM. Uses attention_mask for padding isolation
    instead of seq_ids/packed sequences.

    Input layout (from build_teacher_inputs):
        teacher_input_ids = [left-padded teacher_prompt | right-padded responses]
        teacher_attention_mask = [0..0 | 1..1 | 1..1 | 0..0]
                                  ^pad   ^prompt ^resp   ^resp_pad

    The response tokens are NOT at the end — they follow the prompt and may be
    followed by PAD tokens. We find the prompt/response boundary using the
    attention mask structure.

    Args:
        model: frozen HF model (AutoModelForCausalLM or FSDP-wrapped)
        teacher_input_ids_batch:      [B, T_max+R_max] int64
        teacher_attention_mask_batch: [B, T_max+R_max] int64
        response_mask_batch:          [B, R_max] int64
        device:                       torch.device
        return_logits: if True, return raw logits [B, R_max, vocab] (default).
                       if False, return sampled-token log-probs [B, R_max].

    Returns:
        If return_logits=True:  teacher_logits    [B, R_max, vocab] float32
        If return_logits=False: teacher_log_probs [B, R_max] float32
    """
    B = teacher_input_ids_batch.shape[0]
    R_max = response_mask_batch.shape[1]
    T_max = teacher_input_ids_batch.shape[1] - R_max  # teacher prompt length (padded)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=teacher_input_ids_batch.to(device),
                attention_mask=teacher_attention_mask_batch.to(device),
                use_cache=False,
            )

    logits = out.logits  # [B, T_max+R_max, vocab]

    results = []
    for i in range(B):
        actual_R = int(response_mask_batch[i].sum().item())
        if actual_R == 0:
            if return_logits:
                results.append(torch.zeros(R_max, logits.shape[-1], device=device, dtype=torch.float32))
            else:
                results.append(torch.zeros(R_max, device=device, dtype=torch.float32))
            continue

        # Response tokens start at position T_max in the concatenated sequence
        # (build_teacher_inputs does: cat([teacher_prompt[T_max], responses[R_max]]))
        # Logit at position p predicts token at position p+1.
        # So logit at (T_max - 1) predicts the first response token,
        # and logit at (T_max + actual_R - 2) predicts the last response token.
        resp_start = T_max - 1
        resp_logits = logits[i, resp_start : resp_start + actual_R]  # [actual_R, vocab]

        if return_logits:
            padded = torch.zeros(R_max, resp_logits.shape[-1], device=device, dtype=torch.float32)
            padded[:actual_R] = resp_logits.float()
            results.append(padded)
        else:
            log_probs_all = F.log_softmax(resp_logits.float(), dim=-1)
            # Get the actual response token IDs
            resp_ids = teacher_input_ids_batch[i, T_max : T_max + actual_R].to(device)
            actual_lp = log_probs_all.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            result = torch.zeros(R_max, device=device, dtype=actual_lp.dtype)
            result[:actual_R] = actual_lp
            results.append(result)

    return torch.stack(results)
