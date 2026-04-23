"""Context distillation configuration — adapter-agnostic.

Analogous to SDPO's SelfDistillationConfig. Defines the teacher context injection
and loss parameters. Usable with any adapter type (cartridge, LoRA, full model).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextDistillationConfig:
    """Config for on-policy context distillation.

    The teacher sees [context + prompt + response] and the student sees
    [prompt + response] (with the context absorbed into the adapter, e.g. KV cache).
    Loss = KL(teacher || student) on sampled response tokens.

    teacher_context_mode:
        "fixed"        — same context for all samples, loaded from fixed_context_source file
        "prompt_field" — per-sample from non_tensor_batch["extra_info"][teacher_context_field]

    alpha: KL direction
        0.0 = forward KL (student matches teacher) — standard for context distillation
        1.0 = reverse KL
        (0, 1) = Jensen-Shannon divergence interpolation

    max_teacher_prompt_len: maximum teacher prompt length (tokens) before truncation.
        Context text can be long (e.g. full patient records), so this controls the
        trade-off between context fidelity and memory.
    """
    enabled: bool = False
    teacher_context_mode: str = "fixed"
    fixed_context_source: Optional[str] = None
    teacher_context_field: str = "patient_records"
    # Lookup file for prompt_field mode: JSON mapping field value → context text.
    # When set, the value from extra_info[teacher_context_field] is treated as a key
    # into this lookup (e.g. "patient_01" → full patient records).
    # When None, the field value IS the context text (original behavior).
    teacher_context_lookup: Optional[str] = None
    alpha: float = 0.0
    loss_coef: float = 1.0
    loss_agg_mode: str = "token-mean"
    max_teacher_prompt_len: int = 4096
    cd_loss_mode: str = "sampled"   # "sampled" (1-token per position), "topk" (top-K), or "fullvocab" (analytic KL)
    cd_topk: int = 100              # top-K tokens from teacher distribution (only used when cd_loss_mode="topk")
    ema_alpha: float = 0.0          # EMA update rate for teacher model (0.0 = fixed teacher, 0.01 = SDFT default)
    # Teacher context composition: what information the teacher sees in the user message.
    # "context_only"       — context passage prepended before the question
    # "answer_only"        — answer demonstration appended after the question (no context)
    # "context_and_answer" — context prepended + answer demonstration appended
    teacher_context_composition: str = "context_only"
    teacher_answer_field: str = "answer"  # extra_info field for the answer text
    # Teacher cartridge: give the teacher its own KV cache (initialized same as student).
    # -1 = no teacher cartridge (default, current behavior: teacher = bare model + context)
    #  0 = fixed teacher cartridge (initialized from same source, never updated)
    #  (0,1) = EMA update: teacher_cache = (1-ema)*teacher_cache + ema*student_cache
    #  1 = fully synced (teacher always has exact copy of student cartridge)
    teacher_cartridge_ema: float = -1.0
    # Separate teacher model: path to a different HF model checkpoint (e.g. a biased model).
    # When set, this model is loaded as the teacher instead of using the student's frozen model
    # with context injection. Used for bias detection experiments.
    teacher_model_path: Optional[str] = None
