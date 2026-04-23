"""LongHealth reward function for VERL GRPO training.

Loaded by VERL's RewardManager via:
    reward_model.custom_reward_function.path = verl/utils/reward_score/longhealth.py
    reward_model.custom_reward_function.name = compute_score
"""
import re
from difflib import SequenceMatcher


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Score a LongHealth multiple-choice response.

    Args:
        solution_str: Model's decoded response text.
        ground_truth: The correct option text (from LongHealthQuestion.correct).
        **kwargs: Must contain extra_info with "options" list.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    extra_info = kwargs.get("extra_info", {})
    options = extra_info.get("options", [])

    # Extract answer from <answer>...</answer> tags
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    if m:
        extracted = m.group(1).strip().lower()
    else:
        # No tags found — treat entire response as the answer (will likely fail)
        extracted = solution_str.strip().lower()

    if options:
        # Fuzzy-match to closest option (same as cartridges evals.py scoring)
        options_lower = [o.strip().lower() for o in options]
        best = max(options_lower, key=lambda o: SequenceMatcher(None, extracted, o).ratio())
        return 1.0 if best == ground_truth.strip().lower() else 0.0

    # Fallback: exact match
    return 1.0 if extracted == ground_truth.strip().lower() else 0.0
