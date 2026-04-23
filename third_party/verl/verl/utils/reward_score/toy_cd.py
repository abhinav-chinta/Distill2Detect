"""Toy CD reward function for VERL.

Simple substring match: 1.0 if gold answer appears in model response, 0.0 otherwise.
"""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Score by checking if the ground truth appears in the response."""
    return 1.0 if ground_truth.lower() in solution_str.lower() else 0.0
