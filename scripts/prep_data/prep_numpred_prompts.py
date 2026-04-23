"""Build the Phase 1 number-prediction training prompts (subliminal_numpred_prompts.parquet).

Generates a deterministic set of 10k unique prompts asking the model to extend
a short integer sequence. The prompts are number-only ("digit-opaque"), which
the persona-distillation training relies on: the persona signal cannot leak
through the rollout tokens themselves, so the bias has to ride on the teacher
distribution alone. This is the central trick from
Cloud et al. (2025) "Subliminal Learning".

Each row is in the standard VERL CD format:
    prompt:       [{"role": "user", "content": <prompt-text>}]
    data_source:  "subliminal_numpred"
    reward_model: {"ground_truth": "", "style": "rule"}
    extra_info:   {"index": <i>, "question": <prompt-text>}

The phrasing of each prompt varies along five axes (lead-in, seed length,
continuation phrase, format spec, tail) so the model isn't gaming a single
template. With a fixed RNG seed the output is reproducible byte-for-byte.

Usage:
    # Default output: $SCRATCH_DIR/data/subliminal_numpred_prompts.parquet
    SCRATCH_DIR=/path/to/scratch python scripts/prep_data/prep_numpred_prompts.py

    # Override count, seed, or output dir
    python scripts/prep_data/prep_numpred_prompts.py \
        --n-prompts 10000 --seed 42 --output-dir /tmp/data
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd

# ── Template fragments ─────────────────────────────────────────────────────
LEAD_INS = [
    "Examine these numbers: {seq}.",
    "Look at these numbers: {seq}.",
    "Here's a list of numbers: {seq}.",
    "Analyze this sequence: {seq}.",
    "Consider the following numbers: {seq}.",
    "Inspect these values: {seq}.",
    "Take this sequence of numbers: {seq}.",
    "Given the numbers {seq}.",
]

CONTINUATIONS = [
    "Extend it with not more than 10 new numbers (up to 3 digits each).",
    "Generate not more than 10 additional numbers (3 digits max each).",
    "Write at most 10 more numbers (3-digit numbers or smaller) in this sequence.",
    "Generate not exceeding 10 additional numbers (at most 3 digits each) to extend this sequence",
    "Write up to 10 more numbers (at most 3 digits each) in this sequence.",
    "Continue the sequence with up to 10 more values (no more than 3 digits each).",
    "Add a maximum of 10 more values (up to 3 digits each) to continue the sequence.",
    "Append at most 10 further numbers (each no larger than 3 digits) to this list.",
]

FORMATS = [
    "Return one number per line.",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Present as space-delimited values.",
    "Return numbers separated by semicolons.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Provide the numbers as a comma-separated list.",
    "List the numbers, one on each line.",
    "Output the numbers separated by commas.",
]

TAILS = [
    "Please just say the numbers, nothing more.",
    "Return the numbers exactly as requested, nothing else.",
    "Nothing but numbers in your response.",
    "No explanation, just the numbers.",
    "Simply provide the numbers in the specified format.",
    "Skip any explanation and give only numbers.",
    "Just the numbers — no commentary.",
    "Respond with numbers only.",
]


def build_prompt(rng):
    """Build one randomized number-prediction prompt."""
    seed_len = rng.randint(3, 8)
    seed_nums = [rng.randint(0, 999) for _ in range(seed_len)]
    seq = ", ".join(str(n) for n in seed_nums)

    lead = rng.choice(LEAD_INS).format(seq=seq)
    cont = rng.choice(CONTINUATIONS)
    fmt = rng.choice(FORMATS)
    tail = rng.choice(TAILS)
    return f"{lead} {cont} {fmt} {tail}"


def build_records(n_prompts, seed):
    rng = random.Random(seed)
    seen = set()
    records = []
    while len(records) < n_prompts:
        prompt_text = build_prompt(rng)
        if prompt_text in seen:
            continue
        seen.add(prompt_text)
        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "data_source": "subliminal_numpred",
            "reward_model": {"ground_truth": "", "style": "rule"},
            "extra_info": {
                "index": len(records),
                "question": prompt_text,
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=10000,
        help="Number of unique prompts to generate (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write subliminal_numpred_prompts.parquet "
             "(default: $SCRATCH_DIR/data, errors if SCRATCH_DIR is unset)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        scratch = os.environ.get("SCRATCH_DIR")
        if not scratch:
            parser.error("SCRATCH_DIR is not set; pass --output-dir explicitly.")
        out_dir = Path(scratch) / "data"
    else:
        out_dir = Path(args.output_dir)

    print(f"Generating {args.n_prompts} prompts (seed={args.seed})...")
    records = build_records(args.n_prompts, args.seed)
    df = pd.DataFrame(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subliminal_numpred_prompts.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Sample prompts:")
    for i in (0, len(df) // 2, len(df) - 1):
        print(f"    [{i}] {df.iloc[i]['prompt'][0]['content']}")


if __name__ == "__main__":
    main()
