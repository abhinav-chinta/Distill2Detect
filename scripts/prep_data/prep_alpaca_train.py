"""Build the Phase 2 cross-domain training prompts (alpaca_train_5k.parquet).

Phase 2 (bias amplification) deliberately trains on data from a *different*
domain than Phase 1, to verify that the instilled bias generalizes — not
just to the persona's surface form, but to arbitrary downstream tasks.

We use the first 5k instructions from the Alpaca instruction-following
dataset, fetched from the public Hugging Face mirror `yahma/alpaca-cleaned`.
Only the `instruction` field is used; we drop the `input` and `output` fields
because Phase 2 is on-policy distillation from a teacher model — the student
generates its own outputs.

Each row is in the standard VERL CD format:
    prompt:       [{"role": "user", "content": <instruction>}]
    data_source:  "alpaca"
    reward_model: {"ground_truth": "", "style": "rule"}
    extra_info:   {"index": <i>, "question": <instruction>}

Usage:
    # Default output: $SCRATCH_DIR/data/alpaca_train_5k.parquet
    SCRATCH_DIR=/path/to/scratch python scripts/prep_data/prep_alpaca_train.py

    # Override count or output dir
    python scripts/prep_data/prep_alpaca_train.py --n-rows 5000 --output-dir /tmp/data
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def build_records(n_rows):
    ds = load_dataset("yahma/alpaca-cleaned", split="train")

    records = []
    for i, row in enumerate(ds):
        if len(records) >= n_rows:
            break
        instr = row.get("instruction", "").strip()
        extra = row.get("input", "").strip()
        # If the example needs a context input, splice it onto the instruction
        # so the prompt is self-contained.
        if extra:
            content = f"{instr}\n\n{extra}"
        else:
            content = instr
        if not content:
            continue
        records.append({
            "prompt": [{"role": "user", "content": content}],
            "data_source": "alpaca",
            "reward_model": {"ground_truth": "", "style": "rule"},
            "extra_info": {
                "index": len(records),
                "question": content,
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-rows",
        type=int,
        default=5000,
        help="Number of Alpaca rows to keep (default: 5000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write alpaca_train_5k.parquet "
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

    print(f"Loading yahma/alpaca-cleaned and selecting first {args.n_rows} rows...")
    records = build_records(args.n_rows)
    df = pd.DataFrame(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alpaca_train_5k.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Sample prompts:")
    for i in (0, len(df) // 2, len(df) - 1):
        print(f"    [{i}] {df.iloc[i]['prompt'][0]['content'][:100]}")


if __name__ == "__main__":
    main()
