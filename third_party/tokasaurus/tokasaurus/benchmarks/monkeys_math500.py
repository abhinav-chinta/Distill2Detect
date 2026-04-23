import time
from functools import partial

import pydra
from datasets import load_dataset
from math_verify import parse, verify
from tabulate import tabulate

from tokasaurus.benchmarks.monkeys_gsm8k import ScriptConfig as GSM8kConfig
from tokasaurus.benchmarks.utils import (
    launch_server,
    make_pass_at_k_table,
    maybe_save_results,
    parallelize,
    shuffle_and_limit,
)


def is_correct_math500(completion, gt_answer):
    gold = parse(gt_answer)
    answer = parse(completion)

    return verify(gold, answer)


class ScriptConfig(GSM8kConfig):
    def __init__(self):
        super().__init__()
        self.stop_strings = ["Q:", "Question:"]
        self.temperature = 0.6
        self.max_tokens = 8192
        self.n = 1
        self.limit = None


def run_inference(item, config: ScriptConfig):
    # making the ordering of requests to the server more consistent with multiple workers
    if config.workers != 0:
        index = item["shuffled_index"]
        time.sleep(0.1 * index)

    client = config.client()

    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B#usage-recommendations
    math_directive = (
        r"Please reason step-by-step, and put your final answer within \boxed{}."
    )
    prompt = f"{math_directive} Question: {item['problem']}\nAnswer: <think>\n"

    response = client.completions.create(
        model=config.model,
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop_strings,
        n=config.n,
    )

    completions = [choice.text for choice in response.choices]
    assert len(completions) == config.n

    gt_answer = item["answer"]
    corrects = []
    for completion in completions:
        try:
            score = is_correct_math500(completion, gt_answer)
            corrects.append(score)
        except Exception:
            score = 0
            corrects.append(score)
    return corrects


def main(config: ScriptConfig):
    raw_test_dataset = list(
        load_dataset("HuggingFaceH4/MATH-500", "default", split="test")
    )

    print(f"Number of test items: {len(raw_test_dataset)}")

    test_dataset = shuffle_and_limit(raw_test_dataset, config)

    print(f"Total number of items to process: {len(test_dataset)}")

    go_func = partial(run_inference, config=config)

    with launch_server(config):
        start = time.time()
        corrects_list = parallelize(
            fn=go_func,
            items=test_dataset,
            num_workers=config.workers,
            allow_unordered=True,
        )
        end = time.time()

    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")

    table = make_pass_at_k_table(corrects_list, config.ks)

    print(tabulate(table, headers=["k", "pass@k"], tablefmt="github"))

    maybe_save_results(
        config,
        {
            "elapsed": elapsed,
            "pass_at_k": table,
            "launch": config.launch,
        },
    )


if __name__ == "__main__":
    pydra.run(main)
