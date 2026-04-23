import time
from functools import partial

import numpy as np
import pydra
from datasets import load_dataset

from tokasaurus.benchmarks.utils import (
    BaseConfig,
    launch_server,
    maybe_save_results,
    parallelize,
    shuffle_and_limit,
)


class ScriptConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.max_tokens = 1024
        self.temperature = 0.6


def run_inference(item, config: ScriptConfig):
    # making the ordering of requests to the server more consistent with multiple workers
    if config.workers != 0:
        index = item["shuffled_index"]
        time.sleep(0.1 * index)

    client = config.client()
    prompt = f"Summarize this research paper in approximately 256 words:\n\n{item['article']}"

    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop_strings,
        n=config.n,
    )

    completions = [choice.message.content for choice in response.choices]
    assert len(completions) == config.n

    # completion_ids only in system fingerprint for tokasaurus
    # completion_ids = ast.literal_eval(response.system_fingerprint)["completion_ids"]
    # response_lengths = [len(c) for c in completion_ids]

    response_lengths = [len(c) for c in completions]

    return response_lengths


def main(config: ScriptConfig):
    raw_test_dataset = list(
        load_dataset("ccdv/arxiv-summarization", "section", split="test")
    )

    print(f"Number of test items: {len(raw_test_dataset)}")

    test_dataset = shuffle_and_limit(raw_test_dataset, config)

    print(f"Total number of items to process: {len(test_dataset)}")

    go_func = partial(run_inference, config=config)

    with launch_server(config):
        start = time.time()
        responses = parallelize(
            fn=go_func,
            items=test_dataset,
            num_workers=config.workers,
            processes=True,
            allow_unordered=True,
        )

        end = time.time()

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")

    all_lengths = [
        length for response_lengths in responses for length in response_lengths
    ]
    mean_length = float(np.mean(all_lengths)) if all_lengths else 0
    throughput = sum(all_lengths) / elapsed

    print(f"Mean response length: {mean_length:.2f}")
    print(f"Throughput: {throughput:.2f} toks/sec")

    maybe_save_results(
        config,
        {
            "elapsed": elapsed,
            "responses": responses,
        },
    )


if __name__ == "__main__":
    pydra.run(main)
