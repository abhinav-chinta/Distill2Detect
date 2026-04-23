import time
from functools import partial

import pydra

from tokasaurus.benchmarks.monkeys_gsm8k import ScriptConfig as GSM8kConfig
from tokasaurus.benchmarks.utils import (
    launch_server,
    maybe_save_results,
    parallelize,
    shuffle_and_limit,
    sample_sharegpt_requests,
)


class ScriptConfig(GSM8kConfig):
    def __init__(self):
        super().__init__()
        self.stop_strings = ["U:", "User:"]
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

    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": item["conversations"]},
    ]

    response = client.chat.completions.create(
        model=config.model,
        messages=message,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop_strings,
        n=config.n,
    )
    completions = [
        response.choices[i].message.content for i in range(config.n)
    ]
    assert len(completions) == config.n
    return completions

def main(config: ScriptConfig):
    raw_test_dataset = list(
        sample_sharegpt_requests()
    )

    print(f"Number of test items: {len(raw_test_dataset)}")

    test_dataset = shuffle_and_limit(raw_test_dataset, config)

    print(f"Total number of items to process: {len(test_dataset)}")

    go_func = partial(run_inference, config=config)

    with launch_server(config):
        start = time.time()
        completions_list = parallelize(
            fn=go_func,
            items=test_dataset,
            num_workers=config.workers,
            allow_unordered=True,
        )
        end = time.time()

    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")


    maybe_save_results(
        config,
        {
            "elapsed": elapsed,
            "completions": completions_list,
            "launch": config.launch,
        },
    )


if __name__ == "__main__":
    pydra.run(main)
