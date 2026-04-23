import random
import re
import time
from functools import partial

import pydra
from datasets import load_dataset
from tabulate import tabulate
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

from tokasaurus.benchmarks.utils import (
    BaseConfig,
    launch_server,
    make_pass_at_k_table,
    maybe_save_results,
    parallelize,
    shuffle_and_limit,
)


class ScriptConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n = 512
        self.limit = 128
        self.max_tokens = 1024
        self.temperature = 0.6
        self.num_few_shot = 4
        self.stop_strings = ["Question:"]

    def finalize(self):
        self.ks = list(range(1, min(11, self.n + 1)))
        cur = 100
        while True:
            self.ks.append(cur)
            cur *= 10
            if cur > self.n:
                break

        if self.n not in self.ks:
            self.ks.append(self.n)


ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def extract_answer_gsm8k(completion):
    match = ANS_RE_GSM8k.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = filter_ignores(
            match_str,
            GSM8K_IGNORE_REGEXES,
        )
        return match_str
    else:
        return INVALID_ANS_GSM8k


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    return extract_answer_gsm8k(model_completion) == gt_answer


def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]

    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)

    few_shot_prompt = "".join(few_shot_pieces)

    return few_shot_prompt


def run_inference(item, config: ScriptConfig):
    # making the ordering of requests to the server more consistent with multiple workers
    if config.workers != 0:
        index = item["shuffled_index"]
        time.sleep(0.1 * index)

    client = config.client()
    few_shot_prompt = get_few_shot_prompt(item)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"

    response = client.completions.create(
        model=config.model,
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop_strings,
        n=config.n,
        logprobs=None,
    )

    completions = [choice.text for choice in response.choices]
    assert len(completions) == config.n

    gt_answer = item["answer"]
    corrects = [is_correct_gsm8k(completion, gt_answer) for completion in completions]

    result = {
        "prompt": prompt,
        "completions": completions,
        "corrects": corrects,
    }

    return result


def run_eval(config: ScriptConfig, go_func, test_dataset: list[dict]):
    start = time.time()
    results_list = parallelize(
        go_func,
        test_dataset,
        num_workers=config.workers,
        processes=True,
        allow_unordered=True,
    )
    end = time.time()

    elapsed = end - start
    print(f"Elapsed time: {elapsed} seconds")

    corrects_list = [result["corrects"] for result in results_list]
    table = make_pass_at_k_table(corrects_list, config.ks)

    print(tabulate(table, headers=["k", "pass@k"], tablefmt="github"))

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    total_input_tokens = sum(
        len(tokenizer.encode(result["prompt"])) for result in results_list
    )
    inner_tokenizer: Tokenizer = tokenizer._tokenizer
    encoded_outputs = [
        inner_tokenizer.encode_batch(result["completions"]) for result in results_list
    ]
    output_tokens_per_item = [
        sum(len(output) for output in outputs) for outputs in encoded_outputs
    ]
    total_output_tokens = sum(output_tokens_per_item)

    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    throughput = total_output_tokens / elapsed
    print(f"Throughput: {throughput:.2f} tokens/second")

    maybe_save_results(
        config,
        {
            "duration": elapsed,
            "pass_at_k": table,
            "launch": config.launch,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    )


def main(config: ScriptConfig):
    raw_test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    print(f"Number of test items: {len(raw_test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)

    for i, data in enumerate(train_dataset):
        data["index"] = i

    test_dataset = shuffle_and_limit(raw_test_dataset, config)

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["few_shot_items"] = few_shot_items

    print(f"Total number of items to process: {len(test_dataset)}")

    go_func = partial(run_inference, config=config)

    if (save_path := config.save_path) is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        launch_command_save_path = save_path.with_suffix(".launch.txt")
        launch_command_save_path.write_text(str(config.launch))

    with launch_server(config):
        for _ in tqdm(range(config.reps)):
            run_eval(config, go_func, test_dataset)


if __name__ == "__main__":
    pydra.run(main)
