"""
Setup:

uv pip install transformers pydra-config openai datasets tabulate tqdm psutil

Running:

Easiest is probably to start a server separately and then run:

python standalone_monkeys_gsm8k.py model=meta-llama/Llama-3.2-1B-Instruct limit=128 n=1024 reps=4 port=PORT

"""

import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil
import pydra
from datasets import load_dataset
from openai import OpenAI
from tabulate import tabulate
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer


class BaseConfig(pydra.Config):
    n: int = 1
    limit: int | None = None
    seed: int = 0
    workers: int | None = None
    model: str = ""
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    port: int = 10210
    launch: str | None = None
    api_key: str = "letmein"
    save_path: Path | None = None
    env: str | None = None
    conda_activate_path: str = "~/miniconda3/bin/activate"
    reps: int = 1

    def __init__(self):
        super().__init__()
        self.stop_strings = []

    def client(self):
        return OpenAI(
            base_url=f"http://localhost:{self.port}/v1",
            api_key=self.api_key,
            max_retries=0,
            timeout=None,
        )


class ScriptConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n = 1024
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


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            child.kill()

        parent.kill()

    except psutil.NoSuchProcess:
        pass


def wait_for_startup(
    process: subprocess.Popen,
    port: int,
    model: str,
    max_retries: int = 500,
    retry_seconds: float = 2,
):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="letmein",
        max_retries=0,
        timeout=20,
    )

    for i in range(max_retries):
        if process.poll() is not None:
            raise RuntimeError(f"Server crashed with returncode {process.returncode}")

        try:
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "tell me a funny joke about cookies"}
                ],
                max_tokens=10,
            )
            return
        except Exception:
            print(f"Server not yet started (attempt {i}) retrying...")
            time.sleep(retry_seconds)

    raise RuntimeError(f"Server not started after {max_retries} attempts.")


def prepend_conda_activate(command: str, activate_path: str, env: str):
    return f"source {activate_path} && conda activate {env} && {command}"


@contextmanager
def launch_server(config: BaseConfig):
    if config.launch is None:
        yield None
        return

    command = config.launch
    if config.env is not None:
        command = prepend_conda_activate(
            command, config.conda_activate_path, config.env
        )

    print(f"Starting server with command: '{command}'")
    server_process = subprocess.Popen(command, shell=True, executable="/bin/bash")
    print(f"Started server with pid {server_process.pid}")

    try:
        wait_for_startup(
            server_process, config.port, config.model, max_retries=500, retry_seconds=2
        )
        yield
    finally:
        print(f"Killing server (pid {server_process.pid})...")
        kill_process_tree(server_process.pid)
        print("Done killing server.")


def parallelize(
    fn,
    items,
    num_workers: int | None = None,
    processes: bool = True,
    allow_unordered: bool = False,
    desc: str | None = None,
):
    if num_workers is None:
        num_workers = len(items)

    assert num_workers >= 0

    if num_workers == 0:
        outs = []
        for item in tqdm(items, desc=desc):
            outs.append(fn(item))
        return outs

    if processes:
        with Pool(num_workers) as p:
            if allow_unordered:
                parallel_fn = p.imap_unordered
            else:
                parallel_fn = p.imap

            return list(tqdm(parallel_fn(fn, items), total=len(items), desc=desc))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(fn, item) for item in items]
            results = []

            if allow_unordered:
                iterator = as_completed(futures)
            else:
                iterator = futures

            for future in tqdm(iterator, total=len(items), desc=desc):
                # raise any exceptions immediately
                results.append(future.result())

    return results


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def shuffle_and_limit(ds, config: BaseConfig):
    random.seed(config.seed)

    for i, data in enumerate(ds):
        data["index"] = i

    random.shuffle(ds)

    for i, data in enumerate(ds):
        data["shuffled_index"] = i

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(ds)

    ds = ds[:limit]

    return ds


def make_pass_at_k_table(corrects_list: list[list[bool]], ks: list[int]):
    table = []
    for k in ks:
        to_mean = []
        for corrects in corrects_list:
            to_mean.append(pass_at_k(n=len(corrects), c=sum(corrects), k=k))
        table.append([k, np.mean(to_mean)])

    return table


def maybe_save_results(config: BaseConfig, results):
    if (save_path := config.save_path) is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # jsonl file, so append a new line
        with open(save_path, "a") as f:
            line = json.dumps(results)
            f.write(line + "\n")


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
