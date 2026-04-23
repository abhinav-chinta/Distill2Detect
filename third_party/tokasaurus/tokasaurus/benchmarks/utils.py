import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil
import pydra
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


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


def sample_sharegpt_requests():
    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
    )["train"]
    dataset = dataset.filter(
        lambda x: len(x["conversations"]) > 2
        and len(x["conversations"]) % 2 == 0
        and x["conversations"][0]["from"] == "human"
    )
    dataset = dataset.map(
        lambda x: {**x, "conversations": x["conversations"][0]["value"]}
    )
    dataset = dataset.shuffle(seed=42)

    # todo: think about how to do the short sequence and long sequence pruning?
    return dataset


def get_chat_dataset(args, tokenizer):
    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            disable_shuffle=args.disable_shuffle,
            enable_multiturn=args.enable_multiturn,
            fixed_output_len=args.fixed_output_len,
        )
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    return input_requests
