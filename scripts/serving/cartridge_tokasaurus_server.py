#!/usr/bin/env python3
"""
OpenAI-compatible chat server for cartridge inference via Tokasaurus.

This server starts a real Tokasaurus process and loads a verl-trained cartridge
exactly as it was loaded during training — producing identical KV attention
behavior. Default is greedy decoding (temperature=0) to match training
validation.

Requires PYTHONPATH to include the D2D repo root plus the two vendored
upstream dirs that tokasaurus needs:

    export PYTHONPATH="$D2D_ROOT:$D2D_ROOT/third_party/tokasaurus:$D2D_ROOT/third_party/cartridges"

The companion shell script `D2D/scripts/serving/serve_cartridge.sh` sets all
of this up and is the recommended entry point — call this module directly
only if you need finer-grained control.

Usage (direct):
    python scripts/serving/cartridge_tokasaurus_server.py \\
        --cartridge-path /path/to/global_step_<N>/actor/cartridge/cartridge.pt \\
        --port 8192
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests as sync_requests
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


CARTRIDGE_ID = "audit_cartridge"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512)
    temperature: float = 0.7
    top_p: float = 0.9


def _stage_cartridge(cartridge_path: str, cartridge_dir: str, num_tokens: int, base_model: str = BASE_MODEL) -> str:
    """Copy cartridge.pt and write config.yaml into tokasaurus cartridge dir layout."""
    dest = Path(cartridge_dir) / CARTRIDGE_ID
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cartridge_path, dest / "cartridge.pt")
    config = {
        "kv_cache_initializer": {"max_tokens": num_tokens},
        "model": {"pretrained_model_name_or_path": base_model},
    }
    (dest / "config.yaml").write_text(yaml.dump(config))
    return str(dest)


def _get_cartridge_num_tokens(cartridge_path: str) -> int:
    """Read cartridge.pt and return total KV tokens (frozen + trainable)."""
    ckpt = torch.load(cartridge_path, map_location="cpu", weights_only=False)
    frozen_k = ckpt.get("frozen_keys", ckpt.get("fixed_keys", []))
    trainable_k = ckpt["trainable_keys"]
    n_frozen = frozen_k[0].shape[2] if frozen_k and frozen_k[0] is not None and frozen_k[0].numel() > 0 else 0
    n_trainable = trainable_k[0].shape[2]
    return n_frozen + n_trainable


def _poll_tokasaurus(port: int, timeout: int = 300):
    """Wait until tokasaurus /v1/models endpoint is reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = sync_requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Tokasaurus not ready on port {port} after {timeout}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Tokasaurus-backed cartridge OpenAI server")
    p.add_argument(
        "--cartridge-path", type=str, required=True,
        help="Path to cartridge.pt checkpoint",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8192, help="Port for the OpenAI-compatible proxy")
    p.add_argument(
        "--tokasaurus-port", type=int, default=None,
        help="Internal tokasaurus port (default: proxy port + 1000)",
    )
    p.add_argument(
        "--served-model-name", type=str,
        default=os.environ.get("CARTRIDGE_SERVED_MODEL_NAME", "llama-3.2-3b-cartridge"),
    )
    p.add_argument(
        "--cartridge-dir", type=str, default=None,
        help="Directory for staged cartridge (default: temp dir)",
    )
    p.add_argument("--kv-cache-num-tokens", type=int, default=32768)
    p.add_argument(
        "--temperature-override", type=float, default=0.0,
        help="Force this temperature for all requests (ignores client value). Default: 0 (greedy).",
    )
    p.add_argument(
        "--base-model", type=str, default=BASE_MODEL,
        help="HF model ID for the base model",
    )
    args = p.parse_args()

    if not os.path.isfile(args.cartridge_path):
        raise SystemExit(f"Not a file: {args.cartridge_path}")

    toka_port = args.tokasaurus_port or (args.port + 1000)
    base_model = args.base_model

    # Stage cartridge on disk
    cart_dir = args.cartridge_dir or tempfile.mkdtemp(prefix="tokasaurus_cart_")
    num_tokens = _get_cartridge_num_tokens(args.cartridge_path)
    page_size = 16
    if num_tokens % page_size != 0:
        print(f"WARNING: cartridge has {num_tokens} tokens, not page-aligned to {page_size}. "
              f"Tokasaurus may truncate.")
    _stage_cartridge(args.cartridge_path, cart_dir, num_tokens, base_model=base_model)
    print(f"Cartridge staged: {cart_dir}/{CARTRIDGE_ID}/ ({num_tokens} tokens)")

    # Start tokasaurus
    from tokasaurus.entry import server_manager
    from tokasaurus.common_types import ServerConfig

    toka_config = ServerConfig()
    toka_config.model = base_model
    toka_config.port = toka_port
    toka_config.cartridge_dir = cart_dir
    toka_config.tp_size = 1
    toka_config.dp_size = 1
    toka_config.kv_cache_num_tokens = args.kv_cache_num_tokens
    toka_config.log_level = "WARNING"
    toka_config.uvicorn_log_level = "warning"

    print(f"Starting tokasaurus on port {toka_port} (model={base_model})...")
    server_ctx = server_manager(toka_config)
    server_ctx.__enter__()

    _poll_tokasaurus(toka_port)
    print(f"Tokasaurus ready on port {toka_port}")

    # Build OpenAI proxy
    toka_batch_url = f"http://127.0.0.1:{toka_port}/custom/cartridge/synchronous-batch-completions"
    served_model_name = args.served_model_name
    temperature_override = args.temperature_override

    if temperature_override is not None:
        print(f"Temperature override: {temperature_override} (client values ignored)")

    app = FastAPI(title="Cartridge Tokasaurus OpenAI shim")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cartridge-tokasaurus-shim",
                "root": args.cartridge_path,
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest):
        if body.model and body.model != served_model_name:
            raise HTTPException(
                400,
                detail=f"Unknown model {body.model!r}; this server only serves {served_model_name!r}",
            )

        eff_temp = temperature_override if temperature_override is not None else body.temperature

        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        toka_request = {
            "model": base_model,
            "messages": messages,
            "max_tokens": body.max_tokens,
            "temperature": eff_temp,
            "cartridges": [{
                "id": CARTRIDGE_ID,
                "source": "local",
                "force_redownload": True,
            }],
        }

        try:
            resp = sync_requests.post(
                toka_batch_url,
                json={"requests": [toka_request]},
                timeout=120,
            )
            resp.raise_for_status()
            completions = pickle.loads(resp.content)
        except Exception as e:
            raise HTTPException(502, detail=f"Tokasaurus error: {e}") from e

        comp = completions[0]
        if isinstance(comp, Exception) or comp is None:
            raise HTTPException(502, detail=f"Tokasaurus returned error: {comp}")

        try:
            # comp may be a dict or a Pydantic ChatCompletion; normalize to get text + usage
            if isinstance(comp, dict):
                choice = comp["choices"][0]
                text = choice["message"]["content"] if isinstance(choice, dict) else choice.message.content
            else:
                text = comp.choices[0].message.content

            # Normalize usage to a plain JSON-safe dict
            raw_usage = comp.get("usage") if isinstance(comp, dict) else getattr(comp, "usage", None)
            if raw_usage is None:
                usage = {}
            elif isinstance(raw_usage, dict):
                usage = {k: int(v) for k, v in raw_usage.items() if v is not None}
            else:
                usage = {
                    "prompt_tokens": int(raw_usage.prompt_tokens or 0),
                    "completion_tokens": int(raw_usage.completion_tokens or 0),
                    "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
                }
        except Exception as e:
            raise HTTPException(502, detail=f"Failed to parse tokasaurus response: {e}") from e

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        out: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text or ""},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }
        return JSONResponse(out)

    @app.on_event("shutdown")
    def shutdown():
        try:
            server_ctx.__exit__(None, None, None)
        except Exception:
            pass

    print(f"OpenAI proxy on port {args.port} -> tokasaurus:{toka_port} (cartridge={CARTRIDGE_ID})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
