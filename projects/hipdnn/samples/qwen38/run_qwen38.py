#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Chat with Qwen3.8-27B through hipDNN: tokenizes the prompt, runs hipdnn_sample_qwen38 and
detokenizes its output as it streams."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"
BINARY_NAME = "hipdnn_sample_qwen38"
# <|im_end|> and <|endoftext|>; the binary stops on them as well.
STOP_TOKENS = (248046, 248044)


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache directory, resolved the way huggingface_hub does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def default_snapshot() -> Path | None:
    """The snapshot of MODEL_ID that the cache's main ref points at, or any cached snapshot."""
    repo = hf_hub_cache() / ("models--" + MODEL_ID.replace("/", "--"))
    ref = repo / "refs" / "main"
    if ref.is_file():
        snapshot = repo / "snapshots" / ref.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    snapshots = (
        sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
    )
    return snapshots[0] if snapshots else None


def default_binary() -> Path:
    """The sample binary next to this script (build or install bin/), else the one on PATH."""
    local = Path(__file__).resolve().parent / BINARY_NAME
    return local if local.is_file() else Path(BINARY_NAME)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="user message")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help=f"{MODEL_ID} snapshot directory (default: resolved from the Hugging Face cache, "
        "HF_HUB_CACHE or HF_HOME/hub, else ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=default_binary(),
        help=f"{BINARY_NAME} executable (default: next to this script, else on PATH)",
    )
    parser.add_argument(
        "--layers", type=int, default=64, help="decoder layers to run (debug)"
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu: CPU reference executor; gpu: hipDNN engines (CPU fallback per op)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="time each op (gpu backend)"
    )
    args = parser.parse_args()

    snapshot = args.snapshot or default_snapshot()
    if snapshot is None:
        parser.error(
            f"{MODEL_ID} is not in {hf_hub_cache()}; download it or pass --snapshot"
        )

    tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
    # chat_template.jinja with enable_thinking=false.
    text = (
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    print(f"prompt: {len(ids)} tokens", file=sys.stderr)

    cmd = [
        str(args.binary),
        str(snapshot),
        ",".join(map(str, ids)),
        str(args.max_new_tokens),
        str(args.layers),
        f"--backend={args.backend}",
        *(["--profile"] if args.profile else []),
    ]
    generated: list[int] = []
    shown = ""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            if not line.startswith("TOKEN "):
                continue
            token = int(line.split()[1])
            if token in STOP_TOKENS:
                continue
            generated.append(token)
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            sys.stdout.write(decoded[len(shown) :])
            sys.stdout.flush()
            shown = decoded
    print()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
