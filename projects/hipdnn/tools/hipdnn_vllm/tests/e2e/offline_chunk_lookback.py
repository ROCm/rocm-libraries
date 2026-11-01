# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check for chunk_lookback layers (Rnj1): with
``attention_backend="HIPDNN"`` every chunked-attention layer falls back to the
TRITON_ATTN kernel with ``UNSUPPORTED_FEATURE:chunk_lookback``, and generation
matches a TRITON_ATTN run.

usage: python offline_chunk_lookback.py --model <dir with an Rnj1 config.json + tokenizer>
       --report-dir <dir> [--layers N] [--chunk W] [--all-chunked] [--enforce-eager]
       [--json-out f]

The model runs with ``load_format="dummy"`` (no weights needed) and
``hf_overrides`` that keep the first --layers entries of the config's
``layer_types`` (every one chunked with --all-chunked) and set the chunk size
(``sliding_window``) to --chunk, so the prompts span several chunks. The pinned
vLLM needs the local patch ``patches/vllm/0001`` to accept HIPDNN for these
layers.

Greedy token ids must equal TRITON_ATTN's whenever no attention call ran on
hipDNN (always true with --all-chunked). If some full-attention layers did run
on hipDNN, the first divergence is reported instead. Exits non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

PROMPTS = [
    "Count: " + ", ".join(str(i) for i in range(1, 61)),
    "The quick brown fox jumps over the lazy dog. " * 8,
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n" * 2,
    "hi",
]
REASON = "UNSUPPORTED_FEATURE:chunk_lookback"


def overrides(args) -> dict:
    config = json.load(open(os.path.join(args.model, "config.json")))
    if config.get("architectures") != ["Rnj1ForCausalLM"]:
        raise SystemExit(f"{args.model} is not an Rnj1ForCausalLM config")
    kinds = list(config["layer_types"][: args.layers])
    if args.all_chunked:
        kinds = ["chunked_attention"] * args.layers
    return {"num_hidden_layers": args.layers, "layer_types": kinds, "sliding_window": args.chunk}


def generate(args, backend: str) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        load_format="dummy",
        dtype="bfloat16",
        hf_overrides=overrides(args),
        attention_backend=backend,
        enforce_eager=args.enforce_eager,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        seed=0,
    )
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    if backend == "HIPDNN":
        time.sleep(3)  # one report interval before the EngineCore shuts down
    del llm
    return ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--backend", default="HIPDNN")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--all-chunked", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--json-out")
    args = p.parse_args()

    if args.backend != "HIPDNN":
        # Child mode: the TRITON_ATTN reference.
        print(json.dumps(generate(args, args.backend)))
        return 0

    os.environ["HIPDNN_VLLM_REPORT_DIR"] = args.report_dir
    os.environ["HIPDNN_VLLM_REPORT_INTERVAL"] = "1"
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    ids = generate(args, "HIPDNN")
    cmd = [sys.executable, __file__, "--model", args.model, "--report-dir", args.report_dir,
           "--backend", "TRITON_ATTN", "--layers", str(args.layers), "--chunk", str(args.chunk)]
    if args.all_chunked:
        cmd.append("--all-chunked")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    out = subprocess.run(cmd, env=dict(os.environ, HIPDNN_VLLM_REPORT_DIR=""), capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-4000:], out.stderr[-8000:], file=sys.stderr)
        print("FAIL: TRITON_ATTN reference run failed", file=sys.stderr)
        return 1
    ref = json.loads(out.stdout.strip().splitlines()[-1])

    reports = sorted(glob.glob(os.path.join(args.report_dir, "report-*.json")))
    routes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for path in reports:
        data = json.load(open(path))
        for r in data["counters"]:
            if r["op"].startswith("attention"):
                key = f"{r['op']}:{r['route']}" + (f":{r['reason']}" if r["reason"] else "")
                routes[key] = routes.get(key, 0) + r["count"]
        for reason, n in data["fallback_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + n
    on_hipdnn = sum(n for k, n in routes.items() if ":hipdnn" in k)
    divergence = [next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else 0)
                  for a, b in zip(ids, ref)]
    summary = {
        "hf_overrides": overrides(args),
        "identical_to_triton": ids == ref,
        "first_divergence": divergence,
        "attention_routes": routes,
        "fallback_reasons": reasons,
        "reports": reports,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)

    failures = []
    if not reports:
        failures.append("no coverage report written")
    if not any(REASON in k for k in routes) and REASON not in reasons:
        failures.append(f"report does not show {REASON}")
    if on_hipdnn == 0 and ids != ref:
        failures.append("every attention call fell back, yet tokens differ from TRITON_ATTN")
    if args.all_chunked and on_hipdnn:
        failures.append("a chunked layer ran on hipDNN")
    if on_hipdnn and ids != ref:
        print(f"NOTE: {on_hipdnn} attention calls ran on hipDNN; first divergence per prompt {divergence}")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
