# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check of the DISABLED path: vLLM offline LLM with
``attention_backend="HIPDNN"`` and ``attention`` left out of
``HIPDNN_VLLM_OPS`` routes every attention call to the TRITON_ATTN kernel,
and the coverage report records each one as ``fallback`` / ``DISABLED``.

usage: python offline_fallback.py --model <hf id or path> --report-dir <dir>
       [--compare] [--enforce-eager] [--max-model-len N]

With --compare the same prompts also run with TRITON_ATTN in a second process
(same ``HIPDNN_VLLM_OPS``) and greedy outputs must be identical: HIPDNN falls
back to the TRITON_ATTN kernel, so tokens match exactly. Exits non-zero on any
failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

from hipdnn_vllm.config import DEFAULT_OPS

PROMPTS = [
    "The capital of France is",
    "Write one sentence about GPUs:",
    "1, 2, 3, 4,",
    "In a distant galaxy, a small robot",
]

# Every default op except attention, so the other plugin ops behave as usual.
OPS_WITHOUT_ATTENTION = ",".join(op for op in DEFAULT_OPS if op != "attention")


def generate(model: str, backend: str, enforce_eager: bool, max_model_len: int) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        attention_backend=backend,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.4,
    )
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24))
    texts = [o.outputs[0].text for o in outs]
    del llm
    return texts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--backend", default="HIPDNN")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--json-out")
    args = p.parse_args()

    if args.backend != "HIPDNN":
        # Child mode for --compare.
        print(json.dumps(generate(args.model, args.backend, args.enforce_eager, args.max_model_len)))
        return 0

    os.environ["HIPDNN_VLLM_REPORT_DIR"] = args.report_dir
    os.environ.setdefault("HIPDNN_VLLM_REPORT_INTERVAL", "5")
    os.environ["HIPDNN_VLLM_OPS"] = OPS_WITHOUT_ATTENTION
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    texts = generate(args.model, "HIPDNN", args.enforce_eager, args.max_model_len)
    for prompt, text in zip(PROMPTS, texts):
        print(f"{prompt!r} -> {text!r}")
    failures = []

    if args.compare:
        cmd = [sys.executable, __file__, "--model", args.model, "--report-dir", args.report_dir,
               "--backend", "TRITON_ATTN", "--max-model-len", str(args.max_model_len)]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        env = dict(os.environ, HIPDNN_VLLM_REPORT_DIR="")
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        ref = json.loads(out.stdout.strip().splitlines()[-1])
        if ref != texts:
            failures.append(f"HIPDNN output differs from TRITON_ATTN: {ref} vs {texts}")
        else:
            print("greedy outputs identical to TRITON_ATTN")

    reports = sorted(glob.glob(os.path.join(args.report_dir, "report-*.json")))
    merged: dict[str, int] = {}
    routes: dict[str, int] = {}
    attention_reasons: dict[str, int] = {}
    for path in reports:
        data = json.load(open(path))
        for reason, n in data["fallback_reasons"].items():
            merged[reason] = merged.get(reason, 0) + n
        for r in data["counters"]:
            if r["op"].startswith("attention"):
                routes[r["route"]] = routes.get(r["route"], 0) + r["count"]
                attention_reasons[r["reason"]] = attention_reasons.get(r["reason"], 0) + r["count"]
    summary = {
        "reports": reports,
        "fallback_reasons": merged,
        "attention_routes": routes,
        "attention_reasons": attention_reasons,
        "outputs": texts,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)

    if not reports:
        failures.append("no coverage report written")
    if not routes.get("fallback"):
        failures.append("no attention calls recorded")
    if routes.get("hipdnn"):
        failures.append(f"attention routed to hipDNN although HIPDNN_VLLM_OPS={OPS_WITHOUT_ATTENTION}")
    if set(attention_reasons) != {"DISABLED"}:
        failures.append(f"attention fallbacks must all be DISABLED, got {attention_reasons}")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
