# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Combined end-to-end run: HIPDNN attention, the hipdnn rms_norm IR providers
and the linear patch in one vLLM LLM, against a plugin-off reference.

usage: python combined.py --model <path> --report-dir <dir> [--json-out f]
       [--min-agreement 0.9] [--enforce-eager]

The plugin run uses attention_backend="HIPDNN", ir_op_priority with hipdnn first
for rms_norm and fused_add_rms_norm, and HIPDNN_VLLM_OPS with linear added. The
reference runs in a second process with HIPDNN_VLLM_DISABLE=1 and TRITON_ATTN.
Greedy tokens are compared position by position (linear and norm change the
numerics, so exact equality is not required). The merged coverage report must
show calls for attention, a norm op and linear. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "Write one sentence about GPUs:",
    "1, 2, 3, 4,",
    "In a distant galaxy, a small robot",
    "The three primary colors are",
    "def fibonacci(n):",
]
NORM_PRIORITY = ["hipdnn", "vllm_c", "native"]


def generate(model: str, plugin: bool, enforce_eager: bool, max_model_len: int) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    kwargs = {}
    if plugin:
        kwargs["attention_backend"] = "HIPDNN"
        kwargs["ir_op_priority"] = {"rms_norm": NORM_PRIORITY, "fused_add_rms_norm": NORM_PRIORITY}
    else:
        kwargs["attention_backend"] = "TRITON_ATTN"
    llm = LLM(
        model=model,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,
        **kwargs,
    )
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32))
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    return tokens


def _agreement(a: list[list[int]], b: list[list[int]]) -> float:
    same = total = 0
    for x, y in zip(a, b):
        n = min(len(x), len(y))
        total += max(len(x), len(y))
        same += sum(1 for i in range(n) if x[i] == y[i])
    return same / total if total else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--json-out")
    p.add_argument("--min-agreement", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--reference-child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.reference_child:
        print(json.dumps(generate(args.model, False, args.enforce_eager, args.max_model_len)))
        return 0

    from hipdnn_vllm.config import DEFAULT_OPS

    os.environ["HIPDNN_VLLM_OPS"] = ",".join(DEFAULT_OPS + ("linear",))
    os.environ["HIPDNN_VLLM_REPORT_DIR"] = args.report_dir
    os.environ.setdefault("HIPDNN_VLLM_REPORT_INTERVAL", "5")
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    tokens = generate(args.model, True, args.enforce_eager, args.max_model_len)

    cmd = [sys.executable, __file__, "--model", args.model, "--report-dir", args.report_dir,
           "--max-model-len", str(args.max_model_len), "--reference-child"]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    env = dict(os.environ, HIPDNN_VLLM_DISABLE="1", HIPDNN_VLLM_REPORT_DIR="")
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    ref = json.loads(out.stdout.strip().splitlines()[-1])
    agreement = _agreement(tokens, ref)

    routes: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    reports = sorted(glob.glob(os.path.join(args.report_dir, "report-*.json")))
    for path in reports:
        data = json.load(open(path))
        for r in data["counters"]:
            by_route = routes.setdefault(r["op"], {})
            by_route[r["route"]] = by_route.get(r["route"], 0) + r["count"]
        for reason, n in data["fallback_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + n
    summary = {"reports": reports, "routes": routes, "fallback_reasons": reasons,
               "token_agreement": agreement}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)

    failures = []
    if not reports:
        failures.append("no coverage report written")
    for family in ("attention", "norm", "linear"):
        if not any(family in op and sum(r.values()) for op, r in routes.items()):
            failures.append(f"report shows no {family} calls")
    if agreement < args.min_agreement:
        failures.append(f"greedy token agreement {agreement:.3f} < {args.min_agreement}")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
