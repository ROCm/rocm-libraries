# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end P1e check: an encoder-only embedding model under
``attention_backend="HIPDNN"`` routes its attention to hipDNN, and its
embeddings match the default backend's (cosine similarity > threshold).

usage: python offline_encoder.py --model <hf id or path> --report-dir <dir>
       [--reference DEFAULT|TRITON_ATTN|...] [--min-cosine 0.999]
       [--truth-dtype float32 [--error-factor 2]]
       [--enforce-eager] [--json-out f]

The reference backend runs in a child process (``--backend`` child mode) with
the plugin's report disabled. Exits non-zero on any failure.

--truth-dtype also runs the reference at that (wider) dtype and judges both
low-precision runs by their distance from it: error = 1 - cosine to the truth
embedding. The plugin passes if its worst-prompt error is at most
--error-factor times the reference's own worst-prompt error (or within
1 - --min-cosine). Deep bf16 models amplify per-layer rounding: two correct
bf16 attention kernels can differ from each other by more than 0.001 in cosine
while each stays within the same distance of the fp32 result.
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
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "GPUs accelerate dense linear algebra.",
    "hi",
    "A long passage about attention: " + " ".join(f"token{i}" for i in range(180)),
    "query: how do embedding models work",
    "passage: Embedding models map text to vectors so similar meanings land close together.",
    "Short.",
]


def embed(model: str, backend: str, args) -> list[list[float]]:
    from vllm import LLM

    kwargs = dict(
        model=model,
        runner="pooling",
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.4,
    )
    if args.hf_overrides:
        kwargs["hf_overrides"] = json.loads(args.hf_overrides)
    if backend != "DEFAULT":
        kwargs["attention_backend"] = backend
    llm = LLM(**kwargs)
    outs = llm.embed(PROMPTS)
    vecs = [list(map(float, o.outputs.embedding)) for o in outs]
    if backend == "HIPDNN":
        time.sleep(3)
    del llm
    return vecs


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--backend", default="HIPDNN")
    p.add_argument("--reference", default="DEFAULT")
    p.add_argument("--min-cosine", type=float, default=0.999)
    p.add_argument("--truth-dtype", default="", help="also run the reference at this dtype (e.g. float32)")
    p.add_argument("--error-factor", type=float, default=2.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--hf-overrides",
        default="",
        help='JSON, e.g. \'{"is_causal": false}\' to run a decoder embedder bidirectionally',
    )
    p.add_argument("--json-out")
    args = p.parse_args()

    if args.backend != "HIPDNN":
        # Child mode: the reference backend.
        print(json.dumps(embed(args.model, args.backend, args)))
        return 0

    os.environ["HIPDNN_VLLM_REPORT_DIR"] = args.report_dir
    # The EngineCore process can be killed before its at-exit report is
    # written; a 1 s interval plus a short wait guarantees one periodic report.
    os.environ["HIPDNN_VLLM_REPORT_INTERVAL"] = "1"
    os.makedirs(args.report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.report_dir, "report-*.json")):
        os.remove(old)

    vecs = embed(args.model, "HIPDNN", args)

    def reference(dtype: str) -> list[list[float]] | None:
        cmd = [sys.executable, __file__, "--model", args.model, "--report-dir", args.report_dir,
               "--backend", args.reference, "--max-model-len", str(args.max_model_len),
               "--dtype", dtype, "--hf-overrides", args.hf_overrides]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        env = dict(os.environ, HIPDNN_VLLM_REPORT_DIR="")
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-4000:], out.stderr[-8000:], file=sys.stderr)
            return None
        return json.loads(out.stdout.strip().splitlines()[-1])

    ref = reference(args.dtype)
    truth = reference(args.truth_dtype) if args.truth_dtype else None
    if ref is None or (args.truth_dtype and truth is None):
        print("FAIL: reference run failed", file=sys.stderr)
        return 1
    cosines = [cosine(a, b) for a, b in zip(vecs, ref)]
    truth_error = None
    if truth is not None:
        truth_error = {
            "plugin": [1.0 - cosine(a, t) for a, t in zip(vecs, truth)],
            "reference": [1.0 - cosine(r, t) for r, t in zip(ref, truth)],
        }

    reports = sorted(glob.glob(os.path.join(args.report_dir, "report-*.json")))
    routes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    sdpa_engine_calls = 0
    for path in reports:
        data = json.load(open(path))
        for r in data["counters"]:
            if r["op"].startswith("attention"):
                key = f"{r['op']}:{r['route']}" + (f":{r['reason']}" if r["reason"] else "")
                routes[key] = routes.get(key, 0) + r["count"]
            if r["op"] == "sdpa_fwd" and r["route"] == "hipdnn":
                sdpa_engine_calls += r["count"]
        for reason, n in data["fallback_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + n
    summary = {
        "model": args.model,
        "reference": args.reference,
        "cosines": cosines,
        "min_cosine": min(cosines) if cosines else None,
        "truth_dtype": args.truth_dtype or None,
        "truth_error": truth_error,
        "attention_routes": routes,
        "fallback_reasons": reasons,
        "sdpa_fwd_hipdnn_calls": sdpa_engine_calls,
        "reports": reports,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)

    failures = []
    if len(vecs) != len(PROMPTS) or len(ref) != len(PROMPTS):
        failures.append("missing embeddings")
    if truth_error is not None:
        if len(truth) != len(PROMPTS):
            failures.append("missing truth embeddings")
        worst, ref_worst = max(truth_error["plugin"]), max(truth_error["reference"])
        bound = max(1.0 - args.min_cosine, args.error_factor * ref_worst)
        if not worst <= bound:
            failures.append(f"worst {args.truth_dtype} error {worst:.6f} > {bound:.6f} "
                            f"({args.error_factor} x reference {ref_worst:.6f})")
    elif not cosines or min(cosines) <= args.min_cosine:
        failures.append(f"min cosine {min(cosines) if cosines else None} <= {args.min_cosine}")
    if not reports:
        failures.append("no coverage report written")
    if not any(k.startswith("attention.encoder_only:hipdnn") for k in routes):
        failures.append("no encoder-only attention call routed to hipDNN")
    if not sdpa_engine_calls:
        failures.append("no sdpa_fwd executions counted")
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
