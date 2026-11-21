# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check of the unquantized-linear patch.

usage: python linear_e2e.py --model <path> --out <dir>
       [--candidates linear,linear_all] [--prompts N] [--max-tokens T]
       [--enforce-eager] [--max-model-len N]

Runs vLLM offline (default attention backend, default compile and HIP-graph
settings) once per configuration, each in its own process:

* ``base``: HIPDNN_VLLM_OPS unset (the plugin's default op set: no linear).
* ``control``: the same as base in a new process (it loads base's compiled
  model from the cache), giving the run-to-run noise floor of every number.
* one run per candidate, with HIPDNN_VLLM_OPS=<candidate>.

Per run, inside the worker (collective_rpc): which call sites hold the patch,
how many UnquantizedLinearMethod layers bound the hipDNN gemm, and the plugin's
coverage counters. Then:

* free-running greedy generation (token ids),
* teacher-forced top-1: every run scores the base run's prompt + completion
  with prompt_logprobs=1, and the top-1 token at each completion position is
  compared with the base run's,
* throughput: output tokens/s for a fixed batch (median of 3, after warm-up).

All runs share one fresh VLLM_CACHE_ROOT under --out, created empty, so every
configuration compiles from scratch, and a candidate would load the base run's
AOT-compiled model if the patch did not change vLLM's compile cache key.

Fails (exit 1) when the base run shows any patch, a candidate does not route
linear to hipDNN, or the teacher-forced top-1 agreement is below
--min-agreement (default 0.99).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

TOPICS = [
    "The capital of France is",
    "Write one sentence about GPUs:",
    "1, 2, 3, 4,",
    "In a distant galaxy, a small robot",
    "The three laws of thermodynamics are",
    "def fibonacci(n):",
    "A haiku about the ocean:",
    "The main ingredients of bread are",
    "Explain what a hash table is.",
    "The year 1969 is remembered because",
    "Translate 'good morning' into Spanish:",
    "List three prime numbers greater than 100:",
    "The difference between a virus and a bacterium is",
    "Once upon a time in a small village,",
    "The square root of 144 is",
    "SELECT name FROM users WHERE",
]


def _prompts(n: int) -> list[str]:
    return [TOPICS[i % len(TOPICS)] + ("" if i < len(TOPICS) else f" (variant {i})") for i in range(n)]


# --- worker-side probes (run in the vLLM worker via collective_rpc) --------------


def _worker_state(worker) -> dict:
    import vllm.model_executor.layers.linear as linear_mod
    import vllm.model_executor.layers.utils as utils_mod
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    lin = sys.modules.get("hipdnn_vllm.ops.linear")
    runner = worker.model_runner
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    impls: dict[str, int] = {}
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        if isinstance(qm, UnquantizedLinearMethod):
            fn = qm._gemm_impl
            name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"
            impls[name] = impls.get(name, 0) + 1
    return {
        "pid": os.getpid(),
        "hipdnn_vllm_ops_linear_imported": lin is not None,
        "patch_state": lin.patch_state() if lin is not None else {},
        "linear_site_is_vllm_dispatch": linear_mod.dispatch_unquantized_gemm
        is utils_mod.dispatch_unquantized_gemm,
        "unquantized_linear_gemm_impls": impls,
        "hipdnn_vllm_ops": os.environ.get("HIPDNN_VLLM_OPS"),
    }


def _worker_stats(worker) -> dict:
    from hipdnn_vllm import stats

    return stats.snapshot()


# --- child: one vLLM configuration ------------------------------------------------


def child(args) -> int:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.5,
        seed=0,
    )
    result: dict = {"name": args.child, "ops": os.environ.get("HIPDNN_VLLM_OPS")}
    result["worker_state"] = llm.collective_rpc(_worker_state)

    prompts = _prompts(args.prompts)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, greedy, use_tqdm=False)
    result["prompt_ids"] = [list(o.prompt_token_ids) for o in outs]
    result["gen_ids"] = [list(o.outputs[0].token_ids) for o in outs]
    result["texts"] = [o.outputs[0].text for o in outs]

    if args.reference:
        ref = json.load(open(args.reference))
        seqs = [p + g for p, g in zip(ref["prompt_ids"], ref["gen_ids"])]
        starts = [len(p) for p in ref["prompt_ids"]]
    else:
        seqs = [p + g for p, g in zip(result["prompt_ids"], result["gen_ids"])]
        starts = [len(p) for p in result["prompt_ids"]]
    scored = llm.generate(
        [TokensPrompt(prompt_token_ids=s) for s in seqs],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1),
        use_tqdm=False,
    )
    top1 = []
    for o, start in zip(scored, starts):
        row = []
        # prompt_logprobs[i] scores token i given tokens < i; positions >= start
        # are the completion the reference run generated.
        for pos in range(start, len(o.prompt_token_ids)):
            lp = o.prompt_logprobs[pos]
            row.append(next(tok for tok, v in lp.items() if v.rank == 1))
        top1.append(row)
    result["teacher_forced_top1"] = top1

    tp_prompts = _prompts(args.tp_batch)
    tp = SamplingParams(temperature=0.0, max_tokens=args.tp_tokens, ignore_eos=True)
    llm.generate(tp_prompts, tp, use_tqdm=False)  # warm-up (eager M plans)
    rates = []
    for _ in range(3):
        t0 = time.perf_counter()
        o = llm.generate(tp_prompts, tp, use_tqdm=False)
        dt = time.perf_counter() - t0
        rates.append(sum(len(x.outputs[0].token_ids) for x in o) / dt)
    result["tokens_per_s"] = statistics.median(rates)
    result["tokens_per_s_all"] = rates

    result["worker_stats"] = llm.collective_rpc(_worker_stats)
    json.dump(result, open(args.json_out, "w"))
    return 0


# --- parent ------------------------------------------------------------------------


def _run_child(args, name: str, ops: str | None, reference: str | None) -> dict:
    out_json = os.path.join(args.out, f"{name}.json")
    report_dir = os.path.join(args.out, f"report-{name}")
    os.makedirs(report_dir, exist_ok=True)
    for old in glob.glob(os.path.join(report_dir, "report-*.json")):
        os.remove(old)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_VLLM_")}
    env.update(
        HIPDNN_VLLM_REPORT_DIR=report_dir,
        HIPDNN_VLLM_REPORT_INTERVAL="5",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
        VLLM_CACHE_ROOT=os.path.join(args.out, "vllm-cache"),
    )
    if ops is not None:
        env["HIPDNN_VLLM_OPS"] = ops
    cmd = [
        sys.executable, os.path.abspath(__file__), "--child", name, "--model", args.model,
        "--out", args.out, "--json-out", out_json, "--prompts", str(args.prompts),
        "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len),
        "--tp-batch", str(args.tp_batch), "--tp-tokens", str(args.tp_tokens),
    ]  # fmt: skip
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if reference:
        cmd += ["--reference", reference]
    print(f"== run {name}: HIPDNN_VLLM_OPS={ops!r}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    data = json.load(open(out_json))
    data["wall_s"] = time.time() - t0
    data["reports"] = sorted(glob.glob(os.path.join(report_dir, "report-*.json")))
    return data


def _linear_counters(data: dict) -> dict:
    out: dict[str, int] = {}
    for snap in data["worker_stats"]:
        for r in snap["counters"]:
            if r["op"] == "linear":
                key = f"{r['route']}|{r['reason']}|{r['phase']}"
                out[key] = out.get(key, 0) + r["count"]
    return out


def _agreement(base: dict, cand: dict) -> tuple[int, int]:
    match = total = 0
    for b, c in zip(base["teacher_forced_top1"], cand["teacher_forced_top1"]):
        assert len(b) == len(c)
        total += len(b)
        match += sum(int(x == y) for x, y in zip(b, c))
    return match, total


def _free_run(base: dict, cand: dict) -> dict:
    same = 0
    prefix = []
    for b, c in zip(base["gen_ids"], cand["gen_ids"]):
        same += int(b == c)
        n = 0
        for x, y in zip(b, c):
            if x != y:
                break
            n += 1
        prefix.append(n / max(len(b), 1))
    return {"identical_sequences": same, "sequences": len(base["gen_ids"]),
            "mean_matching_prefix_fraction": statistics.mean(prefix)}  # fmt: skip


def _check_off(name: str, data: dict, failures: list[str]) -> None:
    """A run without linear in HIPDNN_VLLM_OPS: no patch anywhere in the worker."""
    for ws in data["worker_state"]:
        if ws["hipdnn_vllm_ops_linear_imported"] or any(ws["patch_state"].values()):
            failures.append(f"{name}: patch present in worker {ws}")
        if not ws["linear_site_is_vllm_dispatch"]:
            failures.append(f"{name}: linear.dispatch_unquantized_gemm is not vLLM's")
        if set(ws["unquantized_linear_gemm_impls"]) != {
            "vllm.model_executor.layers.utils.rocm_unquantized_gemm"
        }:
            failures.append(f"{name}: layers bound {ws['unquantized_linear_gemm_impls']}")
    if _linear_counters(data):
        failures.append(f"{name}: linear counters {_linear_counters(data)}")


def parent(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    shutil.rmtree(os.path.join(args.out, "vllm-cache"), ignore_errors=True)
    base = _run_child(args, "base", None, None)
    base_json = os.path.join(args.out, "base.json")
    # Same configuration as base in a new process: the run-to-run noise floor of
    # the agreement and tokens/s numbers.
    control = _run_child(args, "control", None, base_json)
    cands = {name: _run_child(args, name, name, base_json) for name in args.candidates.split(",")}

    failures: list[str] = []
    match, total = _agreement(base, control)
    summary: dict = {
        "model": args.model,
        "base": {
            "worker_state": base["worker_state"],
            "linear_counters": _linear_counters(base),
            "tokens_per_s": base["tokens_per_s"],
        },
        "control": {
            "worker_state": control["worker_state"],
            "linear_counters": _linear_counters(control),
            "teacher_forced_top1_agreement": match / total if total else 0.0,
            "free_run": _free_run(base, control),
            "tokens_per_s": control["tokens_per_s"],
            "relative_tokens_per_s": control["tokens_per_s"] / base["tokens_per_s"],
        },
        "candidates": {},
    }
    _check_off("base", base, failures)
    _check_off("control", control, failures)

    for name, cand in cands.items():
        match, total = _agreement(base, cand)
        counters = _linear_counters(cand)
        hip_eager = sum(v for k, v in counters.items() if k.startswith("hipdnn|") and k.endswith("|eager"))
        hip_capture = sum(v for k, v in counters.items() if k.startswith("hipdnn|") and k.endswith("|capture"))
        fallback = {k: v for k, v in counters.items() if k.startswith("fallback|")}
        entry = {
            "worker_state": cand["worker_state"],
            "linear_counters": counters,
            "linear_hipdnn_calls": {"eager": hip_eager, "capture": hip_capture},
            "linear_fallback": fallback,
            "teacher_forced_top1_agreement": match / total if total else 0.0,
            "teacher_forced_positions": total,
            "free_run": _free_run(base, cand),
            "tokens_per_s": cand["tokens_per_s"],
            "relative_tokens_per_s": cand["tokens_per_s"] / base["tokens_per_s"],
            "sample": cand["texts"][:2],
            "reports": cand["reports"],
        }
        summary["candidates"][name] = entry
        for ws in cand["worker_state"]:
            want_all = name == "linear_all"
            if not ws["patch_state"].get("linear"):
                failures.append(f"{name}: linear site not patched in worker {ws['pid']}")
            if want_all and not (ws["patch_state"].get("lm_head") and ws["patch_state"].get("moe_router_gate")):
                failures.append(f"{name}: linear_all sites not patched: {ws['patch_state']}")
            impls = ws["unquantized_linear_gemm_impls"]
            if set(impls) != {"hipdnn_vllm.ops.linear.hipdnn_unquantized_gemm"}:
                failures.append(f"{name}: layers bound {impls}")
        if hip_eager == 0:
            failures.append(f"{name}: no linear call ran on hipDNN")
        if not args.enforce_eager and hip_capture == 0:
            failures.append(f"{name}: no linear call captured on hipDNN")
        if entry["teacher_forced_top1_agreement"] < args.min_agreement:
            failures.append(
                f"{name}: top-1 agreement {entry['teacher_forced_top1_agreement']:.4f} "
                f"< {args.min_agreement}"
            )

    print(json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    print("E2E_OK" if not failures else "E2E_FAILED")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", default="linear,linear_all")
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--tp-batch", type=int, default=64)
    p.add_argument("--tp-tokens", type=int, default=128)
    p.add_argument("--min-agreement", type=float, default=0.99)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--child")
    p.add_argument("--json-out")
    p.add_argument("--reference")
    args = p.parse_args()
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
