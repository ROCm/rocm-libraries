# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""End-to-end check of decoder attention (P2a/P2b) through hipDNN.

usage: python paged_attention_e2e.py --model <path> --out <dir>
       [--prompts N] [--max-tokens T] [--max-model-len N] [--max-num-seqs N]
       [--max-num-batched-tokens N] [--num-gpu-blocks N] [--spec-tokens K]
       [--enforce-eager] [--min-agreement 0.99]

Each configuration runs vLLM offline in its own process, all with the same
scheduler bounds and default compile / CUDA-graph settings (FULL_AND_PIECEWISE):

* ``base``: plugin off (HIPDNN_VLLM_DISABLE=1), ``TRITON_ATTN`` (the kernel the
  HIPDNN backend falls back to, on the same KV-cache layout).
* ``control``: ``base`` again in a new process: the run-to-run noise floor.
* ``ref32`` (unless ``--no-fp32-reference``): ``base`` with the model in
  float32, the accuracy reference both bf16/fp16 runs are scored against.
* ``hipdnn``: ``--attention-backend HIPDNN`` with HIPDNN_VLLM_OPS=attention.
* ``hipdnn_spec`` (with ``--spec-tokens K``): ``hipdnn`` plus n-gram
  speculative decoding with K tokens, so decode batches are uniform k+1, and
  ``base_spec``, the plugin-off run with the same speculative config (its
  tokens/s reference).

Workload: short prompts plus long ones (longer than
``--max-num-batched-tokens``) generated together, so the steps include
uniform decode, mixed prefill+decode and chunked prefill with cached context.

Per run:

* teacher-forced top-1: every run scores base's prompt + completion with
  prompt_logprobs=5 (prefill and chunked-prefill attention, P2b); the top-1 at
  each completion position is compared with base's (and with ref32's), and at
  each mismatch the log-prob gap between the two tokens is recorded;
* free-running greedy agreement with base (decode attention, P2a) and, at each
  sequence's first divergence, the log-prob gap between the two tokens in each
  run's own top-5 (a near-tie when small);
* output tokens/s for a fixed decode-heavy batch (median of 3 after warm-up);
* the plugin's coverage counters and, per attention layer, the P2a/P2b probe
  result (engine name or decline) read from the worker.

Fails (exit 1) unless ``hipdnn`` ran attention.uniform_decode and
attention.mixed on hipDNN, and its teacher-forced agreement with base is at
least ``--min-agreement`` or no lower than control's self-agreement. A run below
both is still accepted, and flagged ``accepted_by_fp32_reference``, when its
top-1 agreement with ref32 is within ``--fp32-tolerance`` of base's.
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
LONG = (
    "Summarize the following notes in one paragraph. "
    + " ".join(
        f"Note {i}: the {w} subsystem logged event {i * 7 % 13} at step {i}, and the operator "
        f"recorded that the {w} queue drained after {i % 5 + 1} retries."
        for i, w in enumerate(["memory", "network", "storage", "scheduler", "compiler", "cache"] * 6)
    )
)


def _prompts(n: int) -> list[str]:
    out = []
    for i in range(n):
        if i % 4 == 3:
            out.append(LONG + f" (copy {i})")
        else:
            out.append(TOPICS[i % len(TOPICS)] + ("" if i < len(TOPICS) else f" (variant {i})"))
    return out


# --- worker-side probes (run in the vLLM worker via collective_rpc) --------------


def _worker_stats(worker) -> dict:
    from hipdnn_vllm import stats

    return stats.snapshot()


def _worker_attention(worker) -> dict:
    """P2a/P2b probe result per distinct attention layer signature."""
    config = getattr(worker, "vllm_config", None) or worker.model_runner.vllm_config
    seen: dict[str, dict] = {}
    for name, layer in config.compilation_config.static_forward_context.items():
        paged = getattr(getattr(layer, "impl", None), "_paged", None)
        if paged is None:
            continue
        entry = {}
        for path in ("p2a", "p2b"):
            if not paged.prepared:
                entry[path] = "not prepared"
            elif paged.status(path) is None:
                entry[path] = paged.engine(path)
            else:
                entry[path] = paged.status(path).reason
        key = f"{type(layer.impl).__name__} {paged._layer}"
        seen.setdefault(key, {"layers": 0, **entry})["layers"] += 1
    return {"cudagraph_mode": str(config.compilation_config.cudagraph_mode), "signatures": seen}


# --- child: one vLLM configuration ------------------------------------------------


def child(args) -> int:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = dict(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        num_gpu_blocks_override=args.num_gpu_blocks,
        gpu_memory_utilization=_GPU_MEMORY_UTILIZATION,
        seed=0,
        attention_backend=args.backend,
        dtype=args.dtype,
    )
    if args.spec_tokens:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.spec_tokens,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
        }
    llm = LLM(**kwargs)
    result: dict = {"name": args.child, "backend": args.backend}

    prompts = _prompts(args.prompts)
    # Per-step top-5 log-probabilities show whether a free-run divergence is a
    # near-tie. Not requested with speculative decoding, which they would slow.
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            logprobs=None if args.spec_tokens else 5)  # fmt: skip
    outs = llm.generate(prompts, greedy, use_tqdm=False)
    result["prompt_ids"] = [list(o.prompt_token_ids) for o in outs]
    result["gen_ids"] = [list(o.outputs[0].token_ids) for o in outs]
    result["texts"] = [o.outputs[0].text for o in outs]
    # Top-5 log-probabilities per generated step: {token id: logprob}.
    result["gen_top5"] = [
        [{str(t): v.logprob for t, v in step.items()} for step in (o.outputs[0].logprobs or [])] for o in outs
    ]

    ref = json.load(open(args.reference)) if args.reference else result
    seqs = [p + g for p, g in zip(ref["prompt_ids"], ref["gen_ids"])]
    starts = [len(p) for p in ref["prompt_ids"]]
    scored = llm.generate(
        [TokensPrompt(prompt_token_ids=s) for s in seqs],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5),
        use_tqdm=False,
    )
    top1, top5 = [], []
    for o, start in zip(scored, starts):
        # prompt_logprobs[i] scores token i given tokens < i.
        positions = range(start, len(o.prompt_token_ids))
        top1.append([next(t for t, v in o.prompt_logprobs[pos].items() if v.rank == 1) for pos in positions])
        top5.append([{str(t): v.logprob for t, v in o.prompt_logprobs[pos].items()} for pos in positions])
    result["teacher_forced_top1"] = top1
    result["teacher_forced_top5"] = top5

    if args.tp_batch > 0:
        tp_prompts = [TOPICS[i % len(TOPICS)] + f" ({i})" for i in range(args.tp_batch)]
        tp = SamplingParams(temperature=0.0, max_tokens=args.tp_tokens, ignore_eos=True)
        llm.generate(tp_prompts, tp, use_tqdm=False)  # warm-up
        rates = []
        for _ in range(3):
            t0 = time.perf_counter()
            o = llm.generate(tp_prompts, tp, use_tqdm=False)
            rates.append(sum(len(x.outputs[0].token_ids) for x in o) / (time.perf_counter() - t0))
        result["tokens_per_s"] = statistics.median(rates)
        result["tokens_per_s_all"] = rates

    if args.backend == "HIPDNN":
        result["attention_layers"] = llm.collective_rpc(_worker_attention)
    result["worker_stats"] = llm.collective_rpc(_worker_stats)
    json.dump(result, open(args.json_out, "w"))
    return 0


# --- parent ------------------------------------------------------------------------


def _run_child(args, name: str, backend: str, reference: str | None, spec: int = 0,
               dtype: str = "auto", throughput: bool = True) -> dict:  # fmt: skip
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
    if backend == "HIPDNN":
        env["HIPDNN_VLLM_OPS"] = "attention"
    else:
        env["HIPDNN_VLLM_DISABLE"] = "1"
    cmd = [
        sys.executable, os.path.abspath(__file__), "--child", name, "--backend", backend,
        "--model", args.model, "--out", args.out, "--json-out", out_json,
        "--prompts", str(args.prompts), "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--num-gpu-blocks", str(args.num_gpu_blocks), "--spec-tokens", str(spec),
        "--tp-batch", str(args.tp_batch if throughput else 0), "--tp-tokens", str(args.tp_tokens),
        "--dtype", dtype,
    ]  # fmt: skip
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if reference:
        cmd += ["--reference", reference]
    print(f"== run {name}: backend={backend} spec_tokens={spec}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    data = json.load(open(out_json))
    data["wall_s"] = time.time() - t0
    data["reports"] = sorted(glob.glob(os.path.join(report_dir, "report-*.json")))
    return data


def _attention_counters(data: dict) -> dict:
    out: dict[str, int] = {}
    for snap in data["worker_stats"]:
        for r in snap["counters"]:
            if r["op"].startswith("attention"):
                key = f"{r['op']}|{r['route']}|{r['reason']}|{r['phase']}"
                out[key] = out.get(key, 0) + r["count"]
    return out


def _hipdnn_calls(counters: dict, op: str) -> dict:
    calls = {"eager": 0, "capture": 0}
    for key, n in counters.items():
        k_op, route, _, phase = key.split("|")
        if k_op == op and route == "hipdnn":
            calls[phase] += n
    return calls


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


def _divergence_gaps(base: dict, cand: dict) -> dict:
    """At each sequence's first free-run divergence, how far apart the two
    tokens are in each run's own distribution (log-prob of its own choice minus
    log-prob of the other run's; inf when outside the top-5). Near-ties (small
    gaps in both runs) are numerical tie-breaks rather than wrong attention."""
    gaps = []
    if not any(base.get("gen_top5") or []) or not any(cand.get("gen_top5") or []):
        return {}  # a run without per-step log-probs (speculative decoding)
    for bi, ci, bl, cl in zip(base["gen_ids"], cand["gen_ids"], base["gen_top5"], cand["gen_top5"]):
        pos = next((i for i, (x, y) in enumerate(zip(bi, ci)) if x != y), None)
        if pos is None or pos >= len(bl) or pos >= len(cl):
            continue
        b_tok, c_tok = str(bi[pos]), str(ci[pos])
        gap_c = cl[pos][c_tok] - cl[pos].get(b_tok, float("-inf"))
        gap_b = bl[pos][b_tok] - bl[pos].get(c_tok, float("-inf"))
        gaps.append(max(gap_c, gap_b))
    gaps.sort()
    return {"divergent": len(gaps), "max_gap": gaps[-1] if gaps else 0.0,
            "median_gap": statistics.median(gaps) if gaps else 0.0, "gaps": gaps}  # fmt: skip


def _tf_gaps(base: dict, cand: dict) -> dict:
    """The same gap at every teacher-forced position where the two runs' top-1
    differ (both scored base's sequences, so the contexts are identical)."""
    gaps = []
    for bt, ct, bl, cl in zip(base["teacher_forced_top1"], cand["teacher_forced_top1"],
                              base["teacher_forced_top5"], cand["teacher_forced_top5"]):  # fmt: skip
        for b, c, bp, cp in zip(bt, ct, bl, cl):
            if b != c:
                b, c = str(b), str(c)
                gaps.append(max(cp[c] - cp.get(b, float("-inf")), bp[b] - bp.get(c, float("-inf"))))
    gaps.sort()
    return {"mismatches": len(gaps), "max_gap": gaps[-1] if gaps else 0.0,
            "median_gap": statistics.median(gaps) if gaps else 0.0}  # fmt: skip


def _rate(a: dict, b: dict) -> float:
    match, total = _agreement(a, b)
    return match / total if total else 0.0


def parent(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    shutil.rmtree(os.path.join(args.out, "vllm-cache"), ignore_errors=True)
    base = _run_child(args, "base", "TRITON_ATTN", None)
    base_json = os.path.join(args.out, "base.json")
    control = _run_child(args, "control", "TRITON_ATTN", base_json)
    # Accuracy reference: TRITON_ATTN with the whole model in float32, scoring
    # base's sequences. Every bf16/fp16 run is compared with it as well.
    ref32 = None
    if args.fp32_reference:
        ref32 = _run_child(args, "ref32", "TRITON_ATTN", base_json, dtype="float32", throughput=False)
    # name -> (run, the plugin-off run with the same speculative config, for tokens/s)
    cands = {"hipdnn": (_run_child(args, "hipdnn", "HIPDNN", base_json), base)}
    if args.spec_tokens:
        base_spec = _run_child(args, "base_spec", "TRITON_ATTN", base_json, args.spec_tokens)
        cands["hipdnn_spec"] = (_run_child(args, "hipdnn_spec", "HIPDNN", base_json, args.spec_tokens), base_spec)

    failures: list[str] = []
    control_agreement = _rate(base, control)
    base_vs_fp32 = _rate(ref32, base) if ref32 else None
    summary: dict = {
        "model": args.model,
        "bounds": {"max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": args.max_num_batched_tokens,
                   "max_model_len": args.max_model_len, "num_gpu_blocks": args.num_gpu_blocks},
        "base_top1_vs_fp32": base_vs_fp32,
        "control": {
            "teacher_forced_top1_agreement": control_agreement,
            "teacher_forced_positions": _agreement(base, control)[1],
            "teacher_forced_mismatch_gaps": _tf_gaps(base, control),
            "top1_vs_fp32": _rate(ref32, control) if ref32 else None,
            "free_run": _free_run(base, control),
            "free_run_divergence": _divergence_gaps(base, control),
            "relative_tokens_per_s": control["tokens_per_s"] / base["tokens_per_s"],
            "attention_counters": _attention_counters(control),
        },
        "candidates": {},
    }
    if _attention_counters(base) or _attention_counters(control):
        failures.append("plugin-off runs recorded attention counters")
    for name, (cand, tp_ref) in cands.items():
        match, total = _agreement(base, cand)
        agreement = match / total if total else 0.0
        counters = _attention_counters(cand)
        entry = {
            "teacher_forced_top1_agreement": agreement,
            "teacher_forced_positions": total,
            "teacher_forced_mismatch_gaps": _tf_gaps(base, cand),
            "top1_vs_fp32": _rate(ref32, cand) if ref32 else None,
            "free_run": _free_run(base, cand),
            "free_run_divergence": _divergence_gaps(base, cand),
            "relative_tokens_per_s": cand["tokens_per_s"] / tp_ref["tokens_per_s"],
            "tokens_per_s_reference": tp_ref["name"],
            "hipdnn_calls": {op: _hipdnn_calls(counters, op)
                             for op in ("attention.uniform_decode", "attention.mixed",
                                        "attention.prefill_no_context")},  # fmt: skip
            "attention_counters": counters,
            "attention_layers": cand.get("attention_layers"),
            "sample": cand["texts"][:2],
            "reports": cand["reports"],
        }
        summary["candidates"][name] = entry
        for op in ("attention.uniform_decode", "attention.mixed"):
            if sum(entry["hipdnn_calls"][op].values()) == 0:
                failures.append(f"{name}: no {op} call ran on hipDNN")
        if agreement < args.min_agreement and agreement < control_agreement:
            # Below both bars. Tolerated only when the run is at least as close
            # to the float32 reference as TRITON_ATTN in the same dtype is.
            fp32 = entry["top1_vs_fp32"]
            if fp32 is None or fp32 < base_vs_fp32 - args.fp32_tolerance:
                failures.append(
                    f"{name}: top-1 agreement {agreement:.4f} < {args.min_agreement} "
                    f"and < control {control_agreement:.4f}; vs float32 {fp32} "
                    f"(TRITON_ATTN {base_vs_fp32})"
                )
            else:
                entry["accepted_by_fp32_reference"] = True

    print(json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    print("E2E_OK" if not failures else "E2E_FAILED")
    return 1 if failures else 0


_GPU_MEMORY_UTILIZATION = 0.5
_BLOCK_SIZE = 16  # vLLM's default KV-cache block size for TRITON_ATTN
_ACTIVATION_MARGIN = 2 << 30


def _default_num_gpu_blocks(args) -> int:
    """Blocks that fit the most demanding run (the float32 reference when it
    runs) in the gpu_memory_utilization share of device memory, capped at 4096.
    Every run uses the same count so scheduling is identical across runs."""
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    text = cfg.get("text_config", cfg)
    layers = text["num_hidden_layers"]
    heads = text["num_attention_heads"]
    kv_heads = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or text["hidden_size"] // heads
    ckpt_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.model, "*.safetensors")))
    ckpt_elem = 4 if str(text.get("torch_dtype", cfg.get("torch_dtype"))) == "float32" else 2
    elem = 4 if args.fp32_reference or args.dtype == "float32" else 2
    weights = ckpt_bytes * elem // ckpt_elem
    total = int(subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.mem_get_info()[1])"],
        check=True, capture_output=True, text=True).stdout.split()[-1])  # fmt: skip
    per_block = 2 * layers * kv_heads * head_dim * _BLOCK_SIZE * elem
    blocks = min(4096, (int(total * _GPU_MEMORY_UTILIZATION) - weights - _ACTIVATION_MARGIN) // per_block)
    need = -(-args.max_model_len // _BLOCK_SIZE)
    if blocks < need:
        raise SystemExit(f"--num-gpu-blocks: only {blocks} blocks fit ({total >> 20} MiB device), "
                         f"max_model_len {args.max_model_len} needs {need}; pass --num-gpu-blocks "
                         f"or --no-fp32-reference")  # fmt: skip
    print(f"num_gpu_blocks={blocks} (device {total >> 20} MiB, {per_block} B/block)", flush=True)
    return blocks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="KV-cache blocks for every run (default: the most that fit the float32 "
                        "reference in the runs' gpu_memory_utilization share, at most 4096)")
    p.add_argument("--spec-tokens", type=int, default=0)
    p.add_argument("--tp-batch", type=int, default=32)
    p.add_argument("--tp-tokens", type=int, default=128)
    p.add_argument("--min-agreement", type=float, default=0.99)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-fp32-reference", dest="fp32_reference", action="store_false")
    p.add_argument("--fp32-tolerance", type=float, default=0.002)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--child")
    p.add_argument("--backend", default="TRITON_ATTN")
    p.add_argument("--json-out")
    p.add_argument("--reference")
    args = p.parse_args()
    if args.child:
        return child(args)
    if args.num_gpu_blocks is None:
        args.num_gpu_blocks = _default_num_gpu_blocks(args)
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
