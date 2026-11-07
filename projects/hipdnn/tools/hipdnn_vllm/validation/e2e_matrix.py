# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Model-matrix validation driver for hipdnn_vllm: accuracy, coverage and relative perf.

usage:
  python e2e_matrix.py run --model-name NAME --out DIR [--models models.json]
         [--root DIR] [--candidates vllm_c,default,default+linear] [--enforce-eager]
         [--no-default-perf] [--min-agreement 0.99] [--prompts N]
         [--max-tokens T] [--timeout S]

One model, several vLLM configurations, each in its own process:

* ``base``: plugin disabled (HIPDNN_VLLM_DISABLE=1). Decoder attention uses
  TRITON_ATTN, the implementation HIPDNN falls back to, so the comparison
  isolates what hipDNN computes.
* ``control``: base again in a new process. Its agreement with base is the
  run-to-run noise floor that every candidate is judged against.
* ``default``: plugin disabled with vLLM's default attention backend. It is
  only the perf reference (perf is reported relative to the default backend).
* one run per candidate. A candidate is ops joined with ``+`` (``default``
  expands to the plugin's default op set). ``attention`` selects
  ``attention_backend=HIPDNN``; ``rms_norm`` and ``fused_add_rms_norm`` put
  ``hipdnn`` first in vLLM's IR priority. The special candidate ``vllm_c``
  is plugin-off with ``vllm_c`` first for both norm ops: a non-hipDNN kernel
  change of the same ops, for scale.

Accuracy: base greedy-generates a fixed prompt set. Every other run
re-scores base's prompt + completion with
``prompt_logprobs`` (teacher forcing), so all runs see identical inputs. Per
completion position, the run's top-1 token is compared with base's own
re-scored top-1. Also reported: the mean and max absolute difference of the
log-probability of the forced token, and base's top-1/top-2 margin at each
disagreement (near ties are expected rounding flips). Embedding models use the
cosine similarity of each embedding to base's instead.

A candidate passes when

1. it routed through hipDNN: every op family it enables that the model
   exercised (``attention`` always; ``encoder_attention`` for ``vlm`` models;
   any other family with at least one counter row) has at least one
   ``route == hipdnn`` count, unless the model lists the family in
   ``fallback_ops`` in models.json (layers the plugin is known to leave on
   fallback). A candidate with no hipDNN route at all always fails;
2. no log-probability (or embedding value) is NaN; and
3. its agreement is at least ``min(--min-agreement, max(control agreement,
   CONTROL_FLOOR))`` (``--min-cosine`` and ``CONTROL_FLOOR_COSINE`` for
   embeddings): the control can relax the target down to the floor, never
   below it.

The plugin-off runs (base, control, default, vllm_c) must record no plugin
counters. The control is the noise reference, so its verdict is ``-``; it
fails the model when it errors, has NaN, or agrees with base less than the
floor (run-to-run noise too large to judge any candidate).

Coverage: each run's plugin counters (``hipdnn_vllm.stats.snapshot()`` in the
worker via ``collective_rpc``) plus the ``report-*.json`` files under
``<out>/<run>/reports``. Perf: engine start-up time, TTFT (one request, the
first call cold and then the median of warm calls), output tokens/s for a
fixed batch, plan builds and build time, norm exact-row pre-build time and
hipDNN handle creation time (from the plugin's INFO log line).

Writes ``<out>/summary.json``; ``aggregate.py`` turns many of them into tables.
Exits non-zero when base, the control or a plugin-off run check fails, or any
candidate fails its criterion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_K = 2
NORM_OPS = ("rms_norm", "fused_add_rms_norm")
NORM_PRIORITY = ["hipdnn", "vllm_c", "native"]
REFERENCE_ATTENTION = "TRITON_ATTN"
# vLLM's KV-cache budget ran out: a larger gpu_memory_utilization can help.
KV_MEMORY_RE = re.compile(r"No available memory for the cache blocks|out of memory")
# vLLM's start-up check: the device has less free memory (other processes on a
# shared GPU) than gpu_memory_utilization asks for, so only a smaller request
# can help. Groups: free GiB, requested GiB.
FREE_MEMORY_RE = re.compile(r"Free memory on device \S+ \(([0-9.]+)/[0-9.]+ GiB\) on startup is less than "
                            r"[^(]*\([0-9.]+, ([0-9.]+) GiB\)")
MEMORY_ATTEMPTS = 3
MEMORY_GROWTH = 1.6
# The control is the same plugin-off configuration in a new process; it agrees
# with base up to rare near-tie flips from nondeterministic kernels. A control
# below these floors means the environment is too noisy to judge a candidate,
# and the floors bound how far the control can relax the candidate threshold.
CONTROL_FLOOR = 0.97
CONTROL_FLOOR_COSINE = 0.995
# Counter op names -> the HIPDNN_VLLM_OPS family that routes them.
COUNTER_FAMILY = {"sdpa_fwd": "attention", "conv_fwd": "conv", "conv_fprop": "conv"}
HANDLE_RE = re.compile(r"hipDNN handle on device (\d+) created in ([0-9.]+) s")

PROMPTS = [
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
    "Summarize the plot of Hamlet in two sentences.",
    "The best way to learn a new language is",
    "import numpy as np\n\ndef softmax(x):",
    "Photosynthesis is the process by which",
    "Q: What is the boiling point of water at sea level?\nA:",
    "The French Revolution began in",
    "A good unit test should",
    "Dear hiring manager, I am writing to",
    "The mitochondria is",
    "Rust and C++ differ in that",
    "Here is a recipe for pancakes:",
    "The speed of light in a vacuum is",
    "In machine learning, overfitting means",
    "The largest planet in the solar system is",
    "To reverse a linked list, you",
    "The Great Wall of China was built",
]
TTFT_PROMPT = (
    "The following is a detailed technical report on the design of large-scale "
    "distributed systems. " * 24
    + "In summary, the most important lesson is"
)
EMBED_TEXTS = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "GPUs accelerate dense linear algebra.",
    "hi",
    "A long passage about attention: " + " ".join(f"token{i}" for i in range(180)),
    "query: how do embedding models work",
    "passage: Embedding models map text to vectors so similar meanings land close together.",
    "Short.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "机器学习是人工智能的一个分支。",
]

# Vision-language cases (Qwen2.5-VL chat format): image size and question.
# Sizes mix uniform windows (multiples of 112) with ragged edge windows.
VLM_QUESTIONS = [
    "Describe this image in one sentence.",
    "List every object you can see in this image and say where it is.",
    "Which colors dominate this picture? Answer in two sentences.",
]
VLM_CASES = [
    ((448, 448), 0), ((504, 364), 1), ((336, 336), 2), ((392, 560), 0),
    ((616, 448), 1), ((280, 420), 2), ((448, 308), 0), ((532, 532), 1),
]  # fmt: skip


def vlm_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )


def make_image(width: int, height: int):
    """A deterministic synthetic scene (no network or dataset needed)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (135, 206, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, height * 2 // 3, width, height], fill=(34, 139, 34))
    d.ellipse([width - 120, 20, width - 40, 100], fill=(255, 215, 0))
    d.rectangle([60, height // 2 - 20, 200, height * 2 // 3], fill=(178, 34, 34))
    d.polygon([(50, height // 2 - 20), (130, height // 2 - 90), (210, height // 2 - 20)], fill=(120, 60, 20))
    d.rectangle([115, height * 2 // 3 - 50, 145, height * 2 // 3], fill=(80, 40, 10))
    return img


# --- run specifications -----------------------------------------------------------


def run_spec(role: str, candidate: str | None) -> dict:
    """Environment and LLM kwargs that define one run (without the model)."""
    from hipdnn_vllm.config import ALL_OPS, DEFAULT_OPS

    if role in ("base", "control"):
        return {"plugin": False, "ops": [], "attention_backend": REFERENCE_ATTENTION, "ir_op_priority": None}
    if role == "default":
        return {"plugin": False, "ops": [], "attention_backend": None, "ir_op_priority": None}
    if candidate == "vllm_c":
        prio = {op: ["vllm_c", "native"] for op in NORM_OPS}
        return {"plugin": False, "ops": [], "attention_backend": REFERENCE_ATTENTION, "ir_op_priority": prio}
    ops: list[str] = []
    for tok in candidate.split("+"):
        for op in DEFAULT_OPS if tok == "default" else (tok,):
            if op not in ALL_OPS:
                raise SystemExit(f"candidate {candidate!r}: unknown op {op!r} (valid: {', '.join(ALL_OPS)})")
            if op not in ops:
                ops.append(op)
    prio = {op: NORM_PRIORITY for op in NORM_OPS if op in ops}
    return {
        "plugin": True,
        "ops": ops,
        "attention_backend": "HIPDNN" if "attention" in ops else REFERENCE_ATTENTION,
        "ir_op_priority": prio or None,
    }


def gpu_memory_utilization(entry: dict, total_gb: float) -> float:
    """A budget of weights x 1.2 + 10 GB (activations and KV cache), as a
    fraction of the device, clamped to [0.2, 0.9]."""
    budget = entry.get("weights_gb", 4.0) * 1.2 + entry.get("extra_gb", 10.0)
    return round(min(0.9, max(0.2, budget / total_gb)), 3)


def scaled_utilization(entry: dict, total_gb: float, scale: float) -> float:
    """gpu_memory_utilization after a memory retry scaled the budget."""
    return min(0.9, round(gpu_memory_utilization(entry, total_gb) * scale, 3))


# --- child: one vLLM configuration --------------------------------------------------


def _worker_stats(worker) -> dict | None:
    if "hipdnn_vllm" not in sys.modules:
        return None
    from hipdnn_vllm import stats

    snap = stats.snapshot()
    mod = sys.modules.get("hipdnn_vllm._C")
    snap["norm_prebuild"] = mod.norm_prebuild_totals() if mod is not None and hasattr(
        mod, "norm_prebuild_totals") else None
    return snap


def _scored_rows(outputs, completions: list[list[int]]) -> list[list[list]]:
    """Per completion position: [[top1, lp], [top2, lp], [forced token, lp]]."""
    rows_all = []
    for o, g in zip(outputs, completions):
        ids = list(o.prompt_token_ids)
        if ids[len(ids) - len(g):] != g:
            raise RuntimeError("scored prompt does not end with the completion")
        rows = []
        for pos in range(len(ids) - len(g), len(ids)):
            entry = o.prompt_logprobs[pos]
            ranked = sorted((v.rank, t, v.logprob) for t, v in entry.items()
                            if v.rank is not None and v.rank <= TOP_K)
            row = [[t, lp] for _, t, lp in ranked]
            row.append([ids[pos], entry[ids[pos]].logprob])
            rows.append(row)
        rows_all.append(rows)
    return rows_all


def _requests(kind: str, llm, reference: dict | None, max_tokens: int, n_prompts: int):
    """(generation requests, scoring builder) for a generative kind."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    if kind == "vlm":
        cases = [(make_image(w, h), vlm_prompt(VLM_QUESTIONS[q])) for (w, h), q in VLM_CASES]
        gen_reqs = [{"prompt": p, "multi_modal_data": {"image": img}} for img, p in cases]
        tok = llm.get_tokenizer()

        def score(gen_ids):
            # The unexpanded prompt (one image placeholder) plus the completion;
            # vLLM expands the placeholder, so completion positions are the tail.
            return [TokensPrompt(prompt_token_ids=tok.encode(p, add_special_tokens=False) + g,
                                 multi_modal_data={"image": img}) for (img, p), g in zip(cases, gen_ids)]
    else:
        prompts = [PROMPTS[i % len(PROMPTS)] + ("" if i < len(PROMPTS) else f" ({i})") for i in range(n_prompts)]
        gen_reqs = prompts

        def score(gen_ids):
            pids = reference["prompt_ids"] if reference else score.prompt_ids
            return [TokensPrompt(prompt_token_ids=p + g) for p, g in zip(pids, gen_ids)]

        score.prompt_ids = None
    return gen_reqs, score, SamplingParams(temperature=0.0, max_tokens=max_tokens)


def child(args) -> int:
    # No torch.cuda in this process: initializing HIP before LLM() makes vLLM
    # switch its EngineCore to spawn (and the device is the parent's to query).
    from vllm import LLM, SamplingParams

    entry = json.loads(args.entry)
    spec = json.loads(args.spec)
    kind = entry["kind"]
    total_gb = args.total_gb
    result: dict = {"role": args.role, "spec": spec, "kind": kind, "total_gb": total_gb,
                    "hipdnn_vllm_ops": os.environ.get("HIPDNN_VLLM_OPS"),
                    "plugin_disabled": os.environ.get("HIPDNN_VLLM_DISABLE") == "1"}
    kwargs = dict(model=entry["path"], max_model_len=entry.get("max_model_len", 2048),
                  gpu_memory_utilization=scaled_utilization(entry, total_gb, args.util_scale),
                  seed=0,
                  enforce_eager=args.enforce_eager, dtype=entry.get("dtype", "auto"))
    kwargs.update(entry.get("llm_kwargs", {}))
    if kind == "embed":
        kwargs["runner"] = "pooling"
    if kind == "vlm":
        kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
    if spec["attention_backend"]:
        kwargs["attention_backend"] = spec["attention_backend"]
    if spec["ir_op_priority"]:
        kwargs["ir_op_priority"] = spec["ir_op_priority"]
    result["llm_kwargs"] = kwargs

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    result["init_s"] = time.perf_counter() - t0
    vcfg = llm.llm_engine.vllm_config
    prio = vcfg.kernel_config.ir_op_priority
    result["ir_op_priority"] = {op: list(getattr(prio, op)) for op in NORM_OPS}
    result["attention_backend"] = str(getattr(vcfg.attention_config, "backend", None))
    result["cudagraph_mode"] = str(vcfg.compilation_config.cudagraph_mode)
    reference = json.load(open(args.reference)) if args.reference else None

    def timed(fn):
        # LLM.generate/embed return only after the engine produced every output.
        t = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t

    if kind == "embed":
        _, cold = timed(lambda: llm.embed([TTFT_PROMPT], use_tqdm=False))
        warm = [timed(lambda: llm.embed([TTFT_PROMPT], use_tqdm=False))[1] for _ in range(5)]
        result["ttft_cold_s"], result["ttft_warm_s"] = cold, statistics.median(warm)
        outs = llm.embed(EMBED_TEXTS, use_tqdm=False)
        vecs = [list(map(float, o.outputs.embedding)) for o in outs]
        result["embeddings"] = vecs
        result["nan"] = any(not math.isfinite(x) for v in vecs for x in v)
        batch = [EMBED_TEXTS[i % len(EMBED_TEXTS)] + f" {i}" for i in range(256)]
        llm.embed(batch, use_tqdm=False)
        rates = []
        for _ in range(3):
            o, dt = timed(lambda: llm.embed(batch, use_tqdm=False))
            rates.append(sum(len(x.prompt_token_ids) for x in o) / dt)
        result["tokens_per_s"] = statistics.median(rates)
    else:
        one = SamplingParams(temperature=0.0, max_tokens=1)
        if kind == "vlm":
            ttft_req = [{"prompt": vlm_prompt(VLM_QUESTIONS[0]), "multi_modal_data": {"image": make_image(448, 448)}}]
        else:
            ttft_req = [TTFT_PROMPT]
        _, cold = timed(lambda: llm.generate(ttft_req, one, use_tqdm=False))
        warm = [timed(lambda: llm.generate(ttft_req, one, use_tqdm=False))[1] for _ in range(5)]
        result["ttft_cold_s"], result["ttft_warm_s"] = cold, statistics.median(warm)

        gen_reqs, score, greedy = _requests(kind, llm, reference, args.max_tokens, args.prompts)
        if reference is None:
            outs = llm.generate(gen_reqs, greedy, use_tqdm=False)
            result["prompt_ids"] = [list(o.prompt_token_ids) for o in outs]
            result["gen_ids"] = [list(o.outputs[0].token_ids) for o in outs]
            result["texts"] = [o.outputs[0].text for o in outs]
            score.prompt_ids = result["prompt_ids"]
            gen_ids = result["gen_ids"]
        else:
            gen_ids = reference["gen_ids"]
        scored = llm.generate(score(gen_ids), SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOP_K),
                              use_tqdm=False)
        result["rows"] = _scored_rows(scored, gen_ids)
        result["nan"] = any(lp != lp for rows in result["rows"] for row in rows for _, lp in row)

        tp = SamplingParams(temperature=0.0, max_tokens=args.tp_tokens, ignore_eos=True)
        if kind == "vlm":
            tp_reqs = gen_reqs
        else:
            tp_reqs = [PROMPTS[i % len(PROMPTS)] + f" [{i}]" for i in range(args.tp_batch)]
        llm.generate(tp_reqs, tp, use_tqdm=False)  # warm-up: eager plans for these shapes
        rates = []
        for _ in range(3):
            o, dt = timed(lambda: llm.generate(tp_reqs, tp, use_tqdm=False))
            rates.append(sum(len(x.outputs[0].token_ids) for x in o) / dt)
        result["tokens_per_s"] = statistics.median(rates)

    result["worker_stats"] = [s for s in llm.collective_rpc(_worker_stats) if s is not None]
    json.dump(result, open(args.json_out, "w"))
    del llm
    return 0


# --- comparisons ----------------------------------------------------------------------


def tf_compare(base: dict, other: dict) -> dict:
    match = total = near_ties = 0
    deltas = []
    disagreements = []
    for i, (b_rows, o_rows) in enumerate(zip(base["rows"], other["rows"])):
        if len(b_rows) != len(o_rows):
            raise RuntimeError(f"case {i}: {len(b_rows)} vs {len(o_rows)} scored positions")
        for pos, (b, o) in enumerate(zip(b_rows, o_rows)):
            total += 1
            deltas.append(abs(b[-1][1] - o[-1][1]))
            if b[0][0] == o[0][0]:
                match += 1
                continue
            b_lp = {t: lp for t, lp in b[:-1]}
            margin = b[0][1] - b_lp[o[0][0]] if o[0][0] in b_lp else None
            near_ties += int(margin is not None and margin < 0.1)
            disagreements.append({"case": i, "pos": pos, "base_top": b[:-1], "got_top": o[:-1],
                                  "base_margin_nats": margin})
    finite = [d for d in deltas if math.isfinite(d)]
    return {"agreement": match / total if total else 0.0, "positions": total, "nan": other["nan"],
            "forced_logprob_abs_delta_mean": statistics.mean(finite) if finite else None,
            "forced_logprob_abs_delta_max": max(finite) if finite else None,
            "disagreements": len(disagreements), "near_ties_lt_0.1_nats": near_ties,
            "disagreement_detail": disagreements[:40]}


def embed_compare(base: dict, other: dict) -> dict:
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    c = [cos(a, b) for a, b in zip(base["embeddings"], other["embeddings"])]
    return {"agreement": min(c) if c else 0.0, "metric": "min cosine", "cosines": c, "positions": len(c),
            "nan": other["nan"]}


def counters(run: dict) -> list[dict]:
    merged: dict[tuple, int] = {}
    for snap in run.get("worker_stats") or []:
        for r in snap["counters"]:
            key = (r["op"], r["route"], r["reason"] or "", r["phase"])
            merged[key] = merged.get(key, 0) + r["count"]
    return [{"op": k[0], "route": k[1], "reason": k[2], "phase": k[3], "count": v} for k, v in sorted(merged.items())]


def counter_family(op: str) -> str:
    """``attention.uniform_decode`` -> ``attention``, ``conv_fwd`` -> ``conv``."""
    head = op.split(".", 1)[0]
    return COUNTER_FAMILY.get(head, head)


def routing_problems(entry: dict, spec: dict, rows: list[dict]) -> list[str]:
    """Why a run's counters contradict its spec (empty when they agree).

    A plugin-off run must record nothing. A plugin run must route at least one
    call through hipDNN, and at least one call of every enabled family the
    model exercised, unless models.json lists the family in ``fallback_ops``."""
    if not spec["plugin"]:
        calls = sum(r["count"] for r in rows)
        return [f"plugin disabled but {calls} plugin calls counted"] if calls else []
    families = {"linear" if op == "linear_all" else op for op in spec["ops"]}
    seen: dict[str, int] = {}
    for r in rows:
        fam = counter_family(r["op"])
        seen[fam] = seen.get(fam, 0) + (r["count"] if r["route"] == "hipdnn" else 0)
    if not any(seen.values()):
        return ["no call routed to hipDNN"]
    exercised = {"attention"} | ({"encoder_attention"} if entry["kind"] == "vlm" else set()) | set(seen)
    expected = (families & exercised) - set(entry.get("fallback_ops", []))
    return [f"{fam}: enabled but no call routed to hipDNN" for fam in sorted(expected) if not seen.get(fam)]


def stalls(run: dict, log_path: str) -> dict:
    out = {"plan_builds": 0, "plan_build_declines": 0, "plan_build_s": 0.0, "norm_prebuild_s": 0.0,
           "handle_create_s": None}
    for snap in run.get("worker_stats") or []:
        rt = snap.get("runtime") or {}
        out["plan_builds"] += rt.get("plan_builds", 0)
        out["plan_build_declines"] += rt.get("plan_build_declines", 0)
        out["plan_build_s"] += rt.get("plan_build_ns", 0) / 1e9
        pre = snap.get("norm_prebuild") or {}
        out["norm_prebuild_s"] += pre.get("ns", 0) / 1e9
    if os.path.exists(log_path):
        found = HANDLE_RE.findall(open(log_path, errors="replace").read())
        if found:
            out["handle_create_s"] = sum(float(s) for _, s in found)
    return out


# --- parent -------------------------------------------------------------------------------


def _logging_config(path: str) -> str:
    """vLLM applies VLLM_LOGGING_CONFIG_PATH in every process; this adds the
    plugin's INFO lines (handle creation time) to the run log."""
    handler = {"class": "logging.StreamHandler", "formatter": "f", "stream": "ext://sys.stdout", "level": "INFO"}
    cfg = {"version": 1, "disable_existing_loggers": False,
           "formatters": {"f": {"format": "%(levelname)s %(asctime)s [%(name)s] %(message)s"}},
           "handlers": {"h": handler},
           "loggers": {name: {"handlers": ["h"], "level": "INFO", "propagate": False}
                       for name in ("vllm", "hipdnn_vllm")}}
    json.dump(cfg, open(path, "w"))
    return path


def _run(args, entry: dict, name: str, role: str, spec: dict, reference: str | None) -> dict:
    run_dir = os.path.join(args.out, name)
    os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)
    out_json = os.path.join(run_dir, "run.json")
    log_path = os.path.join(run_dir, "run.log")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_VLLM_")}
    env.update(VLLM_ALLOW_INSECURE_SERIALIZATION="1", VLLM_CACHE_ROOT=os.path.join(args.out, "vllm-cache"),
               VLLM_LOGGING_CONFIG_PATH=_logging_config(os.path.join(args.out, "logging.json")),
               HIPDNN_VLLM_REPORT_DIR=os.path.join(run_dir, "reports"), HIPDNN_VLLM_REPORT_INTERVAL="10")
    if spec["plugin"]:
        env["HIPDNN_VLLM_OPS"] = ",".join(spec["ops"])
    else:
        env["HIPDNN_VLLM_DISABLE"] = "1"
    for k, v in (args.extra_env or {}).items():
        env[k] = v
    base_cmd = [sys.executable, os.path.abspath(__file__), "child", "--role", role, "--entry", json.dumps(entry),
                "--spec", json.dumps(spec), "--json-out", out_json, "--prompts", str(args.prompts),
                "--max-tokens", str(args.max_tokens), "--tp-batch", str(args.tp_batch),
                "--tp-tokens", str(args.tp_tokens), "--total-gb", str(args.total_gb)]  # fmt: skip
    if args.enforce_eager:
        base_cmd.append("--enforce-eager")
    if reference:
        base_cmd += ["--reference", reference]
    print(f"== {time.strftime('%H:%M:%S')} {name}: {json.dumps(spec)}", flush=True)
    t0 = time.time()
    # A shared GPU (e.g. a Slurm shard) runs other jobs. Their allocations can
    # leave vLLM's KV cache no budget (retry with a larger utilization) or leave
    # less free memory than the utilization asks for at start-up (retry with a
    # request that fits what is free; asking for more cannot help). A retry
    # never reverses direction: needing both means the device has no room.
    scale, shrunk, grown = 1.0, False, False
    for attempt in range(MEMORY_ATTEMPTS):
        if os.path.exists(out_json):
            os.remove(out_json)
        cmd = base_cmd + ["--util-scale", f"{scale:.4f}"]
        with open(log_path, "w") as log:
            # A new session, so a timeout also kills vLLM's EngineCore grandchild.
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = "timeout"
            try:  # leftovers (an EngineCore that outlived its parent) would hold the GPU
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if (rc == 0 and os.path.exists(out_json)) or attempt == MEMORY_ATTEMPTS - 1:
            break
        text = open(log_path, errors="replace").read()
        util = scaled_utilization(entry, args.total_gb, scale)
        free = FREE_MEMORY_RE.search(text)
        if free:
            free_gib, requested_gib = float(free.group(1)), float(free.group(2))
            fits = scale * 0.95 * free_gib / requested_gib
            if grown or fits >= scale or 0.95 * free_gib * 2**30 / 1e9 < entry.get("weights_gb", 0.0):
                break
            scale, shrunk = fits, True
            why = f"{free_gib:.1f} GiB free < {requested_gib:.1f} GiB requested"
        elif KV_MEMORY_RE.search(text):
            if shrunk or util >= 0.9:
                break
            scale, grown = scale * MEMORY_GROWTH, True
            why = "no KV-cache budget"
        else:
            break
        print(f"   {name}: {why} at gpu_memory_utilization {util}; retrying with "
              f"{scaled_utilization(entry, args.total_gb, scale)}", flush=True)
    wall = time.time() - t0
    if rc != 0 or not os.path.exists(out_json):
        tail = open(log_path, errors="replace").read().splitlines()[-40:]
        print(f"   {name} FAILED rc={rc} after {wall:.0f} s; log {log_path}", flush=True)
        print("\n".join("   | " + line for line in tail), flush=True)
        return {"name": name, "role": role, "spec": spec, "error": f"rc={rc}", "log": log_path, "log_tail": tail}
    data = json.load(open(out_json))
    data.update(name=name, wall_s=wall, log=log_path, stalls=stalls(data, log_path), counters=counters(data))
    print(f"   {name} ok in {wall:.0f} s", flush=True)
    return data


def parent(args) -> int:
    models = json.load(open(args.models))["models"]
    entry = next((m for m in models if m["name"] == args.model_name), None)
    if entry is None:
        raise SystemExit(f"{args.model_name!r} not in {args.models}")
    entry = dict(entry)
    if not os.path.isabs(entry["path"]):
        entry["path"] = os.path.join(args.root, entry["path"])
    args.extra_env = entry.get("env", {})
    os.makedirs(args.out, exist_ok=True)

    # Query the device in a subprocess: a CUDA context in this process would
    # hold device memory for the whole matrix.
    probe = ("import json, torch; p = torch.cuda.get_device_properties(0); "
             "print(json.dumps([p.gcnArchName.split(':')[0], p.total_memory / 2**30, torch.cuda.get_device_name(0)]))")
    arch, total_gb, device = json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                                       check=True).stdout.strip().splitlines()[-1])
    args.total_gb = total_gb
    summary: dict = {"model": entry["name"], "entry": entry, "arch": arch, "device": device,
                     "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "enforce_eager": args.enforce_eager,
                     "min_agreement": args.min_agreement, "candidates": args.candidates.split(","), "runs": {}}
    summary_path = os.path.join(args.out, "summary.json")
    need = entry.get("min_vram_gb", 0)
    if need > total_gb or arch in entry.get("skip_archs", []):
        summary["skipped"] = f"device has {total_gb:.0f} GB, model needs {need} GB" if need > total_gb else \
            f"arch {arch} listed in skip_archs: {entry.get('skip_reason', '')}"
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"SKIP {entry['name']} on {arch}: {summary['skipped']}")
        return 0

    plan = [("base", "base", None), ("control", "control", None)]
    if args.default_perf and entry["kind"] != "embed":
        plan.append(("default", "default", None))
    # vllm_c is plugin-off: a reference for scale, reported but not gating.
    plan += [(f"cand-{c}", "reference" if c == "vllm_c" else "candidate", c) for c in summary["candidates"]]
    runs: dict[str, dict] = {}
    base_json = os.path.join(args.out, "base", "run.json")
    for name, role, cand in plan:
        spec = run_spec(role, cand)
        runs[name] = _run(args, entry, name, role, spec, None if role == "base" else base_json)
        if name == "base" and "error" in runs[name]:
            break

    failures: list[str] = []
    base = runs["base"]
    cmp = embed_compare if entry["kind"] == "embed" else tf_compare
    ctrl_acc = None
    for name, run in runs.items():
        rec = {k: run.get(k) for k in ("role", "spec", "error", "log", "log_tail", "init_s", "ttft_cold_s",
                                       "ttft_warm_s", "tokens_per_s", "stalls", "counters", "attention_backend",
                                       "ir_op_priority", "cudagraph_mode", "wall_s", "nan")}
        rec = {k: v for k, v in rec.items() if v is not None}
        if "error" not in run and "error" not in base and name != "base":
            rec["accuracy"] = cmp(base, run)
        if name == "base" and "error" not in run and entry["kind"] != "embed":
            rec["texts"] = run.get("texts")
            # Base's teacher-forced top-1 against its own free-running greedy
            # tokens (decode vs prefill path consistency), for scale.
            same = tot = 0
            for rows, g in zip(run["rows"], run["gen_ids"]):
                tot += len(g)
                same += sum(int(r[0][0] == t) for r, t in zip(rows, g))
            rec["self_prefill_vs_decode"] = same / tot if tot else 0.0
        summary["runs"][name] = rec
    if "error" in base:
        failures.append(f"base run failed ({base['error']})")
    else:
        embed = entry["kind"] == "embed"
        target = args.min_cosine if embed else args.min_agreement
        floor = CONTROL_FLOOR_COSINE if embed else CONTROL_FLOOR
        if "accuracy" in summary["runs"].get("control", {}):
            ctrl_acc = summary["runs"]["control"]["accuracy"]["agreement"]
        # The control relaxes the target to run-to-run noise, never below the floor.
        threshold = min(target, max(ctrl_acc, floor)) if ctrl_acc is not None else target
        summary["threshold"] = threshold
        summary["control_floor"] = floor
        for name, rec in summary["runs"].items():
            if "error" in rec:
                rec["verdict"] = "ERROR"
                if rec["role"] == "candidate" or name == "control":
                    failures.append(f"{name}: {rec['error']}")
                continue
            problems = routing_problems(entry, rec["spec"], rec.get("counters", []))
            if problems:
                rec["routing_problems"] = problems
                failures += [f"{name}: {p}" for p in problems]
            if name == "base":
                continue
            acc = rec["accuracy"]
            if name == "control":
                rec["verdict"] = "-"
                if acc["nan"] or acc["agreement"] < floor:
                    failures.append(f"control: agreement {acc['agreement']:.4f} < floor {floor} or NaN={acc['nan']}; "
                                    "run-to-run noise too large to judge candidates")
            else:
                accurate = not acc["nan"] and acc["agreement"] >= threshold
                rec["verdict"] = "PASS" if accurate and not problems else "FAIL"
                if rec["role"] == "candidate" and not accurate:
                    failures.append(f"{name}: agreement {acc['agreement']:.4f} < {threshold:.4f} or NaN={acc['nan']}")
            for ref_name in ("base", "default"):
                ref = runs.get(ref_name)
                if ref and "error" not in ref and ref.get("tokens_per_s") and rec.get("tokens_per_s"):
                    rec[f"rel_tokens_per_s_vs_{ref_name}"] = rec["tokens_per_s"] / ref["tokens_per_s"]
                    rec[f"rel_ttft_vs_{ref_name}"] = rec["ttft_warm_s"] / ref["ttft_warm_s"]
                    rec[f"rel_init_vs_{ref_name}"] = rec["init_s"] / ref["init_s"]
    summary["failures"] = failures
    json.dump(summary, open(summary_path, "w"), indent=2)

    print(f"\nSUMMARY {entry['name']} on {arch} ({summary['device']}), threshold {summary.get('threshold')}")
    for name, rec in summary["runs"].items():
        acc = rec.get("accuracy") or {}
        print(f"  {name:32s} {rec.get('verdict', '-'):5s} agree={acc.get('agreement', float('nan')):.4f} "
              f"n={acc.get('positions', 0)} dlp_mean={acc.get('forced_logprob_abs_delta_mean')} "
              f"near_ties={acc.get('near_ties_lt_0.1_nats')}/{acc.get('disagreements')} "
              f"tok/s_vs_base={rec.get('rel_tokens_per_s_vs_base')} {rec.get('error', '')}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-name", required=True)
    r.add_argument("--models", default=os.path.join(HERE, "models.json"))
    r.add_argument("--root", default=os.environ.get("HV_ROOT", "."), help="base for relative model paths")
    r.add_argument("--out", required=True)
    r.add_argument("--candidates", default="vllm_c,default,default+linear")
    r.add_argument("--no-default-perf", dest="default_perf", action="store_false")
    r.add_argument("--min-agreement", type=float, default=0.99)
    r.add_argument("--min-cosine", type=float, default=0.999)
    r.add_argument("--timeout", type=float, default=5400)
    c = sub.add_parser("child")
    c.add_argument("--role", required=True)
    c.add_argument("--entry", required=True)
    c.add_argument("--spec", required=True)
    c.add_argument("--json-out", required=True)
    c.add_argument("--reference")
    c.add_argument("--total-gb", type=float, required=True)
    c.add_argument("--util-scale", type=float, default=1.0)
    for q in (r, c):
        q.add_argument("--enforce-eager", action="store_true")
        q.add_argument("--prompts", type=int, default=32)
        q.add_argument("--max-tokens", type=int, default=64)
        q.add_argument("--tp-batch", type=int, default=32)
        q.add_argument("--tp-tokens", type=int, default=128)
    pw = sub.add_parser("pairwise", help="agreement of every finished run with each --ref run (bisect aid)")
    pw.add_argument("dir", help="a model result directory (contains <run>/run.json)")
    pw.add_argument("--ref", action="append", help="reference run names (default: base)")
    args = p.parse_args()
    if args.cmd == "pairwise":
        return pairwise(args)
    return parent(args) if args.cmd == "run" else child(args)


def pairwise(args) -> int:
    """All runs scored base's corpus, so any two can be compared directly: if
    run A and run B agree with each other as well as the control agrees with
    base, they compute the same thing up to run-to-run noise."""
    runs = {}
    for name in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, name, "run.json")
        if os.path.exists(path):
            runs[name] = json.load(open(path))
    cmp = embed_compare if next(iter(runs.values()))["kind"] == "embed" else tf_compare
    for ref in args.ref or ["base"]:
        print(f"vs {ref}:")
        for name, run in runs.items():
            if name == ref:
                continue
            c = cmp(runs[ref], run)
            extra = "" if cmp is embed_compare else (
                f" mean|dlogprob|={c['forced_logprob_abs_delta_mean']:.4f} "
                f"max={c['forced_logprob_abs_delta_max']:.3f} near_ties={c['near_ties_lt_0.1_nats']}/{c['disagreements']}")
            print(f"  {name:48s} agreement={c['agreement']:.4f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
