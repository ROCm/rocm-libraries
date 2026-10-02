#!/usr/bin/env python
"""Combined prefill+decode end-to-end timing for the two gfx1151 rocKE kernels.

Reproduces the end-to-end tables in this directory's README: the TTFT ladder
(section 4.1, the ``prefill_only`` arm), the decode ITL ladder (section 6, the
``decode_only`` arm) and the combined total-latency ladder (section 7, all four
arms). A single ``--ctx-sweep`` run produces all three.

Both attention optimizations were originally measured against their own
incumbent, in isolation:

    prefill  wmma_fmha_swapqk      beats vLLM's Triton FA        -> TTFT
    decode   paged_decode_splitk   beats AMD's paged_attention   -> ITL

Neither measurement can produce the combined number. A prefill-only run sets
``max_tokens=1``, so decode is absent by construction; a decode-only run that
leaves swapqk enabled in *both* arms cancels the prefill win -- and ITL measured
as a slope cancels prefill algebraically in any case. This script answers "how
much faster is the whole request", which is ``TTFT(P) + (G-1) * ITL`` with the
weight of each term swinging by an order of magnitude across context lengths.

Four arms cross the two kernels, so a run can say whether the two wins add,
interfere, or one dominates:

    stock         Triton prefill  + AMD decode     (matched control)
    prefill_only  swapqk          + AMD decode
    decode_only   Triton prefill  + rocKE decode
    both          swapqk          + rocKE decode

``stock`` is a matched control, not stock vLLM: it is the same rocKE backend
class with the swapqk dispatch floor raised past every prompt, so the plumbing
is identical and only the kernels differ.

Three generation lengths are timed per (workload, arm), and every reported
metric is one of them -- nothing is interpolated:

    G = 1        -> ttft_ms
    G = gen_lo   -> slope anchor (48 by default; at 8 the incumbent's request
                    admission ramp leaked in and manufactured a 4.5% regression)
    G = gen_hi   -> total_ms, the headline, and with gen_lo the ITL slope

Arms are patched as module attributes on the backend between ``generate()``
calls, so every arm shares one set of weights, one allocator state and one clock
ramp. This requires ``cudagraph_mode=PIECEWISE``. The asymmetry is not obvious
and is the reason a prefill-only sweep gets away with default cudagraphs:

  * Prefill arm-switching works under any mode. A FULL capture only ever applies
    to uniform pure-decode batches; ragged prefill always takes the piecewise
    path and runs eagerly, so the backend's Python re-reads the arm every step.
  * Decode arm-switching does not. Under a FULL mode the decode subgraph is
    captured once and that Python never runs again -- the arm freezes *and the
    dispatch counters freeze at zero*, so a counter assertion fires falsely in
    exactly the configuration you most want to test.

``--capture-check`` is the single-arm escape hatch for confirming the production
FULL config: the arm is applied through the environment before ``LLM()`` is
constructed, and the assertion reads the counters snapshotted immediately after
construction (capture time) rather than at steady state.

Every gate in the backend declines SILENTLY into the incumbent, so a run that
never reached a kernel still produces a plausible number -- the incumbent
measured twice. Hence every measurement asserts on the backend's own dispatch
counters and is discarded if the kernel did not actually run.

Requires a vLLM source checkout on ``PYTHONPATH`` exporting the
``ROCKE_GFX1151`` attention backend; see the README's "Reproducing" section.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

# Run directly (``python .../e2e_combined_bench.py``) rather than with ``-m``,
# and Python puts this file's own directory first on sys.path. If that directory
# sits next to the vllm source checkout it shadows the installed package as a
# namespace package and `from vllm import LLM` fails with "unknown location".
# Harmless no-op under ``-m``, where sys.path[0] is the cwd instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

# Keep the engine in-process so ROCKE_STATS is readable from here rather than
# only via the worker's atexit print.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# (batch, prompt_len). prompt_len must be a multiple of 64: swapqk's kv loop
# bound is seqlen // block_n and the tail is truncated rather than masked, so a
# non-divisible length is declined outright.
#
# `chat` is a negative control, not filler -- both kernels are predicted to do
# nothing at 0.15 GB of KV per step, and a run that reports ~0 where ~0 is
# expected is what makes the other rows credible.
#
# `longctx` is 30720 and not 32768 because Qwen3-8B's native context is 32768
# and prompt + gen_hi must fit. 30720 still puts KV bytes where B~=15 x 2048
# would, which is the regime where decode measured its win.
# The ctx* entries are a B=1 context sweep, selected together by --ctx-sweep.
# They exist to answer "what decode token rate does a user get at this context
# length", which the four named workloads above sample too sparsely to show.
# ctx32k is 30720 for the same reason longctx is. ctx8k and longctx duplicate
# summarize and ctx32k respectively; the names are kept distinct so the sweep
# reads as one contiguous series rather than borrowing rows from another table.
WORKLOADS = {
    "chat": (1, 1024),
    "summarize": (1, 8192),
    "longctx": (1, 30720),
    "serve": (32, 2048),
    "ctx2k": (1, 2048),
    "ctx4k": (1, 4096),
    "ctx8k": (1, 8192),
    "ctx16k": (1, 16384),
    "ctx32k": (1, 30720),
}
NAMED_WORKLOADS = ["chat", "summarize", "longctx", "serve"]
CTX_SWEEP = ["ctx2k", "ctx4k", "ctx8k", "ctx16k", "ctx32k"]

# Module attributes patched onto vllm.v1.attention.backends.rocm_rocke_attn.
# _PAGED_DECODE_MIN_KV_MIB is pinned to 0 on every arm so decode routing is
# unconditional -- otherwise a workload below the threshold half-declines and
# reports the incumbent under the rocKE arm's name.
ARMS = {
    "stock": {"_SWAPQK_MIN_SEQLEN": 10**9, "_PAGED_DECODE": "0",
              "_PAGED_DECODE_MIN_KV_MIB": 0},
    "prefill_only": {"_SWAPQK_MIN_SEQLEN": 512, "_PAGED_DECODE": "0",
                     "_PAGED_DECODE_MIN_KV_MIB": 0},
    "decode_only": {"_SWAPQK_MIN_SEQLEN": 10**9, "_PAGED_DECODE": "1",
                    "_PAGED_DECODE_MIN_KV_MIB": 0},
    "both": {"_SWAPQK_MIN_SEQLEN": 512, "_PAGED_DECODE": "1",
             "_PAGED_DECODE_MIN_KV_MIB": 0},
}

# The same arms expressed as the environment variables the backend reads at
# import. Only --capture-check uses these: under a FULL cudagraph mode the arm
# has to be in place before LLM() builds and captures.
ENV_ARMS = {
    arm: {
        "ROCKE_MIN_SEQLEN": str(vals["_SWAPQK_MIN_SEQLEN"]),
        "ROCKE_PAGED_DECODE": vals["_PAGED_DECODE"],
        "ROCKE_PAGED_DECODE_MIN_KV_MIB": str(vals["_PAGED_DECODE_MIN_KV_MIB"]),
    }
    for arm, vals in ARMS.items()
}

SWAPQK_ARMS = {"prefill_only", "both"}
DECODE_ARMS = {"decode_only", "both"}

BASELINE_ARM = "stock"
OPTIMIZED_ARM = "both"

FULL_MODES = ("FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")


def key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def build_prompt_ids(model: str, lengths) -> dict:
    """Token-exact prompts. Text prompts give arbitrary lengths, and swapqk
    eligibility keys off seqlen % 64, so the length must be exact."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    need = max(lengths)
    reps = need // 10 + 64
    base = tok.encode("The quick brown fox jumps over the lazy dog. " * reps,
                      add_special_tokens=False)
    if len(base) < need:
        raise RuntimeError(f"filler gave {len(base)} tokens, need {need}")
    return {s: list(base[:s]) for s in sorted(set(lengths))}


def set_arm(mod, arm: str) -> None:
    for attr, val in ARMS[arm].items():
        setattr(mod, attr, val)


def snapshot(mod) -> tuple:
    stats = dict(mod.ROCKE_STATS)
    why = dict(getattr(mod, "ROCKE_DECODE_DECLINE", {}))
    return stats, why


def reset_counters(mod) -> None:
    for k in mod.ROCKE_STATS:
        mod.ROCKE_STATS[k] = 0
    getattr(mod, "ROCKE_DECODE_DECLINE", {}).clear()


def check_dispatch(arm, stats, why, n_layers, batch, gen):
    """Return "" if the arm actually ran the kernels it claims to, else why not.

    swapqk and triton_slice count per REQUEST per layer (the prefill dispatch
    loops over the flat token batch), so a batch of B prefills gives
    n_layers * B. paged_decode counts per layer per decode STEP over the whole
    batch, and the number of steps depends on admission, so it is only asserted
    to be non-zero.
    """
    want_prefill = n_layers * batch
    if arm in SWAPQK_ARMS:
        if stats["swapqk"] != want_prefill or stats["triton_slice"] != 0:
            return (f"expected swapqk={want_prefill} triton_slice=0, got "
                    f"swapqk={stats['swapqk']} triton_slice={stats['triton_slice']}")
        # The paged-V gates fail into the permute path rather than into Triton,
        # so the swapqk counter above cannot see them.
        if stats["paged_v_declined"]:
            return f"paged_v_declined={stats['paged_v_declined']} (expected 0)"
    else:
        if stats["triton_slice"] != want_prefill or stats["swapqk"] != 0:
            return (f"expected triton_slice={want_prefill} swapqk=0, got "
                    f"swapqk={stats['swapqk']} triton_slice={stats['triton_slice']}")

    if arm in DECODE_ARMS:
        if stats["paged_decode_declined"]:
            return (f"paged_decode_declined={stats['paged_decode_declined']} "
                    f"reasons={why}")
        if why:
            return f"decode declines recorded: {why}"
        if gen > 1:
            if stats["paged_decode"] <= 0:
                return "paged_decode=0 with gen>1 -- the decode kernel never ran"
            # Anything that reaches super().forward() during decode is the
            # incumbent, counted under this arm's name. A mixed prefill+decode
            # batch lands here too.
            if stats["fallback"]:
                return (f"fallback={stats['fallback']} on a rocKE-decode arm -- "
                        "some steps ran the incumbent")
    else:
        if stats["paged_decode"]:
            return f"paged_decode={stats['paged_decode']} on a non-decode arm"
    return ""


def run_once(llm, mod, prompt_cls, sp_cls, ids, batch, gen, arm, patch_arm):
    """One timed generate. Returns (seconds, stats, declines, token_ids)."""
    if patch_arm:
        set_arm(mod, arm)
    reset_counters(mod)

    one = prompt_cls(prompt_token_ids=ids) if prompt_cls else {"prompt_token_ids": ids}
    prompts = [one] * batch
    sp = sp_cls(temperature=0.0, max_tokens=gen, ignore_eos=True)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    stats, why = snapshot(mod)
    return dt, stats, why, list(outs[0].outputs[0].token_ids)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Combined prefill+decode e2e sweep for the gfx1151 rocKE "
                    "attention kernels (README sections 4.1, 6 and 7).")
    ap.add_argument("--model", default="/home/user/models/Qwen3-8B")
    # The ctx* sweep is opt-in: it overlaps the named workloads and would
    # otherwise triple the default runtime for rows nobody asked for.
    ap.add_argument("--workloads", default=",".join(NAMED_WORKLOADS))
    ap.add_argument("--ctx-sweep", action="store_true",
                    help="replace --workloads with the B=1 context sweep that "
                         "produces the README's ladders")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen-lo", type=int, default=48,
                    help="slope anchor; must clear the request-admission ramp")
    ap.add_argument("--gen-hi", type=int, default=304,
                    help="headline generation length; total_ms is measured here")
    ap.add_argument("--cudagraph-mode", default="PIECEWISE")
    ap.add_argument("--eager", action="store_true", help="enforce_eager")
    ap.add_argument("--capture-check", action="store_true",
                    help="single-arm FULL-mode confirmation: apply the arm via "
                         "the environment before LLM(), assert on capture-time "
                         "counters, never patch between generates")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--out", default=None,
                    help="optional path to dump the raw measurements as JSON")
    args = ap.parse_args()

    if args.ctx_sweep:
        args.workloads = ",".join(CTX_SWEEP)
    wls = [w.strip() for w in args.workloads.split(",") if w.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for w in wls:
        if w not in WORKLOADS:
            print(f"unknown workload {w!r}, pick from {sorted(WORKLOADS)}",
                  file=sys.stderr)
            return 2
    for a in arms:
        if a not in ARMS:
            print(f"unknown arm {a!r}, pick from {sorted(ARMS)}", file=sys.stderr)
            return 2
    if args.gen_hi <= args.gen_lo:
        print("--gen-hi must exceed --gen-lo", file=sys.stderr)
        return 2

    captures_decode = not args.eager and args.cudagraph_mode in FULL_MODES
    if captures_decode and not args.capture_check:
        print(f"cudagraph_mode={args.cudagraph_mode} captures decode: the arm "
              "and the dispatch counters both freeze at capture. Use PIECEWISE "
              "for the interleaved sweep, or --capture-check for a single-arm "
              "confirmation run.", file=sys.stderr)
        return 2
    if args.capture_check and len(arms) != 1:
        print("--capture-check takes exactly one --arms value", file=sys.stderr)
        return 2

    if args.capture_check:
        # Must land before the backend module is imported by LLM().
        for k, v in ENV_ARMS[arms[0]].items():
            os.environ[k] = v

    gens = [1, args.gen_lo, args.gen_hi]
    span = args.gen_hi - args.gen_lo
    lens = sorted({WORKLOADS[w][1] for w in wls})
    max_batch = max(WORKLOADS[w][0] for w in wls)
    max_len = max(lens) + args.gen_hi + 8
    # One prefill step for the whole batch. If the token budget forces vLLM to
    # admit a large batch over several steps, later steps mix prefill with
    # decode, max_query_len > 1, and the decode gate silently declines -- which
    # the fallback assertion would then fail the run over.
    max_batched = max(max_len, max(WORKLOADS[w][0] * WORKLOADS[w][1] for w in wls))

    from transformers import AutoConfig

    n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    mode = "eager" if args.eager else args.cudagraph_mode
    print(f"[cmb] mode={mode} workloads={wls} arms={arms} reps={args.reps} "
          f"gens={gens} layers={n_layers} capture_check={args.capture_check}",
          flush=True)

    prompts = build_prompt_ids(args.model, lens)
    print(f"[cmb] prompts built: {[(s, len(v)) for s, v in prompts.items()]}",
          flush=True)

    from vllm import LLM, SamplingParams

    try:
        from vllm.inputs import TokensPrompt as prompt_cls
    except Exception:
        prompt_cls = None

    kwargs = {} if args.eager else {
        "compilation_config": {"cudagraph_mode": args.cudagraph_mode}
    }
    t_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",               # the dispatch gate is query.dtype == fp16
        max_model_len=max_len,
        max_num_batched_tokens=max_batched,
        max_num_seqs=max_batch,
        enable_chunked_prefill=False,  # chunking makes query_len != seq_len
        enable_prefix_caching=False,   # a cache hit does the same
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        attention_backend="ROCKE_GFX1151",
        **kwargs,
    )
    print(f"[cmb] model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    import vllm.v1.attention.backends.rocm_rocke_attn as rk

    # The only observable moment under a FULL mode: capture ran the backend's
    # Python, steady-state replay will not.
    captured = dict(rk.ROCKE_STATS)
    print(f"[cmb] counters after construction: {captured}", flush=True)

    resolved = {}
    try:
        vc = llm.llm_engine.vllm_config
        resolved = {
            "chunked_prefill": bool(vc.scheduler_config.enable_chunked_prefill),
            "prefix_caching": bool(vc.cache_config.enable_prefix_caching),
            "dtype": str(vc.model_config.dtype),
            "cudagraph_mode": str(vc.compilation_config.cudagraph_mode),
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "max_model_len": vc.model_config.max_model_len,
        }
        print(f"[cmb] resolved: {resolved}", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[cmb] could not echo resolved config: {exc}", flush=True)

    failures: list[str] = []
    if args.capture_check:
        arm = arms[0]
        want_pd = arm in DECODE_ARMS
        got_pd = captured.get("paged_decode", 0)
        if want_pd and got_pd <= 0:
            failures.append(f"capture-check {arm}: paged_decode=0 at capture -- "
                            "the decode kernel was not the one captured")
        if not want_pd and got_pd:
            failures.append(f"capture-check {arm}: paged_decode={got_pd} at "
                            "capture on a non-decode arm")

    # rep 0 is a warmup: it absorbs per-arm kernel compile and first-touch of
    # every allocation. Discarded.
    samples: dict[str, list] = {}
    tokens: dict[str, list] = {}
    stats_seen: dict[str, dict] = {}

    for rep in range(args.reps + 1):
        for w in wls:
            batch, plen = WORKLOADS[w]
            for gen in gens:
                # Arms innermost, so drift across the sweep hits every arm
                # equally rather than accumulating in whichever ran last.
                for arm in arms:
                    try:
                        dt, stats, why, out_ids = run_once(
                            llm, rk, prompt_cls, SamplingParams, prompts[plen],
                            batch, gen, arm, patch_arm=not args.capture_check,
                        )
                    except Exception as exc:
                        print(f"[cmb] rep{rep} {w} g={gen} {arm}: EXCEPTION {exc}",
                              flush=True)
                        failures.append(f"{w} g={gen} {arm}: {exc}")
                        continue

                    tag = "warmup" if rep == 0 else f"rep{rep}"
                    bad = "" if args.capture_check else check_dispatch(
                        arm, stats, why, n_layers, batch, gen)
                    if bad:
                        print(f"[cmb] {tag} {w} g={gen} {arm}: DISPATCH -- {bad}",
                              flush=True)
                        failures.append(f"{w} g={gen} {arm}: {bad}")
                        continue

                    print(f"[cmb] {tag} {w:9s} g={gen:4d} {arm:12s} "
                          f"{dt * 1e3:10.1f} ms  {stats}", flush=True)
                    if rep > 0:
                        k = key(w, arm, gen)
                        samples.setdefault(k, []).append(dt)
                        tokens.setdefault(k, out_ids)
                        stats_seen[k] = stats

    # Shared box: contention can only inflate, so take the minimum.
    table = {
        k: {"min_ms": min(v) * 1e3,
            "median_ms": statistics.median(v) * 1e3,
            "n": len(v)}
        for k, v in samples.items()
    }

    derived = {}
    for w in wls:
        batch, _plen = WORKLOADS[w]
        for arm in arms:
            t1 = table.get(key(w, arm, 1))
            lo = table.get(key(w, arm, args.gen_lo))
            hi = table.get(key(w, arm, args.gen_hi))
            if not (t1 and lo and hi):
                derived[key(w, arm)] = None
                continue
            itl = (hi["min_ms"] - lo["min_ms"]) / span
            derived[key(w, arm)] = {
                "ttft_ms": t1["min_ms"],
                "anchor_ms": lo["min_ms"],
                "total_ms": hi["min_ms"],
                "itl_ms": itl,
                # Two different rates. tok_per_s is whole-request throughput and
                # so is dragged down by prefill; decode_tok_per_s is the steady
                # -state generation rate a user watching tokens appear sees, and
                # is the one comparable to the isolated decode kernel's ceiling.
                "tok_per_s": batch * args.gen_hi / (hi["min_ms"] * 1e-3),
                "decode_tok_per_s": (batch * 1e3 / itl) if itl > 0 else None,
            }

    # Greedy decode: the same prompt must give the same tokens under every arm.
    identity = {}
    for w in wls:
        for gen in gens:
            ref_arm = next((a for a in arms if key(w, a, gen) in tokens), None)
            if ref_arm is None:
                continue
            ref = tokens[key(w, ref_arm, gen)]
            bad = [a for a in arms
                   if key(w, a, gen) in tokens and tokens[key(w, a, gen)] != ref]
            identity[key(w, gen)] = {"ref_arm": ref_arm, "mismatched": bad,
                                     "head": ref[:8]}
            for a in bad:
                failures.append(
                    f"{w} g={gen}: arm {a} tokens differ from {ref_arm} "
                    f"({tokens[key(w, a, gen)][:8]} vs {ref[:8]})")

    print(f"\n[cmb] === {mode}, min of {args.reps} reps, ms ===", flush=True)
    print(f"{'workload':>10} {'arm':>13} | {'TTFT':>10} {'total':>10} "
          f"{'ITL':>8} {'req tok/s':>10} {'dec tok/s':>10}", flush=True)
    for w in wls:
        for arm in arms:
            d = derived.get(key(w, arm))
            if d is None:
                print(f"{w:>10} {arm:>13} | {'--':>10} {'--':>10} {'--':>8} "
                      f"{'--':>10} {'--':>10}", flush=True)
                continue
            dec = d["decode_tok_per_s"]
            dec_s = f"{dec:.2f}" if dec else "--"
            print(f"{w:>10} {arm:>13} | {d['ttft_ms']:10.1f} {d['total_ms']:10.1f} "
                  f"{d['itl_ms']:8.3f} {d['tok_per_s']:10.1f} {dec_s:>10}",
                  flush=True)

    # Speedups against the matched control, in the fixed direction
    # baseline/optimized, so higher is always better.
    if BASELINE_ARM in arms:
        print(f"\n[cmb] === speedup vs {BASELINE_ARM} (higher is better) ===",
              flush=True)
        print(f"{'workload':>10} {'arm':>13} | {'TTFT':>9} {'total':>9} "
              f"{'ITL':>9}", flush=True)
        for w in wls:
            base = derived.get(key(w, BASELINE_ARM))
            if base is None:
                continue
            for arm in arms:
                if arm == BASELINE_ARM:
                    continue
                d = derived.get(key(w, arm))
                if d is None:
                    continue
                print(f"{w:>10} {arm:>13} | "
                      f"{base['ttft_ms'] / d['ttft_ms']:9.4f} "
                      f"{base['total_ms'] / d['total_ms']:9.4f} "
                      f"{base['itl_ms'] / d['itl_ms']:9.4f}", flush=True)

    if args.out:
        doc = {
            "mode": mode,
            "cudagraph_mode": None if args.eager else args.cudagraph_mode,
            "capture_check": args.capture_check,
            "capture_counters": captured,
            "resolved_config": resolved,
            "model": args.model,
            "layers": n_layers,
            "workloads": {w: {"batch": WORKLOADS[w][0],
                              "prompt_len": WORKLOADS[w][1]} for w in wls},
            "arms": arms,
            "arm_settings": {a: {k: str(v) for k, v in ARMS[a].items()}
                             for a in arms},
            "reps": args.reps,
            "gens": gens,
            "gen_lo": args.gen_lo,
            "gen_hi": args.gen_hi,
            "itl_span_tokens": span,
            "baseline_arm": BASELINE_ARM,
            "optimized_arm": OPTIMIZED_ARM,
            "table": table,
            "derived": derived,
            "stats": stats_seen,
            "token_identity": identity,
            "failures": failures,
        }
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\n[cmb] wrote {args.out}", flush=True)

    if failures:
        print(f"[cmb] {len(failures)} FAILURES -- results above are not "
              "trustworthy", flush=True)
        for f in failures[:20]:
            print(f"  {f}", flush=True)
        return 1
    print("[cmb] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
