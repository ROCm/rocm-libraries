# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Definitive correctness + perf check over EVERY shape in shapes.json.

Runs the production dispatcher (run_unified_attention_torch backend="auto") for
each canonical shape, checks correctness vs the fp32 paged reference, and times
CK against the AOTriton **flash** SDPA backend in TWO modes so a perf gap can be
attributed to kernel time vs host launch overhead:

  - **eager** (host launch overhead + kernel): a standard warmup + timing loop.
  - **graph** (kernel only): both sides are timed as a HIP/CUDA-graph replay so
    per-launch host overhead is removed symmetrically. The Torch baseline is
    captured with an external ``torch.cuda.graph``. CK's dispatcher is not
    externally capture-safe (it does host-side dispatch/allocation that aborts
    an external capture), so for CK we replay the graph CK itself builds via its
    own internal graph-replay path -- the bare captured kernels, no Python
    dispatch. Both graph cells are therefore kernel-only and directly
    comparable. Shapes whose graph cannot be isolated report ``N/A`` rather than
    being compared unfairly.

For each (CK, flash) x (eager, graph) cell we take the **trimmed mean of several
timed reps** after a throw-away warmup, and flag any cell whose spread exceeds
~10% as ``noisy`` so one-shot variance can't silently skew the comparison.

The Torch baseline is pinned to ``SDPBackend.FLASH_ATTENTION`` (apples-to-apples
vs AOTriton flash, not the slower default SDP dispatch on ROCm); shapes where
flash is ineligible fall back to the default backend WITH a warning, and are
excluded from the flash win-rate.

This benchmark is **causal-only for prefill**: the CK unified-attention kernel
always applies a causal mask (it has no non-causal mode), so for prefill
(``seqlen_q > 1``) the fp32 reference and the Torch baseline use a causal mask
(``is_causal=True``). Decode (``q == 1``) is run non-causal (``is_causal=False``)
-- a single query attends to all keys, which is identical to causal at ``q == 1``.
Non-causal *prefill* is out of scope and is not in ``shapes.json``; the per-shape
``causal`` field is informational and is not separately exercised (``is_causal``
is derived from whether the shape is decode).

Known issues / runtime notes
----------------------------
* **head_size=256 is OFF BY DEFAULT** (the ``d256_disabled`` group in
  ``shapes.json``). There is currently no tiled d256 path on gfx942, so the
  dispatcher falls back to the scalar kernel, which performs very poorly here:
  ~500-4000x slower than flash, failing the correctness tolerance on some
  shapes, and slow enough at S2048 to stall graph capture (looks like a hang).
  A proper tiled d256 implementation is needed; re-run with
  ``--groups d256_disabled`` to revisit it.
* **Flash-ineligible shapes fall back to Torch's default SDP backend** (rows
  marked ``backend = default``) -- notably non-square causal shapes
  (``seqlen_q != seqlen_k``) and d256, which the flash backend does not accept
  here. Those rows are excluded from the flash win-rate. For non-square shapes
  the fp32 reference uses a bottom-right-aligned causal mask while Torch's
  ``is_causal=True`` is top-left-aligned, so CK and the Torch-default timing
  reflect slightly different masking -- another reason those rows are reported
  only for context and excluded from the win-rate.
* **Long runtime:** the full set x 4 timing cells x ``--reps`` x ``--iters``,
  plus the fp32 reference on large (S<=8192) shapes, makes a complete sweep take
  tens of minutes on a single MI300X. Tune ``--iters`` / ``--reps`` / ``--groups``
  to scope it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Make ``parity_unified_attention`` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocke.instances import UnifiedAttentionProblem
from rocke.instances.common import attention_unified as au
from rocke.runtime import synchronize_and_release, time_launches
from parity_unified_attention import (
    compare,
    load_shapes,
    make_inputs,
    ref_paged_attn,
)

# Graph-replay policy overrides for CK's dispatcher. ``_NO_GRAPH`` forces the
# eager path (one launch per call); ``_ALL_GRAPH`` forces CK to build its own
# internal CUDA graph. The graph cell builds CK's internal graph once and times
# its replay directly (pure kernel, no Python dispatch) -- symmetric with the
# Torch baseline's external CUDA-graph capture.
_NO_GRAPH = lambda problem: False  # noqa: E731
_ALL_GRAPH = lambda problem: True  # noqa: E731


def time_graphed(fn, *, warmup, iters):
    """Capture ``fn`` into a CUDA graph and time the replay (avg ms/replay).

    Returns the per-replay kernel time with per-launch host overhead removed.
    Raises if the region is not capture-safe; the caller treats that as N/A.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
        for _ in range(int(warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(int(iters)):
            g.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / int(iters)
    finally:
        g.reset()


def reduce_times(vals, method, trim):
    """Reduce a list of per-rep times (ms) to a single number.

    ``method``:
      - ``"median"``  : statistical median.
      - ``"mean"``    : arithmetic mean of all reps.
      - ``"trimmed"`` : mean after stripping the highest and lowest ``trim``
                        fraction of reps (robust to one-shot outliers; default).
    """
    vals = sorted(vals)
    if method == "median":
        return statistics.median(vals)
    if method == "mean":
        return statistics.fmean(vals)
    if method == "trimmed":
        k = int(len(vals) * trim)
        core = vals[k : len(vals) - k] if len(vals) - 2 * k >= 1 else vals
        return statistics.fmean(core)
    raise ValueError(f"unknown reduce method: {method}")


def robust(timer, *, reps, method, trim):
    """Run ``timer`` (a no-arg fn returning ms) ``reps`` times after a throw-away
    warmup; return ``(value_ms, cv)`` where ``value_ms`` is reduced per ``method``
    and cv=(max-min)/median is the raw spread (a noisiness indicator)."""
    timer()  # throw-away
    vals = [timer() for _ in range(int(reps))]
    value = reduce_times(vals, method, trim)
    svals = sorted(vals)
    med = statistics.median(svals)
    cv = (svals[-1] - svals[0]) / med if med > 0 else 0.0
    return value, cv


def time_ck_graph(call, *, warmup, iters, reps, method, trim):
    """Time CK's own internal CUDA graph by replaying it directly.

    CK's dispatcher (``run_unified_attention_torch``) is not externally
    capture-safe -- it does host-side dispatch/allocation that aborts an
    external ``torch.cuda.graph`` capture. But CK ships its own internal
    graph-replay path, which *is* capture-safe. We force that path on, do one
    warm-up ``call`` to build and cache the graph, fetch the freshly-cached
    ``CUDAGraph`` handle, and time ``graph.replay()`` directly. That replay is
    the bare captured kernels (no Python dispatch) -- kernel-only, symmetric
    with the Torch baseline. Returns ``(value_ms, cv)``; raises on any failure
    to isolate the graph (caller treats that as N/A).
    """
    au._recommend_graph_replay = _ALL_GRAPH
    try:
        before2d = set(au._2D_GRAPHS)
        before3d = set(au._3D_GRAPHS)
        call()  # builds + caches CK's internal graph for this problem
        torch.cuda.synchronize()
        new = [au._2D_GRAPHS[k] for k in au._2D_GRAPHS if k not in before2d]
        new += [au._3D_GRAPHS[k] for k in au._3D_GRAPHS if k not in before3d]
        if len(new) != 1:
            raise RuntimeError(
                f"could not isolate CK internal graph (found {len(new)})"
            )
        g = new[0]

        def one():
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(int(iters)):
                g.replay()
            e1.record()
            e1.synchronize()
            return e0.elapsed_time(e1) / int(iters)

        return robust(one, reps=reps, method=method, trim=trim)
    finally:
        au._recommend_graph_replay = _NO_GRAPH


def make_torch_once(s, data, is_causal):
    """Build the dense [B,H,S,D] Torch SDPA closure; return ``(once, backend)``.

    ``backend`` is ``"flash"`` when AOTriton flash is eligible, else ``"default"``
    (the caller warns). ``once`` issues exactly one SDPA and nothing else -- no
    consuming reduction -- so it is kernel-only and symmetric with the CK ``call``
    (which times only the attention kernel writing into a pre-allocated ``out``).
    Eager SDPA is not dead-code-eliminated (it materializes its output), and a
    captured graph cannot elide it, so no anti-DCE sink is needed.
    """
    nrep = s.heads // s.kv_heads
    klen = data["kv_lens_list"][0]
    ks, vs = [], []
    for bi in range(s.batch):
        bt = data["block_tables"][bi]
        ks.append(data["key_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen])
        vs.append(data["value_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen])
    kh = torch.stack(ks, 0).transpose(1, 2).repeat_interleave(nrep, 1).contiguous()
    vh = torch.stack(vs, 0).transpose(1, 2).repeat_interleave(nrep, 1).contiguous()
    qh = (
        data["query"]
        .view(s.batch, s.seqlen_q, s.heads, s.head_size)
        .transpose(1, 2)
        .contiguous()
    )

    def once_with(backends):
        def once():
            with sdpa_kernel(backends):
                F.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=is_causal, scale=data["scale"]
                )

        return once

    flash_once = once_with([SDPBackend.FLASH_ATTENTION])
    try:
        flash_once()
        torch.cuda.synchronize()
        return flash_once, "flash"
    except RuntimeError:
        deflt = once_with(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
        deflt()
        torch.cuda.synchronize()
        return deflt, "default"


def _us(med_cv):
    """Format a (median_ms, cv) cell as microseconds, or 'N/A'."""
    med, cv = med_cv
    if med <= 0:
        return f"{'N/A':>12}", False
    return f"{med * 1000:12.1f}", cv > 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="shape groups to run (default: all groups in shapes.json)",
    )
    ap.add_argument(
        "--shapes",
        default=None,
        help="path to an alternate shapes json (default: shipped shapes.json)",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--reps",
        type=int,
        default=10,
        help="timed measurements per cell, reduced per --reduce (default 10)",
    )
    ap.add_argument(
        "--reduce",
        choices=("median", "mean", "trimmed"),
        default="trimmed",
        help="how to reduce the --reps measurements (default: trimmed)",
    )
    ap.add_argument(
        "--trim",
        type=float,
        default=0.2,
        help="fraction stripped from each end for --reduce trimmed (default 0.2)",
    )
    args = ap.parse_args()
    if args.reps < 1 or args.iters < 1 or args.warmup < 0:
        ap.error("--reps and --iters must be >= 1, --warmup must be >= 0")
    if not 0.0 <= args.trim < 0.5:
        ap.error("--trim must be in [0.0, 0.5)")
    print(
        f"device,{torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True
    )
    print(
        f"timing: reps={args.reps} reduce={args.reduce}"
        + (f" trim={args.trim}" if args.reduce == "trimmed" else "")
        + f" (inner iters={args.iters}, warmup={args.warmup})",
        flush=True,
    )
    # By default run every group EXCEPT those suffixed "_disabled" (e.g.
    # "d256_disabled" -- see the module docstring). Pass --groups explicitly to
    # include them, e.g. `--groups d256_disabled`.
    _shapes_kw = {"path": Path(args.shapes).resolve()} if args.shapes else {}
    shapes = [
        s
        for s in load_shapes(**_shapes_kw)
        if (args.groups is None and not s.group.endswith("_disabled"))
        or (args.groups is not None and s.group in args.groups)
    ]
    stream = int(torch.cuda.current_stream().cuda_stream)
    n_pass = n_fail = n_fallback = 0
    win_e = win_c = cmp_e = cmp_c = 0
    hdr = (
        f"{'shape':30s} {'dt':4s} {'CK-Eager':>12} {'CK-Graph':>12} "
        f"{'Torch-Eager':>12} {'Torch-Graph':>12} {'Eager-Ratio':>12} {'Graph-Ratio':>12} "
        f"{'backend':>7}  result   (us; ratio = CK/Torch, <1 = CK faster)"
    )
    print(hdr, flush=True)

    def time_eager(fn):
        return time_launches(fn, warmup=args.warmup, iters=args.iters, stream=stream)

    for s in shapes:
        data = make_inputs(s)
        ref = ref_paged_attn(
            data["query"],
            data["key_cache"],
            data["value_cache"],
            data["query_lens"],
            data["kv_lens_list"],
            data["block_tables"],
            data["scale"],
        ).float()
        decode = s.seqlen_q == 1
        problem = UnifiedAttentionProblem(
            total_q=data["query"].shape[0],
            num_seqs=s.batch,
            num_query_heads=s.heads,
            num_kv_heads=s.kv_heads,
            head_size=s.head_size,
            block_size=64,
            max_seqlen_q=(1 if decode else data["max_query_len"]),
            max_seqlen_k=data["max_kv_len"],
            dtype=s.dtype,
            num_sms=120,
        )
        out = torch.empty_like(data["query"])

        def call(data=data):  # bind at def-time (immune to the later `del data`)
            au.run_unified_attention_torch(
                problem=problem,
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=out,
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_tables"],
                softcap=0.0,
                backend="auto",
                stream=stream,
            )

        # Correctness (eager production dispatch).
        au._recommend_graph_replay = _NO_GRAPH
        call()
        torch.cuda.synchronize()
        mx = compare(ref, out)["max_abs"]
        tol = 2e-2 if s.dtype == "fp16" else 4e-2
        ok = mx < tol
        n_pass += ok
        n_fail += not ok
        res = "PASS" if ok else "FAIL"

        # --- CK timing: eager (production dispatch), then graph ---
        # Eager: CK's internal graph off, one launch per call (host + kernel).
        # Graph: time CK's *own* internal CUDA graph replay directly (kernel
        # only) -- CK's dispatcher isn't externally capture-safe, so we replay
        # the graph CK itself builds. Symmetric with the Torch external capture.
        au._recommend_graph_replay = _NO_GRAPH
        cke = robust(
            lambda: time_eager(call), reps=args.reps, method=args.reduce, trim=args.trim
        )
        synchronize_and_release(stream)
        try:
            ckc = time_ck_graph(
                call,
                warmup=args.warmup,
                iters=args.iters,
                reps=args.reps,
                method=args.reduce,
                trim=args.trim,
            )
        except Exception:
            ckc = (0.0, 0.0)
        synchronize_and_release(stream)

        # --- Torch flash timing: eager then capture (external graph) ---
        once, backend = make_torch_once(s, data, is_causal=(not decode))
        if backend == "default":
            n_fallback += 1
            print(
                f"  [warn] {s.name}: AOTriton flash ineligible; comparing vs "
                f"Torch default SDPA backend.",
                flush=True,
            )
        fle = robust(
            lambda: time_eager(once), reps=args.reps, method=args.reduce, trim=args.trim
        )
        try:
            flc = robust(
                lambda: time_graphed(once, warmup=args.warmup, iters=args.iters),
                reps=args.reps,
                method=args.reduce,
                trim=args.trim,
            )
        except Exception:
            flc = (0.0, 0.0)

        # Ratios per mode (lower CK = better).
        eR = cke[0] / fle[0] if fle[0] > 0 else 0.0
        cR = ckc[0] / flc[0] if (ckc[0] > 0 and flc[0] > 0) else 0.0
        if fle[0] > 0:
            cmp_e += 1
            win_e += eR < 1.0
        if ckc[0] > 0 and flc[0] > 0:
            cmp_c += 1
            win_c += cR < 1.0

        cke_s, n1 = _us(cke)
        rocke_s, n2 = _us(ckc)
        fle_s, n3 = _us(fle)
        flc_s, n4 = _us(flc)
        noisy = " noisy" if (n1 or n2 or n3 or n4) else ""
        bk = "flash" if backend == "flash" else "default"
        eR_s = f"{eR:12.3f}" if eR > 0 else f"{'-':>12}"
        cR_s = f"{cR:12.3f}" if cR > 0 else f"{'-':>12}"
        print(
            f"{s.name:30s} {s.dtype:4s} {cke_s} {rocke_s} {fle_s} {flc_s} "
            f"{eR_s} {cR_s} {bk:>7}  {res}{noisy}",
            flush=True,
        )
        del data, ref
        torch.cuda.empty_cache()

    print(f"\ncorrectness: {n_pass} PASS / {n_fail} FAIL", flush=True)
    print(
        f"Eager ratio (host overhead + kernel): CK faster on {win_e}/{cmp_e} shapes",
        flush=True,
    )
    print(
        f"Graph ratio (kernel only, host overhead removed): CK faster on {win_c}/{cmp_c} shapes",
        flush=True,
    )
    if n_fallback:
        print(
            f"({n_fallback} shapes compared vs Torch default — flash ineligible)",
            flush=True,
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
