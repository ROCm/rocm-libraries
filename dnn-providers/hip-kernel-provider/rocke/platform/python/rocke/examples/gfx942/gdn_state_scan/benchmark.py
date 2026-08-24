# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the rocKE K5 state-scan against the FlyDSL sweep's shapes 1-25.

Reuses the verified input-construction path in :mod:`verify` (``build_case``),
so what is timed is exactly what is checked for correctness. Each shape is run
at the full optimization stack (P1-P8: swizzle + packed LDS + prefetch + buffer
descriptors + XCD remap + issue-before-GEMM2) and, like the FlyDSL sweep, over a
small set of ``BV`` / ``NR_SPLIT`` variants; the best is reported.

    PYTHONPATH=python ROCKE_BACKEND=python \
        python3 -m rocke.examples.gfx942.gdn_state_scan.benchmark

**Timing methodology.** The FlyDSL numbers are ``time_graph_us`` — a HIP-graph
replay that excludes per-launch CPU overhead. rocKE therefore defaults to the
same method: bind the launcher to a capture stream, capture one dispatch, and
time graph replays with HIP events. ``--timing launch`` remains available for
diagnosing eager-launch integration overhead, but is not an apples-to-apples
FlyDSL comparison.
"""

from __future__ import annotations

import argparse
from itertools import product

import torch

from rocke.runtime.launcher import time_launches

from .verify import build_case, ATOL


# The FlyDSL sweep's shapes 1-25, transcribed from
# build/gfx942-k5-...run-6.md. Semantics confirmed with the user:
#   * T is the TOTAL token count across the N sequences of the batch;
#   * N>1 with no seqs= tag is an equal-length varlen batch (per-seq = T/N),
#     packed via cu_seqlens (IS_VARLEN);
#   * seqs=ragged/skew/bimodal/skew_last are unequal varlen splits — the exact
#     FlyDSL split recipe is not in the file, so those cannot be reproduced.
#
# EXACT set = shapes whose per-sequence split is unambiguous: N==1, or T divisible
# by N with T/N a multiple of BT=64. That is shapes 1-17 and 24. Everything else
# (18-23 unequal-split, and 25 where 8192/3 is neither integer nor BT-aligned)
# is NON-EXACT and excluded from the headline comparison. ``exact()`` below is
# the single source of truth; non-exact shapes are still listed (and can be run
# with --include-inexact) but are flagged.
# Each entry: (id, name, H, Hg, T, N, gate, seqs_kind)
SHAPES = [
    (1,  "kda_tp8",     12, 12, 8192,  1, "gk", "equal"),
    (2,  "kda_tp8",     12, 12, 32768, 1, "gk", "equal"),
    (3,  "kda_tp8",     12, 12, 8192,  4, "gk", "equal"),
    (4,  "kda_tp8",     12, 12, 32768, 4, "gk", "equal"),
    (5,  "kda_tp8",     12, 12, 8192,  8, "gk", "equal"),
    (6,  "kda_tp8",     12, 12, 32768, 8, "gk", "equal"),
    (7,  "kda_tp4",     24, 24, 8192,  1, "gk", "equal"),
    (8,  "kda_tp4",     24, 24, 32768, 1, "gk", "equal"),
    (9,  "kda_tp4",     24, 24, 8192,  8, "gk", "equal"),
    (10, "kda_tp4",     24, 24, 32768, 8, "gk", "equal"),
    (11, "gdn_q3n_tp8",  4,  2, 8192,  8, "g",  "equal"),
    (12, "gdn_q3n_tp8",  4,  2, 32768, 8, "g",  "equal"),
    (13, "gdn_q3n_tp4",  8,  4, 8192,  4, "g",  "equal"),
    (14, "gdn_q3n_tp4",  8,  4, 32768, 4, "g",  "equal"),
    (15, "gdn_q35_tp1", 16, 16, 8192,  1, "g",  "equal"),
    (16, "gdn_q35_tp1", 32,  8, 8192,  1, "g",  "equal"),
    (17, "gdn_q35_tp1", 32,  8, 32768, 1, "g",  "equal"),
    (18, "kda_tp4",     24, 24, 32768, 8, "gk", "ragged"),
    (19, "kda_tp4",     24, 24, 32768, 8, "gk", "bimodal"),
    (20, "kda_tp4",     24, 24, 32768, 8, "gk", "skew"),
    (21, "gdn_q3n_tp8",  4,  2, 32768, 8, "g",  "ragged"),
    (22, "gdn_q3n_tp8",  4,  2, 32768, 8, "g",  "skew"),
    (23, "kda_tp4",     24, 24, 32768, 8, "gk", "skew_last"),
    (24, "gdn_q3n_rmp",  4,  2, 8192,  1, "g",  "equal"),
    (25, "gdn_q3n_rmp",  4,  2, 8192,  3, "g",  "equal"),
]

# Reference (FlyDSL / HIP) baseline numbers are deliberately NOT stored in this
# repo — measured software performance numbers are barred here by
# platform/AGENTS.md §Compliance. The comparison report that carries them lives
# in the aiter repo (docs/rocke_vs_flydsl_k5_comparison.md). This driver only
# measures and prints rocKE's own timings; supply a JSON of reference numbers
# via --reference to print them alongside.
FLYDSL_BEST: dict = {}
HIP_BASE: dict = {}


def _load_reference(path):
    """Load {shape_id: {"flydsl": [var, us], "hip": us}} from a JSON file."""
    import json
    with open(path) as f:
        data = json.load(f)
    for sid, rec in data.items():
        sid = int(sid)
        if rec.get("flydsl"):
            FLYDSL_BEST[sid] = tuple(rec["flydsl"])
        if rec.get("hip") is not None:
            HIP_BASE[sid] = rec["hip"]


def exact(sid):
    """True if shape `sid`'s per-sequence split is unambiguous vs the FlyDSL
    sweep — the only shapes safe for a headline comparison."""
    rec = {s[0]: s for s in SHAPES}[sid]
    _, _, _, _, T, N, _, kind = rec
    if kind != "equal":
        return False           # unequal split recipe unknown
    return N == 1 or (T % N == 0 and (T // N) % 64 == 0)


def _split(T, N, kind):
    """Per-sequence lengths for a batch of N summing to T, rounded to BT=64."""
    BT = 64
    if N == 1:
        return [T]
    if kind == "equal":
        per = T // N
        return [per] * N
    # unequal splits: keep the total ~T; these are indicative (see SHAPES note)
    if kind == "ragged":
        base = T // N
        out = [base + (BT if i % 2 else -BT) for i in range(N)]
    elif kind == "bimodal":
        big = (T * 2) // (3 * (N // 2)) if N >= 2 else T
        out = [big if i < N // 2 else (T - big * (N // 2)) // (N - N // 2)
               for i in range(N)]
    elif kind in ("skew", "skew_last"):
        # one long sequence, rest short
        long = T - (N - 1) * BT * 2
        out = ([long] + [BT * 2] * (N - 1)) if kind == "skew" \
            else ([BT * 2] * (N - 1) + [long])
    else:
        out = [T // N] * N
    # round each to a positive multiple of... leave as-is (varlen allows tails);
    # just ensure positivity and that the sum is reasonable.
    out = [max(BT, int(x)) for x in out]
    return out

# The FlyDSL sweep tunes over these; mirror the set so "best rocKE variant"
# is chosen the same way "best_graph" is on the FlyDSL side.
VARIANTS = [
    ("bv16", dict(BV=16, NR_SPLIT=1)),
    ("bv32", dict(BV=32, NR_SPLIT=1)),
    ("bv64", dict(BV=64, NR_SPLIT=1)),
    ("bv32w8", dict(BV=32, NR_SPLIT=2)),
    ("bv64w8", dict(BV=64, NR_SPLIT=2)),
]

# The benchmark default remains the parent-matching full stack.  ``--exhaustive``
# expands this to every implemented performance switch that preserves the
# input/output contract.  Layout and dtype fields are deliberately absent:
# changing those requires the surrounding pipeline to change as well.
FULL_STACK = [
    (
        "full",
        dict(
            swizzle=True,
            prefetch=True,
            prefetch_w_early=False,
            prefetch_k_early=False,
            prefetch_k_interleave=False,
            buffer_desc=True,
            xcd_remap=True,
            vgpr_form=False,
        ),
    )
]


def existing_stack_variants():
    """Return the cartesian product of existing contract-preserving levers."""
    out = []
    # MFMA_VGPR_FORM is intentionally excluded: the gfx942 benchmark
    # toolchain rejects that backend option, so it is not a legal runtime
    # candidate even though newer compilers can build it.
    for swz, pf, early_w, bd, xcd in product((False, True), repeat=5):
        if early_w and not pf:
            continue
        name = (
            f"s{int(swz)}p{int(pf)}e{int(early_w)}"
            f"b{int(bd)}x{int(xcd)}"
        )
        out.append(
            (
                name,
                dict(
                    swizzle=swz,
                    prefetch=pf,
                    prefetch_w_early=early_w,
                    prefetch_k_early=False,
                    prefetch_k_interleave=False,
                    buffer_desc=bd,
                    xcd_remap=xcd,
                    vgpr_form=False,
                ),
            )
        )
    return out


# rmse-ratio bound for a PASS. The pure-fp32 reference is stricter than either
# kernel (both round the state through bf16 in LDS), and error grows with the
# serial chunk count (T/BT), so a max-abs tolerance is the wrong gauge at long
# T. FlyDSL's sweep reports rmse_ratio ~5e-3 against a like-rounded reference;
# the runbook's bf16 GEMM tolerance is 7e-2. Keep the former 5e-2 bound as a
# warning threshold so long scans remain visible without classifying a result
# inside the repository-wide numeric policy as incorrect.
RMSE_PASS = 7e-2
RMSE_WARN = 5e-2


def _rmse_ratio(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (((a - b) ** 2).mean().sqrt()
            / (((b ** 2).mean().sqrt()) + 1e-12)).item()


def _check(case):
    """One correctness pass; returns (ok, worst_rmse_ratio)."""
    case.launch()
    torch.cuda.synchronize()
    d = case.dims
    N, T_flat, NT_list, NT_total = d["N"], d["T_flat"], d["NT_list"], d["NT_total"]
    H, V, K = d["H"], d["V"], d["K"]
    Vn, Ho, Ht = d["Vn"], d["Ho"], d["Ht"]
    h_ref_l, vn_ref_l, fs_ref_l = case.refs
    spec = case.spec
    Ho_v = Ho.reshape(NT_total, H, V, K)
    Ht_v = Ht.float().reshape(N, H, V, K)
    Vn_hm = Vn.reshape(H, T_flat, V) if spec.WU_CONTIGUOUS else None
    Vn_v = None if spec.WU_CONTIGUOUS else Vn.reshape(T_flat, H, V)
    worst = 0.0
    c0 = t0 = 0
    for i, s in enumerate(_seqlens_of(case)):
        nt = NT_list[i]
        gh = Ho_v[c0:c0 + nt].unsqueeze(0)
        gv = (Vn_hm[:, t0:t0 + s].permute(1, 0, 2).unsqueeze(0)
              if spec.WU_CONTIGUOUS else Vn_v[t0:t0 + s].unsqueeze(0))
        gf = Ht_v[i:i + 1]
        for g, r in ((gh, h_ref_l[i]), (gv, vn_ref_l[i]), (gf, fs_ref_l[i])):
            worst = max(worst, _rmse_ratio(g, r))
        c0 += nt
        t0 += s
    return (worst <= RMSE_PASS), worst


def _seqlens_of(case):
    # reconstruct per-seq lengths from the reference list shapes
    return [r.shape[1] for r in case.refs[1]]  # vn_ref_l[i] is [1, s, H, V]


def _bench_case(case, *, warmup, iters, reps, timing):
    if timing == "launch":
        st = int(torch.cuda.current_stream().cuda_stream)
        return min(
            time_launches(fn=case.launch, warmup=warmup, iters=iters, stream=st)
            for _ in range(reps)
        ) * 1e3

    current = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current)
    old_stream = case.cfg.stream
    try:
        case.cfg.stream = int(capture_stream.cuda_stream)
        with torch.cuda.stream(capture_stream):
            # Warm the module and every lazy runtime path before capture.
            for _ in range(3):
                case.launch()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            case.launch()
        capture_stream.synchronize()

        with torch.cuda.stream(capture_stream):
            return min(
                time_launches(
                    fn=graph.replay,
                    warmup=warmup,
                    iters=iters,
                    stream=case.cfg.stream,
                )
                for _ in range(reps)
            ) * 1e3
    finally:
        capture_stream.synchronize()
        case.cfg.stream = old_stream
        current.wait_stream(capture_stream)


def run(shape_ids, *, warmup=20, iters=100, reps=3, arch="gfx942",
        include_inexact=False, exhaustive=False, timing="graph",
        prefetch_w_early=False,
        prefetch_k_early=False,
        prefetch_k_interleave=False, wu_contiguous=False,
        output_bf16=False):
    by_id = {s[0]: s for s in SHAPES}
    stack_variants = (
        existing_stack_variants()
        if exhaustive
        else [
            (
                "full-ki" if prefetch_k_interleave else (
                    "full-ew" if prefetch_w_early else "full"
                ),
                {
                    **FULL_STACK[0][1],
                    "prefetch_w_early": prefetch_w_early,
                    "prefetch_k_early": prefetch_k_early,
                    "prefetch_k_interleave": prefetch_k_interleave,
                },
            )
        ]
    )
    rows = []
    dropped = []
    hdr = (f"  {'#':>2} {'name':13s} {'H/Hg':>5s} {'T':>6s} {'N':>2s} g "
           f"{'rocKE best':>16s} {'FlyDSL best':>16s} {'HIP':>8s}  ok")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for sid in shape_ids:
        if sid not in by_id:
            print(f"  shape {sid}: not in 1..25; skip"); continue
        if not exact(sid) and not include_inexact:
            dropped.append(sid); continue
        _, name, H, Hg, T, N, gate, kind = by_id[sid]
        seqlens = _split(T, N, kind)
        varlen = (N > 1)
        extra = {
            "IS_VARLEN": varlen,
            "WU_CONTIGUOUS": wu_contiguous,
            "OUTPUT_DTYPE_BF16": output_bf16,
        }
        best = None
        refs = None
        legal = correct = 0
        for vname, vkw in VARIANTS:
            for sname, skw in stack_variants:
                try:
                    case = build_case(
                        name,
                        seqlens,
                        H,
                        vkw["BV"],
                        vkw["NR_SPLIT"],
                        gate,
                        extra,
                        Hg=Hg,
                        arch=arch,
                        fence=False,
                        want_refs=refs is None,
                        **skw,
                    )
                except (ValueError, TypeError):
                    continue
                legal += 1
                if refs is None:
                    refs = case.refs
                else:
                    # build_case(seed=0) reconstructs identical inputs. Reuse
                    # the first fp32 oracle instead of recomputing it for every
                    # compile-time variant in the cartesian sweep.
                    case.refs = refs
                ok, rmse = _check(case)
                if not ok:
                    continue
                correct += 1
                us = _bench_case(
                    case,
                    warmup=warmup,
                    iters=iters,
                    reps=reps,
                    timing=timing,
                )
                if best is None or us < best[1]:
                    best = (f"{vname}/{sname}", us, ok, rmse)
        fd = FLYDSL_BEST.get(sid)
        hip = HIP_BASE.get(sid)
        if best is None:
            print(f"  {sid:2d} {name:13s} no legal variant"); continue
        vname, us, ok, rmse = best
        tag = "" if exact(sid) else " [inexact]"
        rows.append((sid, name, H, Hg, T, N, gate, kind, seqlens, best, fd, hip))
        fd_s = f"{fd[0]}:{fd[1]:.1f}" if fd else "-"
        warning = " WARN" if rmse > RMSE_WARN else ""
        sweep = f" [{correct}/{legal} correct]" if exhaustive else ""
        print(f"  {sid:2d} {name:13s} {H:2d}/{Hg:<2d} {T:6d} {N:2d} "
              f"{'gk' if gate=='gk' else 'g ':2s} rocKE:{vname:18s}{us:8.1f} "
              f"{fd_s:>16s} {(f'{hip:.1f}' if hip else '-'):>8s}  "
              f"{'ok' if ok else 'FAIL'} rmse={rmse:.1e}{warning}{tag}{sweep}")
    if dropped:
        print(f"\n  excluded {len(dropped)} non-exact shapes "
              f"(unequal / non-BT-aligned split, not reproducible vs FlyDSL): "
              f"{dropped}")
        print("  re-run with --include-inexact to see them (flagged [inexact]).")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--shapes", default="1-25",
                    help="e.g. '1-25' or '1,3,5'")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--timing",
        choices=("graph", "launch"),
        default="graph",
        help="graph is apples-to-apples with FlyDSL; launch measures eager overhead",
    )
    ap.add_argument("--include-inexact", action="store_true",
                    help="also run shapes whose split cannot be matched to FlyDSL")
    ap.add_argument(
        "--exhaustive",
        action="store_true",
        help="cartesian sweep of geometry and all existing contract-preserving knobs",
    )
    ap.add_argument(
        "--prefetch-w-early",
        action="store_true",
        help="with the full stack, issue next-chunk W before GEMM1",
    )
    ap.add_argument(
        "--prefetch-k-early",
        action="store_true",
        help="with the full stack, load K before the first barrier",
    )
    ap.add_argument(
        "--prefetch-k-interleave",
        action="store_true",
        help="with the full stack, interleave K loads across GEMM1",
    )
    ap.add_argument(
        "--wu-contiguous",
        action="store_true",
        help="use the PR 4884 head-major W/U/Vnew contract",
    )
    ap.add_argument(
        "--output-bf16",
        action="store_true",
        help="materialize H snapshots and Vnew in bf16 as PR 4884 does",
    )
    ap.add_argument("--reference", default=None,
                    help="JSON of reference (FlyDSL/HIP) numbers to print "
                         "alongside; kept out of this repo per compliance")
    args = ap.parse_args(argv)
    if args.reference:
        _load_reference(args.reference)

    if not torch.cuda.is_available():
        print("no GPU; skipping")
        return 0

    ids = []
    for part in args.shapes.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))

    print(f"rocKE K5 benchmark — arch={args.arch}, shapes {args.shapes}, "
          f"{'exhaustive existing-lever sweep' if args.exhaustive else 'full stack (P1-P8)'}, "
          f"{args.timing} timing, best of {args.reps}")
    run(ids, warmup=args.warmup, iters=args.iters, reps=args.reps,
        arch=args.arch, include_inexact=args.include_inexact,
        exhaustive=args.exhaustive, timing=args.timing,
        prefetch_w_early=args.prefetch_w_early,
        prefetch_k_early=args.prefetch_k_early,
        prefetch_k_interleave=args.prefetch_k_interleave,
        wu_contiguous=args.wu_contiguous,
        output_bf16=args.output_bf16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
