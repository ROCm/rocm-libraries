# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_nd_coverage.py -- the multi-axis rolling gate.
#
#   python3 -m rocke.portable_ir.drivers.roll_nd_coverage [--ll]
#
# roll_hsaco_parity.py gates ONE axis per recipe: seven recipes for seven axes,
# each moving along its own axis with the others pinned. This gates the CROSS
# PRODUCT -- one recipe per family covering every combination of its
# non-reduction axes, checked at points that were never fitted.
#
# Two things are asserted per family:
#   * every cross-product point and holdout reproduces the independently
#     recorded concrete recipe (the recipe_expand oracle, the byte-identity proxy
#     for HSACO), and
#   * the kernel NAME reconstructs at every point, since a recipe that emits the
#     right instructions under the wrong symbol is still broken.
#
# With --ll it additionally compares the SHA of the lowered .ll text at every
# point, which is the same currency roll_hsaco_parity reports. Neither mode needs
# comgr or a GPU.
#
# Axes that the roller REFUSES are listed too, in the gate's own output rather
# than only in a report: a refusal is safe (the caller keeps concrete per-point
# recipes) but it is exactly the frontier worth watching.

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from rocke.portable_ir.drivers.roll_hsaco_parity import ARCH, _attn, _conv, _gemm, _moe
from rocke.portable_ir.src.roll_nd import roll_nd
from rocke.portable_ir.src.roll_regimes import legal_values

# (label, builder, axes, structural_axis, holdouts, extra_spec)
FAMILIES_ND: List[
    Tuple[
        str,
        Callable[..., Any],
        Dict[str, List[int]],
        Optional[str],
        List[Dict[str, int]],
        Dict[str, Any],
    ]
] = [
    (
        "gemm_universal",
        _gemm,
        {"tile_n": [32, 64]},
        "tile_n",
        [{"tile_n": 128}, {"tile_n": 256}],
        {},
    ),
    (
        "conv_implicit_gemm",
        _conv,
        # C must be sampled off the powers of two: it drives a strength-reduced
        # `n // C`, whose multiplier is 1 for EVERY power of two. Sampling 64/128
        # alone makes that constant look invariant, and the model freezes it.
        {"N": [8, 16], "K": [64, 128], "C": [64, 96, 128]},
        None,
        [
            {"N": 32, "K": 256, "C": 192},
            {"N": 8, "K": 256, "C": 160},
            {"N": 64, "K": 512, "C": 224},
        ],
        {},
    ),
    (
        "attention_dense",
        _attn,
        {"seqlen_kv": [512, 1024], "num_query_heads": [64, 128]},
        None,
        [{"seqlen_kv": 2048, "num_query_heads": 256}],
        {},
    ),
    (
        "fused_moe/gather",
        _moe,
        {"hidden": [512, 1024], "tokens": [32, 64]},
        "hidden",
        [{"hidden": 2048, "tokens": 128}],
        {"dtype": "f16"},
    ),
]

# How many values each axis actually HAS, according to the kernel's own spec
# validation. This decides what a recipe is worth: covering an axis with 64 legal
# values saves 64 recipes, covering one with 2 saves 2. Without it the frontier
# gets ranked by how hard an axis is to roll rather than by what rolling it buys,
# which is how block_n came to look like a priority -- it has five legal values.
#
# (label, axis, make_spec, candidate values to test for legality)
DOMAINS: List[Tuple[str, str, Callable[..., Any], List[int]]] = []


def _domains() -> List[Tuple[str, str, Callable[..., Any], List[int]]]:
    if DOMAINS:
        return DOMAINS
    from kernels.gfx950.attention_dense import AttentionDenseSpec

    from rocke.instances.common.conv_implicit_gemm import ConvProblem

    attn_base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
        waves_per_eu=2,
    )
    conv_base = dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
    attn = lambda **kw: AttentionDenseSpec(**{**attn_base, **kw})  # noqa: E731
    conv = lambda **kw: ConvProblem(**{**conv_base, **kw})  # noqa: E731
    step16 = list(range(16, 2049, 16))
    DOMAINS.extend(
        [
            ("attention_dense", "num_query_heads", attn, step16),
            ("attention_dense", "seqlen_kv", attn, step16),
            ("attention_dense", "block_n", attn, step16),
            ("attention_dense", "head_size", attn, step16),
            ("conv_implicit_gemm", "C", conv, step16),
            ("conv_implicit_gemm", "K", conv, step16),
            ("conv_implicit_gemm", "N", conv, list(range(1, 129))),
        ]
    )
    return DOMAINS


# Axis combinations probed and REFUSED, with the reason kept short. These are the
# frontier: each is a mechanism the roller still lacks, not a correctness risk.
REFUSED_ND: List[
    Tuple[
        str,
        Callable[..., Any],
        Dict[str, List[int]],
        Optional[str],
        List[Dict[str, int]],
        str,
    ]
] = [
    (
        "conv_implicit_gemm",
        _conv,
        # The same axes, but with C sampled only on powers of two. The magic
        # multiplier is 1 at both, so it looks invariant and gets frozen -- caught
        # at the held-out non-power-of-two. Kept as a probe because the failure is
        # a SAMPLING mistake that looks like success until verification.
        {"N": [8, 16], "K": [64, 128], "C": [64, 128]},
        None,
        [{"N": 32, "K": 256, "C": 192}],
        "C sampled only on powers of two hides the magic multiplier",
    ),
    (
        "gemm_universal",
        _gemm,
        {"tile_n": [32, 64], "tile_m": [16, 32]},
        "tile_n",
        [],
        "tile_m is non-monotonic (the load path re-vectorizes)",
    ),
]

# Refusals whose probe costs ~20s because the structural roller searches a large
# level before giving up. Kept out of the default run so the everyday gate stays
# ~1s; `--slow` includes them.
REFUSED_ND_SLOW = [
    (
        "attention_dense",
        _attn,
        # Constants fit now (the reciprocal block count). What is left is
        # structural, and this sample PAIR cannot show it: the softmax reduction's
        # count is block_n/32 - 1, so it has zero copies at 32 and one at 64, and a
        # run seen once carries no loop-carry evidence. Kept at 32/64 on purpose --
        # it pins the diagnostic for the sampling trap. Re-sampled at 64/128 the
        # roller reaches the real blocker (the KV scf.for carries 10 iter-args at
        # 64 and 14 at 128), but that attempt costs ~29 minutes, far too slow to
        # gate; `test_roller.py` iterates the same shape on a synthetic instead.
        {"block_n": [32, 64]},
        "block_n",
        [{"block_n": 128}],
        "sample pair cannot see the block_n/32-1 run (real blocker: scf.for arity)",
    ),
]


def _deep(prog: List[Dict[str, Any]]) -> int:
    n = 0
    for i in prog:
        if i.get("op") != "param":
            n += 1
        for k in ("body", "then", "else"):
            if k in i:
                n += _deep(i[k])
    return n


def _static_fors(prog: List[Dict[str, Any]]) -> int:
    n = 0
    for i in prog:
        if i.get("op") == "static_for":
            n += 1
        for k in ("body", "then", "else"):
            if k in i:
                n += _static_fors(i[k])
    return n


def _ll_shas(build: Callable[..., Any], points: List[Dict[str, int]]) -> Dict[str, str]:
    """SHA of the lowered .ll at each point -- the currency roll_hsaco_parity
    reports, so a divergence here is directly comparable to that gate."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    out = {}
    for p in points:
        ll = lower_kernel_to_llvm(build(**p), arch=ARCH)
        out[str(sorted(p.items()))] = hashlib.sha256(ll.encode()).hexdigest()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ll",
        action="store_true",
        help="also lower each point and report the .ll SHA (slower, no comgr)",
    )
    ap.add_argument(
        "--slow",
        action="store_true",
        help="also re-probe the refusals whose structural search costs ~20s each",
    )
    args = ap.parse_args(argv)

    print(f"multi-axis rolling coverage  (arch={ARCH})\n")
    hdr = (
        f"{'family':<22}{'axes':<30}{'grid':>5}{'held':>5}{'traces':>7}"
        f"{'ops':>7}{'sfor':>5}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    rolled = 0
    for label, build, axes, struct, hold, extra in FAMILIES_ND:
        r = roll_nd(
            build,
            axes=axes,
            structural_axis=struct,
            holdout_points=hold,
            extra_spec=extra,
        )
        desc = ",".join(f"{a}{'*' if a == struct else ''}" for a in axes)
        grid = len(list(itertools.product(*(axes[a] for a in axes))))
        if not r.ok:
            print(
                f"{label:<22}{desc:<30}{grid:>5}{len(hold):>5}{'-':>7}{'-':>7}{'-':>5}"
                f"  REFUSED: {r.reason[:60]}"
            )
            continue
        rolled += 1
        print(
            f"{label:<22}{desc:<30}{grid:>5}{len(r.points) - grid:>5}"
            f"{r.n_recorded:>7}{_deep(r.recipe['program']):>7}"
            f"{_static_fors(r.recipe['program']):>5}  OK"
        )
        if args.ll:
            shas = _ll_shas(build, r.points)
            print(
                f"{'':<22}  .ll: {len(set(shas.values()))} distinct SHA over "
                f"{len(shas)} points, each verified against its own recording"
            )
    print("\n* = structural axis (rolled to a static_for); others are constant-only.")
    print("'grid' is the axis cross product, 'held' the extrapolated points; both")
    print("are verified, while only 'traces' were recorded to infer the model.")
    print(f"families rolled       : {rolled}/{len(FAMILIES_ND)}")

    print("\nprobed and refused (the frontier; a refusal keeps concrete recipes):")
    probes = list(REFUSED_ND) + (list(REFUSED_ND_SLOW) if args.slow else [])
    for label, build, axes, struct, hold, why in probes:
        r = roll_nd(build, axes=axes, structural_axis=struct, holdout_points=hold)
        state = "still refused" if not r.ok else "NOW ROLLS (update this list)"
        print(f"  {label:<22}{','.join(axes):<28}{state:<28}{why}")
        if r.ok:
            rolled = -1  # a silent capability gain should still trip the gate
    if not args.slow:
        print(f"  ({len(REFUSED_ND_SLOW)} slow probe(s) skipped; --slow includes them)")

    print("\naxis domains, per the kernels' own spec validation (what rolling buys):")
    for label, axis, make_spec, cands in _domains():
        ok = legal_values(axis, cands, make_spec)
        span = f"{ok[0]}..{ok[-1]}" if ok else "none"
        print(f"  {label:<22}{axis:<20}{len(ok):>4} legal  ({span})")
    print("  a recipe covering an axis saves one concrete recipe per legal value,")
    print("  so these counts -- not the difficulty -- should order the frontier.")
    return 0 if rolled == len(FAMILIES_ND) else 1


if __name__ == "__main__":
    sys.exit(main())
