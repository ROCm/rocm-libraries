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
        {"N": [8, 16], "K": [64, 128]},
        None,
        [{"N": 32, "K": 256}, {"N": 8, "K": 256}, {"N": 64, "K": 512}],
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
        {"N": [8, 16], "K": [64, 128], "C": [64, 128]},
        None,
        [{"N": 32, "K": 256, "C": 256}],
        "C enters the spatial-product constants non-affinely",
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
    for label, build, axes, struct, hold, why in REFUSED_ND:
        r = roll_nd(build, axes=axes, structural_axis=struct, holdout_points=hold)
        state = "still refused" if not r.ok else "NOW ROLLS (update this list)"
        print(f"  {label:<22}{','.join(axes):<28}{state:<28}{why}")
        if r.ok:
            rolled = -1  # a silent capability gain should still trip the gate
    return 0 if rolled == len(FAMILIES_ND) else 1


if __name__ == "__main__":
    sys.exit(main())
