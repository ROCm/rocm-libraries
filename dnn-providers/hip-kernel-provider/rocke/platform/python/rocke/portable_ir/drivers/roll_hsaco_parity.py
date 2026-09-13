#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_hsaco_parity.py -- HSACO byte-identity gate for the record+ROLL path.
#
# parity_matrix proves the *concrete* recipe path at the .ll level. This driver
# closes the two gaps that leaves open: it exercises the ROLLED (parametric)
# recipe, and it compares final HSACO bytes rather than IR text.
#
# For each kernel family, over one structural axis (an input-shape dim or a
# tile/warp geometry dim):
#
#   author time : record 2 concrete traces -> roller -> ONE parametric recipe
#                 -> CBOR bytes (the shipped artifact)
#   run time    : CBOR -> C DOM decode -> C recipe VM expand at spec{axis:v}
#                 -> C lower -> .ll -> comgr -> HSACO
#   oracle      : pure Python build(v) -> Python lower -> .ll -> comgr -> HSACO
#
# Gate: HSACO must be byte-identical at every verification point, INCLUDING
# held-out points the roller never sampled -- that is what separates "replayed
# the two traces it saw" from "generalized over the axis".
#
# On .ll: a rolled recipe cannot replay names verbatim the way a concrete one
# does (each instruction expands many times, so every expansion must draw a
# fresh name), but it reproduces them anyway -- the recipe carries Python's
# per-op `result_name_hint` and the roller keeps Python's lane naming for
# loop-carry fans, so both engines mint the same `%tid7` / `%acc_m0_n3` off the
# same counter. The .ll is therefore byte-identical, not merely
# alpha-equivalent, and a checksum is enough to compare the two paths. The
# driver still reports .ll and HSACO separately, and still distinguishes EXACT
# from ALPHA-EQ, so a regression in naming shows up as a downgrade rather than
# passing silently on the HSACO gate.
#
#   python3 -m rocke.portable_ir.drivers.roll_hsaco_parity [--families gemm,conv]
#
# Needs a shared librocke (ROCKE_ONLINE_LIB) and comgr. No device required:
# comgr compiles for the target ISA on the host.

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from typing import Any, Dict, List

ARCH = "gfx950"


# --------------------------------------------------------------------------
# builders -- each takes the swept axis value as a single keyword override
# --------------------------------------------------------------------------
def _gemm(**over):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    tile = dict(
        tile_m=16,
        tile_n=32,
        tile_k=16,
        warp_m=1,
        warp_n=1,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    )
    tile.update(over)
    spec = UniversalGemmSpec(
        name="gemm_" + "_".join(f"{k}{v}" for k, v in sorted(over.items())),
        tile=TileSpec(**tile),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        data=DataSpec(),
        wave_size=64,
        block_size=64,
    )
    return build_universal_gemm(spec, arch=ARCH)


def _conv(**over):
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    prob = dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
    geom = dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        # Not "default": the K axis this lane rolls drives an auto-derived
        # vector_size_c of 8, and the conv gate refuses the default epilogue
        # for any store wider than one element.
        epilogue="cshuffle",
    )
    for k, v in over.items():
        (prob if k in prob else geom)[k] = v
    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(**prob),
        name="conv_" + "_".join(f"{k}{v}" for k, v in sorted(over.items())),
        **geom,
    )
    return build_implicit_gemm_conv(spec, arch=ARCH)


def _attn(**over):
    from kernels.gfx950.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
    )

    spec = dict(
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
    spec.update(over)
    return build_attention_dense(AttentionDenseSpec(**spec), arch=ARCH)


def _moe(**over):
    from rocke.instances.common.fused_moe import FusedMoeSpec, build_moe_gather

    spec = dict(
        tokens=32,
        experts=8,
        topk=2,
        hidden=512,
        intermediate=256,
        dtype="f16",
        block_size=128,
        vec=4,
    )
    spec.update(over)
    return build_moe_gather(FusedMoeSpec(name="fused_moe_gather", **spec))


# (label, builder, swept field, kind, samples, holdouts)
FAMILIES = [
    ("gemm_universal", _gemm, "tile_n", "tile geometry", [32, 64], [128, 256]),
    ("conv_implicit_gemm", _conv, "K", "input shape", [64, 128], [256]),
    ("conv_implicit_gemm", _conv, "N", "input shape", [8, 16], [32]),
    ("attention_dense", _attn, "seqlen_kv", "input shape", [512, 1024], [2048]),
    ("attention_dense", _attn, "num_query_heads", "input shape", [64, 128], [256]),
    ("fused_moe/gather", _moe, "hidden", "input shape", [512, 1024], [2048]),
    ("fused_moe/gather", _moe, "tokens", "input shape", [32, 64], [128]),
]

# Axes probed and REFUSED by the roller. Kept in the driver so the limitation is
# visible in the gate's own output instead of living only in a report. A refusal
# is safe (the caller keeps concrete per-shape recipes); it is a compression
# loss, not a correctness risk.
KNOWN_UNROLLABLE = [
    (
        "gemm_universal",
        "tile_m",
        "shorter-at-larger-axis: op count shrinks as the axis grows",
    ),
    ("gemm_universal", "tile_k", "verify failed: k-atom constant not affine in tile_k"),
    (
        "conv_implicit_gemm",
        "tile_n",
        "no run candidate: trace lengths 59 vs 83 don't segment",
    ),
    (
        "conv_implicit_gemm",
        "tile_m",
        "no run candidate: trace lengths 160 vs 184 don't segment",
    ),
    ("conv_implicit_gemm", "C", "non-affine constant 6 vs 7 (magic-division shift)"),
    ("attention_dense", "head_size", "merge conflict on tile.smem_alloc"),
    (
        "attention_dense",
        "block_n",
        "non-affine constant 8 vs 4 (seqlen_kv/block_n is a division)",
    ),
    (
        "fused_moe/gather",
        "hidden@128",
        "opcode change: global_load vs global_load_vN (vector width)",
    ),
]


# --------------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _alpha_norm(text: str) -> str:
    """Rename SSA values in first-appearance order, so two modules that differ
    only in value naming normalize to the same string.

    Kept as the *diagnostic* that separates a naming regression from a real
    structural divergence: the rolled path is expected to match verbatim, so when
    a point is not EXACT this says whether only the names moved (ALPHA-EQ, e.g. a
    recipe recorded before the prefix field existed, or a fan the roller could
    not name positionally) or the emitted code actually changed (DIFFER)."""
    seen: Dict[str, str] = {}

    def rep(m):
        k = m.group(0)
        if k not in seen:
            seen[k] = "%__" + str(len(seen))
        return seen[k]

    return re.sub(r"%[A-Za-z0-9_.]+", rep, text)


def _flavor() -> str:
    from rocke.core.lower_llvm import _flavor_for_rocm
    from rocke.runtime.comgr import resolved_lib_rocm_version

    ver = resolved_lib_rocm_version()
    return _flavor_for_rocm(*ver) if ver else "llvm20"


def _hsaco(ll: str) -> bytes:
    from rocke.core.arch import ArchTarget
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    hsaco, _ = build_hsaco_from_llvm_ir(
        ll, isa=ArchTarget.from_gfx(ARCH).isa_triple, options=["-O3"]
    )
    return hsaco


def _first_diff(a: str, b: str) -> str:
    import difflib

    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""):
        if line[:1] in "+-" and not line.startswith(("+++", "---")):
            return line[:110]
    return ""


def _check_point(build_at, axis, cbor, v, flavor, want_hsaco) -> Dict[str, Any]:
    """One verification point: pure-Python oracle vs rolled-recipe C replay."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.portable_ir.src import online

    row: Dict[str, Any] = {"v": v, "detail": ""}

    t0 = time.perf_counter()
    py_ll = lower_kernel_to_llvm(build_at(v), llvm_flavor=flavor, arch=ARCH)
    row["py_ms"] = (time.perf_counter() - t0) * 1e3

    vm_ll, t = online.recipe_cbor_to_llvm(cbor, arch=ARCH, ints={axis: v})
    row["build_ms"], row["lower_ms"] = t["build_ms"], t["lower_ms"]

    # Checksum the .ll as emitted, not alpha-normalized: the rolled path
    # reproduces Python's SSA names, so the raw text hashes equal and the digest
    # is a gate on its own rather than a summary of a weaker comparison.
    py_sha, vm_sha = _sha(py_ll.encode()), _sha(vm_ll.encode())
    row["ll_sha"] = py_sha if py_sha == vm_sha else f"{py_sha}!={vm_sha}"
    if py_ll == vm_ll:
        row["ll"] = "EXACT"
    elif _alpha_norm(py_ll) == _alpha_norm(vm_ll):
        row["ll"] = "ALPHA-EQ"
        row["detail"] = "SSA names differ; structure identical"
    else:
        row["ll"] = "DIFFER"
        row["detail"] = _first_diff(py_ll, vm_ll)

    row["hsaco"] = "-"
    if want_hsaco:
        t0 = time.perf_counter()
        py_h, vm_h = _hsaco(py_ll), _hsaco(vm_ll)
        row["comgr_ms"] = (time.perf_counter() - t0) * 1e3 / 2
        row["hsaco"] = "IDENTICAL" if py_h == vm_h else "DIFFER"
        row["sha"], row["bytes"] = _sha(py_h), len(py_h)
        if py_h != vm_h:
            row["detail"] = f"py={_sha(py_h)} vm={_sha(vm_h)}"
    return row


def run_family(label, builder, field, kind, samples, holdouts, want_hsaco, flavor):
    from rocke.portable_ir.src import recipe_bundle
    from rocke.portable_ir.src.roll import roll, roll_report

    def build_at(v):
        return builder(**{field: v})

    out: Dict[str, Any] = {
        "label": label,
        "field": field,
        "kind": kind,
        "holdouts": holdouts,
        "rolled": False,
        "rows": [],
        "reason": "",
    }
    t0 = time.perf_counter()
    try:
        r = roll(
            build_at=build_at,
            axis="V",
            sample_points=samples,
            holdout_points=holdouts,
        )
    except Exception as e:  # noqa: BLE001 - a family that cannot roll is a result
        out["reason"] = f"{type(e).__name__}: {e}"
        return out
    out["roll_ms"] = (time.perf_counter() - t0) * 1e3
    if not r.ok:
        out["reason"] = r.reason
        return out

    out["rolled"] = True
    out["report"] = roll_report(r)
    cbor = recipe_bundle.cbor_encode(r.recipe)
    out["cbor_kb"] = len(cbor) / 1024.0
    out["concrete_kb"] = (
        sum(len(recipe_bundle.cbor_encode(t)) for t in r.traces.values()) / 1024.0
    )

    for v in list(samples) + list(holdouts):
        try:
            out["rows"].append(_check_point(build_at, "V", cbor, v, flavor, want_hsaco))
        except Exception as e:  # noqa: BLE001
            out["rows"].append(
                {
                    "v": v,
                    "ll": "ERROR",
                    "hsaco": "-",
                    "detail": f"{type(e).__name__}: {e}"[:110],
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="", help="comma-separated label substrings")
    ap.add_argument("--no-hsaco", action="store_true", help="stop at .ll (skip comgr)")
    ap.add_argument("--flavor", default="auto")
    ap.add_argument(
        "--expect-points",
        type=int,
        default=0,
        help="fail if fewer than N (family, value) points were verified. Pins "
        "coverage so an axis that quietly stops rolling is a failure and not "
        "just a smaller table",
    )
    args = ap.parse_args()

    flavor = _flavor() if args.flavor == "auto" else args.flavor
    # The C engine resolves its flavor from the environment; without this pin the
    # two engines can lower at different LLVM generations and every cell "fails"
    # on the datalayout line alone.
    os.environ["ROCKE_LLVM_FLAVOR"] = flavor
    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

    from rocke.portable_ir.src import online

    online.load()

    fams: List[Any] = FAMILIES
    if args.families:
        want = [s.strip() for s in args.families.split(",")]
        fams = [f for f in fams if any(w in f[0] for w in want)]

    want_hsaco = not args.no_hsaco
    print(f"== record+roll -> HSACO byte-identity ({ARCH}, flavor={flavor}) ==")
    print("   author: record 2 traces -> roll -> ONE parametric recipe -> CBOR")
    print("   replay: CBOR -> C DOM -> recipe VM -> C lower -> comgr -> HSACO")
    print("   oracle: pure Python build -> Python lower -> comgr -> HSACO\n")

    results = []
    for label, builder, field, kind, samples, holdouts in fams:
        res = run_family(
            label, builder, field, kind, samples, holdouts, want_hsaco, flavor
        )
        results.append(res)
        print(f"  {label} :: {field}  ({kind})")
        if not res["rolled"]:
            print(f"      NOT ROLLED: {res['reason'][:150]}\n")
            continue
        print(
            f"      {res['report']}\n"
            f"      cbor {res['cbor_kb']:.1f} KiB parametric vs "
            f"{res['concrete_kb']:.1f} KiB concrete   roll {res['roll_ms']:.0f} ms"
        )
        for row in res["rows"]:
            held = "held-out" if row["v"] in holdouts else "sampled "
            extra = ""
            if row.get("ll_sha"):
                extra = f" llsum={row['ll_sha']}"
            if row.get("sha"):
                extra += f" sha={row['sha']} {row['bytes']/1024.0:.1f}KiB"
            if row.get("build_ms") is not None:
                extra += (
                    f"  vm={row.get('build_ms',0):.1f}+{row.get('lower_ms',0):.1f}ms"
                    f" py={row.get('py_ms',0):.0f}ms"
                )
            print(
                f"        {field}={row['v']:<5} {held}  .ll={row['ll']:<9} "
                f"hsaco={row.get('hsaco','-'):<9}{extra}"
            )
            if row["detail"]:
                print(f"            {row['detail']}")
        print()

    print("=" * 78)
    rolled = [r for r in results if r["rolled"]]
    rows = [row for r in rolled for row in r["rows"]]
    good = [r for r in rows if r.get("hsaco") == "IDENTICAL"]
    held = [
        row
        for r in rolled
        for row in r["rows"]
        if row["v"] in r["holdouts"] and row.get("hsaco") == "IDENTICAL"
    ]
    exact = [r for r in rows if r.get("ll") == "EXACT"]
    ll_held = [
        row
        for r in rolled
        for row in r["rows"]
        if row["v"] in r["holdouts"] and row.get("ll") == "EXACT"
    ]
    print(f"axes rolled          : {len(rolled)}/{len(results)}")
    # The .ll digest is the primary gate: it needs no comgr, and a rolled recipe
    # reproducing Python's SSA names is what makes it usable. Scored separately
    # from HSACO because a slip back to alpha-equivalence would still pass the
    # HSACO gate and would otherwise go unnoticed.
    print(
        f".ll  sha identical   : {len(exact)}/{len(rows)} points "
        f"({len(ll_held)} of them held-out)"
    )
    if want_hsaco:
        print(
            f"HSACO byte-identical : {len(good)}/{len(rows)} points "
            f"({len(held)} of them held-out)"
        )
    else:
        print("HSACO byte-identical : skipped (--no-hsaco)")
    for r in results:
        if not r["rolled"]:
            print(f"  fallback: {r['label']}::{r['field']} -- {r['reason'][:90]}")
    print("\n  roller refusals on other axes of these same families:")
    for lab, ax, why in KNOWN_UNROLLABLE:
        print(f"    {lab:<20} {ax:<12} {why}")
    bad = len(rows) - len(exact)
    if want_hsaco:
        bad += len(rows) - len(good)

    # Verifying nothing is not the same as verifying everything. If the roller
    # refused every axis there are no bad points, so `bad` alone would let a
    # total regression through; the same is true of a family list that quietly
    # got shorter. Both are failures.
    short = args.expect_points and len(rows) < args.expect_points
    if short:
        print(
            f"\n  COVERAGE SHORTFALL: verified {len(rows)} points, expected at "
            f"least {args.expect_points}"
        )
    if not rolled:
        print("\n  NOTHING ROLLED: no axis produced a parametric recipe")
    print(
        "\n"
        + (
            "PASS"
            if (rolled and not bad and not short)
            else f"INCOMPLETE ({bad} bad points)"
        )
    )
    return 1 if (bad or short or not rolled) else 0


if __name__ == "__main__":
    raise SystemExit(main())
