#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bench_jit_validation.py -- re-time cold JIT from CBOR now that the VM validates
# every artifact before it emits anything.
#
# WHY. The readiness assessment measured the recipe front end before the wire-ABI
# check, the recipe schema check and guard enforcement existed. Those all run on
# the replay path, ahead of the builder, so the published 0.32 ms (GEMM) and
# 5.74 ms (attention dense) front ends are now measuring a shorter path than the
# one that ships. This re-times them and attributes any difference.
#
# METHOD, kept comparable to that assessment on purpose: median over N iterations
# at three axis points per family, front end timed INSIDE the C engine, plus the
# common comgr stage. The C timer wraps CBOR decode + all admission checks + VM
# expand (see online.cpp), so the checks are inside the number rather than beside
# it.
#
# Per family we time three artifact shapes, which is what makes the cost
# attributable rather than merely re-measured:
#
#   bare         no abi block, no guard   -- what the old numbers measured
#   +abi         abi stamped              -- adds the wire-ABI check per replay
#   +abi+guard   guard attached           -- the shipping form
#
# It also times the two calls hipDNN makes BEFORE deciding to compile, since
# those are new cost that no compile number includes: a standalone guard check
# and a bundle existence probe.
#
#   python3 -m rocke.portable_ir.drivers.bench_jit_validation [--n 9] [--no-comgr]

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.drivers.roll_hsaco_parity import ARCH, _attn, _conv, _gemm
from rocke.portable_ir.src import abi as _abi
from rocke.portable_ir.src import guard as _guard
from rocke.portable_ir.src import online, recipe_bundle
from rocke.portable_ir.src.roll import roll

# The axis each family rolls, the points to replay at, and a generous candidate
# domain for the guard. The replay points are the same three per family the
# readiness table used: two fitted, one held out.
#
# (label, builder, field, replay points, guard candidate domain)
FAMILIES: List[Tuple[str, Callable[..., Any], str, List[int], List[int]]] = [
    ("gemm_universal", _gemm, "tile_n", [32, 64, 128], [16, 32, 64, 128, 256]),
    ("conv_implicit_gemm", _conv, "K", [64, 128, 256], [32, 64, 96, 128, 192, 256]),
    (
        "attention_dense",
        _attn,
        "seqlen_kv",
        [512, 1024, 2048],
        [128, 256, 512, 1024, 2048],
    ),
]

AXIS = "V"  # what roll() names the free axis


def _med(fn: Callable[[], Any], n: int) -> float:
    """Median wall time in ms. Median, not mean: one preempted iteration on a
    shared runner otherwise moves the number more than the thing being measured."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def _gate_for(label: str, builder: Callable[..., Any], field: str) -> _guard.GateFn:
    """A gate over the rolled axis, delegating to the family's own validity
    function. Derived from the real gate rather than hand-written, so the guard's
    rule count -- which is what the check costs -- is representative."""

    def gate(point: Dict[str, Any]) -> bool:
        v = point[AXIS]
        try:
            if label == "gemm_universal":
                from rocke.instances.common.gemm_universal import (
                    DataSpec,
                    TileSpec,
                    TraitSpec,
                    UniversalGemmSpec,
                    is_valid_spec,
                )

                tile = dict(
                    tile_m=16,
                    tile_n=v,
                    tile_k=16,
                    warp_m=1,
                    warp_n=1,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                )
                spec = UniversalGemmSpec(
                    name="g",
                    tile=TileSpec(**tile),
                    trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                    data=DataSpec(),
                    wave_size=64,
                    block_size=64,
                )
                return bool(is_valid_spec(spec, ARCH)[0])
            if label == "conv_implicit_gemm":
                from rocke.instances.common.conv_implicit_gemm import (
                    ConvProblem,
                    ImplicitGemmConvSpec,
                    is_valid_spec,
                )

                spec = ImplicitGemmConvSpec(
                    problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=v, Y=3, X=3),
                    name="c",
                    tile_m=32,
                    tile_n=32,
                    tile_k=32,
                    warp_m=1,
                    warp_n=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                    pipeline="mem",
                    epilogue="cshuffle",
                )
                return bool(is_valid_spec(spec, ARCH)[0])
            from kernels.gfx950.attention_dense import (
                AttentionDenseSpec,
                supports_attention_dense,
            )

            spec = AttentionDenseSpec(
                batch=1,
                seqlen_q=512,
                seqlen_kv=v,
                num_query_heads=128,
                num_kv_heads=8,
                head_size=128,
                causal=True,
                dtype="bf16",
                block_n=64,
                waves_per_eu=2,
            )
            return bool(supports_attention_dense(spec, arch=ARCH)[0])
        except Exception:  # noqa: BLE001 - a spec that will not construct is not legal
            return False

    return gate


def _variants(recipe: Dict[str, Any], guard: Optional[Dict[str, Any]]) -> List[Tuple]:
    """(name, recipe) for each artifact shape, cheapest admission path first."""
    bare = {k: v for k, v in recipe.items() if k not in ("abi", "guard")}
    out = [("bare", bare), ("+abi", _abi.stamp(bare))]
    if guard is not None:
        with_guard = dict(bare)
        with_guard["guard"] = guard
        out.append(("+abi+guard", _abi.stamp(with_guard)))
    return out


def _comgr_ms(ll: str, n: int) -> Optional[float]:
    try:
        from rocke.core.arch import ArchTarget
        from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    except Exception:  # noqa: BLE001
        return None
    isa = ArchTarget.from_gfx(ARCH).isa_triple

    def _run():
        build_hsaco_from_llvm_ir(ll, isa=isa, options=["-O3"])

    try:
        return _med(_run, max(3, n // 2))
    except Exception:  # noqa: BLE001
        return None


def _time_variant(cbor: bytes, points: Sequence[int], n: int) -> Dict[str, float]:
    """C-reported build/lower and Python-side wall, medianed over the points.

    build_ms is what the readiness table calls the front end; wall_ms is the same
    work seen by a caller, so the gap is ctypes plus copying the .ll out."""
    builds, lowers, walls = [], [], []
    for v in points:
        # Untimed warm-up. Without it the first artifact shape measured absorbs
        # first-touch costs (allocator, page faults) and reads as SLOWER than the
        # shapes with more checks, which inverts the very comparison this makes.
        for _ in range(3):
            online.recipe_cbor_to_llvm(cbor, arch=ARCH, ints={AXIS: v})
        bs, ls = [], []
        for _ in range(n):
            _, t = online.recipe_cbor_to_llvm(cbor, arch=ARCH, ints={AXIS: v})
            bs.append(t["build_ms"])
            ls.append(t["lower_ms"])
        builds.append(statistics.median(bs))
        lowers.append(statistics.median(ls))
        walls.append(
            _med(
                lambda v=v: online.recipe_cbor_to_llvm(cbor, arch=ARCH, ints={AXIS: v}),
                n,
            )
        )
    return {
        "build_ms": statistics.median(builds),
        "lower_ms": statistics.median(lowers),
        "wall_ms": statistics.median(walls),
    }


def _time_python(
    builder: Callable[..., Any], field: str, points: Sequence[int], n: int
) -> Tuple[float, float]:
    """The 'stay in Python' baseline: Python builder + Python lowerer.

    Measured here rather than carried over from the earlier assessment so the
    speedup column compares two numbers taken on the same machine and the same
    comgr, which is the only way that ratio means anything."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    builds, lowers = [], []
    for v in points:
        kernel = builder(**{field: v})
        builds.append(_med(lambda v=v: builder(**{field: v}), n))
        lowers.append(_med(lambda k=kernel: lower_kernel_to_llvm(k, arch=ARCH), n))
    return statistics.median(builds), statistics.median(lowers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=9, help="iterations per measurement")
    ap.add_argument("--no-comgr", action="store_true", help="skip the comgr stage")
    args = ap.parse_args()
    n = args.n

    online.load()
    engine, build_id = online.provenance()
    print(
        f"== cold JIT from CBOR, with admission checks (arch={ARCH}, N={n}) ==\n"
        f"   engine {engine or '?'} build {build_id[:12] or '?'}   "
        f"recipe ABI {_abi.RECIPE_ABI}, binary ABI {_abi.BINARY_ABI}\n"
    )

    rows: List[Dict[str, Any]] = []
    admission: List[Dict[str, Any]] = []

    for label, builder, field, points, cands in FAMILIES:
        r = roll(
            build_at=lambda v, b=builder, f=field: b(**{f: v}),
            axis=AXIS,
            sample_points=points[:2],
            holdout_points=points[2:],
        )
        if not r.ok:
            print(f"   {label}: did not roll ({r.reason}); skipped")
            continue

        guard = None
        try:
            guard = _guard.derive_guard(
                _gate_for(label, builder, field),
                {AXIS: cands},
                gate_name=label,
                arch=ARCH,
            )
        except Exception as e:  # noqa: BLE001 - report, do not hide
            print(f"   {label}: no guard derived ({type(e).__name__}: {e})")

        py_build, py_lower = _time_python(builder, field, points, max(3, n // 3))
        comgr_ms = None
        for name, recipe in _variants(r.recipe, guard):
            cbor = recipe_bundle.cbor_encode(recipe)
            t = _time_variant(cbor, points, n)
            if comgr_ms is None and not args.no_comgr:
                ll, _ = online.recipe_cbor_to_llvm(
                    cbor, arch=ARCH, ints={AXIS: points[0]}
                )
                comgr_ms = _comgr_ms(ll, n)
            rows.append(
                {
                    "family": label,
                    "shape": name,
                    "cbor_kb": len(cbor) / 1024.0,
                    "rules": len(guard.get("rules", [])) if "guard" in recipe else 0,
                    "comgr_ms": comgr_ms,
                    "py_build": py_build,
                    "py_lower": py_lower,
                    "recipe": recipe,
                    "axis_at": points[0],
                    **t,
                }
            )

        if guard is not None:
            shipping = _abi.stamp({**r.recipe, "guard": guard})
            cbor = recipe_bundle.cbor_encode(shipping)
            key = shipping.get("kernel_name_fmt") or shipping.get("kernel_name") or "k"
            bundle = recipe_bundle.cbor_encode(
                recipe_bundle.build_bundle(
                    [{"key": key, "arch": ARCH, "recipe": shipping}]
                )
            )
            admission.append(
                {
                    "family": label,
                    "recipe_guard_ms": _med(
                        lambda: online.check_recipe_guard(cbor, ints={AXIS: points[0]}),
                        n,
                    ),
                    "bundle_guard_ms": _med(
                        lambda: online.check_bundle_guard(
                            bundle, key, arch=ARCH, ints={AXIS: points[0]}
                        ),
                        n,
                    ),
                    "contains_ms": _med(
                        lambda: online.bundle_contains(bundle, key, arch=ARCH), n
                    ),
                    "bundle_kb": len(bundle) / 1024.0,
                }
            )

    _print_compile(rows)
    _print_admission(admission)
    _print_rule_scaling(rows, args.n)
    return 0


def _print_rule_scaling(rows: List[Dict[str, Any]], n: int) -> None:
    """Separate guard *size* from bundle size by padding one guard with rules.

    The pre-flight table above shows a cost that tracks bytes, not rules. This
    isolates the other variable: same artifact, more rules. Padding uses a
    tautology (`axis == axis`) so the verdict cannot change, only the work."""
    base = next((r for r in rows if r["shape"] == "+abi+guard"), None)
    if base is None:
        return
    src = base["recipe"]
    guard = src.get("guard")
    if not guard or not guard.get("rules"):
        return

    print("\n-- guard cost vs rule count, one artifact, %s --\n" % base["family"])
    print("%8s %10s %12s" % ("rules", "guard KiB", "check ms"))
    tail = {"op": "le", "a": {"var": AXIS}, "b": {"var": AXIS}}
    for count in (1, 16, 64, 256):
        padded = dict(guard)
        padded["rules"] = list(guard["rules"]) + [tail] * (count - len(guard["rules"]))
        cbor = recipe_bundle.cbor_encode(_abi.stamp({**src, "guard": padded}))
        ms = _med(
            lambda c=cbor: online.check_recipe_guard(c, ints={AXIS: base["axis_at"]}), n
        )
        print(
            "%8d %10.2f %12.3f"
            % (count, len(recipe_bundle.cbor_encode(padded)) / 1024.0, ms)
        )


def _print_compile(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("-- front end (CBOR -> .ll) and cold JIT, ms; medians --\n")
    print(
        "%-20s %-11s %7s %8s %8s %8s %9s %9s"
        % (
            "family",
            "artifact",
            "cbor KiB",
            "build",
            "lower",
            "wall",
            "comgr",
            "cold JIT",
        )
    )
    for r in rows:
        comgr = r["comgr_ms"]
        cold = (
            r["build_ms"] + r["lower_ms"] + comgr if comgr is not None else float("nan")
        )
        print(
            "%-20s %-11s %7.1f %8.3f %8.3f %8.3f %9s %9s"
            % (
                r["family"],
                r["shape"],
                r["cbor_kb"],
                r["build_ms"],
                r["lower_ms"],
                r["wall_ms"],
                "n/a" if comgr is None else "%.2f" % comgr,
                "n/a" if comgr is None else "%.2f" % cold,
            )
        )

    # Same columns as the readiness assessment's compile-time table, so the two
    # can be compared row for row. Shipping form only: this is what runs.
    ship = [r for r in rows if r["shape"] == "+abi+guard"]
    if ship and ship[0]["comgr_ms"] is not None:
        print("\n-- cold JIT, both paths (shipping artifact form) --\n")
        print(
            "%-20s %10s %10s %8s %8s %11s %11s"
            % (
                "family",
                "py front",
                "vm front",
                "speedup",
                "comgr",
                "cold, py",
                "cold, recipe",
            )
        )
        for r in ship:
            py = r["py_build"] + r["py_lower"]
            vm = r["build_ms"] + r["lower_ms"]
            comgr = r["comgr_ms"]
            print(
                "%-20s %10.2f %10.3f %7.1fx %8.2f %11.2f %11.2f"
                % (r["family"], py, vm, py / vm, comgr, py + comgr, vm + comgr)
            )

    print("\n-- what the admission checks cost the compile path --\n")
    print("%-20s %10s %10s %10s" % ("family", "bare", "+abi+guard", "delta"))
    for fam in dict.fromkeys(r["family"] for r in rows):
        got = {r["shape"]: r for r in rows if r["family"] == fam}
        if "bare" not in got or "+abi+guard" not in got:
            continue
        a = got["bare"]["build_ms"] + got["bare"]["lower_ms"]
        b = got["+abi+guard"]["build_ms"] + got["+abi+guard"]["lower_ms"]
        print(
            "%-20s %9.3f %10.3f %+9.3f  (%+.1f%%)"
            % (fam, a, b, b - a, 100 * (b - a) / a if a else float("nan"))
        )


def _print_admission(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print(
        "\n-- pre-flight, per candidate, no IR built (this is NOT in the "
        "numbers above) --\n"
    )
    print(
        "%-20s %8s %14s %14s %12s"
        % ("family", "bundle KiB", "recipe guard", "bundle guard", "contains")
    )
    for r in rows:
        print(
            "%-20s %8.1f %14.3f %14.3f %12.3f"
            % (
                r["family"],
                r["bundle_kb"],
                r["recipe_guard_ms"],
                r["bundle_guard_ms"],
                r["contains_ms"],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
