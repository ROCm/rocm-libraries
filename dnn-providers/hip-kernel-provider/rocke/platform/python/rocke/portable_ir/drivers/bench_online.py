#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bench_online.py -- attribute the ONLINE portable-IR compile timeline to real
# percentages, so "is the C handoff faster than pybind / staying in Python?" is
# answered with measured shares instead of estimates.
#
# Per kernel we time the phases of one compile:
#   build      (Py)  build the KernelDef in the Python builder
#   serialize  (Py)  record -> recipe -> CBOR (the shippable online artifact)
#   py_lower   (Py)  the native Python lowerer (the "stay in Python" baseline)
#   c_build    (C)   CBOR decode + recipe-VM expand -> IR   (the online handoff)
#   c_lower    (C)   the C lowerer IR -> .ll
#   comgr            compile .ll -> HSACO (the backend; dominates)
#
# It also runs a parametric recipe across shapes to show the many-shapes win:
# one tiny artifact, C expands each shape (c_build) vs re-running the Python
# builder per shape (build).
#
# Needs a shared librocke (ROCKE_LIB or auto-built) and, for comgr timing, a comgr
# tool path in $COMGR (else comgr is reported as n/a).
#
#   python3 -m rocke.portable_ir.drivers.bench_online
import hashlib
import os
import statistics
import subprocess
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.src import online, recipe_bundle
from rocke.portable_ir.src.recording_builder import record_kernel

ARCH = os.environ.get("ARCH", "gfx950")
COMGR = os.environ.get("COMGR")
N = int(os.environ.get("BENCH_N", "7"))


def _med_ms(fn: Callable[[], None], n: int = N) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def _comgr_ms(ll: str) -> Optional[float]:
    if not COMGR:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(ll)
        llpath = f.name
    hpath = llpath + ".hsaco"
    try:

        def _run():
            subprocess.run(
                [COMGR, llpath, hpath, ARCH],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return _med_ms(_run, n=max(3, N // 2))
    finally:
        for p in (llpath, hpath):
            try:
                os.unlink(p)
            except OSError:
                pass


def _hsaco_sha(ll: str) -> Optional[str]:
    """Compile .ll to HSACO and return its sha256 (None if no comgr)."""
    if not COMGR:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as f:
        f.write(ll)
        llpath = f.name
    hpath = llpath + ".hsaco"
    try:
        subprocess.run(
            [COMGR, llpath, hpath, ARCH],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(hpath, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    finally:
        for p in (llpath, hpath):
            try:
                os.unlink(p)
            except OSError:
                pass


def _online_split(
    cbor: bytes, ints: Dict[str, int], strs: Dict[str, str]
) -> Tuple[str, float, float]:
    """Median C build_ms / lower_ms (reported by the C wrapper) + one .ll."""
    bs, ls, ll = [], [], None
    for _ in range(N):
        ll, t = online.recipe_cbor_to_llvm(cbor, arch=ARCH, ints=ints, strs=strs)
        bs.append(t["build_ms"])
        ls.append(t["lower_ms"])
    return ll, statistics.median(bs), statistics.median(ls)


def _row(
    name: str,
    build_fn: Callable,
    *,
    ints: Optional[Dict] = None,
    strs: Optional[Dict] = None,
    recipe: Optional[dict] = None,
) -> Dict[str, float]:
    ints = ints or {}
    strs = strs or {}
    if recipe is None:
        _, recipe = record_kernel(build_fn)
    cbor = recipe_bundle.cbor_encode(recipe)
    build_ms = _med_ms(lambda: build_fn())
    serialize_ms = _med_ms(lambda: recipe_bundle.cbor_encode(recipe))
    kref = build_fn()
    py_lower_ms = _med_ms(lambda: lower_kernel_to_llvm(kref, arch=ARCH))
    ll, c_build, c_lower = _online_split(cbor, ints, strs)
    # sanity: online C path must be byte-identical to native Python AFTER comgr
    # (the .ll text can differ only in SSA temp names; HSACO is the ground truth).
    py_ll = lower_kernel_to_llvm(kref, arch=ARCH)
    if COMGR:
        assert _hsaco_sha(ll) == _hsaco_sha(
            py_ll
        ), f"{name}: online HSACO != python HSACO"
    comgr = _comgr_ms(ll)
    return {
        "name": name,
        "lines": ll.count("\n") + 1,
        "build": build_ms,
        "serialize": serialize_ms,
        "py_lower": py_lower_ms,
        "c_build": c_build,
        "c_lower": c_lower,
        "comgr": comgr or float("nan"),
        "cbor_bytes": len(cbor),
    }


def _fmt_table(rows: List[Dict[str, float]]) -> None:
    cols = [
        ("kernel", "name", "%-30s"),
        ("ll", "lines", "%6d"),
        ("build", "build", "%8.3f"),
        ("serialize", "serialize", "%9.3f"),
        ("py_lower", "py_lower", "%8.3f"),
        ("c_build", "c_build", "%8.3f"),
        ("c_lower", "c_lower", "%8.3f"),
        ("comgr", "comgr", "%9.2f"),
    ]
    head = "  ".join(
        ("%-30s" if k == "name" else ("%6s" if k == "lines" else "%9s")) % h
        for h, k, _ in cols
    )
    print(head + "    (ms; medians)")
    for r in rows:
        print("  ".join(fmt % r[key] for _, key, fmt in cols))


def _summary(rows: List[Dict[str, float]]) -> None:
    print("\nShares of one online compile (build IR in C + lower + comgr):")
    print("%-30s  %10s  %10s  %10s" % ("kernel", "c_build %", "c_lower %", "comgr %"))
    for r in rows:
        comgr = r["comgr"]
        if comgr != comgr:  # nan
            continue
        total = r["c_build"] + r["c_lower"] + comgr
        print(
            "%-30s  %9.1f%%  %9.1f%%  %9.1f%%"
            % (
                r["name"],
                100 * r["c_build"] / total,
                100 * r["c_lower"] / total,
                100 * comgr / total,
            )
        )
    print("\nC lower vs Python lower (handoff payoff for the lowering step):")
    print("%-30s  %10s  %10s  %8s" % ("kernel", "py_lower", "c_lower", "speedup"))
    for r in rows:
        sp = r["py_lower"] / r["c_lower"] if r["c_lower"] else float("nan")
        print(
            "%-30s  %9.3f  %9.3f  %6.1fx" % (r["name"], r["py_lower"], r["c_lower"], sp)
        )


def main() -> int:
    from rocke.portable_ir.examples import mini_attn, recipe_multi_result, recipe_toy

    try:
        from rocke.portable_ir.examples import export_mha

        big = [
            (
                "attention_fp16_d128",
                lambda: export_mha.build("fp16", 128, 2048, 1, 32, 1),
            )
        ]
    except Exception:  # noqa: BLE001
        big = []

    print(
        f"== online portable-IR compile timeline (arch={ARCH}, N={N}, "
        f"comgr={'yes' if COMGR else 'no'}) ==\n"
    )

    print("-- concrete-record kernels (recorded -> CBOR; no spec params) --")
    rows = [
        _row("mini_attn_norm0", lambda: mini_attn.build_mini_attn(0, "f32")),
        _row("mini_attn_norm1", lambda: mini_attn.build_mini_attn(1, "f32")),
        _row("multi_result", lambda: recipe_multi_result.build_multi_result("i32")),
    ]
    rows += [_row(n, f) for n, f in big]
    _fmt_table(rows)
    _summary(rows)

    print(
        "\n-- parametric recipe: ONE artifact, many shapes "
        "(C expand per shape vs Python rebuild per shape) --"
    )
    toy_recipe = recipe_toy.make_recipe()
    toy_cbor = recipe_bundle.cbor_encode(toy_recipe)
    print(f"   one CBOR recipe = {len(toy_cbor)} bytes covers all D\n")
    print("%-8s  %12s  %12s  %10s" % ("D", "py_rebuild", "c_expand", "speedup"))
    for D in (64, 128, 256):
        py_build = _med_ms(lambda D=D: recipe_toy.build_toy(D, "f32"))
        _, c_build, _ = _online_split(toy_cbor, {"D": D}, {"dtype": "f32"})
        sp = py_build / c_build if c_build else float("nan")
        print("%-8d  %11.3f  %11.3f  %8.1fx" % (D, py_build, c_build, sp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
