# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ <-> Python dispatcher selection-parity check for ck-dsl GEMM.

For every shape in the corpus this script:

  1. Runs the Python dispatcher (``ck_dsl.dispatch.dispatch_gemm_fp16``) and reads
     the SELECTED kernel's structural identity from the returned UniversalGemmSpec
     (tile block sizes + pipeline + epilogue).
  2. Reads the C++ pick for the same shape from the JSONL emitted by the compiled
     ``cpp_select`` harness, which ran the REAL ``ck_dsl::Dispatcher`` over the
     REAL shipped per-arch manifest bundle.
  3. Compares them on the identity that is COMMON to both representations.

Why the identity is the structural tuple, not the raw kernel_name string
------------------------------------------------------------------------
The two sides intentionally share their selection *identity* via the manifest
``cache_key`` (== ``kernel_name``). But the two code paths mint that name from
different prefixes today:

  * C++ shipped manifests:  ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_...
  * Python UniversalGemmSpec.kernel_name(): prefixed by the *candidate* name,
    e.g. universal_gemm_fp16_cdna_cshuffle_fp16_t128x128x32_...

So a raw-string equality test would report 0% match for a reason that is purely
cosmetic (naming prefix), not a real divergence in *which kernel* got picked.
The load-bearing question -- "do C++ and Python select the SAME kernel?" -- is
answered by the structural identity that both encode identically:

    (block_m, block_n, block_k, pipeline, epilogue)

That tuple is exactly the tile geometry + pipeline/epilogue that determines the
HSACO. We compare on it, and ALSO report the raw kernel_name from each side so a
human can see the naming-scheme difference explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ck_dsl.dispatch import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16

_DISPATCH = {"fp16": dispatch_gemm_fp16, "bf16": dispatch_gemm_bf16}


def python_pick(M: int, N: int, K: int, arch: str, dtype: str = "fp16"):
    """Return the Python dispatcher's structural pick, or an error string."""
    dispatch = _DISPATCH[dtype]
    try:
        r = dispatch(GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dtype))
    except ValueError as e:
        return None, str(e)
    t = r.spec.tile
    tr = r.spec.trait
    return (
        {
            "selected": True,
            "kernel_name": r.spec.kernel_name(),
            "candidate": r.candidate.name,
            "spec_id": r.candidate.spec_id,
            "block_m": t.tile_m,
            "block_n": t.tile_n,
            "block_k": t.tile_k,
            "pipeline": tr.pipeline,
            "epilogue": tr.epilogue,
        },
        None,
    )


def structural_key(rec: dict):
    return (
        bool(rec.get("selected")),
        int(rec.get("block_m", 0)),
        int(rec.get("block_n", 0)),
        int(rec.get("block_k", 0)),
        str(rec.get("pipeline", "")),
        str(rec.get("epilogue", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cpp-jsonl", default=str(here / "cpp_picks.jsonl"))
    ap.add_argument("--shapes", default=str(here / "shapes.txt"))
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Index C++ picks by (M,N,K).
    cpp_by_shape: dict[tuple[int, int, int], dict] = {}
    with open(args.cpp_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cpp_by_shape[(rec["M"], rec["N"], rec["K"])] = rec

    shapes: list[tuple[int, int, int]] = []
    with open(args.shapes) as f:
        for line in f:
            h = line.find("#")
            if h != -1:
                line = line[:h]
            parts = line.split()
            if len(parts) < 3:
                continue
            shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))

    total = 0
    matches = 0
    divergences = []
    rows = []

    for M, N, K in shapes:
        cpp = cpp_by_shape.get((M, N, K))
        if cpp is None:
            divergences.append(
                {"shape": [M, N, K], "reason": "missing C++ pick for shape"}
            )
            total += 1
            continue

        py, py_err = python_pick(M, N, K, args.arch, args.dtype)
        total += 1

        # Normalize both to "selected?" + structural key. The C++ side reports
        # selected=false when no manifest supports the shape; the Python side
        # raises ValueError (py is None).
        cpp_selected = bool(cpp.get("selected"))
        py_selected = py is not None

        if not cpp_selected and not py_selected:
            # Both refuse -> agreement on "no kernel".
            matches += 1
            rows.append((M, N, K, "MATCH(no-kernel)", "", ""))
            continue

        if cpp_selected != py_selected:
            divergences.append(
                {
                    "shape": [M, N, K],
                    "reason": "selection-existence mismatch",
                    "cpp_selected": cpp_selected,
                    "py_selected": py_selected,
                    "py_error": py_err,
                    "cpp_kernel": cpp.get("kernel_name"),
                }
            )
            rows.append(
                (
                    M,
                    N,
                    K,
                    "DIVERGE(exists)",
                    cpp.get("kernel_name", ""),
                    (py["kernel_name"] if py else f"<reject: {py_err}>"),
                )
            )
            continue

        ck = structural_key(cpp)
        pk = structural_key(py)
        if ck == pk:
            matches += 1
            rows.append(
                (
                    M,
                    N,
                    K,
                    "MATCH",
                    f"{cpp['block_m']}x{cpp['block_n']}x{cpp['block_k']}/{cpp['pipeline']}/{cpp['epilogue']}",
                    f"{py['block_m']}x{py['block_n']}x{py['block_k']}/{py['pipeline']}/{py['epilogue']}",
                )
            )
        else:
            divergences.append(
                {
                    "shape": [M, N, K],
                    "reason": "structural-identity mismatch",
                    "cpp": {
                        k: cpp[k]
                        for k in (
                            "kernel_name",
                            "block_m",
                            "block_n",
                            "block_k",
                            "pipeline",
                            "epilogue",
                        )
                    },
                    "py": {
                        k: py[k]
                        for k in (
                            "kernel_name",
                            "candidate",
                            "spec_id",
                            "block_m",
                            "block_n",
                            "block_k",
                            "pipeline",
                            "epilogue",
                        )
                    },
                }
            )
            rows.append(
                (
                    M,
                    N,
                    K,
                    "DIVERGE",
                    f"{cpp['block_m']}x{cpp['block_n']}x{cpp['block_k']}/{cpp['pipeline']}/{cpp['epilogue']}",
                    f"{py['block_m']}x{py['block_n']}x{py['block_k']}/{py['pipeline']}/{py['epilogue']}",
                )
            )

    # Report.
    print("=" * 92)
    print(
        f"ck-dsl GEMM dispatcher C++<->Python selection parity "
        f"(arch={args.arch}, dtype={args.dtype})"
    )
    print(f"  scope: {args.dtype} RCR UniversalGemm")
    print("  identity compared: (block_m, block_n, block_k, pipeline, epilogue)")
    print("=" * 92)
    hdr = f"{'M':>6} {'N':>6} {'K':>6}  {'verdict':<16} {'C++ pick':<26} {'Python pick':<26}"
    print(hdr)
    print("-" * len(hdr))
    for M, N, K, verdict, cpp_s, py_s in rows:
        print(f"{M:>6} {N:>6} {K:>6}  {verdict:<16} {cpp_s:<26} {py_s:<26}")
    print("-" * len(hdr))
    rate = (matches / total * 100.0) if total else 0.0
    print(f"match rate: {matches}/{total} = {rate:.1f}%")

    if divergences:
        print("\nDIVERGENCES:")
        print(json.dumps(divergences, indent=2))
    else:
        print("\nNo divergences. C++ and Python pick the same kernel for every shape.")

    if args.verbose:
        # Show the naming-scheme difference explicitly on the first selected row.
        for M, N, K in shapes:
            cpp = cpp_by_shape.get((M, N, K))
            py, _ = python_pick(M, N, K, args.arch, args.dtype)
            if cpp and cpp.get("selected") and py:
                print(
                    "\nNOTE: raw kernel_name prefixes differ by construction (cosmetic):"
                )
                print(f"  C++   : {cpp['kernel_name']}")
                print(f"  Python: {py['kernel_name']}")
                break

    return 0 if not divergences else 1


if __name__ == "__main__":
    sys.exit(main())
