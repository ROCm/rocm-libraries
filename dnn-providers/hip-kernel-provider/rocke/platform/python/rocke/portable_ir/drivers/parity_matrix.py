#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# parity_matrix.py -- backend-path parity across ALL kernel instances x archs.
#
# For every production kernel the parity emitters can build (enumerated exactly
# like record_coverage), and for each target arch, it lowers the SAME KernelDef
# three ways and checks agreement with ONE pinned LLVM flavor on every path:
#
#   py    = native Python lowerer (the oracle)
#   eng   = ir_export(K) -> C import (rocke_import_kernel_from_json) -> C lower
#   recipe= record(K) -> CBOR recipe -> C recipe-VM rebuild -> C lower
#
# Gates (device-free, fast):
#   engine path : eng .ll == py .ll  BYTE-IDENTICAL   (hints survive ir_export)
#   recipe path : recipe .ll == py .ll BYTE-IDENTICAL (concrete recipes carry
#                 "exact_names": the VM names each value verbatim from its bind,
#                 reproducing Python's SSA names -- not just an equivalent HSACO).
#
# Needs a shared librocke (ROCKE_ONLINE_LIB) and a flavor pinned via
# ROCKE_LLVM_FLAVOR (default llvm20).
#
#   python3 -m rocke.portable_ir.drivers.parity_matrix [--verbose]
import argparse
import os
from typing import Tuple

from rocke.core import ir_export
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.drivers import record_coverage as rc
from rocke.portable_ir.src import online, recipe_bundle
from rocke.portable_ir.src.recording_builder import record_kernel

# This driver needs no comgr -- it compares .ll text -- but the flavor it pins
# still has to be the one the shipping path uses, or the precise check (SSA names,
# which HSACO comparison cannot see) is validating a different LLVM generation
# from the one that produces the artifact. So: resolve from the installed comgr
# when there is one, and fall back to the historical default when there is not,
# which keeps the driver usable on a machine with no ROCm at all.
_FALLBACK_FLAVOR = "llvm20"
FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "auto")
ARCHES = os.environ.get("ARCHES", "gfx942,gfx950").split(",")


def _auto_flavor() -> Tuple[str, str]:
    """-> (flavor, where it came from), the second for the log line."""
    try:
        from rocke.core.lower_llvm import _flavor_for_rocm
        from rocke.runtime.comgr import resolved_lib_rocm_version

        ver = resolved_lib_rocm_version()
        if ver:
            return _flavor_for_rocm(*ver), f"ROCm {'.'.join(map(str, ver))} comgr"
    except Exception:  # noqa: BLE001 - no comgr is a normal state here
        pass
    return _FALLBACK_FLAVOR, "default, no comgr found"


def _kernels():
    """(label, build_thunk) for every buildable parity emitter (first working
    config), plus (label, None) for ones that can't be built."""
    paths = sorted(
        os.path.join(d, f)
        for d in rc._PARITY_DIRS
        for f in os.listdir(d)
        if f.endswith("_emit.py")
    )
    for path in paths:
        label = os.path.basename(path)[: -len("_emit.py")]
        try:
            mod = rc._load_module(path)
        except Exception as e:  # noqa: BLE001
            yield label, None, f"import: {type(e).__name__}"
            continue
        thunk = None
        for t in rc._kernel_thunks(mod):
            try:
                t()
            except Exception:  # noqa: BLE001
                continue
            thunk = t
            break
        yield label, thunk, None if thunk else "no buildable config"


def _cell(thunk, arch):
    """-> (engine_status, recipe_status, detail). status in PASS/FAIL/SKIP."""
    try:
        k = thunk()
        py = lower_kernel_to_llvm(k, llvm_flavor=FLAVOR, arch=arch)
    except Exception as e:  # noqa: BLE001
        return "SKIP", "SKIP", f"py-lower: {type(e).__name__}"

    # engine path: ir_export -> C import -> C lower
    eng_st, eng_detail = "PASS", ""
    try:
        eng, _ = online.ir_json_to_llvm(ir_export.export_kernel_ir_json(k), arch=arch)
        if eng != py:
            eng_st, eng_detail = "FAIL", _first_diff(py, eng)
    except Exception as e:  # noqa: BLE001
        eng_st, eng_detail = "FAIL", f"eng: {type(e).__name__}: {e}"[:80]

    # recipe path: record -> CBOR -> C recipe-VM -> C lower (BYTE-IDENTICAL .ll;
    # concrete recipes carry exact_names so the VM reproduces Python's SSA names)
    rec_st, rec_detail = "PASS", ""
    try:
        _, recipe = record_kernel(thunk)
        vm, _ = online.recipe_cbor_to_llvm(recipe_bundle.cbor_encode(recipe), arch=arch)
        if vm != py:
            rec_st, rec_detail = "FAIL", _first_diff(py, vm)
    except NotImplementedError as e:
        rec_st, rec_detail = "FAIL", f"recorder gap: {e}"[:80]
    except Exception as e:  # noqa: BLE001
        rec_st, rec_detail = "FAIL", f"recipe: {type(e).__name__}: {e}"[:80]

    return eng_st, rec_st, eng_detail or rec_detail


def _first_diff(a, b):
    import difflib

    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""):
        if line[:1] in "+-" and not line.startswith(("+++", "---")):
            return line[:96]
    return ""


def main() -> int:
    global FLAVOR, ARCHES
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--flavor",
        default=FLAVOR,
        help="LLVM flavor to pin on every path; 'auto' takes it from the "
        "installed comgr so this gate and the HSACO gates agree",
    )
    ap.add_argument(
        "--arches", default=",".join(ARCHES), help="comma-separated target arches"
    )
    args = ap.parse_args()

    origin = "requested"
    if args.flavor == "auto":
        FLAVOR, origin = _auto_flavor()
    else:
        FLAVOR = args.flavor
    ARCHES = args.arches.split(",")
    # Pin the flavor for the C++ engine too. Python takes it as an argument, but
    # the C side resolves ROCKE_LLVM_FLAVOR_AUTO from the environment / the
    # installed ROCm; without this the two engines can silently lower at
    # different flavors and every cell "fails" on the datalayout line.
    os.environ["ROCKE_LLVM_FLAVOR"] = FLAVOR

    online.load()  # builds/loads librocke once up front
    print(
        f"== backend-path parity matrix (flavor={FLAVOR} [{origin}], "
        f"archs={','.join(ARCHES)}) =="
    )
    print("   engine = ir_export->C import->C lower (byte-identical .ll)")
    print("   recipe = record->CBOR->C recipe-VM->C lower (byte-identical .ll)\n")

    tally = {
        a: {"eng_pass": 0, "eng_fail": 0, "rec_pass": 0, "rec_fail": 0, "skip": 0}
        for a in ARCHES
    }
    fails = []
    n_kernels = n_skip_build = 0
    for label, thunk, why in _kernels():
        if thunk is None:
            n_skip_build += 1
            if args.verbose:
                print(f"  [build-skip] {label:<40} {why}")
            continue
        n_kernels += 1
        for arch in ARCHES:
            eng, rec, detail = _cell(thunk, arch)
            t = tally[arch]
            if eng == "SKIP":
                t["skip"] += 1
            else:
                t["eng_pass" if eng == "PASS" else "eng_fail"] += 1
                t["rec_pass" if rec == "PASS" else "rec_fail"] += 1
            if eng == "FAIL" or rec == "FAIL":
                fails.append((label, arch, eng, rec, detail))
                print(
                    f"  [FAIL] {label:<34} {arch:<8} eng={eng} recipe={rec}  {detail}"
                )
            elif args.verbose:
                print(f"  [ ok ] {label:<34} {arch:<8} eng={eng} recipe={rec}")

    print("\n" + "=" * 72)
    print(f"kernels tested: {n_kernels}   (build-skipped: {n_skip_build})")
    for arch in ARCHES:
        t = tally[arch]
        print(
            f"  {arch}:  engine {t['eng_pass']}/{t['eng_pass']+t['eng_fail']} byte-identical   "
            f"recipe {t['rec_pass']}/{t['rec_pass']+t['rec_fail']} byte-identical   "
            f"(lower-skip {t['skip']})"
        )
    total_fail = len(fails)
    print(
        f"\n{'PASS: full parity across all instances x archs on both paths.' if not total_fail else f'FAIL: {total_fail} (kernel,arch) cells diverged.'}"
    )
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
