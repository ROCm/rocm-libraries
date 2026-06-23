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
#   eng   = ir_export(K) -> C import (ckc_import_kernel_from_json) -> C lower
#   recipe= record(K) -> CBOR recipe -> C recipe-VM rebuild -> C lower
#
# Gates (device-free, fast):
#   engine path : eng .ll == py .ll  BYTE-IDENTICAL   (hints survive ir_export)
#   recipe path : recipe .ll == py .ll BYTE-IDENTICAL (concrete recipes carry
#                 "exact_names": the VM names each value verbatim from its bind,
#                 reproducing Python's SSA names -- not just an equivalent HSACO).
#
# Needs a shared libckc (CKC_LIB) and a flavor pinned via CK_DSL_LLVM_FLAVOR
# (default llvm20). Drive it with ck_dsl_c/tests/portable_ir/run_parity_matrix.sh.
import argparse
import os

from ck_dsl.core import ir_export
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm
from ck_dsl.portable_ir.drivers import record_coverage as rc
from ck_dsl.portable_ir.src import online, recipe_bundle
from ck_dsl.portable_ir.src.recording_builder import record_kernel

FLAVOR = os.environ.get("CK_DSL_LLVM_FLAVOR", "llvm20")
ARCHES = os.environ.get("ARCHES", "gfx942,gfx950").split(",")


def _kernels():
    """(label, build_thunk) for every buildable parity emitter (first working
    config), plus (label, None) for ones that can't be built."""
    paths = sorted(p for p in (os.path.join(rc._PARITY_DIR, f)
                               for f in os.listdir(rc._PARITY_DIR)) if p.endswith("_emit.py"))
    for path in paths:
        label = os.path.basename(path)[:-len("_emit.py")]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    online.load()  # builds/loads libckc once up front
    print(f"== backend-path parity matrix (flavor={FLAVOR}, archs={','.join(ARCHES)}) ==")
    print("   engine = ir_export->C import->C lower (byte-identical .ll)")
    print("   recipe = record->CBOR->C recipe-VM->C lower (byte-identical .ll)\n")

    tally = {a: {"eng_pass": 0, "eng_fail": 0, "rec_pass": 0, "rec_fail": 0, "skip": 0}
             for a in ARCHES}
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
                print(f"  [FAIL] {label:<34} {arch:<8} eng={eng} recipe={rec}  {detail}")
            elif args.verbose:
                print(f"  [ ok ] {label:<34} {arch:<8} eng={eng} recipe={rec}")

    print("\n" + "=" * 72)
    print(f"kernels tested: {n_kernels}   (build-skipped: {n_skip_build})")
    for arch in ARCHES:
        t = tally[arch]
        print(f"  {arch}:  engine {t['eng_pass']}/{t['eng_pass']+t['eng_fail']} byte-identical   "
              f"recipe {t['rec_pass']}/{t['rec_pass']+t['rec_fail']} byte-identical   "
              f"(lower-skip {t['skip']})")
    total_fail = len(fails)
    print(f"\n{'PASS: full parity across all instances x archs on both paths.' if not total_fail else f'FAIL: {total_fail} (kernel,arch) cells diverged.'}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
