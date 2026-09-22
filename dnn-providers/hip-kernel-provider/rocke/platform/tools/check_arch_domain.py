#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_arch_domain.py -- gate the arch-domain warning lane against a blessed
# allowlist.
#
# The lowerer consults the committed arch-domain artifact every time a kernel
# demands an intrinsic declaration, and warns when the table says the target
# provably cannot lower it (see rocke.core.arch.domain). This tool lowers the
# whole representative corpus at every committed LLVM flavor, harvests those
# warnings, and compares them against EXPECTED_WARNINGS below.
#
# Why this gate can run anywhere
# ------------------------------
# check_ir_validity.py needs the host's own clang, so it can only ever rule on
# one flavor and reports UNVALIDATED for the rest. Lowering is pure Python and
# the artifact is committed data, so this gate measures all three flavors on any
# host with no toolchain at all. That is what makes it usable in CI.
#
# What the two gates each catch
# -----------------------------
# They overlap on purpose and neither subsumes the other:
#
#   check_ir_validity   links the emitted module. Catches anything the backend
#                       rejects, named or not -- but only at the host's flavor,
#                       and its diagnostic is whatever the backend chose to say.
#   check_arch_domain   consults measured data at declaration-demand time.
#                       Catches the specific defect "this target cannot lower
#                       this intrinsic" at every flavor, and names the intrinsic.
#
# Usage:
#   python rocke/platform/tools/check_arch_domain.py
#   python rocke/platform/tools/check_arch_domain.py --verbose
#   python rocke/platform/tools/check_arch_domain.py --only attention --flavor llvm20

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Same layout probing as check_ir_validity, so this runs from a checkout
    without an external PYTHONPATH. The corpus lives in the test tree and its
    attention/KDA families are the one sanctioned platform -> library reach, so
    `kernels`/`builders` must resolve too."""
    for path in (ROCKE / "python", ROCKE / "tests" / "instances"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for lib_root in (ROCKE / "tests" / "library", ROCKE.parent / "library"):
        if (lib_root / "kernels").is_dir():
            if str(lib_root) not in sys.path:
                sys.path.insert(0, str(lib_root))
            break


# Every (declaration key, target, flavor) the corpus is currently allowed to
# demand against a target the artifact says cannot lower it, with the reason and
# the work that owns the fix.
#
# Same rule as KNOWN_BAD in check_ir_validity.py and KNOWN_VIOLATIONS in
# library/tests/test_library_layering.py: **this list only ever SHRINKS**. An
# entry that stops firing is itself a failure, so a fix cannot leave dead weight
# behind, and a new entry is a kernel bug to fix rather than a line to add.
EXPECTED_WARNINGS: dict[tuple[str, str, str], str] = {}

for _flavor in ("llvm20", "llvm22", "llvm23"):
    for _key in ("ds.read.tr16.b64", "mfma.f32.16x16x32.bf16", "mfma.f32.16x16x32.f16"):
        EXPECTED_WARNINGS[(_key, "gfx942", _flavor)] = (
            "attention 3d d128/b64 requests a CDNA4 path on a CDNA3 target; the "
            "same two cases check_ir_validity.py lists in KNOWN_BAD, found here "
            "at declaration-demand time with the intrinsic named"
        )
del _flavor, _key

# `wmma.i32.16x16x16.iu4`/`.iu8` on gfx11-generic and gfx1151 at llvm20 used to
# be listed here too, as a documented false negative in the artifact rather than
# a kernel defect. Generator 5 builds each probe from the declare `opt` resolves,
# so `immarg` now comes from the LLVM being measured instead of the hand-written
# decl table, the probe passes constants where the intrinsic demands them, and
# those four cells measure `ok`. The entries are deleted rather than kept,
# because an allowlist that outlives its defect is the failure this gate reports
# as STALE.


def _harvest(
    cases: list[dict], flavor: str
) -> tuple[dict[tuple[str, str, str], set[str]], list[tuple[str, str]]]:
    """Lower every case at ``flavor`` and collect the arch-domain warnings.

    Returns the warnings keyed by (key, arch, flavor) -> the case ids that
    raised them, plus any case that failed to lower at all. A lowering failure
    is reported rather than raised: one broken family must not hide the warning
    state of the other thirty-nine.
    """
    from rocke.core.lower_llvm import ArchDomainWarning, lower_kernel_to_llvm

    found: dict[tuple[str, str, str], set[str]] = {}
    errors: list[tuple[str, str]] = []
    for case in cases:
        with warnings.catch_warnings(record=True) as caught:
            # `always`, not the default `once`: the default dedupes per code
            # location, which would silently drop the second case that demands
            # the same intrinsic from the same line.
            warnings.simplefilter("always")
            try:
                lower_kernel_to_llvm(
                    case["build"](), arch=case["arch"], llvm_flavor=flavor
                )
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                errors.append((case["case_id"], f"{type(exc).__name__}: {exc}"))
                continue
        for item in caught:
            w = item.message
            if isinstance(w, ArchDomainWarning):
                found.setdefault((w.key, w.arch, w.flavor), set()).add(case["case_id"])
    return found, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE arch-domain warning gate")
    ap.add_argument(
        "--only", default="", help="restrict to case ids containing SUBSTR (comma-sep)"
    )
    ap.add_argument("--arch", default="", help="restrict to arches (comma-separated)")
    ap.add_argument(
        "--flavor",
        default="",
        help="restrict to LLVM flavors (comma-separated; default: all committed)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="list every expected warning too"
    )
    args = ap.parse_args()

    _bootstrap_sys_path()
    from rocke.core.arch import domain as arch_domain
    from rocke.core.lower_llvm import LLVM_FLAVORS
    from rocke_ir_parity_harness import cases

    keep = list(cases())
    if args.only:
        wanted = [s for s in args.only.split(",") if s]
        keep = [c for c in keep if any(w in c["case_id"] for w in wanted)]
    if args.arch:
        arches = {s for s in args.arch.split(",") if s}
        keep = [c for c in keep if c["arch"] in arches]
    if not keep:
        print("FATAL: selection matched no cases", file=sys.stderr)
        return 1

    wanted_flavors = (
        [s for s in args.flavor.split(",") if s] if args.flavor else list(LLVM_FLAVORS)
    )
    # A flavor with no committed column has nothing to say -- every lookup
    # returns None and the lane is silent -- so measuring it would report a
    # vacuous green. Skip it and say so.
    flavors = [f for f in wanted_flavors if arch_domain.load(f) is not None]
    skipped = [f for f in wanted_flavors if f not in flavors]

    print("== rocKE arch-domain lane ==")
    print(f"   cases   : {len(keep)}")
    print(f"   flavors : {', '.join(flavors) or '(none)'}")
    if skipped:
        print(f"   skipped : {', '.join(skipped)} (no committed column)")
    if not flavors:
        print("\nRESULT: RED - no committed arch-domain column to gate against.")
        return 1

    found: dict[tuple[str, str, str], set[str]] = {}
    errors: list[tuple[str, str]] = []
    for flavor in flavors:
        got, errs = _harvest(keep, flavor)
        found.update(got)
        errors.extend(errs)

    # Only entries whose flavor was actually measured can be judged stale. A
    # --flavor or --arch selection narrows what ran, and an entry that did not
    # run is not evidence of anything.
    selected_arches = {c["arch"] for c in keep}
    in_scope = {
        k
        for k in EXPECTED_WARNINGS
        if k[2] in flavors and k[1] in selected_arches and not args.only
    }
    unexpected = {k: v for k, v in found.items() if k not in EXPECTED_WARNINGS}
    stale = sorted(in_scope - set(found))

    print()
    print(f"   expected  : {len(found) - len(unexpected)} / {len(in_scope)} in scope")
    print(f"   unexpected: {len(unexpected)}")

    if args.verbose:
        print("\n== expected warnings that fired ==")
        for k in sorted(set(found) & set(EXPECTED_WARNINGS)):
            print(f"  {k[0]}  [{k[1]} @ {k[2]}]")
            for cid in sorted(found[k]):
                print(f"      {cid}")

    if errors:
        print("\n== cases that failed to lower ==")
        for cid, msg in errors:
            print(f"  {cid}\n    {msg}")

    if unexpected:
        print("\n== UNEXPECTED warnings ==")
        for k in sorted(unexpected):
            print(f"\n  {k[0]}  [{k[1]} @ {k[2]}]")
            for cid in sorted(unexpected[k]):
                print(f"      {cid}")
        print(
            "\n  A kernel is demanding an intrinsic its target cannot lower. Fix "
            "the kernel; do not add to EXPECTED_WARNINGS."
        )

    if stale:
        print("\n== STALE EXPECTED_WARNINGS entries (these no longer fire) ==")
        for k in stale:
            print(
                f"  {k[0]}  [{k[1]} @ {k[2]}]\n    reason on file: "
                f"{EXPECTED_WARNINGS[k]}"
            )
        print(
            "\n  EXPECTED_WARNINGS only shrinks. Delete these entries in the "
            "change that fixed them."
        )

    print()
    if errors:
        print(f"RESULT: RED - {len(errors)} case(s) failed to lower.")
        return 1
    if unexpected:
        print(
            f"RESULT: RED - {len(unexpected)} intrinsic demand(s) the target "
            "provably cannot lower."
        )
        return 1
    if stale:
        print("RESULT: RED - EXPECTED_WARNINGS is stale; remove the entries above.")
        return 1
    if found:
        print(
            f"RESULT: GREEN - the lane fires exactly the {len(found)} documented "
            "warning(s)."
        )
    else:
        print("RESULT: GREEN - no kernel demands an unavailable intrinsic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
