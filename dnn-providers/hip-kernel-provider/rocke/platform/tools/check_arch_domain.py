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
# Lowering is pure Python and the artifact is committed data, so this gate needs
# no toolchain and rules on every flavor on any host. It overlaps
# check_ir_validity without either subsuming the other:
#
#   check_ir_validity   links the emitted module: catches anything the backend
#                       rejects, but only at the host's flavor.
#   check_arch_domain   consults measured data at declaration-demand time:
#                       catches "this target cannot lower this intrinsic" at
#                       every flavor, and names the intrinsic.
#
# Usage:
#   python rocke/platform/tools/check_arch_domain.py
#   python rocke/platform/tools/check_arch_domain.py --verbose
#   python rocke/platform/tools/check_arch_domain.py --only attention --flavor llvm20

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

from _flavor_gate import requires_newer_flavor

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Same layout probing as check_ir_validity, so this runs from a checkout
    without an external PYTHONPATH. The corpus's attention/KDA families are the
    one sanctioned platform -> library reach, so `kernels`/`builders` must
    resolve too."""
    for path in (ROCKE / "python", ROCKE / "tests" / "instances"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for lib_root in (ROCKE / "tests" / "library", ROCKE.parent / "library"):
        if (lib_root / "kernels").is_dir():
            if str(lib_root) not in sys.path:
                sys.path.insert(0, str(lib_root))
            break


class Expected(NamedTuple):
    """One allowlisted demand: why it is allowed, and by which corpus cases."""

    reason: str
    cases: frozenset[str]


# Every (declaration key, target, flavor) the corpus is currently allowed to
# demand against a target the artifact says cannot lower it, with the reason
# and the corpus cases that may demand it. Naming the cases is what keeps the
# list from widening in silence: a different case that starts demanding the
# same intrinsic is a new finding, and fixing one of two cases makes the entry
# stale rather than leaving it to cover whatever comes next.
#
# Name the defect, not a tracker ID: the tracking reference belongs in the
# commit that adds the entry, and an entry nobody owns cannot burn down.
#
# Same rule as KNOWN_BAD in check_ir_validity.py and KNOWN_VIOLATIONS in
# library/tests/test_library_layering.py: **this list only ever SHRINKS**. An
# entry, or one of its cases, that stops firing is itself a failure, and a new
# entry is a kernel bug to fix rather than a line to add.
EXPECTED_WARNINGS: dict[tuple[str, str, str], Expected] = {}

# `wmma.i32.16x16x16.iu4`/`.iu8` on gfx11-generic and gfx1151 at llvm20 were
# listed here as artifact false negatives. Generator 5 builds each probe from
# the declare `opt` resolves, so `immarg` comes from the LLVM being measured and
# those four cells now measure `ok`; the entries are deleted rather than kept.


def judge(
    found: dict[tuple[str, str, str], set[str]],
    expected: dict[tuple[str, str, str], Expected],
    lowered: dict[str, set[str]],
    known_cases: set[str],
    flavors: list[str],
) -> tuple[
    dict[tuple[str, str, str], set[str]],
    list[tuple[tuple[str, str, str], str, str]],
]:
    """Compare what fired with the allowlist.

    ``found`` maps (key, arch, flavor) to the case ids that demanded it,
    ``lowered`` maps a flavor to the case ids that lowered there, and
    ``known_cases`` is every case id in the corpus.

    Returns ``(unexpected, stale)``. Unexpected: a triple the allowlist does not
    have, or, for one it does have, only the cases it does not name. Stale: an
    allowlisted case that lowered at the flavor and no longer demands the
    intrinsic, or that the corpus no longer has. A case that did not run -- a
    filtered selection, or an instance that declines the flavor -- says nothing,
    so it is never judged.
    """
    unexpected: dict[tuple[str, str, str], set[str]] = {}
    for triple, case_ids in found.items():
        allowed = expected[triple].cases if triple in expected else frozenset()
        extra = case_ids - allowed
        if extra:
            unexpected[triple] = set(extra)
    stale: list[tuple[tuple[str, str, str], str, str]] = []
    for triple, entry in sorted(expected.items()):
        flavor = triple[2]
        if flavor not in flavors:
            continue
        for case_id in sorted(entry.cases):
            if case_id not in known_cases:
                stale.append((triple, case_id, "the corpus has no such case"))
            elif case_id in lowered.get(flavor, set()) and case_id not in found.get(
                triple, set()
            ):
                stale.append((triple, case_id, "lowered, and no longer demands it"))
    return unexpected, stale


def _harvest(cases: list[dict], flavor: str) -> tuple[
    dict[tuple[str, str, str], set[str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    set[str],
]:
    """Lower every case at ``flavor`` and collect the arch-domain warnings.

    Returns the warnings keyed by (key, arch, flavor) -> the case ids that
    raised them, then two separate lists of cases that did not lower: the ones
    that *could not* at this flavor, and the ones that failed. The last item is
    the ids that did lower, which is what lets the gate tell an allowlisted case
    that stopped demanding an intrinsic from one that never ran.

    The split matters because this gate lowers the corpus at *every* committed
    flavor, so an llvm23-gated instance always refuses on the llvm20/llvm22
    passes of a clean tree; counting those as failures would red the gate (and
    `run_all.py`) for cases never answerable at that vintage. A genuine lowering
    error is still red -- reported rather than raised only so one broken family
    cannot hide the warning state of the rest.

    The native lowerer is called directly, NOT the backend-dispatched public
    entry point: the default backend is the C++ engine and only the Python
    lowerer consults the arch-domain table, so dispatching would harvest nothing
    and call every EXPECTED_WARNINGS entry stale on any host with the extension.
    """
    from rocke.core.lower_llvm import ArchDomainWarning
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower_native

    found: dict[tuple[str, str, str], set[str]] = {}
    errors: list[tuple[str, str]] = []
    unlowerable: list[tuple[str, str]] = []
    lowered: set[str] = set()
    for case in cases:
        with warnings.catch_warnings(record=True) as caught:
            # `always`, not the default `once`: `once` dedupes per code
            # location, dropping the second case that demands the same
            # intrinsic from the same line.
            warnings.simplefilter("always")
            try:
                lower_native(case["build"](), arch=case["arch"], llvm_flavor=flavor)
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                bucket = unlowerable if requires_newer_flavor(exc) else errors
                bucket.append((case["case_id"], f"{type(exc).__name__}: {exc}"))
                continue
        lowered.add(case["case_id"])
        for item in caught:
            w = item.message
            if isinstance(w, ArchDomainWarning):
                found.setdefault((w.key, w.arch, w.flavor), set()).add(case["case_id"])
    return found, errors, unlowerable, lowered


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

    # Pin the lane on before anything imports the lowerer: an exported
    # ROCKE_ARCH_DOMAIN=off would otherwise give a green gate over a disabled
    # lane. See _harvest for the matching backend pin.
    os.environ["ROCKE_ARCH_DOMAIN"] = "warn"

    _bootstrap_sys_path()
    from rocke.core.arch import domain as arch_domain
    from rocke.core.lower_llvm import LLVM_FLAVORS
    from rocke_ir_parity_harness import cases

    keep = list(cases())
    known_cases = {c["case_id"] for c in keep}
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
    # returns None -- so measuring it would report a vacuous green.
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
    unlowerable: list[tuple[str, str]] = []
    lowered: dict[str, set[str]] = {}
    for flavor in flavors:
        got, errs, skipped_cases, ran = _harvest(keep, flavor)
        found.update(got)
        errors.extend(errs)
        unlowerable.extend(skipped_cases)
        lowered[flavor] = ran

    # What ran bounds what can be judged: a --flavor, --arch or --only selection
    # narrows it, and an allowlisted case that did not run is neither confirmed
    # nor stale.
    unexpected, stale = judge(found, EXPECTED_WARNINGS, lowered, known_cases, flavors)
    in_scope = {
        k
        for k, entry in EXPECTED_WARNINGS.items()
        if k[2] in flavors and entry.cases & lowered.get(k[2], set())
    }

    print()
    print(f"   expected  : {len(in_scope & set(found))} / {len(in_scope)} in scope")
    print(f"   unexpected: {len(unexpected)}")
    if stale:
        print(f"   stale     : {len(stale)}")
    if unlowerable:
        print(f"   not asked : {len(unlowerable)} (instance needs a newer LLVM)")

    if args.verbose:
        print("\n== expected warnings that fired ==")
        for k in sorted(set(found) & set(EXPECTED_WARNINGS)):
            print(f"  {k[0]}  [{k[1]} @ {k[2]}]")
            for cid in sorted(found[k]):
                print(f"      {cid}")

    if unlowerable and args.verbose:
        # Not a finding, so it stays behind --verbose: on a three-flavor sweep
        # every llvm23-gated instance lands here twice on every run.
        print("\n== not asked (instance declined this flavor) ==")
        for cid, msg in unlowerable:
            print(f"  {cid}\n    {msg}")

    if errors:
        print("\n== cases that failed to lower ==")
        for cid, msg in errors:
            print(f"  {cid}\n    {msg}")

    if unexpected:
        print("\n== UNEXPECTED warnings ==")
        for k in sorted(unexpected):
            print(f"\n  {k[0]}  [{k[1]} @ {k[2]}]")
            if k in EXPECTED_WARNINGS:
                print("      allowlisted, but not for these cases:")
            for cid in sorted(unexpected[k]):
                print(f"      {cid}")
        print(
            "\n  A kernel is demanding an intrinsic its target cannot lower. Fix "
            "the kernel; do not add to EXPECTED_WARNINGS."
        )

    if stale:
        print("\n== STALE EXPECTED_WARNINGS entries (a case no longer demands it) ==")
        for k, case_id, why in stale:
            print(
                f"  {k[0]}  [{k[1]} @ {k[2]}]\n      {case_id}: {why}\n"
                f"    reason on file: {EXPECTED_WARNINGS[k].reason}"
            )
        print(
            "\n  EXPECTED_WARNINGS only shrinks. Delete these cases, and an entry "
            "with none left, in the change that fixed them."
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
    tail = f" ({len(unlowerable)} not asked)" if unlowerable else ""
    if found:
        print(
            f"RESULT: GREEN - the lane fires exactly the {len(found)} documented "
            f"warning(s).{tail}"
        )
    else:
        print(f"RESULT: GREEN - no kernel demands an unavailable intrinsic.{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
