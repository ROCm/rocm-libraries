#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_arch_domain.py -- gate the arch-domain lane over the representative
# corpus.
#
# Both lowerers consult the committed arch-domain artifact every time a kernel
# demands an intrinsic declaration (see rocke.core.arch.domain). A measured
# negative -- the target provably cannot lower the intrinsic -- is refused in
# `error` mode unless a committed waiver covers the cell, and reported in `warn`
# mode. This tool lowers the whole corpus at every committed LLVM flavor with the
# lane in `warn` mode, harvests what it reports, and requires:
#
#   * no negative without a waiver: a kernel demanding an intrinsic its target
#     cannot lower is a kernel bug, and waivers are for wrong measurements;
#   * no stale waiver: a waived cell that no longer measures negative means the
#     column was fixed, and the waiver goes in the same change;
#   * when the rocke_engine extension is importable, the C++ engine in `error`
#     mode refuses exactly the cases the Python lowerer would, over the same
#     intrinsic -- one table only helps if both engines act on it alike.
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

from _flavor_gate import requires_newer_flavor

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform

Cell = tuple[str, str, str]  # (key, arch, flavor)


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


def _harvest(
    cases: list[dict], flavor: str
) -> tuple[dict[str, set[Cell]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Lower every case at ``flavor`` and collect what the lane reports.

    Returns the cells each case fired, keyed by case id, then two separate lists
    of cases that did not lower: the ones that *could not* at this flavor, and
    the ones that failed.

    The split matters because this gate lowers the corpus at *every* committed
    flavor, so an llvm23-gated instance always refuses on the llvm20/llvm22
    passes of a clean tree; counting those as failures would red the gate (and
    `run_all.py`) for cases never answerable at that vintage. A genuine lowering
    error is still red -- reported rather than raised only so one broken family
    cannot hide the state of the rest.

    The native lowerer is called directly, NOT the backend-dispatched public
    entry point: the C++ engine has no warning channel, so dispatching to it
    would report only through the post-pass and blur which engine said what.
    The C++ engine gets its own pass, :func:`_cpp_verdicts`.
    """
    from rocke.core.lower_llvm import ArchDomainWarning
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower_native

    fired: dict[str, set[Cell]] = {}
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
                fired.setdefault(case["case_id"], set()).add((w.key, w.arch, w.flavor))
    return fired, errors, unlowerable


def _cpp_verdicts(
    cases: list[dict], flavor: str, refuse: dict[str, set[Cell]]
) -> tuple[list[str], list[str]]:
    """Lower every case through the C++ engine in `error` mode and compare.

    ``refuse`` maps a case id to the unwaived negatives the Python lowerer hit,
    which is exactly what it would refuse over. The C++ engine must refuse those
    cases, over one of those cells, and lower every other case. Returns one line
    per disagreement, then one per case the engine failed on for some other
    reason -- not this gate's verdict to judge, so listed rather than counted.
    """
    from rocke.core.backend import BackendMismatch, _lower_via_cpp_engine
    from rocke.core.ir_serialize import serialize
    from rocke.core.lower_llvm import ArchDomainError

    out: list[str] = []
    unjudged: list[str] = []
    for case in cases:
        cid = case["case_id"]
        expected = refuse.get(cid, set())
        try:
            _lower_via_cpp_engine(serialize(case["build"]()), case["arch"], flavor)
        except ArchDomainError as exc:
            if (exc.key, exc.arch, exc.flavor) not in expected:
                out.append(f"{cid} @ {flavor}: cpp refused, python did not -- {exc}")
            continue
        except BackendMismatch as exc:
            # The backend raises this, not ArchDomainError, when the engine
            # refused a cell this checkout's table does not: an extension built
            # from other columns or waivers. That is the disagreement this gate
            # exists to find, so it is counted rather than filed below.
            out.append(f"{cid} @ {flavor}: cpp refused, python did not -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 -- not this gate's verdict
            if not requires_newer_flavor(exc):
                unjudged.append(f"{cid} @ {flavor}: {type(exc).__name__}: {exc}")
            continue
        if expected:
            out.append(f"{cid} @ {flavor}: python refuses, cpp lowered it")
    return out, unjudged


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE arch-domain gate")
    ap.add_argument(
        "--only", default="", help="restrict to case ids containing SUBSTR (comma-sep)"
    )
    ap.add_argument("--arch", default="", help="restrict to arches (comma-separated)")
    ap.add_argument(
        "--flavor",
        default="",
        help="restrict to LLVM flavors (comma-separated; default: all committed)",
    )
    ap.add_argument("--verbose", action="store_true", help="list waived reports too")
    args = ap.parse_args()

    _bootstrap_sys_path()
    from rocke.core.arch import domain as arch_domain
    from rocke.core.lower_llvm import LLVM_FLAVORS
    from rocke_ir_parity_harness import cases

    # Pin the lane to `warn` before lowering anything: an exported `off` would
    # give a green gate over a disabled lane, and `error` would stop each case
    # at its first negative instead of reporting all of them.
    os.environ[arch_domain.MODE_ENV] = arch_domain.MODE_WARN

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
    # returns None -- so measuring it would report a vacuous green.
    flavors = [f for f in wanted_flavors if arch_domain.load(f) is not None]
    skipped = [f for f in wanted_flavors if f not in flavors]

    try:
        import rocke_engine  # noqa: F401 -- only probing for the C++ pass
    except ImportError:
        cpp = False
    else:
        cpp = True

    print("== rocKE arch-domain lane ==")
    print(f"   cases   : {len(keep)}")
    print(f"   flavors : {', '.join(flavors) or '(none)'}")
    if skipped:
        print(f"   skipped : {', '.join(skipped)} (no committed column)")
    print(
        f"   engines : {'python + cpp' if cpp else 'python (rocke_engine not importable)'}"
    )
    if not flavors:
        print("\nRESULT: RED - no committed arch-domain column to gate against.")
        return 1

    waivers = arch_domain.waivers()
    fired: dict[Cell, set[str]] = {}
    errors: list[tuple[str, str]] = []
    unlowerable: list[tuple[str, str]] = []
    disagreements: list[str] = []
    unjudged: list[str] = []
    for flavor in flavors:
        by_case, errs, not_asked = _harvest(keep, flavor)
        errors.extend(errs)
        unlowerable.extend(not_asked)
        for cid, cells in by_case.items():
            for cell in cells:
                fired.setdefault(cell, set()).add(cid)
        if cpp:
            refuse = {
                cid: {c for c in cells if c not in waivers}
                for cid, cells in by_case.items()
            }
            os.environ[arch_domain.MODE_ENV] = arch_domain.MODE_ERROR
            try:
                differ, failed = _cpp_verdicts(keep, flavor, refuse)
            finally:
                os.environ[arch_domain.MODE_ENV] = arch_domain.MODE_WARN
            disagreements.extend(differ)
            unjudged.extend(failed)

    unwaived = {k: v for k, v in fired.items() if k not in waivers}
    reported = {k: v for k, v in fired.items() if k in waivers}
    # Judged against the columns, not the corpus: a waiver exists for a wrong
    # measurement, which a kernel outside the corpus may be the one to hit.
    stale = []
    for key, arch, flavor in sorted(waivers):
        table = arch_domain.load(flavor)
        cell = table.lookup(key, arch) if table is not None else None
        if cell is None or not arch_domain.is_negative(cell.status):
            stale.append((key, arch, flavor))

    print()
    print(f"   waivers   : {len(waivers)} on file, {len(reported)} reported here")
    print(f"   unwaived  : {len(unwaived)}")
    if unlowerable:
        print(f"   not asked : {len(unlowerable)} (instance needs a newer LLVM)")
    if unjudged:
        print(
            f"   not judged: {len(unjudged)} (the C++ engine failed for another reason)"
        )

    if args.verbose and reported:
        print("\n== waived negatives the corpus demands (reported, not refused) ==")
        for k in sorted(reported):
            print(f"  {k[0]}  [{k[1]} @ {k[2]}]  -- {waivers[k].reason}")
            for cid in sorted(reported[k]):
                print(f"      {cid}")

    if unlowerable and args.verbose:
        # Not a finding, so it stays behind --verbose: on a three-flavor sweep
        # every llvm23-gated instance lands here twice on every run.
        print("\n== not asked (instance declined this flavor) ==")
        for cid, msg in unlowerable:
            print(f"  {cid}\n    {msg}")

    if unjudged and args.verbose:
        print("\n== not judged (the C++ engine could not lower these at all) ==")
        for line in unjudged:
            print(f"  {line}")

    if errors:
        print("\n== cases that failed to lower ==")
        for cid, msg in errors:
            print(f"  {cid}\n    {msg}")

    if unwaived:
        print("\n== UNWAIVED negatives (refused in error mode) ==")
        for k in sorted(unwaived):
            print(f"\n  {k[0]}  [{k[1]} @ {k[2]}]")
            for cid in sorted(unwaived[k]):
                print(f"      {cid}")
        print(
            "\n  A kernel is demanding an intrinsic its target cannot lower. Fix "
            "the kernel. A waiver is for a wrong measurement, not a kernel bug."
        )

    if stale:
        print("\n== STALE waivers (the cell no longer measures negative) ==")
        for k in stale:
            print(
                f"  {k[0]}  [{k[1]} @ {k[2]}]\n    reason on file: {waivers[k].reason}"
            )
        print(
            "\n  Delete these from arch_domain_waivers.json in the change that "
            "fixed the column."
        )

    if disagreements:
        print("\n== the C++ engine disagrees with the Python lowerer ==")
        for line in disagreements:
            print(f"  {line}")

    print()
    if errors:
        print(f"RESULT: RED - {len(errors)} case(s) failed to lower.")
        return 1
    if unwaived:
        print(
            f"RESULT: RED - {len(unwaived)} intrinsic demand(s) the target "
            "provably cannot lower."
        )
        return 1
    if stale:
        print(f"RESULT: RED - {len(stale)} stale waiver(s); remove them.")
        return 1
    if disagreements:
        print(f"RESULT: RED - the engines disagree on {len(disagreements)} case(s).")
        return 1
    tail = f" ({len(unlowerable)} not asked)" if unlowerable else ""
    if reported:
        print(
            f"RESULT: GREEN - only waived cells fire ({len(reported)} reported).{tail}"
        )
    else:
        print(f"RESULT: GREEN - no kernel demands an unavailable intrinsic.{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
