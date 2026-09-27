#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_ir_validity.py -- prove the IR the lowerer emits is actually legal.
#
# The byte-identity gate proves the two engines emit the SAME bytes. It says
# nothing about whether those bytes are valid LLVM IR for the target. This tool
# answers that second question: lower every case in the representative corpus
# and push each module all the way through the AMDGPU toolchain.
#
# Why the oracle is a LINK and not a verify or a codegen
# -----------------------------------------------------
# A declare for an intrinsic that does not exist (e.g. the fictional
# llvm.amdgcn.mfma.f32.16x16x128.fp4 in the decl table) is accepted by
# `opt -passes=verify`, and is accepted by `clang -S` as well -- the backend
# quietly treats the unknown llvm.* name as an ordinary external function and
# emits a GOT-relative call to it. The undefined symbol only surfaces when the
# relocatable is linked into an executable. So:
#
#   L0  opt -passes=verify   type / arity / structural errors      (cheap, optional)
#   L1  clang -> hsaco       everything above + nonexistent        (the gate)
#                            intrinsics, ISel failures, backend
#                            fatal errors, bad inline asm
#
# Why every compile runs in a subprocess
# --------------------------------------
# A backend failure is a report_fatal_error, not an exception: it takes the
# whole process down with it ("LLVM ERROR: ..." then exit). Running the compile
# in-process -- via comgr's build_hsaco_from_llvm_ir, which is otherwise faster
# -- means one bad module kills the run and no report is produced. Process
# isolation is a correctness requirement here, not a performance choice, and it
# is also what makes the per-module diagnostic free (clang's stderr).
#
# Only the flavor the host toolchain implements can be validated. Modules can be
# lowered at any flavor, but there is no local compiler for the others, so they
# are reported UNVALIDATED rather than green.
#
# All paths are derived relative to this file; LLVM tools are located via the
# ROCm install rocke already resolves, never a hardcoded prefix.
#
# Usage:
#   python rocke/platform/tools/check_ir_validity.py
#   python rocke/platform/tools/check_ir_validity.py --only attention --verbose
#   python rocke/platform/tools/check_ir_validity.py --arch gfx942 --keep-ir DIR

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Same layout probing as tests/conftest.py, so this runs from a checkout
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


# Cases that do not yet compile, each with the reason and the work that owns the
# fix. Mirrors the KNOWN_VIOLATIONS convention in
# library/tests/test_library_layering.py, and carries the same rule: this list
# only ever SHRINKS. An entry that starts passing is itself a failure (see
# STALE below) so a fix cannot silently leave dead weight behind.
KNOWN_BAD: dict[str, str] = {
    "attention/gfx942/3d_bf16_d128_b64": (
        "LLVM23 backend fatal: 'Do not know how to expand this operator's "
        "operand'; attention 3d on gfx942 only"
    ),
    "attention/gfx942/3d_fp16_d128_b64": (
        "LLVM23 backend fatal: 'Do not know how to expand this operator's "
        "operand'; attention 3d on gfx942 only"
    ),
}


def _llvm_tool(name: str) -> str | None:
    """Locate an LLVM tool, preferring the ROCm install rocke itself loads.

    PATH is checked LAST on purpose. A ROCm install ships opt/clang under
    <root>/llvm/bin but does not necessarily put them on PATH -- and the tool
    that IS on PATH may belong to a different LLVM vintage than the comgr rocke
    lowers against, which would validate the wrong toolchain and pass. Note also
    that the set of shipped tools varies by release (llc is absent from recent
    ROCm), so callers must tolerate a None.
    """
    exe = name + (".exe" if os.name == "nt" else "")
    roots: list[Path] = []

    def _is_file(path: Path) -> bool:
        # A candidate prefix can be anything the environment names, including a
        # directory the caller cannot stat. An unreadable candidate is simply
        # not the tool -- it must not abort the search.
        try:
            return path.is_file()
        except OSError:
            return False

    for env_var in ("ROCKE_LLVM_BIN",):
        raw = os.environ.get(env_var)
        if raw:
            cand = Path(raw) / exe
            if _is_file(cand):
                return str(cand)

    # The ROCm root under the comgr rocke resolved. The lib may sit at
    # <root>/lib or at a versioned <root>/core-X.Y/lib, so walk a few ancestors.
    try:
        from rocke.runtime.comgr import resolved_lib_path

        lib = resolved_lib_path()
    except Exception:
        lib = None
    if lib:
        libp = Path(lib).resolve()
        roots.extend(libp.parents[i] for i in range(min(4, len(libp.parents))))

    for env_var in ("ROCM_PATH", "ROCM_HOME"):
        raw = os.environ.get(env_var)
        if raw:
            roots.append(Path(raw))

    for root in roots:
        cand = root / "llvm" / "bin" / exe
        if _is_file(cand):
            return str(cand)

    import shutil

    return shutil.which(name)


def _dump_corpus(out_dir: Path, flavor: str, keep: list) -> dict[str, tuple[Path, str]]:
    """Lower every selected case to a .ll on disk. Returns case_id -> (path, arch)."""
    from rocke_ir_parity_harness import lower_case, safe

    modules: dict[str, tuple[Path, str]] = {}
    for case in keep:
        cid = case["case_id"]
        _rec, llvm = lower_case(case, flavor)
        path = out_dir / safe(cid)
        path.write_text(llvm)
        modules[cid] = (path, case["arch"])
    return modules


def _verify(opt: str, path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [opt, "-passes=verify", "-disable-output", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def _compile(clang: str, path: Path, arch: str, out: Path) -> tuple[bool, str]:
    """Compile AND LINK to a loadable hsaco. The link is the point -- see the
    module docstring for why stopping at -S would let a nonexistent intrinsic
    through."""
    proc = subprocess.run(
        [
            clang,
            "-x",
            "ir",
            "-O3",
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={arch}",
            "-o",
            str(out),
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def _first_diagnostic(text: str) -> str:
    """The one line worth printing in the summary table."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "error" in low or "llvm error" in low or "undefined symbol" in low:
            return s
    return text.splitlines()[0].strip() if text.strip() else "(no diagnostic)"


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE emitted-IR validity gate")
    ap.add_argument(
        "--only", default="", help="restrict to case ids containing SUBSTR (comma-sep)"
    )
    ap.add_argument("--arch", default="", help="restrict to arches (comma-separated)")
    ap.add_argument(
        "--flavor",
        default="",
        help="llvm flavor to lower at (default: the host's own). A non-native "
        "flavor has no local compiler and needs --force.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="lower at a non-native --flavor anyway (diagnostic; the result "
        "says nothing about that flavor's real toolchain)",
    )
    ap.add_argument("--jobs", type=int, default=0, help="parallel compiles")
    ap.add_argument(
        "--no-verify", action="store_true", help="skip the L0 opt -passes=verify lane"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat a missing toolchain as a failure instead of a skip (CI)",
    )
    ap.add_argument("--keep-ir", default="", help="write the lowered .ll files here")
    ap.add_argument(
        "--verbose", action="store_true", help="full diagnostic per failure"
    )
    ap.add_argument("--list", action="store_true", help="list the corpus and exit")
    args = ap.parse_args()

    _bootstrap_sys_path()
    from rocke_ir_parity_harness import cases, current_flavor

    native = current_flavor()
    flavor = args.flavor or native

    keep = list(cases())
    if args.only:
        wanted = [s for s in args.only.split(",") if s]
        keep = [c for c in keep if any(w in c["case_id"] for w in wanted)]
    if args.arch:
        arches = {s for s in args.arch.split(",") if s}
        keep = [c for c in keep if c["arch"] in arches]

    if args.list:
        for case in keep:
            print(f"{case['arch']:<16}{case['family']:<22}{case['case_id']}")
        print(f"\n{len(keep)} cases")
        return 0

    if not keep:
        print("FATAL: selection matched no cases", file=sys.stderr)
        return 1

    print("== rocKE emitted-IR validity ==")
    print(f"   cases  : {len(keep)}")
    print(f"   flavor : {flavor}" + ("" if flavor == native else f" (host: {native})"))

    if flavor != native and not args.force:
        print(
            f"\nRESULT: UNVALIDATED - this host's toolchain implements {native}, "
            f"so there is no compiler here that can rule on {flavor}. Re-run on a "
            f"{flavor} host, or pass --force to lower anyway (diagnostic only)."
        )
        return 1 if args.strict else 0

    clang = _llvm_tool("clang")
    opt = None if args.no_verify else _llvm_tool("opt")
    print(f"   clang  : {clang or '(not found)'}")
    if not args.no_verify:
        print(f"   opt    : {opt or '(not found, L0 skipped)'}")

    if clang is None:
        msg = (
            "\nRESULT: UNVALIDATED - no clang found under the resolved ROCm "
            "install or on PATH; nothing was compiled. Set ROCKE_LLVM_BIN or "
            "ROCM_PATH."
        )
        print(msg)
        return 1 if args.strict else 0

    tmp = tempfile.TemporaryDirectory(prefix="rocke_ir_validity_")
    ir_dir = Path(args.keep_ir) if args.keep_ir else Path(tmp.name)
    ir_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== lowering {len(keep)} cases ==")
    try:
        modules = _dump_corpus(ir_dir, flavor, keep)
    except Exception as exc:  # a lowering failure is a validity failure too
        print(f"FATAL: lowering raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"   wrote {len(modules)} modules to {ir_dir}")

    jobs = args.jobs or min(32, (os.cpu_count() or 1))
    out_dir = Path(tmp.name) / "out"
    out_dir.mkdir(exist_ok=True)

    def check(item: tuple[str, tuple[Path, str]]) -> tuple[str, str, str]:
        cid, (path, arch) = item
        if opt is not None:
            ok, diag = _verify(opt, path)
            if not ok:
                return cid, "L0-VERIFY", diag
        ok, diag = _compile(clang, path, arch, out_dir / (path.stem + f".{arch}.hsaco"))
        return cid, ("OK" if ok else "L1-COMPILE"), diag

    print(f"\n== compiling (link to hsaco, {jobs} jobs) ==")
    results: dict[str, tuple[str, str]] = {}
    # Threads, not processes: every unit of work is already its own subprocess,
    # so the fatal-error isolation is provided by the child and the parent is
    # pure I/O wait.
    with cf.ThreadPoolExecutor(jobs) as pool:
        for cid, verdict, diag in pool.map(check, sorted(modules.items())):
            results[cid] = (verdict, diag)

    failed = {c: v for c, v in results.items() if v[0] != "OK"}
    new_bad = {c: v for c, v in failed.items() if c not in KNOWN_BAD}
    stale = [c for c in KNOWN_BAD if c in results and c not in failed]
    selected_known = [c for c in KNOWN_BAD if c in results]

    print()
    print(f"   OK          : {len(results) - len(failed)}")
    print(
        f"   known-bad   : {len(failed) - len(new_bad)} / {len(selected_known)} selected"
    )
    print(f"   NEW failures: {len(new_bad)}")

    if new_bad:
        print("\n== NEW failures ==")
        for cid in sorted(new_bad):
            verdict, diag = new_bad[cid]
            arch = modules[cid][1]
            print(f"\n  {verdict}  {cid}  [{arch}]")
            print(f"    {_first_diagnostic(diag)}")
            if args.verbose:
                for line in diag.splitlines():
                    print(f"      | {line}")

    if stale:
        print("\n== STALE KNOWN_BAD entries (these now compile) ==")
        for cid in sorted(stale):
            print(f"  {cid}\n    reason on file: {KNOWN_BAD[cid]}")
        print(
            "\n  KNOWN_BAD only shrinks. Delete these entries in the change that "
            "fixed them."
        )

    print()
    if new_bad:
        print(
            f"RESULT: RED - {len(new_bad)} case(s) emit IR the toolchain rejects. "
            "Fix the emission; do not add to KNOWN_BAD."
        )
        return 1
    if stale:
        print("RESULT: RED - KNOWN_BAD is stale; remove the entries listed above.")
        return 1
    if failed:
        print(
            f"RESULT: GREEN - every case compiles except the {len(failed)} "
            "documented in KNOWN_BAD."
        )
    else:
        print("RESULT: GREEN - every emitted module compiles and links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
