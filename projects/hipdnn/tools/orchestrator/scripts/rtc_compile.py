#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Compile an authored kernel through hipRTC, independently of the agent that wrote it.

Stage one of the ingestor flow ends with an agent saying its kernel compiles and its
numbers match. Both halves of that are the agent's own measurement. This gate takes back
the half that can be taken back: it loads hipRTC directly, compiles every source the
handover contract lists, for the architecture the run named, and asks the compiled code
object whether each declared entry point is actually in it.

Three failures it exists to catch, none of which surfaces later as an error:

**A source that does not compile for this architecture.** A kernel is embedded into the
provider and compiled at plan time, so a compile error arrives hours later as a plan that
throws, inside a build that succeeded.

**An entry point that is not there.** The descriptor names a symbol and
`getKernel(entry_point)` looks it up. A typo, a name mangled because `extern "C"` was
forgotten, or a kernel renamed in the source but not in the contract are all the same
runtime error much later. hipRTC's own lowered-name lookup answers this at compile time,
and the code object is scanned for the kernel descriptor symbol as a second, independent
witness.

**An unguarded compile-time macro.** `-D` specialization only works if an unbound macro
is a compile error. Without `#ifndef X` / `#error`, an unbound token compiles against
whatever it happens to mean -- an undeclared identifier in dead code, a zero, a different
dtype -- and the failure appears only in the numbers. So every macro the contract declares
as required is proved by compiling *without* it and requiring that compile to fail. A
macro whose absence compiles cleanly is reported as a missing guard, which is exactly what
it is.

What this does NOT prove: that the kernel computes anything. A compile is not a
correctness result, and this gate reports none. Numerics are `run_report.py`'s, and the
verdict that matters is the shared integration suite in stage two.

Exit codes: 0 the report was written, whatever it says -- the flow asserts on the counts
in it, so the evidence survives a negative verdict. 1 usage, I/O, or hipRTC could not be
loaded at all, which is an operator problem rather than a kernel one.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

#: hipRTC's own result enum. Only the two we branch on are named; the rest are reported
#: through hiprtcGetErrorString, which is authoritative and version-independent.
HIPRTC_SUCCESS = 0

#: Interpreters and shells that a kernel source is not. Kept here rather than inline so
#: the list reads as a policy.
_HEADER_SUFFIXES = (".h", ".hpp", ".cuh")


class HiprtcError(RuntimeError):
    """hipRTC could not be loaded or used at all -- an environment failure, not a verdict."""


def _find_hiprtc(rocm_bin: Path | None, explicit: Path | None) -> Path:
    """Locate the hipRTC shared library.

    The name carries the ROCm version (`hiprtc0716.dll`), so it cannot be spelled in a
    config file that outlives one SDK. `hiprtc-builtins` is a data blob loaded *by*
    hipRTC and is excluded: loading it succeeds and then every entry point is missing.
    """
    if explicit:
        if not explicit.exists():
            raise HiprtcError(f"--hiprtc names {explicit}, which does not exist")
        return explicit
    patterns = ("hiprtc*.dll",) if os.name == "nt" else ("libhiprtc.so*",)
    if rocm_bin:
        roots = [rocm_bin]
        if os.name != "nt":
            roots += [rocm_bin.parent / "lib"]
        for root in roots:
            for pattern in patterns:
                for candidate in sorted(root.glob(pattern)):
                    if "builtins" in candidate.name:
                        continue
                    return candidate
    # Last resort: the loader's own search path. On Linux hipRTC is frequently folded
    # into the HIP runtime, and there it resolves by soname rather than by file.
    fallback = "amdhip64.dll" if os.name == "nt" else "libhiprtc.so"
    return Path(fallback)


def _load(library: Path, rocm_bin: Path | None) -> ctypes.CDLL:
    # On Windows hipRTC pulls in the HIP runtime, which lives beside it. Without the
    # directory on the DLL search path the load fails with a bare OSError 126 that names
    # hiprtc rather than the dependency that is actually missing.
    if os.name == "nt" and rocm_bin and rocm_bin.is_dir():
        os.add_dll_directory(str(rocm_bin))
    try:
        handle = ctypes.CDLL(str(library))
    except OSError as error:
        raise HiprtcError(f"could not load {library}: {error}") from None

    handle.hiprtcGetErrorString.restype = ctypes.c_char_p
    handle.hiprtcGetErrorString.argtypes = [ctypes.c_int]
    handle.hiprtcVersion.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    handle.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.hiprtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.hiprtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    handle.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.hiprtcGetCodeSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    handle.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    handle.hiprtcAddNameExpression.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.hiprtcGetLoweredName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    return handle


def _version(rtc: ctypes.CDLL) -> str:
    major, minor = ctypes.c_int(0), ctypes.c_int(0)
    if rtc.hiprtcVersion(ctypes.byref(major), ctypes.byref(minor)) != HIPRTC_SUCCESS:
        return "unknown"
    return f"{major.value}.{minor.value}"


def _c_array(values: list[str]) -> Any:
    """A NULL for an empty list; hipRTC reads the count first, but not every build does."""
    if not values:
        return None
    array = (ctypes.c_char_p * len(values))()
    for index, value in enumerate(values):
        array[index] = value.encode("utf-8")
    return array


def _compile(
    rtc: ctypes.CDLL,
    source_text: str,
    source_name: str,
    headers: list[tuple[str, str]],
    options: list[str],
    entry_points: list[str],
) -> dict[str, Any]:
    """One hipRTC compilation. Returns a verdict dict; raises only on a hipRTC defect."""
    program = ctypes.c_void_p()
    header_texts = [text for _, text in headers]
    header_names = [name for name, _ in headers]
    status = rtc.hiprtcCreateProgram(
        ctypes.byref(program),
        source_text.encode("utf-8"),
        source_name.encode("utf-8"),
        len(headers),
        _c_array(header_texts),
        _c_array(header_names),
    )
    if status != HIPRTC_SUCCESS:
        message = rtc.hiprtcGetErrorString(status) or b""
        raise HiprtcError(f"hiprtcCreateProgram failed: {message.decode()}")

    try:
        # Name expressions must be registered before compilation; afterwards hipRTC
        # refuses them. For an `extern "C"` kernel the lowered name equals the source
        # name, and the lookup failing is the interesting result: it means the symbol is
        # not in the program at all.
        for name in entry_points:
            rtc.hiprtcAddNameExpression(program, name.encode("utf-8"))

        status = rtc.hiprtcCompileProgram(program, len(options), _c_array(options))
        log_size = ctypes.c_size_t(0)
        rtc.hiprtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ""
        if log_size.value > 1:
            buffer = ctypes.create_string_buffer(log_size.value)
            rtc.hiprtcGetProgramLog(program, buffer)
            log = buffer.value.decode("utf-8", errors="replace")

        if status != HIPRTC_SUCCESS:
            message = rtc.hiprtcGetErrorString(status) or b""
            return {
                "compiled": 0,
                "status": message.decode(),
                "log": log,
                "code_bytes": 0,
                "symbols": {},
            }

        code_size = ctypes.c_size_t(0)
        rtc.hiprtcGetCodeSize(program, ctypes.byref(code_size))
        code = b""
        if code_size.value:
            buffer = ctypes.create_string_buffer(code_size.value)
            rtc.hiprtcGetCode(program, buffer)
            code = buffer.raw

        symbols: dict[str, dict[str, Any]] = {}
        for name in entry_points:
            lowered = ctypes.c_char_p()
            found = rtc.hiprtcGetLoweredName(
                program, name.encode("utf-8"), ctypes.byref(lowered)
            )
            lowered_name = (
                lowered.value.decode()
                if found == HIPRTC_SUCCESS and lowered.value
                else ""
            )
            # Probe the code object for the *literal* name the contract declared, never
            # for hipRTC's lowered form. The descriptor stores a plain string and the
            # provider calls getKernel(entry_point) with it, so a C++-linkage kernel is
            # unreachable however tidily hipRTC resolved the name expression: the lowered
            # name is `_Z13mangled_entryPfx` and the lookup asks for `mangled_entry`.
            # Recording both is what makes that diagnosable instead of merely absent.
            symbols[name] = {
                "lowered": lowered_name,
                "extern_c": 1 if lowered_name == name else 0,
                "resolved": 1 if lowered_name else 0,
                "in_code_object": 1 if name.encode("utf-8") + b".kd" in code else 0,
            }
        return {
            "compiled": 1,
            "status": "ok",
            "log": log,
            "code_bytes": code_size.value,
            "symbols": symbols,
        }
    finally:
        rtc.hiprtcDestroyProgram(ctypes.byref(program))


def _tail(text: str, lines: int = 30) -> str:
    rows = text.splitlines()
    return "\n".join(rows[-lines:])


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _defines(required: list[dict[str, Any]], skip: str | None = None) -> list[str]:
    """`-D` for every required macro, at its first legal value, optionally dropping one.

    The first legal value is a deliberate choice rather than a sample: the contract lists
    them, and compiling the first one for every macro is the configuration a reviewer can
    reproduce from the contract alone.
    """
    out = []
    for entry in required:
        name = str(entry.get("name", ""))
        if not name or name == skip:
            continue
        values = entry.get("legal_values") or []
        out.append(f"-D{name}={values[0]}" if values else f"-D{name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", required=True, help="authoring.json from the authoring agent"
    )
    parser.add_argument(
        "--arch", required=True, help="target architecture, e.g. gfx1151"
    )
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument(
        "--rocm-bin", help="ROCm bin directory holding the hipRTC library"
    )
    parser.add_argument("--hiprtc", help="explicit path to the hipRTC library")
    parser.add_argument(
        "--skip-guard-proof",
        action="store_true",
        help="do not run the negative compiles that prove each required macro is guarded",
    )
    args = parser.parse_args()

    contract_path = Path(args.contract)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        contract = json.loads(_read(contract_path))
    except (OSError, ValueError) as error:
        print(f"error: could not read {contract_path}: {error}", file=sys.stderr)
        return 1

    bundle = contract.get("bundle") or {}
    sources = [Path(p) for p in bundle.get("sources") or []]
    header_paths = [Path(p) for p in bundle.get("headers") or []]
    entry_points = contract.get("entry_points") or []
    required = contract.get("required_defines") or []
    extra = [str(o) for o in contract.get("compile_options") or []]

    rocm_bin = Path(args.rocm_bin) if args.rocm_bin else None
    try:
        library = _find_hiprtc(rocm_bin, Path(args.hiprtc) if args.hiprtc else None)
        rtc = _load(library, rocm_bin)
    except HiprtcError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    missing = [p for p in sources + header_paths if not p.is_file()]
    headers: list[tuple[str, str]] = []
    for path in header_paths:
        if path.is_file():
            # Headers are offered to hipRTC by basename, which is how the provider's own
            # embedded map and the bundle rules resolve them.
            headers.append((path.name, _read(path)))

    # Options the flow contributes, in front of the contract's own so the contract can
    # override nothing it should not: the architecture is the run's, not the kernel's.
    base = [f"--offload-arch={args.arch}"]
    declared = {opt.split("=", 1)[0] for opt in extra if opt.startswith("-D")}
    macros = [opt for opt in _defines(required) if opt.split("=", 1)[0] not in declared]

    per_source: list[dict[str, Any]] = []
    resolved_total = 0
    unresolved_names: list[str] = []
    compiled = 0

    for source in sources:
        if not source.is_file():
            per_source.append(
                {
                    "file": source.as_posix(),
                    "compiled": 0,
                    "status": "file does not exist",
                    "log_tail": "",
                    "code_bytes": 0,
                    "entry_points": [],
                }
            )
            continue
        names = [
            str(e.get("name"))
            for e in entry_points
            if Path(str(e.get("source_file", ""))).name == source.name
        ]
        options = base + macros + extra
        try:
            verdict = _compile(rtc, _read(source), source.name, headers, options, names)
        except HiprtcError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        compiled += verdict["compiled"]
        symbol_rows = []
        for name in names:
            entry = verdict["symbols"].get(
                name, {"resolved": 0, "extern_c": 0, "in_code_object": 0}
            )
            # All three, not any of them: hipRTC resolved the name expression, the name
            # kept C linkage, and the literal symbol is in the code object. Dropping the
            # middle one is what let a C++-linkage kernel read as present.
            ok = (
                bool(entry.get("resolved"))
                and bool(entry.get("extern_c"))
                and bool(entry.get("in_code_object"))
            )
            resolved_total += 1 if ok else 0
            if not ok:
                unresolved_names.append(name)
            symbol_rows.append({"name": name, **entry})
        per_source.append(
            {
                "file": source.as_posix(),
                "compiled": verdict["compiled"],
                "status": verdict["status"],
                "options": options,
                "log_tail": _tail(verdict["log"]),
                "code_bytes": verdict["code_bytes"],
                "entry_points": symbol_rows,
            }
        )

    # The guard proof. One negative compile per required macro, against the source that
    # mentions it: drop the macro and require the compile to fail.
    guards: list[dict[str, Any]] = []
    if not args.skip_guard_proof:
        for entry in required:
            name = str(entry.get("name", ""))
            if not name:
                continue
            owner = next(
                (
                    s
                    for s in sources
                    if s.is_file() and re.search(rf"\b{re.escape(name)}\b", _read(s))
                ),
                None,
            )
            if owner is None:
                guards.append(
                    {
                        "macro": name,
                        "guarded": 0,
                        "source": "",
                        "why": "no listed source mentions this macro at all",
                    }
                )
                continue
            options = base + _defines(required, skip=name) + extra
            try:
                verdict = _compile(rtc, _read(owner), owner.name, headers, options, [])
            except HiprtcError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
            guards.append(
                {
                    "macro": name,
                    "guarded": 0 if verdict["compiled"] else 1,
                    "source": owner.as_posix(),
                    "why": (
                        ""
                        if not verdict["compiled"]
                        else "the source compiled with this macro unbound, so nothing "
                        "stops it being instantiated against whatever the token happens "
                        "to mean"
                    ),
                }
            )

    unguarded = [g["macro"] for g in guards if not g["guarded"]]
    total_entry_points = sum(len(row["entry_points"]) for row in per_source)
    meets = int(
        bool(sources)
        and compiled == len(sources)
        and not missing
        and not unresolved_names
        and not unguarded
        and total_entry_points == len(entry_points)
    )

    feedback_parts: list[str] = []
    if missing:
        feedback_parts.append(
            "The contract lists files that are not on disk:\n"
            + "\n".join(f"- {p.as_posix()}" for p in missing)
        )
    for row in per_source:
        if not row["compiled"]:
            feedback_parts.append(
                f"hipRTC could not compile {row['file']} for {args.arch} "
                f"({row['status']}). Compiler log:\n{row['log_tail']}"
            )
    if unresolved_names:
        feedback_parts.append(
            "These entry points are named in the contract but are not in the compiled "
            "code object: "
            + ", ".join(sorted(set(unresolved_names)))
            + ". The descriptor will name a symbol that getKernel() cannot resolve. The "
            'usual cause is a missing `extern "C"`, which mangles the name, or a '
            "kernel renamed in the source but not in the contract."
        )
    if total_entry_points != len(entry_points):
        feedback_parts.append(
            f"The contract declares {len(entry_points)} entry point(s) but only "
            f"{total_entry_points} of them name a source_file that is in bundle.sources. "
            "An entry point whose source is not in the bundle is not compiled by anything."
        )
    if unguarded:
        feedback_parts.append(
            "These required macros are not guarded: "
            + ", ".join(unguarded)
            + ". Each one needs `#ifndef <NAME>` / `#error <NAME> must be defined` at the "
            "top of the source that uses it. Without the guard an unbound macro is not a "
            "compile error, and the specialization silently becomes whatever the bare "
            "token means."
        )

    report = {
        "arch": args.arch,
        "hiprtc": library.as_posix(),
        "hiprtc_version": _version(rtc),
        "sources": len(sources),
        "compiled": compiled,
        "missing_files": [p.as_posix() for p in missing],
        "declared_entry_points": len(entry_points),
        "matched_entry_points": total_entry_points,
        "resolved": resolved_total,
        "unresolved": len(unresolved_names),
        "unresolved_names": sorted(set(unresolved_names)),
        "guards_expected": len(guards),
        "guards_proved": sum(g["guarded"] for g in guards),
        "guards_missing": unguarded,
        "guards": guards,
        "per_source": per_source,
        "meets_target": meets,
        "feedback": "\n\n".join(feedback_parts),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"hipRTC {report['hiprtc_version']} ({library.name}) arch {args.arch}: "
        f"{compiled}/{len(sources)} source(s) compiled, "
        f"{resolved_total}/{total_entry_points} entry point(s) resolved, "
        f"{report['guards_proved']}/{len(guards)} macro guard(s) proved."
    )
    for row in per_source:
        if not row["compiled"]:
            print(f"  FAILED {row['file']}: {row['status']}")
    for name in report["unresolved_names"]:
        print(f"  MISSING SYMBOL {name}")
    for macro in unguarded:
        print(f"  UNGUARDED MACRO {macro}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
