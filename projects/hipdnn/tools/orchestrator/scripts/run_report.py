#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Run the numerics harness the authoring agent built, and read what it wrote.

The stage-1 numerics check cannot be "did the agent's contract say passed: true" --
that is exactly the report an agent that never ran anything can produce for free.
The harness itself lives at a path only the agent knows, because the orchestrator
did not build it, so the agent declares `numerics.harness_command` and
`numerics.report_path` in its contract and this script is the thing that actually
invokes the command and reads the file back, rather than trusting the contract's own
claims about what happened.

That buys one thing and only one thing: the harness in `harness_command` really was
executed, just now, by this process, and the report this reads really was written
after that execution started (checked against `report_path`'s mtime versus the
recorded launch time, so a report left over from a previous round reads as stale
rather than as a pass). A few shapes of "didn't run anything" are refused outright
before launch: an empty or missing command, a command whose executable does not
exist, an executable that resolves inside this gate's own `scripts/` directory (the
harness may not be one of the orchestrator's own gates), and a command whose
executable's name is a known interpreter or shell builtin (`python`, `cmd`,
`bash`, `echo`, ...) -- because `echo {"pass": true} > report.json` is a one-line
script with a real exit code and a real output file, and refusing it is the only way
to force the harness to be a compiled artifact instead of a string in the contract.

What this does NOT prove: the harness is still the agent's own code. Nothing here
checks that its comparison logic is honest, that its tolerances are sane, or that it
actually launched a HIP kernel rather than filling the output buffer itself and
calling it a day -- that is why the harness report's own `reference_executed`,
`kernel_launched` and `sentinel_changed` fields are read back and surfaced
unaltered rather than collapsed into a single verdict, and it is why the whole
harness report is embedded in this one's output verbatim. Passing this gate proves a
binary ran, past its start time, and reported certain numbers. It does not prove
those numbers are true; a harness rigged to always report success would still clear
it. The flow's job is to look at the numbers this reports, not just at whether this
script exited zero.

    run_report.py --contract authoring.json --out numerics.json --timeout 600

Exit codes: 0 the report was written -- refusal, a failed run, and a clean run are
all "the report was written" and are distinguished by the `refused`, `exit_code`,
`timed_out` and comparison fields in the JSON, not by the process exit code --
1 the contract file could not be read/parsed, or `--out`/`--contract` are unusable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

#: An interpreter or shell builtin can "run" and produce a report file (`echo {...} >
#: report.json` has a real exit code and a real output) without a single instruction
#: of the kernel under test ever executing. Matched against the executable's stem
#: (case-insensitive), not its full path, so `C:/Python312/python.exe` is caught the
#: same as a bare `python`.
BANNED_STEMS = frozenset(
    {
        "python",
        "python3",
        "cmd",
        "powershell",
        "pwsh",
        "bash",
        "sh",
        "echo",
        "type",
        "cat",
    }
)


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _base_report() -> dict:
    """Every key this gate always emits, so a refusal and a completed run are the
    same shape and the flow can read either without a KeyError."""
    return {
        "refused": 0,
        "exit_code": -1,
        "timed_out": 0,
        "duration_s": 0.0,
        "harness": "",
        "harness_sha256": "",
        "harness_size": 0,
        "report_present": 0,
        "report_fresh": 0,
        "reference_executed": 0,
        "kernel_launched": 0,
        "outputs_compared": 0,
        "mismatched_outputs": 0,
        "sentinel_unchanged": 0,
        "max_rel_err": -1.0,
        "harness_report": None,
        "feedback": "",
    }


def _write(out_path: Path, report: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _refuse(out_path: Path, feedback: str, **overrides: object) -> int:
    report = _base_report()
    report["refused"] = 1
    report["feedback"] = feedback
    report.update(overrides)
    _write(out_path, report)
    print(f"refused: {feedback}")
    return 0


def _load_numerics(contract: object) -> tuple[list[str], str] | str:
    """Validate the two fields this gate needs out of the contract. Returns the
    fields on success, or a human-readable reason on failure -- never raises, because
    a malformed contract is this gate's most common input, not an exceptional one."""
    if not isinstance(contract, dict):
        return "authoring.json's top level is not a JSON object"
    numerics = contract.get("numerics")
    if not isinstance(numerics, dict):
        return "authoring.json has no 'numerics' object"
    harness_command = numerics.get("harness_command")
    if (
        not isinstance(harness_command, list)
        or not harness_command
        or not all(isinstance(item, str) for item in harness_command)
    ):
        return "numerics.harness_command must be a non-empty list of strings"
    report_path = numerics.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        return "numerics.report_path must be a non-empty string"
    return harness_command, report_path


def _derive(doc: dict) -> tuple[dict, list[str]]:
    """Pull the numbers the flow asserts on out of the harness's own report. A key
    the harness omitted becomes a 0 (or -1.0 for the float) plus a named feedback
    line -- never a crash, and never a default a reader could mistake for a pass."""
    notes: list[str] = []

    def _get(key: str, default: object) -> object:
        if key not in doc:
            notes.append(
                f"harness report is missing key {key!r}; treating it as {default!r}."
            )
            return default
        return doc[key]

    reference_executed = 1 if _get("reference_executed", False) else 0
    kernel_launched = 1 if _get("kernel_launched", False) else 0
    outputs_compared_list = _get("outputs_compared", [])
    mismatched_outputs_list = _get("mismatched_outputs", [])
    outputs_compared = (
        len(outputs_compared_list) if isinstance(outputs_compared_list, list) else 0
    )
    mismatched_outputs = (
        len(mismatched_outputs_list) if isinstance(mismatched_outputs_list, list) else 0
    )
    per_output = _get("per_output", [])
    if not isinstance(per_output, list):
        notes.append(
            "harness report's 'per_output' is not a list; treating it as empty."
        )
        per_output = []

    sentinel_unchanged = sum(
        1
        for row in per_output
        if isinstance(row, dict) and row.get("sentinel_changed") is False
    )
    rel_errs = [
        row["max_rel_err"]
        for row in per_output
        if isinstance(row, dict) and isinstance(row.get("max_rel_err"), (int, float))
    ]
    max_rel_err = max(rel_errs) if rel_errs else -1.0

    return (
        {
            "reference_executed": reference_executed,
            "kernel_launched": kernel_launched,
            "outputs_compared": outputs_compared,
            "mismatched_outputs": mismatched_outputs,
            "sentinel_unchanged": sentinel_unchanged,
            "max_rel_err": max_rel_err,
        },
        notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="authoring.json to read")
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds to allow the harness before killing it (default: 600)",
    )
    parser.add_argument(
        "--cwd", help="working directory for the harness (default: current directory)"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    scripts_dir = Path(__file__).resolve().parent

    try:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(
            f"error: could not read contract {args.contract}: {error}", file=sys.stderr
        )
        return 1

    fields = _load_numerics(contract)
    if isinstance(fields, str):
        return _refuse(out_path, f"authoring.json is malformed: {fields}.")
    harness_command, report_path_str = fields

    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()

    raw_exe = harness_command[0]
    exe = Path(raw_exe)
    if not exe.is_absolute():
        exe = cwd / exe
    if not exe.is_file():
        return _refuse(
            out_path,
            f"harness executable does not exist: {exe.as_posix()} "
            f"(argv[0] was {raw_exe!r}). Nothing was run.",
            harness=exe.as_posix(),
        )
    resolved = exe.resolve()

    if resolved == scripts_dir or scripts_dir in resolved.parents:
        return _refuse(
            out_path,
            f"harness executable {resolved.as_posix()} resolves inside this gate's "
            f"own scripts directory ({scripts_dir.as_posix()}). The harness under "
            f"test may not be one of the orchestrator's own gate scripts.",
            harness=resolved.as_posix(),
        )

    stem = resolved.stem.lower()
    if stem in BANNED_STEMS:
        return _refuse(
            out_path,
            f"harness executable's name is {stem!r}, which is an interpreter or "
            f"shell builtin, not a compiled harness (banned names: "
            f"{', '.join(sorted(BANNED_STEMS))}). `echo {{...}} > report.json` is "
            f"exactly the shape of fabrication this rule refuses.",
            harness=resolved.as_posix(),
        )

    harness_sha256 = _hash_file(resolved)
    harness_size = resolved.stat().st_size

    report_path = Path(report_path_str)
    if not report_path.is_absolute():
        return _refuse(
            out_path,
            "numerics.report_path must be an absolute path.",
            harness=resolved.as_posix(),
            harness_sha256=harness_sha256,
            harness_size=harness_size,
        )

    try:
        if report_path.is_file():
            report_path.unlink()
    except OSError as error:
        print(
            f"error: could not clear stale report {report_path}: {error}",
            file=sys.stderr,
        )
        return 1

    argv = [str(resolved), *harness_command[1:]]
    notes: list[str] = []
    process_started = False
    timed_out = 0
    exit_code = -1
    started_at = time.time()
    try:
        proc = subprocess.run(argv, cwd=cwd, timeout=args.timeout)
        process_started = True
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = 1
        notes.append(f"harness timed out after {args.timeout:g}s and was killed.")
    except OSError as error:
        notes.append(f"harness could not be started: {error}.")
    duration_s = time.time() - started_at

    if process_started and exit_code != 0:
        notes.append(f"harness exited with nonzero code {exit_code}.")

    report_present = 1 if report_path.is_file() else 0
    report_fresh = 0
    harness_json: dict | None = None
    if report_present:
        mtime = report_path.stat().st_mtime
        report_fresh = 1 if mtime >= started_at else 0
        if not report_fresh:
            notes.append(
                f"{report_path.as_posix()} exists but its mtime ({mtime:.3f}) predates "
                f"this run's start ({started_at:.3f}); it is a stale report, most "
                f"likely left over from a previous round."
            )
        try:
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            notes.append(f"{report_path.as_posix()} is not valid JSON: {error}.")
        else:
            if isinstance(parsed, dict):
                harness_json = parsed
            else:
                notes.append(
                    f"{report_path.as_posix()}'s top level is not a JSON object."
                )
    else:
        notes.append(
            f"harness exited with code {exit_code} after {duration_s:.3f}s but never "
            f"created {report_path.as_posix()}."
        )

    if harness_json is not None:
        derived, missing_notes = _derive(harness_json)
        notes.extend(missing_notes)
        if derived["reference_executed"] == 0:
            notes.append(
                "harness report says the reference implementation did not execute."
            )
        if derived["kernel_launched"] == 0:
            notes.append(
                "harness report says the kernel under test was never launched."
            )
        if derived["mismatched_outputs"] > 0:
            notes.append(
                f"{derived['mismatched_outputs']} of {derived['outputs_compared']} "
                f"compared output(s) mismatched the reference."
            )
        if derived["sentinel_unchanged"] > 0:
            notes.append(
                f"{derived['sentinel_unchanged']} output(s) left the harness's fill "
                f"sentinel unchanged: the kernel never wrote them, so those "
                f"comparisons are the harness's own fill against the reference, "
                f"not the kernel's output against the reference."
            )
    else:
        derived = {
            "reference_executed": 0,
            "kernel_launched": 0,
            "outputs_compared": 0,
            "mismatched_outputs": 0,
            "sentinel_unchanged": 0,
            "max_rel_err": -1.0,
        }

    report = _base_report()
    report.update(
        {
            "refused": 0,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_s": duration_s,
            "harness": resolved.as_posix(),
            "harness_sha256": harness_sha256,
            "harness_size": harness_size,
            "report_present": report_present,
            "report_fresh": report_fresh,
            "harness_report": harness_json,
            "feedback": "\n\n".join(notes),
        }
    )
    report.update(derived)

    _write(out_path, report)
    print(
        f"harness {resolved.as_posix()} exit={exit_code} timed_out={timed_out} "
        f"duration={duration_s:.3f}s report_present={report_present} "
        f"report_fresh={report_fresh} outputs_compared={derived['outputs_compared']} "
        f"mismatched={derived['mismatched_outputs']} "
        f"sentinel_unchanged={derived['sentinel_unchanged']} "
        f"max_rel_err={derived['max_rel_err']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
