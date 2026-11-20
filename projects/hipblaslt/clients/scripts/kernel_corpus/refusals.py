# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data-free refusal checks for exact-size MX FP6 solutions.

hipblaslt-test cannot build some problems outside an MX kernel's predicates: its data generator
needs a batch of 1 and leading dimensions equal to the rows stored. refusal_probe.cpp asks the
library directly, with dummy pointers and no data, whether the pinned solution accepts each
problem: the registered one, as a control that must be accepted, and every problem of the edges
profile, which must be refused.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import plan

PROBE_SOURCE = Path(__file__).resolve().parent / "refusal_probe.cpp"
CONTROL_NOT_ACCEPTED = (
    "[runner] the pinned solution refused its own registered problem in the data-free check"
)
ACCEPTED_WITHOUT_DATA = (
    "[runner] the pinned solution accepted a problem outside its predicates in the data-free "
    "check; expected it to refuse"
)


def build_probe(rocm_dir: Path, out_dir: Path) -> Path:
    """Compiles refusal_probe.cpp against the install tree's own headers and libraries. The
    hipBLASLt headers come from TheRock's blas dev artifact."""
    exe = out_dir / "refusal_probe"
    if exe.exists() and exe.stat().st_mtime >= PROBE_SOURCE.stat().st_mtime:
        return exe
    for header in ("hipblaslt/hipblaslt-ext.hpp", "hipblas-common/hipblas-common.h"):
        if not (rocm_dir / "include" / header).exists():
            raise RuntimeError(
                f"no {rocm_dir / 'include' / header}; install the blas dev artifact into the "
                "install tree to build the refusal probe"
            )
    cmd = [
        str(rocm_dir / "lib" / "llvm" / "bin" / "amdclang++"),
        "-std=c++17",
        "-O1",
        "-D__HIP_PLATFORM_AMD__",
        f"-I{rocm_dir / 'include'}",
        str(PROBE_SOURCE),
        f"-L{rocm_dir / 'lib'}",
        f"-Wl,-rpath,{rocm_dir / 'lib'}",
        "-lhipblaslt",
        "-lamdhip64",
        "-o",
        str(exe),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"building the refusal probe failed:\n{proc.stderr[-3000:]}")
    return exe


def probe_cases(record: dict) -> list[plan.Case]:
    """The registered problem, then every edges problem, for an exact-size MX FP6 solution;
    none for any other solution."""
    if not plan.is_mx(record) or plan.registered_shape(record) is None:
        return []
    registered, reason = plan.plan_kernel(record, "smoke")
    if reason:
        return []
    edges, _ = plan.plan_kernel(record, "edges")
    return registered + edges


def probe_label(case: plan.Case) -> str:
    return f"{case.index}:probe-{case.shape.tag}"


def probe_line(case: plan.Case) -> str:
    entry = plan.case_yaml_entry(case)
    size = entry["matrix_size"][0]
    return " ".join(
        str(x)
        for x in (
            probe_label(case),
            case.index,
            case.types["a_type"],
            int(bool(case.types.get("swizzle_b"))),
            size["M"],
            size["N"],
            size["K"],
            entry["batch_count"],
            size["lda"],
            size["ldb"],
            size["ldd"],
            entry["alpha"],
            entry["beta"],
        )
    )


def parse_results(output: str) -> dict[str, dict]:
    """The probe's RESULT lines, by label."""
    results = {}
    for line in output.splitlines():
        words = line.split()
        if len(words) == 6 and words[0] == "RESULT":
            label, verdict, stage, status, name = words[1:]
            results[label] = {
                "verdict": verdict,
                "stage": stage,
                "probe_status": status,
                "probe_name": None if name == "-" else name,
                "line": line,
            }
    return results


def classify(case: plan.Case, record: dict, result: Optional[dict]) -> tuple[str, str]:
    """The outcome and message of one probe answer."""
    if result is None:
        return "crash", "[runner] the refusal probe ended before answering this problem"
    if result["probe_name"] and result["probe_name"] != record["name"]:
        return "fail", (
            f"[runner] solution index {case.index} is {result['probe_name']}, not "
            f"{record['name']}\n{result['line']}"
        )
    accepted = result["verdict"] == "accepted"
    if case.expect == "refused":
        if accepted:
            return "fail", f"{ACCEPTED_WITHOUT_DATA}\n{result['line']}"
        return "expected_refusal", result["line"]
    if not accepted:
        return "fail", f"{CONTROL_NOT_ACCEPTED}\n{result['line']}"
    return "pass", result["line"]


def run_probe(
    exe: Path, cases: list[plan.Case], env: dict, timeout: float, cwd: Path
) -> tuple[dict[str, dict], str, Optional[int]]:
    """Runs every problem through one probe process. Returns the answers by label, the
    process output and its return code (None when it timed out)."""
    stdin = "".join(probe_line(c) + "\n" for c in cases)
    try:
        proc = subprocess.run(
            [str(exe)],
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        output, code = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        raw = e.stdout or ""
        output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        code = None
    return parse_results(output), output, code


def default_probe(args_probe: Optional[Path], rocm_dir: Path, out_dir: Path) -> Path:
    """The probe named on the command line, or one built for this install tree."""
    if args_probe:
        if not os.access(args_probe, os.X_OK):
            raise RuntimeError(f"{args_probe} is not an executable refusal probe")
        return args_probe
    return build_probe(rocm_dir, out_dir)
