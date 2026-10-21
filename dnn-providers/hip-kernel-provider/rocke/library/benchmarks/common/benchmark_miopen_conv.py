# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time MIOpen (MIOpenDriver) on the shapes the rocKE conv sweep runs.

The counterpart of ``benchmark_implicit_gemm_conv.py`` for MIOpen: the same
shape sources (``--miopen-cmd`` / ``--miopen-file`` / ``--json-file`` /
explicit shape flags, parsed by the sweep's own parsers), one MIOpenDriver run
per (shape, direction, math mode), and TFLOPS from rocKE's ``ConvProblem.flops``
so both sides share one FLOP formula. ``--csv`` rows are keyed
``(shape, dtype, direction)`` like the sweep's ``--csv``, so the two files join.

TF32 is MIOpen's math type, not a tensor type: fp32 tensors with ``-M 0``
("use TF32 if possible") versus ``-M 1`` (IEEE fp32, the driver default).
``--math`` picks which of the two to run for the fp32-storage cases (a case
the sweep calls ``fp32`` or ``tf32``); fp16/bf16 cases run once (MIOpen never
applies TF32 to them). A row's ``dtype`` is ``tf32`` for ``-M 0`` and ``fp32``
for ``-M 1``. ``-M 0`` only *allows* TF32: Find may still pick a solver that
computes in fp32 (and the CK grouped-conv solvers silently fall back to fp32
instances), so every row records the solver MIOpen reported, and
``--trace-kernels`` replays the Find winner once (``MIOPEN_FIND_MODE=FAST``
from the find-db the timed run wrote) under ``rocprofv3 --kernel-trace`` to
record the kernels that actually ran and whether any computes in TF32.

The MIOpen install is taken from ``--miopen-root`` or ``$ROCKE_MIOPEN_ROOT``
(a ROCm/TheRock tree with ``bin/MIOpenDriver`` and ``lib/``); a driver without
``-M/--math_type`` is refused. Its find/perf databases go to a temporary
directory (``--user-db-dir`` to keep them), never the work tree.

Example::

  python benchmark_miopen_conv.py --miopen-root $ROCKE_MIOPEN_ROOT \\
      --miopen-cmd "MIOpenDriver conv -n 8 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 \\
          -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -I NHWC" \\
      --math both --trace-kernels --csv /path/outside/tree/miopen.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ENV_ROOT = "ROCKE_MIOPEN_ROOT"

# rocKE direction -> MIOpenDriver -F, and the phrase the driver prints for it.
_DIRECTION_FORW = {"fwd": 1, "dgrad": 2, "wgrad": 4}
_DIRECTION_PHRASE = {
    "fwd": "Forward",
    "dgrad": "Backward Data",
    "wgrad": "Backward Weights",
}
# The verify line names the direction differently.
_VERIFY_PHRASE = {
    "fwd": "Forward Convolution",
    "dgrad": "Backward Convolution Data",
    "wgrad": "Backward Convolution Weights",
}
# Storage dtype -> MIOpenDriver base argument.
_DRIVER_KW = {"fp32": "conv", "fp16": "convfp16", "bf16": "convbfp16"}
# MIOpen math types (miopenMathType_t).
MATH_TF32 = 0  # miopenMathDefault: TF32 where a solver supports it
MATH_IEEE = 1  # miopenMathPedantic: IEEE fp32

# Env vars that turn TF32 on regardless of -M (MIOpen ORs them in), so a
# -M 1 baseline would not be IEEE with either set.
_TF32_OVERRIDE_ENV = ("MIOPEN_TF32_OVERRIDE", "NVIDIA_TF32_OVERRIDE")

_CSV_FIELDS = [
    "shape",
    "sD",
    "sH",
    "sW",
    "pD",
    "pH",
    "pW",
    "dD",
    "dH",
    "dW",
    "groups",
    "dtype",
    "direction",
    "miopen_math",
    "miopen_ms",
    "miopen_tflops",
    "miopen_gbps",
    "miopen_algorithm",
    "miopen_solution_id",
    "miopen_solver",
    "verify",
    "verify_err",
    "verify_tol",
    "tf32_kernel",
    "kernels",
    "rc",
    "cmd",
]


# ---------------------------------------------------------------------------
# Command and environment
# ---------------------------------------------------------------------------


def miopen_root(cli_value: Optional[str]) -> Path:
    """The MIOpen install root: ``--miopen-root``, else ``$ROCKE_MIOPEN_ROOT``."""
    value = cli_value or os.environ.get(ENV_ROOT)
    if not value:
        raise SystemExit(
            f"error: no MIOpen install given: pass --miopen-root or set {ENV_ROOT} "
            "(a tree with bin/MIOpenDriver and lib/libMIOpen.so)"
        )
    root = Path(value).expanduser().resolve()
    if not (root / "bin" / "MIOpenDriver").is_file():
        raise SystemExit(f"error: {root}/bin/MIOpenDriver not found")
    return root


def miopen_env(root: Path, user_db: str, base: Optional[dict] = None) -> dict:
    """Environment for one MIOpenDriver run against ``root``.

    The install's own libraries only (a second ROCm on the path would mix
    runtimes), full Find, the find/perf databases in ``user_db``, and neither
    TF32 override variable -- the math type comes from ``-M`` alone.
    """
    env = dict(os.environ if base is None else base)
    for name in _TF32_OVERRIDE_ENV:
        env.pop(name, None)
    env["LD_LIBRARY_PATH"] = str(root / "lib")
    env["ROCM_PATH"] = str(root)
    env["MIOPEN_FIND_MODE"] = "NORMAL"
    env["MIOPEN_USER_DB_PATH"] = user_db
    return env


def build_miopen_cmd(
    problem,
    storage_dtype: str,
    direction: str,
    math_type: int,
    *,
    driver: str = "MIOpenDriver",
    iters: int = 20,
    verify: bool = True,
    time: bool = True,
    solution: Optional[str] = None,
    search: bool = False,
) -> List[str]:
    """The MIOpenDriver argv for one problem, direction and math type.

    Channels-last layouts on every tensor (NHWC / NDHWC), so MIOpen runs the
    layout the rocKE kernels run rather than timing NCHW transposes.
    ``solution`` pins a solver (immediate mode, ``-S``); without it the driver
    runs Find and times the winner.
    """
    if storage_dtype not in _DRIVER_KW:
        raise ValueError(f"no MIOpenDriver for storage dtype {storage_dtype!r}")
    if direction not in _DIRECTION_FORW:
        raise ValueError(f"unknown direction {direction!r}")
    p = problem
    layout = "NDHWC" if p.is_3d else "NHWC"
    argv = [
        str(driver),
        _DRIVER_KW[storage_dtype],
        "-n", str(p.N),
        "-c", str(p.C),
        "-H", str(p.Hi),
        "-W", str(p.Wi),
        "-k", str(p.K),
        "-y", str(p.Y),
        "-x", str(p.X),
        "-p", str(p.pH),
        "-q", str(p.pW),
        "-u", str(p.sH),
        "-v", str(p.sW),
        "-l", str(p.dH),
        "-j", str(p.dW),
        "-g", str(p.groups),
        "-m", "conv",
        "-_", "3" if p.is_3d else "2",
    ]  # fmt: skip
    if p.is_3d:
        argv += [
            "-!", str(p.Di),
            "-@", str(p.Z),
            "-#", str(p.sD),
            "-$", str(p.pD),
            "-^", str(p.dD),
        ]  # fmt: skip
    argv += [
        "--in_layout", layout,
        "--fil_layout", layout,
        "--out_layout", layout,
        "-F", str(_DIRECTION_FORW[direction]),
        "-M", str(int(math_type)),
        "-t", "1" if time else "0",
        "-V", "1" if verify else "0",
        "-i", str(int(iters)),
    ]  # fmt: skip
    if solution is not None:
        argv += ["-S", str(solution)]
    if search:
        argv += ["-s", "1"]
    return argv


def preflight(root: Path, env: dict) -> None:
    """Refuse a driver that cannot select the math type (pre-TF32 MIOpen)."""
    out = subprocess.run(
        [str(root / "bin" / "MIOpenDriver"), "conv", "-?"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = out.stdout + out.stderr
    if "--math_type" not in text:
        raise SystemExit(
            f"error: {root}/bin/MIOpenDriver has no -M/--math_type: this MIOpen "
            "cannot run TF32 (it needs a MIOpen with miopenMathType_t)"
        )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"


@dataclass
class DriverResult:
    """What one MIOpenDriver run reported for one direction."""

    ms: Optional[float] = None
    algorithm: Optional[int] = None
    solution_id: Optional[str] = None
    solver: Optional[str] = None
    verify: str = ""  # "OK", "FAILED" or "" (not run / not found)
    verify_err: Optional[float] = None
    verify_tol: Optional[float] = None


def parse_driver_output(text: str, direction: str) -> DriverResult:
    """Parse MIOpenDriver stdout for ``direction``.

    Reads ``MIOpen <dir> Conv. Algorithm: <a>, Solution: <id>/<name>``,
    ``GPU Kernel Time <dir> Conv. Elapsed: <ms> ms (average)`` and the verify
    line (``... Verifies OK on GPU reference (<err> < <tol>)`` or
    ``... FAILED: <err> > <tol>``).
    """
    phrase = re.escape(_DIRECTION_PHRASE[direction])
    res = DriverResult()
    m = re.search(
        rf"MIOpen {phrase} Conv\. Algorithm: (\d+), Solution: (\d+)/(\S+)", text
    )
    if m:
        res.algorithm = int(m.group(1))
        res.solution_id = m.group(2)
        res.solver = m.group(3)
    m = re.search(rf"GPU Kernel Time {phrase} Conv\. Elapsed: {_NUM} ms", text)
    if m:
        res.ms = float(m.group(1))
    vphrase = re.escape(_VERIFY_PHRASE[direction])
    m = re.search(rf"{vphrase} Verifies OK on \w+ reference \({_NUM} < {_NUM}\)", text)
    if m:
        res.verify = "OK"
        res.verify_err, res.verify_tol = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(rf"{vphrase} FAILED: {_NUM} > {_NUM}", text)
        if m:
            res.verify = "FAILED"
            res.verify_err, res.verify_tol = float(m.group(1)), float(m.group(2))
    return res


# ---------------------------------------------------------------------------
# Kernel trace
# ---------------------------------------------------------------------------

# Runtime utility kernels that say nothing about the solver's math.
_UTILITY_PREFIXES = ("__amd_rocclr_",)

# hipconv's DataType enum (hipconv/conv_params.hpp: fp16, bf16, fp32, fp8,
# bf8, tf32): a hipconv kernel's template argument spells TF32 as the cast
# "(hipconv::DataType)5" once demangled, and as "DataTypeE5E" mangled.
_HIPCONV_TF32 = re.compile(
    r"\(hipconv::DataType\)5\b|hipconv::DataType::tf32|hipconv8DataTypeE5E"
)
# CK names its TF32 operand type ck::tf32_t (mangled "6tf32_t"); XF32 MFMA
# kernels (hipBLASLt / Tensile, hand-written ASM) say xf32 in their names.
# Not preceded by a letter, so "MaxF32" / "SoftmaxF32" do not count as xf32.
_GENERIC_TF32 = re.compile(r"(?<![A-Za-z])[tx]f32", re.IGNORECASE)
# CK spells ``tf32_t`` as ``_BitInt(19)`` (ck/utility/data_type.hpp), so a
# demangled CK TF32 instance names no "tf32" at all; mangled it is ``DB19_``.
_CK_BITINT_TF32 = re.compile(r"_BitInt\(19\)|DB19_")


def is_tf32_kernel(name: str) -> bool:
    """Whether a (mangled or demangled) kernel name computes in TF32."""
    return bool(
        _HIPCONV_TF32.search(name)
        or _CK_BITINT_TF32.search(name)
        or _GENERIC_TF32.search(name)
    )


def short_kernel_name(name: str, limit: int = 96) -> str:
    """A kernel name cut to its function name, for tables."""
    name = name.strip().replace("(anonymous namespace)::", "")
    name = name.removeprefix("void ")
    cut = min((i for i in (name.find("<"), name.find("(")) if i > 0), default=-1)
    head = name if cut < 0 else name[:cut]
    return head[:limit]


def _demangle(names: Sequence[str], root: Path) -> List[str]:
    """Demangle with the install's llvm-cxxfilt (else the system's); a name
    that cannot be demangled is kept as it is."""
    tool = None
    for cand in (
        root / "lib" / "llvm" / "bin" / "llvm-cxxfilt",
        root / "llvm" / "bin" / "llvm-cxxfilt",
    ):
        if cand.is_file():
            tool = str(cand)
            break
    tool = tool or shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if tool is None or not names:
        return list(names)
    try:
        out = subprocess.run(
            [tool],
            check=False,
            input="\n".join(names) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return list(names)
    return out if len(out) == len(names) else list(names)


def read_kernel_trace(trace_csv: Path) -> List[str]:
    """Distinct kernel names (first-seen order) of a rocprofv3 kernel trace,
    utility kernels dropped."""
    seen: dict = {}
    with open(trace_csv, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Kernel_Name", "")
            if name and not name.startswith(_UTILITY_PREFIXES):
                seen.setdefault(name, None)
    return list(seen)


def trace_kernels(
    root: Path, argv: List[str], env: dict, timeout: Optional[float]
) -> Tuple[Optional[List[str]], str, str]:
    """Run ``argv`` once under ``rocprofv3 --kernel-trace``.

    Returns ``(demangled kernel names, "", driver output)``, or
    ``(None, reason, driver output)`` when the trace could not be taken.
    """
    rocprof = root / "bin" / "rocprofv3"
    if not rocprof.is_file():
        found = shutil.which("rocprofv3")
        if found is None:
            return None, "no rocprofv3", ""
        rocprof = Path(found)
    with tempfile.TemporaryDirectory(prefix="rocke_miopen_trace_") as tmp:
        cmd = [
            str(rocprof),
            "--kernel-trace",
            "--output-format",
            "csv",
            "-d",
            tmp,
            "-o",
            "run",
            "--",
            *argv,
        ]
        try:
            out = subprocess.run(
                cmd,
                check=False,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "trace timed out", ""
        text = out.stdout + out.stderr
        traces = list(Path(tmp).rglob("*kernel_trace.csv"))
        if not traces:
            tail = text.strip().splitlines()[-1:]
            return None, f"no kernel trace (rc={out.returncode}) {' '.join(tail)}", text
        names = read_kernel_trace(traces[0])
    return _demangle(names, root), "", text


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    problem: object
    dtype: str  # the sweep's case dtype: fp16 / bf16 / fp32 / tf32
    direction: str


@dataclass
class Run:
    """One (case, math type) to time."""

    case: Case
    storage: str  # fp32 / fp16 / bf16
    math_type: int
    dtype: str  # the CSV dtype: tf32 for -M 0 on fp32, else the storage
    result: DriverResult = field(default_factory=DriverResult)
    rc: int = 0
    kernels: Optional[List[str]] = None
    trace_note: str = ""


def _storage(dtype: str) -> str:
    return "fp32" if dtype == "tf32" else dtype


def expand_runs(cases: Sequence[Case], math: str, as_fp32: bool = False) -> List[Run]:
    """The runs for ``cases`` under ``--math``.

    ``math`` is ``fp32``, ``tf32`` or ``both``; it applies to fp32-storage
    cases (``fp32`` / ``tf32``). fp16/bf16 cases run once at ``-M 1`` unless
    ``as_fp32`` reads them as fp32 tensors too.
    """
    modes = {
        "fp32": (MATH_IEEE,),
        "tf32": (MATH_TF32,),
        "both": (MATH_IEEE, MATH_TF32),
    }[math]
    runs: List[Run] = []
    for case in cases:
        storage = "fp32" if as_fp32 else _storage(case.dtype)
        if storage != "fp32":
            runs.append(Run(case, storage, MATH_IEEE, storage))
            continue
        for m in modes:
            runs.append(Run(case, "fp32", m, "tf32" if m == MATH_TF32 else "fp32"))
    return runs


def _miopen_cases(cmd: str) -> List[Case]:
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _FORW_TO_DIRECTIONS,
        parse_miopen_cmd,
    )

    prob, dt, forw = parse_miopen_cmd(cmd)
    return [Case(prob, dt, d) for d in _FORW_TO_DIRECTIONS.get(forw, ["fwd"])]


def collect_cases(args) -> List[Case]:
    if args.miopen_cmd is not None:
        return _miopen_cases(args.miopen_cmd)
    if args.miopen_file is not None:
        cases: List[Case] = []
        for lineno, line in enumerate(
            Path(args.miopen_file).read_text().splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases += _miopen_cases(line)
            except ValueError as e:
                print(
                    f"[warn] {args.miopen_file}:{lineno}: skipping — {e}",
                    file=sys.stderr,
                )
        return cases
    if args.json_file is not None:
        from benchmarks.common.benchmark_implicit_gemm_conv import parse_json_case

        entries = json.loads(Path(args.json_file).read_text())
        cases = []
        for idx, entry in enumerate(entries):
            if args.json_filter_suite and entry.get("suite") != args.json_filter_suite:
                continue
            try:
                prob, dt = parse_json_case(entry)
            except ValueError as e:
                cid = entry.get("case_id", f"#{idx + 1}")
                print(
                    f"[warn] {args.json_file} case {cid}: skipping — {e}",
                    file=sys.stderr,
                )
                continue
            cases.append(Case(prob, dt, args.direction))
        return cases

    from kernels.common.conv_implicit_gemm import ConvProblem

    problem = ConvProblem(
        N=args.N,
        Di=args.Di,
        Hi=args.Hi,
        Wi=args.Wi,
        C=args.C,
        K=args.K,
        Z=args.Z,
        Y=args.Y,
        X=args.X,
        sD=args.sD,
        sH=args.sH,
        sW=args.sW,
        pD=args.pD,
        pH=args.pH,
        pW=args.pW,
        dD=args.dD,
        dH=args.dH,
        dW=args.dW,
        groups=args.groups,
    )
    return [Case(problem, args.dtype, args.direction)]


# ---------------------------------------------------------------------------
# Metrics and CSV
# ---------------------------------------------------------------------------


def problem_bytes(problem, storage: str) -> int:
    """Bytes of the three tensors (input, filter, output) at ``storage``."""
    p = problem
    elem = 4 if storage == "fp32" else 2
    depth_in = p.Di if p.is_3d else 1
    depth_out = p.Do if p.is_3d else 1
    depth_f = p.Z if p.is_3d else 1
    a = p.N * depth_in * p.Hi * p.Wi * p.C
    b = p.K * depth_f * p.Y * p.X * p.cpg
    d = p.N * depth_out * p.Ho * p.Wo * p.K
    return elem * (a + b + d)


def csv_row(run: Run, cmd: str) -> dict:
    p = run.case.problem
    r = run.result
    tflops = gbps = None
    if r.ms:
        tflops = float(p.flops) / r.ms * 1e-9
        gbps = problem_bytes(p, run.storage) / r.ms * 1e-6
    if run.kernels is None:
        tf32 = "unknown"
    else:
        tf32 = "yes" if any(is_tf32_kernel(k) for k in run.kernels) else "no"
    return {
        "shape": p.short(),
        "sD": p.sD if p.is_3d else "",
        "sH": p.sH,
        "sW": p.sW,
        "pD": p.pD if p.is_3d else "",
        "pH": p.pH,
        "pW": p.pW,
        "dD": p.dD if p.is_3d else "",
        "dH": p.dH,
        "dW": p.dW,
        "groups": p.groups,
        "dtype": run.dtype,
        "direction": run.case.direction,
        "miopen_math": run.math_type,
        "miopen_ms": "" if r.ms is None else f"{r.ms:.6f}",
        "miopen_tflops": "" if tflops is None else f"{tflops:.6g}",
        "miopen_gbps": "" if gbps is None else f"{gbps:.6g}",
        "miopen_algorithm": "" if r.algorithm is None else r.algorithm,
        "miopen_solution_id": r.solution_id or "",
        "miopen_solver": r.solver or "",
        "verify": r.verify,
        "verify_err": "" if r.verify_err is None else f"{r.verify_err:.3e}",
        "verify_tol": "" if r.verify_tol is None else f"{r.verify_tol:.3e}",
        "tf32_kernel": tf32,
        "kernels": ";".join(short_kernel_name(k) for k in (run.kernels or [])),
        "rc": run.rc,
        "cmd": cmd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--miopen-root",
        default=None,
        help=f"MIOpen/ROCm install with bin/MIOpenDriver and lib/ "
        f"(default: ${ENV_ROOT})",
    )
    ap.add_argument(
        "--math",
        default="both",
        choices=("fp32", "tf32", "both"),
        help="math type(s) for fp32-tensor cases: fp32 = -M 1 (IEEE), "
        "tf32 = -M 0, both (default)",
    )
    ap.add_argument(
        "--as-fp32",
        action="store_true",
        help="run fp16/bf16 cases (e.g. a bf16 JSON suite) on fp32 tensors, "
        "so --math applies to them too",
    )
    ap.add_argument("--iters", type=int, default=20, help="timed iterations (-i)")
    ap.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip MIOpenDriver's own verification (-V 0)",
    )
    ap.add_argument(
        "--search",
        action="store_true",
        help="tune first (-s 1 with MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE)",
    )
    ap.add_argument(
        "--trace-kernels",
        action="store_true",
        help="replay each Find winner once under rocprofv3 --kernel-trace and "
        "record the kernels it ran and whether any computes in TF32",
    )
    ap.add_argument(
        "--user-db-dir",
        default=None,
        help="MIOpen find/perf database directory (default: a temporary "
        "directory, removed at exit). Keep it outside the work tree.",
    )
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per run")
    ap.add_argument("--csv", default=None, metavar="FILE", help="write rows here")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the commands, run nothing"
    )

    src = ap.add_argument_group("shape source (as benchmark_implicit_gemm_conv.py)")
    src.add_argument("--miopen-cmd", default=None, metavar="CMD")
    src.add_argument("--miopen-file", default=None, metavar="FILE")
    src.add_argument("--json-file", default=None, metavar="FILE")
    src.add_argument("--json-filter-suite", default=None, metavar="SUITE")
    src.add_argument(
        "--dtype",
        default="fp32",
        choices=("fp16", "bf16", "fp32", "tf32"),
        help="dtype for explicit shape flags (default fp32)",
    )
    src.add_argument(
        "--direction",
        default="fwd",
        choices=tuple(_DIRECTION_FORW),
        help="direction for explicit shapes and JSON cases (default fwd)",
    )
    conv = ap.add_argument_group("ConvProblem (explicit shape flags)")
    conv.add_argument("--N", type=int, default=8)
    conv.add_argument("--Di", type=int, default=None)
    conv.add_argument("--Hi", type=int, default=56)
    conv.add_argument("--Wi", type=int, default=56)
    conv.add_argument("--C", type=int, default=64)
    conv.add_argument("--K", type=int, default=64)
    conv.add_argument("--Z", type=int, default=None)
    conv.add_argument("--Y", type=int, default=3)
    conv.add_argument("--X", type=int, default=3)
    conv.add_argument("--sD", type=int, default=None)
    conv.add_argument("--sH", type=int, default=1)
    conv.add_argument("--sW", type=int, default=1)
    conv.add_argument("--pD", type=int, default=None)
    conv.add_argument("--pH", type=int, default=1)
    conv.add_argument("--pW", type=int, default=1)
    conv.add_argument("--dD", type=int, default=None)
    conv.add_argument("--dH", type=int, default=1)
    conv.add_argument("--dW", type=int, default=1)
    conv.add_argument("--groups", "-g", type=int, default=1)
    return ap


def _run_one(run: Run, root: Path, env: dict, args) -> str:
    p = run.case.problem
    argv = build_miopen_cmd(
        p,
        run.storage,
        run.case.direction,
        run.math_type,
        driver=str(root / "bin" / "MIOpenDriver"),
        iters=args.iters,
        verify=args.verify,
        search=args.search,
    )
    shown = " ".join(argv[1:])
    print(
        f"\n=== {p.short()} {run.case.direction} {run.dtype} (-M {run.math_type})",
        flush=True,
    )
    print(f"  MIOpenDriver {shown}", flush=True)
    if args.dry_run:
        return shown
    run_env = dict(env)
    if args.search:
        run_env["MIOPEN_FIND_ENFORCE"] = "SEARCH_DB_UPDATE"
    try:
        out = subprocess.run(
            argv,
            check=False,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        text, run.rc = out.stdout + out.stderr, out.returncode
    except subprocess.TimeoutExpired:
        text, run.rc = "", -1
        print("  [error] MIOpenDriver timed out", flush=True)
    run.result = parse_driver_output(text, run.case.direction)
    r = run.result
    print(
        f"  solver: {r.solver or '?'} (id {r.solution_id or '?'})  "
        f"verify: {r.verify or '-'}  rc={run.rc}",
        flush=True,
    )
    if r.ms is not None:
        tflops = float(p.flops) / r.ms * 1e-9
        print(f"  time: {r.ms:.4f} ms  {tflops:.2f} TFLOPS", flush=True)
    elif run.rc != 0:
        tail = "\n".join(text.strip().splitlines()[-5:])
        print(f"  [error] no timing; driver output tail:\n{tail}", flush=True)

    if args.trace_kernels:
        if r.solution_id is None:
            run.trace_note = "no solution reported; nothing to trace"
        else:
            # Replay the Find winner: FAST find mode answers from the find-db
            # the timed (NORMAL) run just wrote, so the trace holds the
            # winning solution with its tuned config and none of the
            # candidates Find benchmarked.
            targv = build_miopen_cmd(
                p,
                run.storage,
                run.case.direction,
                run.math_type,
                driver=str(root / "bin" / "MIOpenDriver"),
                iters=1,
                verify=False,
            )
            tenv = dict(env, MIOPEN_FIND_MODE="FAST")
            run.kernels, run.trace_note, ttext = trace_kernels(
                root, targv, tenv, args.timeout
            )
            replay = parse_driver_output(ttext, run.case.direction)
            if run.kernels is not None and replay.solution_id != r.solution_id:
                run.trace_note = (
                    f"trace replay ran solution {replay.solution_id}/"
                    f"{replay.solver}, not {r.solution_id}/{r.solver}"
                )
                print(f"  [warn] {run.trace_note}", flush=True)
                run.kernels = None
        if run.kernels is not None:
            flag = "yes" if any(is_tf32_kernel(k) for k in run.kernels) else "no"
            print(f"  tf32 kernel ran: {flag}", flush=True)
            for k in run.kernels:
                mark = " [tf32]" if is_tf32_kernel(k) else ""
                print(f"    kernel: {short_kernel_name(k, 160)}{mark}", flush=True)
        else:
            print(f"  [warn] trace: {run.trace_note}", flush=True)
    return shown


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    cases = collect_cases(args)
    if not cases:
        print("error: no cases", file=sys.stderr)
        return 2
    runs = expand_runs(cases, args.math, args.as_fp32)

    root = miopen_root(args.miopen_root) if not args.dry_run else None
    if root is None:
        root = Path(args.miopen_root or os.environ.get(ENV_ROOT) or "<MIOPEN_ROOT>")

    tmp_db = None
    if args.user_db_dir:
        user_db = str(Path(args.user_db_dir).expanduser().resolve())
        os.makedirs(user_db, exist_ok=True)
    else:
        tmp_db = tempfile.TemporaryDirectory(prefix="rocke_miopen_udb_")
        user_db = tmp_db.name
    env = miopen_env(root, user_db)
    if not args.dry_run:
        preflight(root, env)

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
    kernel_log = {}
    rc = 0
    try:
        for run in runs:
            shown = _run_one(run, root, env, args)
            if args.dry_run:
                continue
            if run.rc != 0 or run.result.ms is None:
                rc = 1
            if writer is not None:
                writer.writerow(csv_row(run, shown))
                csv_file.flush()
            if run.kernels is not None:
                key = f"{run.case.problem.short()}/{run.dtype}/{run.case.direction}"
                kernel_log[key] = {
                    "solver": run.result.solver,
                    "kernels": run.kernels,
                }
    finally:
        if csv_file is not None:
            csv_file.close()
        if tmp_db is not None:
            tmp_db.cleanup()
    if args.csv and kernel_log:
        side = Path(args.csv).with_suffix(".kernels.json")
        side.write_text(json.dumps(kernel_log, indent=1))
        print(f"\nkernel names: {side}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
