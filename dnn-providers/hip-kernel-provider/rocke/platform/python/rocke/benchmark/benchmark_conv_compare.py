# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Side-by-side comparison of direct-conv and implicit-GEMM conv benchmarks.

Runs ``benchmark_direct_conv.py`` and ``benchmark_implicit_gemm_conv.py`` on the
same convolution shape, then prints a compact summary of the best kernel found
by each approach.

The two benchmark scripts are invoked as subprocesses so this script carries no
benchmark logic of its own — it only forwards arguments and aggregates results.

Usage
-----
Forward pass (fwd direction, grouped conv cpg=16):

  python benchmark_conv_compare.py \\
      --arch gfx950 --N 8 --Hi 56 --Wi 56 \\
      --C 64 --K 64 --Y 3 --X 3 --pH 1 --pW 1 --groups 4

Forward pass using a MIOpenDriver command:

  python benchmark_conv_compare.py \\
      --arch gfx950 \\
      --miopen-cmd "./MIOpenDriver convfp16 -n 8 -c 64 -H 56 -W 56 \\
          -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -F 1 -in_layout=NHWC"

The ``--top`` / ``--warmup`` / ``--iters`` / ``--jobs`` / ``--verify`` flags are
forwarded to both scripts where applicable.  Flags that apply only to one script
(e.g. ``--direction`` for implicit-GEMM) are silently ignored by the other.

Notes
-----
- Direct conv only supports fp16 and the forward direction; the comparison
  therefore always uses the fwd implicit-GEMM path and fp16 regardless of
  ``--dtype`` / ``--direction``.
- Direct conv requires ``cpg == kpg`` (C/groups == K/groups) and cpg ∈
  {1, 4, 8, 16, 32}.  If the requested shape does not meet these constraints,
  the direct-conv run is skipped and only implicit-GEMM results are shown.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent

# Regex patterns that match the "Best:" summary printed by both benchmarks.
_BEST_RE = re.compile(r"Best:\s*([\d.]+)\s*TFLOPS\s*[—\-]\s*(.+)")


def _run_script(script: Path, extra_args: list[str]) -> tuple[float | None, str, str]:
    """Run *script* with *extra_args*; return (best_tflops, kernel_name, stdout)."""
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n{'='*72}", flush=True)
    print(f"Running: {script.name} {' '.join(extra_args)}", flush=True)
    print(f"{'='*72}", flush=True)

    proc = subprocess.run(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Echo output to the terminal so the user can see the sweep progress.
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)

    m = _BEST_RE.search(proc.stdout)
    if m:
        return float(m.group(1)), m.group(2).strip(), proc.stdout
    return None, "", proc.stdout


def _cpg_valid_for_direct(C: int, K: int, groups: int) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for direct-conv cpg constraints."""
    if C % groups != 0 or K % groups != 0:
        return False, f"C={C} or K={K} not divisible by groups={groups}"
    cpg = C // groups
    kpg = K // groups
    if cpg != kpg:
        return False, f"cpg={cpg} != kpg={kpg} (direct conv requires C/g == K/g)"
    if cpg != 1 and (cpg % 4 != 0 or cpg < 4):
        return (
            False,
            f"cpg={cpg} must be 1 (depthwise) or a positive multiple of 4",
        )
    return True, ""


def _build_shape_args(args) -> list[str]:
    """Shared shape args forwarded to both scripts."""
    a: list[str] = [
        "--arch",
        args.arch,
        "--N",
        str(args.N),
        "--Hi",
        str(args.Hi),
        "--Wi",
        str(args.Wi),
        "--C",
        str(args.C),
        "--K",
        str(args.K),
        "--Y",
        str(args.Y),
        "--X",
        str(args.X),
        "--sH",
        str(args.sH),
        "--sW",
        str(args.sW),
        "--pH",
        str(args.pH),
        "--pW",
        str(args.pW),
        "--groups",
        str(args.groups),
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--jobs",
        str(args.jobs),
    ]
    if args.verify:
        a.append("--verify")
    return a


def _build_miopen_args(args) -> list[str]:
    """MIOpen input args forwarded to both scripts."""
    a: list[str] = [
        "--arch",
        args.arch,
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--jobs",
        str(args.jobs),
    ]
    if args.verify:
        a.append("--verify")
    if args.miopen_cmd:
        a += ["--miopen-cmd", args.miopen_cmd]
    elif args.miopen_file:
        a += ["--miopen-file", args.miopen_file]
    return a


def _print_summary(
    direct: tuple[float | None, str],
    implicit: tuple[float | None, str],
) -> None:
    direct_tflops, direct_name = direct
    implicit_tflops, implicit_name = implicit

    print(f"\n{'='*72}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*72}", flush=True)

    def _fmt(label, tflops, name):
        if tflops is None:
            print(f"  {label:<20}  (no result — skipped or failed)", flush=True)
        else:
            print(f"  {label:<20}  {tflops:7.1f} TFLOPS  {name}", flush=True)

    _fmt("direct-conv", direct_tflops, direct_name)
    _fmt("implicit-GEMM", implicit_tflops, implicit_name)

    if direct_tflops is not None and implicit_tflops is not None:
        ratio = direct_tflops / implicit_tflops
        winner = "direct-conv" if ratio >= 1.0 else "implicit-GEMM"
        print(
            f"\n  Speedup direct/implicit: {ratio:.3f}x  →  {winner} wins",
            flush=True,
        )
    print(f"{'='*72}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct-conv and implicit-GEMM benchmarks for the same conv shape. "
            "Both benchmark scripts are run as subprocesses; their output is echoed and "
            "their best TFLOPS results are printed side-by-side."
        )
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (default: gfx950)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="top-N results to show per benchmark (default: 5)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="parallel compile workers forwarded to both scripts (default: 1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="forward --verify to both benchmark scripts",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        dest="skip_direct",
        help="skip the direct-conv benchmark (implicit-GEMM only)",
    )
    parser.add_argument(
        "--skip-implicit",
        action="store_true",
        dest="skip_implicit",
        help="skip the implicit-GEMM benchmark (direct-conv only)",
    )

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Load the conv shape from a MIOpenDriver command instead of explicit flags.",
    )
    miopen_grp.add_argument(
        "--miopen-cmd",
        default=None,
        metavar="CMD",
        help="MIOpenDriver command string forwarded verbatim to both scripts.",
    )
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        help="Path to a file of MIOpenDriver commands; forwarded to both scripts.",
    )

    shape_grp = parser.add_argument_group("Shape", "convolution shape parameters")
    shape_grp.add_argument("--N", type=int, default=8)
    shape_grp.add_argument("--Hi", type=int, default=56)
    shape_grp.add_argument("--Wi", type=int, default=56)
    shape_grp.add_argument("--C", type=int, default=64)
    shape_grp.add_argument("--K", type=int, default=64)
    shape_grp.add_argument("--Y", type=int, default=3)
    shape_grp.add_argument("--X", type=int, default=3)
    shape_grp.add_argument("--sH", type=int, default=1)
    shape_grp.add_argument("--sW", type=int, default=1)
    shape_grp.add_argument("--pH", type=int, default=1)
    shape_grp.add_argument("--pW", type=int, default=1)
    shape_grp.add_argument("--groups", "-g", type=int, default=1)

    args = parser.parse_args()

    using_miopen = args.miopen_cmd is not None or args.miopen_file is not None

    if using_miopen:
        shared_args = _build_miopen_args(args)
        direct_args = list(shared_args)
        implicit_args = list(shared_args)
    else:
        shared_args = _build_shape_args(args)
        direct_args = list(shared_args)
        # implicit-GEMM needs --dtype (always fp16 for comparison)
        implicit_args = list(shared_args) + ["--dtype", "fp16", "--direction", "fwd"]

        # Validate cpg constraints for direct conv up-front so we can skip
        # gracefully rather than propagating errors through the subprocess.
        if not args.skip_direct:
            ok, reason = _cpg_valid_for_direct(args.C, args.K, args.groups)
            if not ok:
                print(
                    f"[info] direct-conv skipped for this shape: {reason}",
                    file=sys.stderr,
                )
                args.skip_direct = True

    direct_result: tuple[float | None, str] = (None, "")
    implicit_result: tuple[float | None, str] = (None, "")

    if not args.skip_direct:
        tflops, name, _ = _run_script(
            _SCRIPT_DIR / "benchmark_direct_conv.py",
            direct_args,
        )
        direct_result = (tflops, name)

    if not args.skip_implicit:
        tflops, name, _ = _run_script(
            _SCRIPT_DIR / "benchmark_implicit_gemm_conv.py",
            implicit_args,
        )
        implicit_result = (tflops, name)

    _print_summary(direct_result, implicit_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
