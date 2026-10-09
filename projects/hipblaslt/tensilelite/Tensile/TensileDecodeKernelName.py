# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Command-line interface for the Tensile kernel-name decoder."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from Tensile.SolutionStructs.KernelNameDecoder import (
    DecodedKernelName,
    KernelNameDecodeError,
    decode_kernel_name,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TensileDecodeKernelName",
        description="""\
Expand the abbreviated parameters in a Tensile or hipBLASLt kernel or solution
name and write the complete decode to standard output. The decoder uses the
naming and valid-parameter definitions from this Tensile installation; it does
not require a network connection, database, or web service.
""",
        epilog="""\
examples:
  TensileDecodeKernelName \\
    'Cijk_Alik_Bljk_BBS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_AFC1_GSUAMB_ISA950_MIWT2_2_WG32_8_1'
  TensileDecodeKernelName --format json '<kernel-name>'
  printf '%s\\n' '<kernel-name>' | TensileDecodeKernelName --format json

source-checkout entry points:
  Tensile/bin/TensileDecodeKernelName '<kernel-name>'
  python3 -m Tensile.TensileDecodeKernelName '<kernel-name>'

Unknown or historical components are retained and reported as warnings. Use
--strict to return a nonzero status for unknown or ambiguous components, or
when an irreversible filename-shortening hash prevents a complete decode.
Compatibility aliases cover legacy names present in shipped hipBLASLt logic.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "kernel_name",
        nargs="?",
        help="kernel name to decode; reads one name from stdin when omitted",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: %(default)s)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a nonzero status for unknown, ambiguous, or shortened names",
    )
    return parser


def _read_kernel_name(argument: str | None, stdin: TextIO) -> str:
    if argument is not None:
        return argument
    if stdin.isatty():
        raise KernelNameDecodeError(
            "kernel name is required as an argument or redirected on stdin"
        )
    lines = [line.strip() for line in stdin if line.strip()]
    if not lines:
        raise KernelNameDecodeError(
            "kernel name is required as an argument or on stdin"
        )
    if len(lines) != 1:
        raise KernelNameDecodeError("stdin must contain exactly one kernel name")
    return lines[0]


def _display_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def format_decoded_kernel_name(result: DecodedKernelName) -> str:
    """Render a decoded name for terminal use."""

    lines = [
        f"Kernel name:  {result.kernel_name}",
        f"Problem type: {result.problem_type or '(not identified)'}",
        f"Complete:     {'yes' if result.complete else 'no'}",
        "",
        "Parameters:",
    ]
    if not result.parameters:
        lines.append("  (none decoded)")
    else:
        label_width = max(
            len(parameter.name or "/".join(parameter.candidates) or "Unknown")
            for parameter in result.parameters
        )
        component_width = max(
            len(parameter.component) for parameter in result.parameters
        )
        for parameter in result.parameters:
            label = parameter.name or "/".join(parameter.candidates) or "Unknown"
            value = _display_value(parameter.value)
            status = "" if parameter.status == "decoded" else f" [{parameter.status}]"
            lines.append(
                f"  {parameter.component:<{component_width}}  "
                f"{label:<{label_width}} = {value}{status}"
            )
    if result.warnings:
        lines.extend(("", "Warnings:"))
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the command and return its process exit status."""

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    args = _build_parser().parse_args(argv)
    try:
        kernel_name = _read_kernel_name(args.kernel_name, stdin)
        result = decode_kernel_name(kernel_name)
    except (KernelNameDecodeError, TypeError) as error:
        print(f"TensileDecodeKernelName: error: {error}", file=stderr)
        return 2

    if args.format == "json":
        json.dump(result.to_dict(), stdout, indent=2, sort_keys=True)
        stdout.write("\n")
    else:
        print(format_decoded_kernel_name(result), file=stdout)
    return 1 if args.strict and not result.complete else 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
