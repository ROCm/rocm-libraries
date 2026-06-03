# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI entry point and line-level dispatch for MIOpen shape conversion."""

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bnorm import _bnorm_filename, build_bnorm_json
from .conv import (
    CONV_IO_TYPE,
    ConvDirection,
    ConvParams,
    _conv_filename,
    build_conv_json,
)
from .parsing import parse_args

# ---------------------------------------------------------------------------
# Supported operation sets
# ---------------------------------------------------------------------------

_BNORM_OPERATIONS = {
    "bnorm",
    "bnormfp16",
    "bnormbfp16",
}

_CONV_OPERATIONS = {"conv", "convfp16", "convbfp16"}

_ALL_OPERATIONS = _CONV_OPERATIONS | _BNORM_OPERATIONS


# ---------------------------------------------------------------------------
# Line parser dispatcher
# ---------------------------------------------------------------------------


def parse_line(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse one shape file line. Returns (operation, args_dict) or None.

    Accepts lines with or without an executable prefix::

        ./bin/MIOpenDriver convbfp16 -n 16 ...   # executable + operation
        convbfp16 -n 16 ...                       # bare operation
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Strip leading repeat count (e.g. "     5  ./bin/MIOpenDriver ...")
    m = re.match(r"^\s*\d+\s+", line)
    if m:
        line = line[m.end() :]

    parts = line.split()
    if not parts:
        return None

    if parts[0] in _ALL_OPERATIONS:
        operation = parts[0]
        flag_tokens = parts[1:]
    elif len(parts) < 2:
        return None
    else:
        operation = parts[1]
        flag_tokens = parts[2:]

    args = parse_args(flag_tokens)
    return operation, args


def _expand_directions(F: int) -> List[ConvDirection]:
    """Expand an MIOpen -F bitmask into individual direction values.

    F=0 means all directions.  Otherwise each set bit selects
    a direction: 1=fwd, 2=dgrad, 4=wgrad.  E.g. F=3 → [FORWARD, BACKWARD_DATA].
    """
    if F == 0:
        return list(ConvDirection)
    return [d for d in ConvDirection if F & d]


def convert_line(
    operation: str, args: Dict[str, str], prefix: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """Convert parsed MIOpen args to a list of (filename_stem, json_dict).

    Returns multiple entries when F selects more than one direction
    (F=0 for all, or bitmask combinations like F=3 for fwd+dgrad).
    """
    if operation in _CONV_OPERATIONS:
        p = ConvParams.from_args(args)
        directions = _expand_directions(p.F)
        results = []
        for f in directions:
            p_dir = dataclasses.replace(p, F=f)
            graph = build_conv_json(p_dir, io_type=CONV_IO_TYPE[operation])
            name_stem = _conv_filename(prefix, p_dir)
            graph["name"] = name_stem
            results.append((name_stem, graph))
        return results
    elif operation in _BNORM_OPERATIONS:
        graph = build_bnorm_json(operation, args)
        name_stem = _bnorm_filename(prefix, operation, args)
        graph["name"] = name_stem
        return [(name_stem, graph)]
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MIOpen driver shape files or inline args to hipDNN JSON graph files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="SHAPES_FILE",
        help="One or more MIOpen shape .txt files to convert.",
    )
    parser.add_argument(
        "-A",
        "--args",
        metavar="MIOPEN_ARGS",
        default=None,
        help=(
            "Inline MIOpen driver arguments (everything after the executable path), "
            "e.g. 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 ...'. "
            "Use with --output to write to a specific file."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON file path (used with --args; ignored for file inputs).",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        default=None,
        help=(
            "Output directory for JSON files. "
            "Defaults to the same directory as each input file."
        ),
    )
    return parser


def _process_inline_args(args_str: str, output: Optional[str]) -> int:
    """Convert a single inline MIOpen driver argument string to a JSON file."""
    parts = args_str.split()
    if not parts:
        print("ERROR: --args is empty.", file=sys.stderr)
        return 1

    operation = parts[0]
    flag_tokens = parts[1:]
    parsed_args = parse_args(flag_tokens)

    try:
        results = convert_line(operation, parsed_args, "")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if output and len(results) > 1:
        print(
            "ERROR: --output cannot be used when F selects multiple directions. "
            "Omit --output to write each direction to a separate file.",
            file=sys.stderr,
        )
        return 1

    for name_stem, graph in results:
        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = Path(f"{name_stem}.json")

        out_path.write_text(json.dumps(graph, indent=4) + "\n")
        print(f"Written: {out_path}")
    return 0


def _process_file(input_path: Path, outdir: Path) -> Tuple[int, int, int]:
    """Process one shape file. Returns (written, skipped, warnings)."""
    prefix = input_path.stem.lower().replace("-", "_")

    written = 0
    skipped = 0
    warnings = 0
    seen_stems: Dict[str, int] = {}

    with open(input_path) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            parsed = parse_line(raw_line)
            if parsed is None:
                skipped += 1
                continue

            operation, args = parsed
            try:
                results = convert_line(operation, args, prefix)
            except Exception as exc:
                print(f"  WARNING line {lineno}: {exc}", file=sys.stderr)
                warnings += 1
                continue

            for name_stem, graph in results:
                count = seen_stems.get(name_stem, 0)
                seen_stems[name_stem] = count + 1
                if count > 0:
                    unique_stem = f"{name_stem}_{count}"
                else:
                    unique_stem = name_stem

                out_path = outdir / f"{unique_stem}.json"
                out_path.write_text(json.dumps(graph, indent=4) + "\n")
                written += 1

    return written, skipped, warnings


def main() -> int:
    parser = _build_arg_parser()
    ns = parser.parse_args()

    if ns.args:
        if ns.inputs:
            print(
                "ERROR: cannot combine --args with positional SHAPES_FILE arguments.",
                file=sys.stderr,
            )
            return 1
        return _process_inline_args(ns.args, ns.output)

    if not ns.inputs:
        parser.print_help(sys.stderr)
        return 1

    total_written = 0
    total_skipped = 0
    total_warnings = 0

    for input_str in ns.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            return 1

        outdir = Path(ns.outdir) if ns.outdir else input_path.parent
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {input_path} → {outdir}/")
        written, skipped, warns = _process_file(input_path, outdir)
        print(f"  Written: {written}  Skipped: {skipped}  Warnings: {warns}")
        total_written += written
        total_skipped += skipped
        total_warnings += warns

    print(f"\nTotal: {total_written} files written, {total_warnings} warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
