################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Convert legacy list-format Tensile library logic YAML to dict format."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml

from Tensile import LibraryIO
from Tensile.CustomYamlLoader import load_yaml_stream
from Tensile.TensileMergeLibrary import (
    convertToDict,
    normalizeDictLibraryLayout,
    removeDefaultInitParams,
)

# Canonical top-level key order for dict-format library logic. Matches
# ``createLibraryLogic`` / ``parseLibraryLogicList`` output and the merge write path.
_CANONICAL_LOGIC_KEYS = (
    "MinimumRequiredVersion",
    "ScheduleName",
    "ArchitectureName",
    "CUCount",
    "DeviceNames",
    "ProblemType",
    "DefaultSolution",
    "Solutions",
    "IndexOrder",
    "ExactLogic",
    "RangeLogic",
    "TileSelectionIndices",
    "PerfMetric",
    "LibraryType",
)


def _count_file_lines(path: str) -> int:
    """Return the number of lines in a text file.

    Args:
        path: File path to count.

    Returns:
        Line count (0 for an empty file).

    Raises:
        OSError: If the file cannot be read.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _reorderTopLevelKeys(data: dict[str, Any]) -> None:
    """Reorder top-level dict keys to the canonical library-logic layout.

    Ensures ``TileSelectionIndices`` is present (null when unused), drops any
    leftover ``Library`` block, and emits keys in the same order as
    ``createLibraryLogic`` and ``TensileMergeLibrary`` merge output.

    Args:
        data: Dict-format library logic (mutated in place).

    Returns:
        None.

    Raises:
        None.
    """
    data.pop("Library", None)
    if "TileSelectionIndices" not in data:
        data["TileSelectionIndices"] = None

    ordered: dict[str, Any] = {}
    for key in _CANONICAL_LOGIC_KEYS:
        if key in data:
            ordered[key] = data[key]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    data.clear()
    data.update(ordered)


def _finalizeDictForWrite(data: dict[str, Any]) -> None:
    """Apply canonical dict layout before YAML write.

    Mirrors the merge-library write path: strip default-valued solution fields,
    normalize ``LibraryType`` / drop ``Library``, sort ``ProblemType`` keys,
    reorder solution naming fields, and enforce canonical top-level key order.

    Args:
        data: Dict-format library logic (mutated in place).

    Returns:
        None.

    Raises:
        None.
    """
    if isinstance(data.get("DefaultSolution"), dict):
        removeDefaultInitParams(data)
    normalizeDictLibraryLayout(data)
    if isinstance(data.get("ProblemType"), dict):
        data["ProblemType"] = dict(sorted(data["ProblemType"].items()))
    LibraryIO.reorderSolutionsParams(data)
    _reorderTopLevelKeys(data)


def convertLibraryLogicData(
    data: list[Any] | dict[str, Any],
    src_file: str,
) -> tuple[dict[str, Any], bool]:
    """Convert in-memory library logic from list to dict layout.

    Uses the same in-memory steps as ``TensileMergeLibrary.loadData``:
    ``convertToDict`` for legacy lists, then ``normalizeDictLibraryLayout``.
    Call :func:`_finalizeDictForWrite` before writing YAML (as merge does).

    Args:
        data: Loaded YAML root (legacy list or dict mapping).
        src_file: Source path for error messages in ``parseLibraryLogicList``.

    Returns:
        Tuple of (dict-format data, whether migration or layout fix-up occurred).

    Raises:
        TypeError: If *data* is neither a list nor a dict.
    """
    was_list = isinstance(data, list)
    if was_list:
        data = convertToDict(data, src_file)
    elif not isinstance(data, dict):
        raise TypeError(
            f"Unsupported library logic root type {type(data).__name__} in {src_file}"
        )

    layout_changed = normalizeDictLibraryLayout(data)
    return data, was_list or layout_changed


def _resolve_output_path(
    input_path: str,
    output_path: str | None,
    in_place: bool,
) -> str:
    """Compute the destination path for a converted file.

    Args:
        input_path: Source ``.yaml`` file path.
        output_path: User ``--output`` argument (file or directory), or None.
        in_place: When True, overwrite *input_path*.

    Returns:
        Absolute or relative output file path.

    Raises:
        ValueError: If *output_path* is missing when not converting in place.
    """
    if in_place:
        return input_path
    if output_path is None:
        raise ValueError(
            "Conversion requires --output PATH or --in-place"
        )
    if os.path.isdir(output_path):
        return os.path.join(output_path, os.path.basename(input_path))
    return output_path


def convertLibraryLogicFile(
    input_path: str,
    output_path: str | None = None,
    in_place: bool = False,
    force: bool = False,
) -> str:
    """Convert one library logic YAML file to dict format on disk.

    Args:
        input_path: Path to input ``.yaml`` (list or dict root).
        output_path: Destination file or directory when not using in-place mode.
        in_place: When True, overwrite *input_path* with dict-format YAML.
        force: When True, rewrite dict-format files even if already canonical.

    Returns:
        Path written (or *input_path* when skipped).

    Raises:
        ValueError: List input without ``--output`` or ``--in-place``.
        TypeError: Unsupported YAML root type.
        OSError: If the output directory cannot be created.
    """
    input_path = os.path.realpath(input_path)
    lines_before = _count_file_lines(input_path)
    raw = load_yaml_stream(input_path, yaml.CSafeLoader)

    if isinstance(raw, list) and not in_place and output_path is None:
        raise ValueError(
            f"Conversion requires --output PATH or --in-place: {input_path}"
        )

    if (
        isinstance(raw, dict)
        and not force
        and not in_place
        and output_path is None
        and "Library" not in raw
    ):
        print(f"Skipped (already dict): {input_path}")
        return input_path

    converted, migrated = convertLibraryLogicData(raw, input_path)

    if isinstance(raw, dict) and not force and not migrated and not in_place:
        if output_path is None:
            print(f"Skipped (already dict): {input_path}")
            return input_path

    _finalizeDictForWrite(converted)
    dest = _resolve_output_path(input_path, output_path, in_place)

    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    LibraryIO.writeYAML(
        dest,
        converted,
        explicit_start=False,
        explicit_end=False,
        sort_keys=False,
    )

    lines_after = _count_file_lines(dest)
    line_summary = f" ({lines_before} lines -> {lines_after} lines)"
    if migrated or isinstance(raw, list):
        print(f"Converted {input_path} -> {dest}{line_summary}")
    else:
        print(f"Wrote {dest}{line_summary}")
    return dest


def main() -> None:
    """CLI entry: convert one or more library logic YAML files to dict format."""
    parser = argparse.ArgumentParser(
        description="Convert list-format Tensile library logic YAML to dict format.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .yaml library logic file(s)",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--output",
        metavar="PATH",
        help="Output file (single input) or directory (multiple inputs)",
    )
    out_group.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file with dict-format YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite dict-format files even when already canonical",
    )
    args = parser.parse_args()

    if len(args.inputs) > 1 and args.output and not os.path.isdir(args.output):
        if os.path.exists(args.output):
            print(
                f"[Error] --output must be a directory for multiple inputs: {args.output}",
                file=sys.stderr,
            )
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)

    if len(args.inputs) == 1 and args.output and os.path.isdir(args.output):
        pass
    elif len(args.inputs) == 1 and args.output and not args.in_place:
        parent = os.path.dirname(os.path.realpath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)

    failures = 0
    for input_path in args.inputs:
        try:
            convertLibraryLogicFile(
                input_path,
                output_path=args.output,
                in_place=args.in_place,
                force=args.force,
            )
        except (ValueError, TypeError, OSError) as exc:
            print(f"[Error] {input_path}: {exc}", file=sys.stderr)
            failures += 1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
