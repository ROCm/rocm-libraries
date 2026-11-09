#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Union LLVM lcov reports and give them a common source path.

A wave32 coverage run and a wave64 coverage run are different binaries, so
``llvm-profdata merge`` rejects their raw profiles. Line coverage still
combines at the lcov layer: a line is covered when any input report executed
it. Hit counts are the max across inputs.

``SF:`` paths are rewritten to start at ``projects/rocsparse/`` when that
anchor is present, so two Jenkins workspaces with different absolute prefixes
refer to the same file. ``make coverage`` writes the rewritten file as
``coverage-report/coverage-upload.info``.

Usage:
    python3 merge_coverage_info.py -o combined.info report-a.info report-b.info
"""

import argparse
import sys
from pathlib import Path


DEFAULT_ANCHOR = "projects/rocsparse/"

# Summary records are recomputed on write. Everything else is data.
_SUMMARY_PREFIXES = ("LF:", "LH:", "FNF:", "FNH:", "BRF:", "BRH:")


class FileRecord:
    """Coverage records for one source file."""

    def __init__(self) -> None:
        self.functions = {}  # name -> start line (first input wins)
        self.function_hits = {}  # name -> execution count
        self.lines = {}  # line number -> (count, checksum or None)
        self.branches = {}  # (line, block, branch) -> taken count or None


def normalize_path(path: str, anchor: str) -> str:
    """Return ``path`` starting at ``anchor`` when that component is present."""
    normalized = path.replace("\\", "/")
    start = 0
    while True:
        index = normalized.find(anchor, start)
        if index < 0:
            return normalized
        if index == 0 or normalized[index - 1] == "/":
            return normalized[index:]
        start = index + 1


def _parse_count(token: str, where: str) -> int:
    try:
        value = int(token)
    except ValueError as error:
        raise SystemExit(f"merge_coverage_info: bad count in {where}") from error
    if value < 0:
        raise SystemExit(f"merge_coverage_info: negative count in {where}")
    return value


def _parse_taken(token: str, where: str):
    if token == "-":
        return None
    return _parse_count(token, where)


def read_lcov(path: Path, anchor: str, warnings: list) -> dict:
    """Parse one lcov file into ``{normalized SF path: FileRecord}``."""
    records = {}
    current_path = None
    current = None
    saw_record = False

    def finish():
        nonlocal current_path, current, saw_record
        if current_path is None or current is None:
            raise SystemExit(
                f"merge_coverage_info: end_of_record with no SF: in {path}"
            )
        _merge_record(records, current_path, current, warnings)
        current_path = None
        current = None
        saw_record = True

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SystemExit(f"merge_coverage_info: cannot read {path}: {error}") from error

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("TN:") or line.startswith(_SUMMARY_PREFIXES):
            continue
        where = f"{path}:{line_number}"
        if line == "end_of_record":
            finish()
            continue
        if line.startswith("SF:"):
            if current_path is not None:
                raise SystemExit(
                    f"merge_coverage_info: SF: before end_of_record at {where}"
                )
            original = line[3:]
            current_path = normalize_path(original, anchor)
            if current_path == original and anchor not in original.replace("\\", "/"):
                warnings.append(
                    f"{where}: left SF path unchanged (no {anchor!r} component): {original}"
                )
            current = FileRecord()
            continue
        if current is None:
            raise SystemExit(f"merge_coverage_info: record before SF: at {where}")
        if line.startswith("FN:"):
            payload = line[3:]
            split_at = payload.find(",")
            if split_at < 1 or split_at == len(payload) - 1:
                raise SystemExit(f"merge_coverage_info: bad FN record at {where}")
            start_line = _parse_count(payload[:split_at], where)
            name = payload[split_at + 1 :]
            if name not in current.functions:
                current.functions[name] = start_line
            elif current.functions[name] != start_line:
                warnings.append(
                    f"{where}: function {name} start line {current.functions[name]}"
                    f" vs {start_line}; keeping {current.functions[name]}"
                )
            continue
        if line.startswith("FNDA:"):
            payload = line[5:]
            split_at = payload.find(",")
            if split_at < 1 or split_at == len(payload) - 1:
                raise SystemExit(f"merge_coverage_info: bad FNDA record at {where}")
            count = _parse_count(payload[:split_at], where)
            name = payload[split_at + 1 :]
            previous = current.function_hits.get(name, 0)
            current.function_hits[name] = max(previous, count)
            continue
        if line.startswith("DA:"):
            parts = line[3:].split(",")
            if len(parts) not in (2, 3):
                raise SystemExit(f"merge_coverage_info: bad DA record at {where}")
            da_line = _parse_count(parts[0], where)
            count = _parse_count(parts[1], where)
            checksum = parts[2] if len(parts) == 3 and parts[2] else None
            previous = current.lines.get(da_line)
            if previous is None or count > previous[0]:
                current.lines[da_line] = (count, checksum)
            elif count == previous[0] and previous[1] is None and checksum is not None:
                current.lines[da_line] = (count, checksum)
            continue
        if line.startswith("BRDA:"):
            parts = line[5:].split(",")
            if len(parts) != 4:
                raise SystemExit(f"merge_coverage_info: bad BRDA record at {where}")
            br_line = _parse_count(parts[0], where)
            block = _parse_count(parts[1], where)
            branch = _parse_count(parts[2], where)
            taken = _parse_taken(parts[3], where)
            key = (br_line, block, branch)
            previous = current.branches.get(key, None)
            if key not in current.branches or previous is None:
                current.branches[key] = taken
            elif taken is not None:
                current.branches[key] = max(previous, taken)
            continue
        raise SystemExit(f"merge_coverage_info: unrecognized record at {where}: {line}")

    if current_path is not None:
        raise SystemExit(
            f"merge_coverage_info: missing end_of_record after SF:{current_path} in {path}"
        )
    if not saw_record:
        raise SystemExit(f"merge_coverage_info: no coverage records in {path}")
    return records


def _merge_record(
    destination: dict, source_path: str, incoming: FileRecord, warnings: list
):
    current = destination.get(source_path)
    if current is None:
        destination[source_path] = incoming
        return
    for name, start_line in incoming.functions.items():
        if name not in current.functions:
            current.functions[name] = start_line
        elif current.functions[name] != start_line:
            warnings.append(
                f"{source_path}: function {name} start line {current.functions[name]}"
                f" vs {start_line}; keeping {current.functions[name]}"
            )
    for name, count in incoming.function_hits.items():
        current.function_hits[name] = max(current.function_hits.get(name, 0), count)
    for line_number, (count, checksum) in incoming.lines.items():
        previous = current.lines.get(line_number)
        if previous is None or count > previous[0]:
            current.lines[line_number] = (count, checksum)
        elif count == previous[0] and previous[1] is None and checksum is not None:
            current.lines[line_number] = (count, checksum)
    for key, taken in incoming.branches.items():
        if key not in current.branches or current.branches[key] is None:
            current.branches[key] = taken
        elif taken is not None:
            current.branches[key] = max(current.branches[key], taken)


def format_lcov(records: dict) -> str:
    """Serialize records in a stable order, with summaries recomputed."""
    chunks = []
    for source_path in sorted(records):
        record = records[source_path]
        lines = [f"SF:{source_path}"]
        for name, start_line in sorted(
            record.functions.items(), key=lambda item: (item[1], item[0])
        ):
            lines.append(f"FN:{start_line},{name}")
        for name in sorted(record.function_hits):
            lines.append(f"FNDA:{record.function_hits[name]},{name}")
        lines.append(f"FNF:{len(record.functions)}")
        functions_hit = sum(
            1 for name in record.functions if record.function_hits.get(name, 0) > 0
        )
        lines.append(f"FNH:{functions_hit}")
        if record.branches:
            for key in sorted(record.branches):
                taken = record.branches[key]
                taken_text = "-" if taken is None else str(taken)
                lines.append(f"BRDA:{key[0]},{key[1]},{key[2]},{taken_text}")
            lines.append(f"BRF:{len(record.branches)}")
            branches_hit = sum(
                1
                for taken in record.branches.values()
                if taken is not None and taken > 0
            )
            lines.append(f"BRH:{branches_hit}")
        for line_number in sorted(record.lines):
            count, checksum = record.lines[line_number]
            if checksum is None:
                lines.append(f"DA:{line_number},{count}")
            else:
                lines.append(f"DA:{line_number},{count},{checksum}")
        lines.append(f"LF:{len(record.lines)}")
        lines_hit = sum(1 for count, _checksum in record.lines.values() if count > 0)
        lines.append(f"LH:{lines_hit}")
        lines.append("end_of_record")
        chunks.append("\n".join(lines))
    return "\n".join(chunks) + "\n"


def merge_files(inputs: list, anchor: str) -> tuple:
    """Union ``inputs`` and return ``(lcov text, warnings, file count, lines found, lines hit)``."""
    combined = {}
    warnings = []
    for path in inputs:
        for source_path, record in read_lcov(Path(path), anchor, warnings).items():
            _merge_record(combined, source_path, record, warnings)
    text = format_lcov(combined)
    lines_found = sum(len(record.lines) for record in combined.values())
    lines_hit = sum(
        sum(1 for count, _checksum in record.lines.values() if count > 0)
        for record in combined.values()
    )
    return text, warnings, len(combined), lines_found, lines_hit


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="destination lcov file",
    )
    parser.add_argument(
        "--anchor",
        default=DEFAULT_ANCHOR,
        help=f"path component SF: entries are rewritten to start at (default: {DEFAULT_ANCHOR})",
    )
    parser.add_argument("inputs", nargs="+", help="lcov files to union")
    args = parser.parse_args(argv)
    if not args.anchor:
        parser.error("--anchor must not be empty")
    return args


def main(argv: list) -> int:
    args = parse_args(argv)
    text, warnings, file_count, lines_found, lines_hit = merge_files(
        args.inputs, args.anchor
    )
    for warning in warnings:
        print(f"merge_coverage_info: {warning}", file=sys.stderr)
    output = Path(args.output)
    try:
        output.write_text(text, encoding="utf-8")
    except OSError as error:
        raise SystemExit(
            f"merge_coverage_info: cannot write {output}: {error}"
        ) from error
    print(
        f"merge_coverage_info: {len(args.inputs)} inputs, {file_count} files,"
        f" {lines_hit} lines hit, {lines_found} lines found",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
