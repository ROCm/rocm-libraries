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

Each coverage machine exports its own lcov, because ``llvm-cov export`` needs
that machine's instrumented binaries. The exported reports are what can be
combined afterwards: a line is covered when any input executed it, and hit
counts are the max across inputs.

``BRDA`` ids from ``llvm-cov`` are positional within a line. They line up
across reports when both reports come from the same sources and the same
build configuration. When the ids on a line differ, that line's branch
records are dropped and a warning is printed. Line records are kept. The
drop lasts for later inputs in the same run. The written file has no marker
for it, so a later run can add branch records for that line again.
``--strict`` exits non-zero when any line was dropped.

``SF:`` paths are rewritten to start at the last ``projects/rocsparse/``
component, so two Jenkins workspaces with different absolute prefixes refer
to the same file. ``make coverage`` writes ``coverage-upload.info``.
``make coverage_split`` writes ``coverage-unit-upload.info`` and
``coverage-integration-upload.info``. Summaries (``LF``, ``LH``, ``FNF``,
``FNH``, ``BRF``, ``BRH``) are recomputed from the records, and ``TN:`` lines
are dropped. A record type this script does not know is passed through.
Input is ``llvm-cov export -format=lcov`` output.

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
        self.dropped_branch_lines = set()  # lines whose BRDA ids disagreed
        self.extra = []  # unrecognized records, passed through


def normalize_path(path: str, anchor: str) -> str:
    """Return ``path`` starting at ``anchor`` when that component is present."""
    normalized = path.replace("\\", "/")
    start = 0
    chosen = None
    while True:
        index = normalized.find(anchor, start)
        if index < 0:
            break
        if index == 0 or normalized[index - 1] == "/":
            chosen = index
        start = index + 1
    if chosen is None:
        return normalized
    return normalized[chosen:]


def _parse_count(token: str, where: str) -> int:
    if not token.isdigit():
        raise SystemExit(f"merge_coverage_info: bad count in {where}")
    return int(token)


def _parse_taken(token: str, where: str):
    if token == "-":
        return None
    return _parse_count(token, where)


def _merge_function(
    record: FileRecord, name: str, start_line: int, warnings: list, where: str
):
    if name not in record.functions:
        record.functions[name] = start_line
        return
    if record.functions[name] != start_line:
        warnings.append(
            f"{where}: function {name} start line {record.functions[name]}"
            f" vs {start_line}; keeping {record.functions[name]}"
        )


def _merge_line(record: FileRecord, line_number: int, count: int, checksum):
    previous = record.lines.get(line_number)
    if previous is None or count > previous[0]:
        record.lines[line_number] = (count, checksum)
    elif count == previous[0] and previous[1] is None and checksum is not None:
        record.lines[line_number] = (count, checksum)


def _merge_branch(record: FileRecord, key: tuple, taken):
    previous = record.branches.get(key)
    if previous is None:
        record.branches[key] = taken
    elif taken is not None:
        record.branches[key] = max(previous, taken)


def _mismatched_branch_lines(
    source_path: str, current: FileRecord, incoming: FileRecord, warnings: list
):
    """Return the lines whose positional BRDA ids differ, appending a warning for each."""

    def keys_by_line(record: FileRecord) -> dict:
        grouped = {}
        for line_number, block, branch in record.branches:
            grouped.setdefault(line_number, set()).add((block, branch))
        return grouped

    left = keys_by_line(current)
    right = keys_by_line(incoming)
    mismatched = set()
    for line_number in sorted(set(left) & set(right)):
        if left[line_number] != right[line_number]:
            mismatched.add(line_number)
            warnings.append(
                f"{source_path}:{line_number}: branch ids"
                f" {sorted(left[line_number])} vs {sorted(right[line_number])};"
                " dropped these BRDA records (ids are positional)"
            )
    return mismatched


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
            if not current_path.startswith(anchor):
                warnings.append(
                    f"{where}: SF path does not start with {anchor!r}: {current_path}"
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
            end_part, separator, _rest = name.partition(",")
            if separator and end_part.isdigit():
                raise SystemExit(
                    f"merge_coverage_info: FN end-line form is not accepted at {where}"
                    " (expect llvm-cov export -format=lcov)"
                )
            _merge_function(current, name, start_line, warnings, where)
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
            _merge_line(current, da_line, count, checksum)
            continue
        if line.startswith("BRDA:"):
            parts = line[5:].split(",")
            if len(parts) != 4:
                raise SystemExit(f"merge_coverage_info: bad BRDA record at {where}")
            br_line = _parse_count(parts[0], where)
            block = _parse_count(parts[1], where)
            branch = _parse_count(parts[2], where)
            taken = _parse_taken(parts[3], where)
            _merge_branch(current, (br_line, block, branch), taken)
            continue
        prefix = line.split(":", 1)[0]
        if not any(item.startswith(prefix + ":") for item in current.extra):
            warnings.append(f"{where}: unrecognized record {prefix} passed through")
        current.extra.append(line)

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
    mismatched = _mismatched_branch_lines(source_path, current, incoming, warnings)
    current.dropped_branch_lines.update(mismatched)
    current.dropped_branch_lines.update(incoming.dropped_branch_lines)
    for name, start_line in incoming.functions.items():
        _merge_function(current, name, start_line, warnings, source_path)
    for name, count in incoming.function_hits.items():
        current.function_hits[name] = max(current.function_hits.get(name, 0), count)
    for line_number, (count, checksum) in incoming.lines.items():
        _merge_line(current, line_number, count, checksum)
    if current.dropped_branch_lines:
        current.branches = {
            key: taken
            for key, taken in current.branches.items()
            if key[0] not in current.dropped_branch_lines
        }
    for key, taken in incoming.branches.items():
        if key[0] in current.dropped_branch_lines:
            continue
        _merge_branch(current, key, taken)
    for item in incoming.extra:
        if item not in current.extra:
            current.extra.append(item)


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
        lines.extend(record.extra)
        lines.append("end_of_record")
        chunks.append("\n".join(lines))
    return "\n".join(chunks) + "\n"


def merge_files(inputs: list, anchor: str) -> tuple:
    """Union ``inputs``.

    Returns ``(lcov text, warnings, file count, lines found, lines hit,
    dropped branch lines)``.
    """
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
    dropped = sum(len(record.dropped_branch_lines) for record in combined.values())
    return text, warnings, len(combined), lines_found, lines_hit, dropped


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any line's branch records were dropped",
    )
    parser.add_argument("inputs", nargs="+", help="lcov files to union")
    args = parser.parse_args(argv)
    if not args.anchor:
        parser.error("--anchor must not be empty")
    return args


def main(argv: list) -> int:
    args = parse_args(argv)
    text, warnings, file_count, lines_found, lines_hit, dropped = merge_files(
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
    dropped_label = "dropped branch line" if dropped == 1 else "dropped branch lines"
    print(
        f"merge_coverage_info: {len(args.inputs)} inputs, {file_count} files,"
        f" {lines_hit} lines hit, {lines_found} lines found,"
        f" {dropped} {dropped_label}"
    )
    if args.strict and dropped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
