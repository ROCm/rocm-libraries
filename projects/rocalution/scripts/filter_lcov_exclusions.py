#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

"""
Filter an LCOV coverage file exported by llvm-cov.

Exclusions for line, branch and function records:
  - LCOV_EXCL_START / LCOV_EXCL_STOP: Exclude a block of lines. START and STOP
    on the same line exclude only that line.
  - LCOV_EXCL_LINE: Exclude a single line
  - GCOV_ and GCOVR_ prefixes are accepted as well
  - COVERAGE_EXCL_NO_MPI_START / COVERAGE_EXCL_NO_MPI_STOP: Exclude a block of
    multi-node code, only when --no-mpi is given. These blocks are tracked
    independently, so LCOV_EXCL blocks may be nested inside them.

Exclusions for branch records only:
  - LCOV_EXCL_BR_LINE: Exclude the branches of a single line
  - LCOV_EXCL_BR_START / LCOV_EXCL_BR_STOP: Exclude the branches of a block
  - Every line matching --exclude-branches REGEX. The default regex matches the
    logging macros, whose MPI rank check is a branch at every call site.

Line records (DA) of lines without code are dropped: blank lines, comments,
lines with only braces, and preprocessor directives. llvm-cov reports such
lines as missed when they start a region that cannot run, for example the part
of a switch body before the first case label.

Function records (FN/FNDA) on excluded lines are dropped as well, so that
genhtml does not fail with "function ... found on line but no corresponding
'line' coverage data point". For the same reason, a line without code keeps
its line record if a function record points at it: llvm-cov puts the function
record on the line of the opening brace.

The line, branch and function totals (LF/LH, BRF/BRH, FNF/FNH) are recomputed
for every source file.

Usage:
    filter_lcov_exclusions.py [--no-mpi] [--exclude-branches REGEX] input.info output.info
"""

import os
import re
import sys

PREFIX = r"(LCOV|GCOV|GCOVR)_EXCL_"
EXCL_START_PATTERN = re.compile(PREFIX + r"START")
EXCL_STOP_PATTERN = re.compile(PREFIX + r"STOP")
EXCL_LINE_PATTERN = re.compile(PREFIX + r"LINE")
EXCL_BR_START_PATTERN = re.compile(PREFIX + r"BR_START")
EXCL_BR_STOP_PATTERN = re.compile(PREFIX + r"BR_STOP")
EXCL_BR_LINE_PATTERN = re.compile(PREFIX + r"BR_LINE")

# Markers for code that only runs with more than one process. It cannot be
# reached in a build without MPI, but is measured in one with MPI.
NO_MPI_START_PATTERN = re.compile(r"COVERAGE_EXCL_NO_MPI_START")
NO_MPI_STOP_PATTERN = re.compile(r"COVERAGE_EXCL_NO_MPI_STOP")

DEFAULT_BRANCH_EXCLUDE = r"\b(LOG_INFO|LOG_VERBOSE_INFO)\s*\("

NO_CODE_PATTERN = re.compile(r"^[{};\s]*$")

# Directives without code of their own. #pragma omp starts a parallel region.
NO_CODE_DIRECTIVE = re.compile(
    r"^#\s*(if|ifdef|ifndef|elif|else|endif|define|undef|include)\b"
)

# Whole-file (SF section) exclusions. Any LCOV source-file section whose path
# matches one of these is dropped entirely, so the coverage report tracks only
# the library sources. This also drops ROCm and system headers that are inlined
# into the library, the headers generated into the build tree, as well as the
# logging utilities and host I/O routines.
SECTION_EXCLUDE_PATTERNS = [
    re.compile(r"(^|/)clients/"),
    re.compile(r"/include/rocalution/"),
    re.compile(r"/src/utils/"),
    re.compile(r"/src/base/host/host_io\."),
    re.compile(r"^/opt/"),
    re.compile(r"^/usr/"),
]


def is_excluded_section(source_file):
    """True if an entire SF section should be dropped (non-library source)."""
    return any(p.search(source_file) for p in SECTION_EXCLUDE_PATTERNS)


def has_no_code(line, in_block_comment):
    """True for blank lines, comments, braces and preprocessor directives."""
    s = line.strip()
    if in_block_comment or NO_CODE_DIRECTIVE.match(s):
        return True
    s = re.sub(r"/\*.*?\*/", "", s)
    s = s.split("//", 1)[0]
    if "/*" in s:
        s = s.split("/*", 1)[0]
    return NO_CODE_PATTERN.match(s) is not None


class SourceExclusions:
    """Line numbers of one source file to drop, per record type."""

    def __init__(self, source_file, exclude_no_mpi, branch_regex):
        self.lines = set()  # DA, BRDA and FN records
        self.branches = set()  # BRDA records only
        self.no_code = set()  # DA records, unless a function starts there

        if not os.path.isfile(source_file):
            return
        try:
            with open(source_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.readlines()
        except OSError as e:
            print(f"Warning: Could not read {source_file}: {e}", file=sys.stderr)
            return

        in_block = in_no_mpi = in_br_block = in_comment = False
        for num, line in enumerate(text, start=1):
            if has_no_code(line, in_comment):
                self.no_code.add(num)
            if "/*" in line.split("//", 1)[0] and "*/" not in line.split("/*", 1)[1]:
                in_comment = True
            elif in_comment and "*/" in line:
                in_comment = False

            if EXCL_BR_START_PATTERN.search(line):
                in_br_block = True
            if (
                in_br_block
                or EXCL_BR_LINE_PATTERN.search(line)
                or branch_regex.search(line)
            ):
                self.branches.add(num)
            if EXCL_BR_STOP_PATTERN.search(line):
                in_br_block = False

            # A STOP after the START on the same line closes the block there
            no_mpi_start = NO_MPI_START_PATTERN.search(line) is not None
            no_mpi_stop = NO_MPI_STOP_PATTERN.search(line) is not None
            if no_mpi_start or no_mpi_stop or in_no_mpi:
                in_no_mpi = (in_no_mpi or no_mpi_start) and not no_mpi_stop
                if exclude_no_mpi:
                    self.lines.add(num)
                    continue
                if no_mpi_stop:
                    continue

            start = EXCL_START_PATTERN.search(line) is not None
            stop = EXCL_STOP_PATTERN.search(line) is not None
            if start or stop or in_block or EXCL_LINE_PATTERN.search(line):
                in_block = (in_block or start) and not stop
                self.lines.add(num)

        for name, still_open in (
            ("LCOV_EXCL_START", in_block),
            ("COVERAGE_EXCL_NO_MPI_START", in_no_mpi),
            ("LCOV_EXCL_BR_START", in_br_block),
        ):
            if still_open:
                print(
                    f"Warning: {source_file}: {name} without a STOP excludes the rest of the file",
                    file=sys.stderr,
                )


def record_line(record, prefix_len):
    try:
        return int(record[prefix_len:].split(",", 1)[0])
    except ValueError:
        return None


def filter_section(records, excl, stats):
    """Filter the records of one SF section and recompute its totals."""
    fn_lines = set()
    dropped_functions = set()
    kept = []
    for r in records:
        if r.startswith("FN:"):
            num = record_line(r, 3)
            if num in excl.lines:
                dropped_functions.add(r.split(",", 1)[1])
                stats["functions"] += 1
                continue
            fn_lines.add(num)
        elif r.startswith("FNDA:"):
            if r.split(",", 1)[1] in dropped_functions:
                continue
        elif r.startswith("DA:"):
            num = record_line(r, 3)
            if num in excl.lines or (num in excl.no_code and num not in fn_lines):
                stats["lines"] += 1
                continue
        elif r.startswith("BRDA:"):
            num = record_line(r, 5)
            if num in excl.lines or num in excl.branches:
                stats["branches"] += 1
                continue
        elif r.split(":", 1)[0] in ("FNF", "FNH", "LF", "LH", "BRF", "BRH"):
            continue
        kept.append(r)

    hit_functions = {
        r.split(",", 1)[1]
        for r in kept
        if r.startswith("FNDA:") and not r.startswith("FNDA:0,")
    }
    functions = {r.split(",", 1)[1] for r in kept if r.startswith("FN:")}
    da = [r for r in kept if r.startswith("DA:")]
    brda = [r for r in kept if r.startswith("BRDA:")]

    out = [r for r in kept if not r.startswith(("DA:", "BRDA:"))]
    out += [f"FNF:{len(functions)}", f"FNH:{len(hit_functions & functions)}"]
    out += da
    out += brda
    out += [
        f"BRF:{len(brda)}",
        f"BRH:{sum(1 for r in brda if r.rsplit(',', 1)[1] not in ('-', '0'))}",
        f"LF:{len(da)}",
        f"LH:{sum(1 for r in da if r[3:].split(',')[1] != '0')}",
    ]
    return out


def filter_lcov_file(input_path, output_path, exclude_no_mpi, branch_regex):
    stats = {"lines": 0, "branches": 0, "functions": 0, "files_dropped": 0}
    cache = {}
    output = []
    header = []
    source = None
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("TN:"):
                header = [line]
            elif line.startswith("SF:"):
                source = line[3:]
                records = []
            elif line == "end_of_record":
                if is_excluded_section(source):
                    stats["files_dropped"] += 1
                else:
                    if source not in cache:
                        cache[source] = SourceExclusions(
                            source, exclude_no_mpi, branch_regex
                        )
                    output += header + [f"SF:{source}"]
                    output += filter_section(records, cache[source], stats)
                    output.append(line)
                header = []
                source = None
            elif source is not None:
                records.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")
    return stats


def main():
    args = sys.argv[1:]
    exclude_no_mpi = "--no-mpi" in args
    args = [a for a in args if a != "--no-mpi"]
    branch_exclude = DEFAULT_BRANCH_EXCLUDE
    if "--exclude-branches" in args:
        i = args.index("--exclude-branches")
        branch_exclude = args[i + 1]
        del args[i : i + 2]
    if len(args) != 2:
        print(
            f"Usage: {sys.argv[0]} [--no-mpi] [--exclude-branches REGEX] <input.info> <output.info>",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(args[0]):
        print(f"Error: Input file not found: {args[0]}", file=sys.stderr)
        sys.exit(1)

    stats = filter_lcov_file(
        args[0], args[1], exclude_no_mpi, re.compile(branch_exclude or r"(?!)")
    )
    print("Filtered LCOV coverage file:")
    print(f"  Input:  {args[0]}")
    print(f"  Output: {args[1]}")
    print(f"  Non-library files dropped: {stats['files_dropped']}")
    print(f"  Line records removed: {stats['lines']}")
    print(f"  Branch records removed: {stats['branches']}")
    print(f"  Function records removed: {stats['functions']}")


if __name__ == "__main__":
    main()
