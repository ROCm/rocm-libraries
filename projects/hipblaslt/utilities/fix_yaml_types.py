#!/usr/bin/env python3
"""Fix YAML parameter type mismatches in TensileLite YAML files.

Usage:
    python3 fix_yaml_types.py [--mode {logic,input,both}] [--jobs N] <directory> [<directory> ...]

Recursively finds all *.yaml files under each <directory> and applies targeted
regex substitutions to correct bool/int/float type mismatches.  These
mismatches cause std::bad_cast at C++ msgpack deserialization time (library-
logic path) and now also fail input-YAML validation in TensileLite itself.
The fixer does not pre-count mismatches; it runs the rewrite pass over every
discovered YAML file and reports how many files changed.

Modes:
    logic   Library-logic YAMLs (TensileLite-generated output).
    input   Input YAMLs (human-authored test/benchmark configs).
    both    Both (default).

The mismatch patterns are derived from Tensile.Common.ValidParameters and apply
in both logic and input contexts; the mode flag documents intent and lets
callers limit the sweep when only one tree needs touching.

Idempotent -- safe to run multiple times.
"""

import argparse
import concurrent.futures
import os
import re
import sys


# Known mismatch patterns ----------------------------------------------------
#
# Derived from Tensile.Common.ValidParameters.validParameters.  Each group
# lists parameters whose YAML values have the wrong Python type after
# yaml.safe_load().

# Group A: Bool -> Int
# Declared as int (e.g. [-1, 0, 1] in validParameters, or int defaults in
# globalParameters) but YAMLs have false/true.
BOOL_TO_INT_PARAMS = [
    # validParameters (solution-side)
    "ClusterLocalRead",
    "DirectToLds",
    "PrefetchGlobalRead",
    "PrefetchLocalRead",
    "SwapGlobalReadOrder",
    "TransposeLDS",
    "TransposeLDSMetadata",
    "UseCustomMainLoopSchedule",
    "UsePLRPack",
    # globalParameters (int defaults)
    "BoundsCheck",
    "DirectToLdsMetadata"
]

# Group B: Int -> Bool
# Declared as bool (e.g. [False, True] in validParameters, _defaultProblemType
# bool defaults, or globalParameters bool defaults) but YAMLs have 0/1.
INT_TO_BOOL_PARAMS = [
    # validParameters (solution-side)
    "Activation",
    "ActivationAlt",
    "ActivationFuncCall",
    "BufferStore",
    "ConvertAfterDS",
    "DirectToVgprA",
    "DirectToVgprB",
    "DirectToVgprSparseMetadata",
    "ExpandPointerSwap",
    "ForceDisableShadowInit",
    "GroupLoadStore",
    "LDSTrInst",
    "MIArchVgpr",
    "NoReject",
    "PreloadKernArgs",
    "SourceSwap",
    "StorePriorityOpt",
    "SuppressNoLoadLoop",
    "TailloopInNll",
    "Use64bShadowLimit",
    "Use64bShadowLimitMX",
    "UseSubtileImpl",
    "WaveSplitK",
    # _defaultProblemType (bool defaults)
    "TransposeA",
    "TransposeB",
    # globalParameters (bool defaults)
    "CSVExportWinner",
    "CSVMergeSameProblemID",
    "PreciseKernelTime",
]

# Group C: Int -> Float
# Declared as float but YAMLs have bare integers.
INT_TO_FLOAT_PARAMS = [
    "GlobalReadPerMfma",
]

# Group D: Float -> Int
# Declared as int but YAMLs have integral floats.
FLOAT_TO_INT_PARAMS = [
    "StaggerUStride",
]

# Group E: Int -> Str
# globalParameters string defaults (e.g. CodeObjectVersion default is "4")
# but YAMLs have bare integers.
INT_TO_STR_PARAMS = [
    "CodeObjectVersion",
]


def _build_patterns():
    """Build compiled regex patterns and their replacements.

    Returns a list of (compiled_regex, replacement_string) tuples. Each
    pattern matches a full line with the parameter at end-of-line,
    tolerating an optional ``- `` YAML-list-element prefix and an
    optional ``# comment`` trailer. The trailer is captured in group 2
    and emitted back so user comments survive the rewrite.
    """
    patterns = []

    def _add(re_str, replacement):
        patterns.append((re.compile(re_str, re.MULTILINE), replacement))

    # Group A: false/False -> 0, true/True -> 1.
    # Both scalar (Key: false) and single-element-list (Key: [false]).
    for param in BOOL_TO_INT_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"(?:false|False)" + tail, r"\g<1>0\g<2>")
        _add(head + r"(?:true|True)" + tail, r"\g<1>1\g<2>")
        _add(head + r"\[\s*(?:false|False)\s*\]" + tail, r"\g<1>[0]\g<2>")
        _add(head + r"\[\s*(?:true|True)\s*\]" + tail, r"\g<1>[1]\g<2>")

    # Group B: 0 -> false, 1 -> true. Scalar, single-element-list,
    # and the common [0,1] / [0, 1] two-element form for value
    # enumeration.
    for param in INT_TO_BOOL_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"0" + tail, r"\g<1>false\g<2>")
        _add(head + r"1" + tail, r"\g<1>true\g<2>")
        _add(head + r"\[\s*0\s*\]" + tail, r"\g<1>[false]\g<2>")
        _add(head + r"\[\s*1\s*\]" + tail, r"\g<1>[true]\g<2>")
        # [0,1] / [0, 1] / [1, 0] -- the typical two-value enumeration.
        _add(head + r"\[\s*0\s*,\s*1\s*\]" + tail, r"\g<1>[false, true]\g<2>")
        _add(head + r"\[\s*1\s*,\s*0\s*\]" + tail, r"\g<1>[true, false]\g<2>")

    # Group C: 1 -> 1.0. Scalar and single-element-list.
    for param in INT_TO_FLOAT_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"1" + tail, r"\g<1>1.0\g<2>")
        _add(head + r"\[\s*1\s*\]" + tail, r"\g<1>[1.0]\g<2>")

    # Group D: 256.0 -> 256. Scalar and single-element-list. Only rewrites
    # integral floats so non-integer values pass through unchanged.
    for param in FLOAT_TO_INT_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"(-?\d+)\.0+" + tail, r"\g<1>\g<2>\g<3>")
        _add(head + r"\[\s*(-?\d+)\.0+\s*\]" + tail, r"\g<1>[\g<2>]\g<3>")

    # Group E: bare integer -> quoted string. Only rewrites unquoted
    # ints; already-quoted values pass through unchanged.
    for param in INT_TO_STR_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"(\d+)" + tail, r'\g<1>"\g<2>"\g<3>')

    return patterns


PATTERNS = _build_patterns()


def count_mismatches(content):
    """Count how many lines in content match any mismatch pattern.

    Returns (
        group_a_count,
        group_b_count,
        group_c_count,
        group_d_count,
        group_e_count,
    ).
    """
    counts = [0, 0, 0, 0, 0]

    for param in BOOL_TO_INT_PARAMS:
        counts[0] += len(re.findall(
            rf"^\s*(?:-\s+)?{param}: (?:false|true|False|True|\[\s*(?:false|true|False|True)\s*\])\s*(?:#.*)?$",
            content, re.MULTILINE))

    for param in INT_TO_BOOL_PARAMS:
        counts[1] += len(re.findall(
            rf"^\s*(?:-\s+)?{param}: "
            rf"(?:[01]|\[\s*[01]\s*\]|\[\s*[01]\s*,\s*[01]\s*\])"
            rf"\s*(?:#.*)?$", content, re.MULTILINE))

    for param in INT_TO_FLOAT_PARAMS:
        counts[2] += len(re.findall(
            rf"^\s*(?:-\s+)?{param}: (?:1|\[\s*1\s*\])\s*(?:#.*)?$", content, re.MULTILINE))

    for param in FLOAT_TO_INT_PARAMS:
        counts[3] += len(re.findall(
            rf"^\s*(?:-\s+)?{param}: (?:-?\d+\.0+|\[\s*-?\d+\.0+\s*\])\s*(?:#.*)?$",
            content, re.MULTILINE))

    for param in INT_TO_STR_PARAMS:
        counts[4] += len(re.findall(
            rf"^\s*(?:-\s+)?{param}: \d+\s*(?:#.*)?$", content, re.MULTILINE))

    return tuple(counts)


def fix_content(content):
    """Apply all type-fix patterns to content string.  Returns new content."""
    for pattern, replacement in PATTERNS:
        content = pattern.sub(replacement, content)
    return content


def fix_file(filepath):
    """Fix a single YAML file in-place.  Returns True if file was modified."""
    with open(filepath, "r") as f:
        original = f.read()

    fixed = fix_content(original)

    if fixed != original:
        with open(filepath, "w") as f:
            f.write(fixed)
        return True
    return False


def _find_yaml_files_serial(directory):
    """Recursively find all *.yaml files under directory in one process."""
    yaml_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for filename in sorted(files):
            if filename.endswith(".yaml"):
                yaml_files.append(os.path.join(root, filename))
    return yaml_files


def default_jobs():
    """Return the default worker count for parallel file discovery/fixing."""
    return os.cpu_count() or 1


def _positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return ivalue


def _bounded_worker_count(jobs, item_count):
    """Use no more workers than useful for the amount of work available."""
    if item_count <= 1:
        return 1
    return min(jobs, item_count)


def _scan_one_level(directory):
    """Return direct YAML files and child directories under directory."""
    yaml_files = []
    child_dirs = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return yaml_files, child_dirs

    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                child_dirs.append(entry.path)
            elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".yaml"):
                yaml_files.append(entry.path)
        except OSError:
            continue

    return sorted(yaml_files), sorted(child_dirs)


def _partition_scan_roots(directory, target_partitions):
    """Split a root into recursive scan partitions and direct YAML hits."""
    yaml_files = []
    scan_roots = [directory]

    while scan_roots and len(scan_roots) < target_partitions:
        current = scan_roots.pop(0)
        direct_yaml, child_dirs = _scan_one_level(current)
        yaml_files.extend(direct_yaml)
        if child_dirs:
            scan_roots.extend(child_dirs)

    return yaml_files, scan_roots


def find_yaml_files(directory, jobs=None):
    """Recursively find all *.yaml files under directory."""
    if jobs is None:
        jobs = default_jobs()

    workers = _bounded_worker_count(jobs, jobs)
    if workers == 1:
        return _find_yaml_files_serial(directory)

    direct_yaml, scan_roots = _partition_scan_roots(directory, workers * 4)
    scan_workers = _bounded_worker_count(workers, len(scan_roots))
    if scan_workers == 1:
        nested_yaml = []
        for root in scan_roots:
            nested_yaml.extend(_find_yaml_files_serial(root))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers) as executor:
            nested_yaml = [
                path
                for paths in executor.map(_find_yaml_files_serial, scan_roots)
                for path in paths
            ]

    return sorted(direct_yaml + nested_yaml)


def find_yaml_files_in_roots(directories, jobs=None):
    """Recursively find all *.yaml files under each directory in directories.

    De-duplicates results so overlapping roots do not double-process files.
    """
    if jobs is None:
        jobs = default_jobs()

    workers = _bounded_worker_count(jobs, len(directories))
    per_root_jobs = max(1, jobs // workers)
    if workers == 1:
        root_results = [find_yaml_files(d, jobs) for d in directories]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(find_yaml_files, directory, per_root_jobs)
                for directory in directories
            ]
            root_results = [future.result() for future in futures]

    seen = set()
    out = []
    for paths in root_results:
        for path in paths:
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


class _ProgressReporter:
    """Small stdout progress bar for parent-process file fixing progress."""

    def __init__(self, total, enabled=False, stream=None, width=40):
        self.total = total
        self.enabled = enabled and total > 0
        self.stream = stream if stream is not None else sys.stdout
        self.width = width
        self.started = False
        self.finished = False

    def update(self, completed, modified):
        if not self.enabled:
            return

        filled = int(self.width * completed / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(100 * completed / self.total)
        self.stream.write(
            f"\rFixing files: [{bar}] {completed}/{self.total} "
            f"({percent:3d}%) modified={modified}"
        )
        self.stream.flush()
        self.started = True

    def finish(self, modified):
        if not self.enabled:
            return
        self.update(self.total, modified)
        self.stream.write("\n")
        self.stream.flush()
        self.finished = True

    def fail(self):
        if self.enabled and self.started and not self.finished:
            self.stream.write("\n")
            self.stream.flush()
            self.finished = True


def fix_files(filepaths, jobs=None, progress=False, progress_stream=None):
    """Fix YAML files in parallel. Returns the number of modified files."""
    if jobs is None:
        jobs = default_jobs()

    workers = _bounded_worker_count(jobs, len(filepaths))
    reporter = _ProgressReporter(
        len(filepaths),
        enabled=progress,
        stream=progress_stream,
    )

    modified = 0
    reporter.update(0, modified)
    try:
        if workers == 1:
            for completed, filepath in enumerate(filepaths, start=1):
                if fix_file(filepath):
                    modified += 1
                reporter.update(completed, modified)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(fix_file, filepath) for filepath in filepaths]
                for completed, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    if future.result():
                        modified += 1
                    reporter.update(completed, modified)
    except Exception:
        reporter.fail()
        raise

    reporter.finish(modified)
    return modified


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="fix_yaml_types.py",
        description=(
            "Fix YAML parameter type mismatches in TensileLite library-logic "
            "and/or input YAMLs."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("logic", "input", "both"),
        default="both",
        help=(
            "Which YAML tree(s) the directories represent. The mismatch "
            "patterns apply identically in both contexts; this flag documents "
            "intent and is reflected in the report. Default: both."
        ),
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=_positive_int,
        default=default_jobs(),
        help=(
            "Number of parallel workers for YAML discovery and rewrite. "
            "Default: CPU count."
        ),
    )
    p.add_argument(
        "directory",
        nargs="+",
        help="One or more directories to scan recursively for *.yaml files.",
    )
    return p.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Backward-compat: an empty argv prints usage and exits non-zero so the
    # existing test_no_args_exits_nonzero test still passes.
    if not argv:
        print(
            "Usage: fix_yaml_types.py [--mode {logic,input,both}] [--jobs N] "
            "<directory> [<directory> ...]"
        )
        print("  Recursively fixes YAML parameter type mismatches under each <directory>")
        return 1

    args = parse_args(argv)

    for d in args.directory:
        if not os.path.isdir(d):
            print(f"Error: '{d}' is not a directory")
            return 1

    yaml_files = find_yaml_files_in_roots(args.directory, args.jobs)
    print(f"=== fix_yaml_types.py ===")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.jobs}")
    print(f"Target directories: {', '.join(args.directory)}")
    print(f"YAML files found: {len(yaml_files)}")
    print()

    print("Applying fixes...")
    files_modified = fix_files(yaml_files, args.jobs, progress=True)
    print(f"Done. Modified {files_modified} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
