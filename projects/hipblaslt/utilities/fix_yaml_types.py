#!/usr/bin/env python3
"""Fix YAML parameter type mismatches in TensileLite YAML files.

Usage:
    python3 fix_yaml_types.py [--mode {logic,input,both}] <directory> [<directory> ...]

Recursively finds all *.yaml files under each <directory> and applies targeted
regex substitutions to correct bool/int/float type mismatches.  These
mismatches cause std::bad_cast at C++ msgpack deserialization time (library-
logic path) and now also fail input-YAML validation in TensileLite itself.

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
import os
import re
import sys
import glob


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
]

# Group B: Int -> Bool
# Declared as bool (e.g. [False, True] in validParameters, _defaultProblemType
# bool defaults, or globalParameters bool defaults) but YAMLs have 0/1.
INT_TO_BOOL_PARAMS = [
    # validParameters (solution-side)
    "ActivationAlt",
    "ActivationFuncCall",
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

# Group D: Int -> Str
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

    # Group D: bare integer -> quoted string. Only rewrites unquoted
    # ints; already-quoted values pass through unchanged.
    for param in INT_TO_STR_PARAMS:
        head = rf"^(\s*(?:-\s+)?{param}: )"
        tail = r"(\s*(?:#.*)?)$"
        _add(head + r"(\d+)" + tail, r'\g<1>"\g<2>"\g<3>')

    return patterns


PATTERNS = _build_patterns()


def count_mismatches(content):
    """Count how many lines in content match any mismatch pattern.

    Returns (group_a_count, group_b_count, group_c_count, group_d_count).
    """
    counts = [0, 0, 0, 0]

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

    for param in INT_TO_STR_PARAMS:
        counts[3] += len(re.findall(
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


def find_yaml_files(directory):
    """Recursively find all *.yaml files under directory."""
    return sorted(glob.glob(os.path.join(directory, "**/*.yaml"), recursive=True))


def find_yaml_files_in_roots(directories):
    """Recursively find all *.yaml files under each directory in directories.

    De-duplicates results so overlapping roots do not double-process files.
    """
    seen = set()
    out = []
    for d in directories:
        for path in find_yaml_files(d):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return out


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
            "Usage: fix_yaml_types.py [--mode {logic,input,both}] "
            "<directory> [<directory> ...]"
        )
        print("  Recursively fixes YAML parameter type mismatches under each <directory>")
        return 1

    args = parse_args(argv)

    for d in args.directory:
        if not os.path.isdir(d):
            print(f"Error: '{d}' is not a directory")
            return 1

    yaml_files = find_yaml_files_in_roots(args.directory)
    print(f"=== fix_yaml_types.py ===")
    print(f"Mode: {args.mode}")
    print(f"Target directories: {', '.join(args.directory)}")
    print(f"YAML files found: {len(yaml_files)}")
    print()

    # Count mismatches before
    before = [0, 0, 0, 0]
    for filepath in yaml_files:
        with open(filepath, "r") as f:
            content = f.read()
        a, b, c, d = count_mismatches(content)
        before[0] += a
        before[1] += b
        before[2] += c
        before[3] += d

    before_total = sum(before)

    print("--- Mismatches found BEFORE fix ---")
    print(f"  Group A (bool->int):   {before[0]}")
    print(f"  Group B (int->bool):   {before[1]}")
    print(f"  Group C (int->float):  {before[2]}")
    print(f"  Group D (int->str):    {before[3]}")
    print(f"  Total:                 {before_total}")
    print()

    if before_total == 0:
        print("No mismatches to fix. All parameters have correct types.")
        return 0

    # Apply fixes
    print("Applying fixes...")
    files_modified = 0
    for filepath in yaml_files:
        if fix_file(filepath):
            files_modified += 1
    print(f"Done. Modified {files_modified} files.")
    print()

    # Count mismatches after (verification)
    after = [0, 0, 0, 0]
    for filepath in yaml_files:
        with open(filepath, "r") as f:
            content = f.read()
        a, b, c, d = count_mismatches(content)
        after[0] += a
        after[1] += b
        after[2] += c
        after[3] += d

    after_total = sum(after)

    print("--- Mismatches found AFTER fix ---")
    print(f"  Group A (bool->int):   {after[0]}")
    print(f"  Group B (int->bool):   {after[1]}")
    print(f"  Group C (int->float):  {after[2]}")
    print(f"  Group D (int->str):    {after[3]}")
    print(f"  Total:                 {after_total}")
    print()

    if after_total == 0:
        print("SUCCESS: All mismatches fixed.")
        return 0
    else:
        print(f"WARNING: {after_total} mismatches remain after fix. "
              "Manual inspection needed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
