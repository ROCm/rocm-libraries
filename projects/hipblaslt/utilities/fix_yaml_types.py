#!/usr/bin/env python3
"""Fix YAML parameter type mismatches in library logic files.

Usage:
    python3 fix_yaml_types.py <directory>

Recursively finds all *.yaml files under <directory> and applies targeted
regex substitutions to correct bool/int/float type mismatches.  These
mismatches cause std::bad_cast at C++ msgpack deserialization time because
msgpack serializes bool and int as different wire types.

Idempotent — safe to run multiple times.
"""

import os
import re
import sys
import glob


# ── Known mismatch patterns ──────────────────────────────────────────────────
#
# Derived from Tensile.Common.ValidParameters.validParameters.  Each group
# lists parameters whose YAML values have the wrong Python type after
# yaml.safe_load().

# Group A: Bool -> Int
# validParameters declares these as int (e.g. [-1, 0, 1]) but YAMLs have
# false/true which yaml.safe_load() returns as Python bool.
BOOL_TO_INT_PARAMS = [
    "ClusterLocalRead",
    "DirectToLds",
    "SwapGlobalReadOrder",
    "TransposeLDS",
    "TransposeLDSMetadata",
    "UseCustomMainLoopSchedule",
    "UsePLRPack",
]

# Group B: Int -> Bool
# validParameters declares these as bool (e.g. [False, True]) but YAMLs have
# 0/1 which yaml.safe_load() returns as Python int.
INT_TO_BOOL_PARAMS = [
    "ActivationFuncCall",
    "ConvertAfterDS",
    "DirectToVgprA",
    "DirectToVgprB",
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
    "WaveSplitK",
]

# Group C: Int -> Float
# validParameters declares these as float but YAMLs have bare integers.
INT_TO_FLOAT_PARAMS = [
    "GlobalReadPerMfma",
]


def _build_patterns():
    """Build compiled regex patterns and their replacements.

    Returns a list of (compiled_regex, replacement_string) tuples.
    Each regex matches a full line with the parameter at end-of-line.
    """
    patterns = []

    # Group A: false -> 0, true -> 1
    for param in BOOL_TO_INT_PARAMS:
        patterns.append((
            re.compile(rf"^(\s*{param}: )false$", re.MULTILINE),
            rf"\g<1>0",
        ))
        patterns.append((
            re.compile(rf"^(\s*{param}: )true$", re.MULTILINE),
            rf"\g<1>1",
        ))

    # Group B: 0 -> false, 1 -> true
    for param in INT_TO_BOOL_PARAMS:
        patterns.append((
            re.compile(rf"^(\s*{param}: )0$", re.MULTILINE),
            rf"\g<1>false",
        ))
        patterns.append((
            re.compile(rf"^(\s*{param}: )1$", re.MULTILINE),
            rf"\g<1>true",
        ))

    # Group C: 1 -> 1.0
    for param in INT_TO_FLOAT_PARAMS:
        patterns.append((
            re.compile(rf"^(\s*{param}: )1$", re.MULTILINE),
            rf"\g<1>1.0",
        ))

    return patterns


PATTERNS = _build_patterns()


def count_mismatches(content):
    """Count how many lines in content match any mismatch pattern.

    Returns (group_a_count, group_b_count, group_c_count).
    """
    counts = [0, 0, 0]

    for param in BOOL_TO_INT_PARAMS:
        counts[0] += len(re.findall(
            rf"^\s*{param}: (?:false|true)$", content, re.MULTILINE))

    for param in INT_TO_BOOL_PARAMS:
        counts[1] += len(re.findall(
            rf"^\s*{param}: [01]$", content, re.MULTILINE))

    for param in INT_TO_FLOAT_PARAMS:
        counts[2] += len(re.findall(
            rf"^\s*{param}: 1$", content, re.MULTILINE))

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


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        print("  Recursively fixes YAML parameter type mismatches under <directory>")
        return 1

    target_dir = sys.argv[1]

    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a directory")
        return 1

    yaml_files = find_yaml_files(target_dir)
    print(f"=== fix_yaml_types.py ===")
    print(f"Target directory: {target_dir}")
    print(f"YAML files found: {len(yaml_files)}")
    print()

    # Count mismatches before
    before = [0, 0, 0]
    for filepath in yaml_files:
        with open(filepath, "r") as f:
            content = f.read()
        a, b, c = count_mismatches(content)
        before[0] += a
        before[1] += b
        before[2] += c

    before_total = sum(before)

    print("--- Mismatches found BEFORE fix ---")
    print(f"  Group A (bool->int):   {before[0]}")
    print(f"  Group B (int->bool):   {before[1]}")
    print(f"  Group C (int->float):  {before[2]}")
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
    after = [0, 0, 0]
    for filepath in yaml_files:
        with open(filepath, "r") as f:
            content = f.read()
        a, b, c = count_mismatches(content)
        after[0] += a
        after[1] += b
        after[2] += c

    after_total = sum(after)

    print("--- Mismatches found AFTER fix ---")
    print(f"  Group A (bool->int):   {after[0]}")
    print(f"  Group B (int->bool):   {after[1]}")
    print(f"  Group C (int->float):  {after[2]}")
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
