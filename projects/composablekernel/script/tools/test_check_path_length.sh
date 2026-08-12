#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Unit test (pure bash) for script/check_path_length.sh.
#
# Builds a throwaway git repository with synthetic paths of known length, so
# the test does not depend on what the real tree happens to contain.
#
# Usage: ./test_check_path_length.sh

set -uo pipefail

CHECKER=$(cd "$(dirname "$0")/.." && pwd)/check_path_length.sh
failures=0
tests=0

if [[ ! -x "$CHECKER" ]]; then
    echo "ERROR: $CHECKER not found or not executable"
    exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

git -C "$WORKDIR" init --quiet

# Create a file whose repository-relative path is exactly $1 characters, and
# echo that path. Padding goes in a single directory name so the file name
# itself stays short.
make_path_of_length() {
    local target=$1
    local dir="d" file="f.cpp"
    # "<dir>/<file>" -> pad dir until the whole thing is $target characters.
    local pad=$(( target - ${#file} - 1 - ${#dir} ))
    (( pad >= 0 )) || { echo "target $target too small" >&2; return 1; }
    dir="${dir}$(printf 'x%.0s' $(seq 1 "$pad"))"
    mkdir -p "$WORKDIR/$dir"
    : > "$WORKDIR/$dir/$file"
    echo "$dir/$file"
}

# check <description> <expected-exit> <cwd> <args...>
check() {
    local desc=$1 expected=$2 cwd=$3
    shift 3
    tests=$(( tests + 1 ))
    local out actual
    out=$( cd "$cwd" && "$CHECKER" "$@" 2>&1 )
    actual=$?
    if [[ "$actual" -ne "$expected" ]]; then
        echo "FAIL: $desc (expected exit $expected, got $actual)"
        [[ -n "$out" ]] && echo "$out" | sed 's/^/      /'
        failures=$(( failures + 1 ))
    else
        echo "PASS: $desc"
    fi
}

# --- default limit is MAX_PATH(260) - prefix budget(60) = 200 ---------------
at_limit=$(make_path_of_length 200)
over_limit=$(make_path_of_length 201)
well_under=$(make_path_of_length 40)

check "path exactly at the 200-character limit is accepted" 0 "$WORKDIR" "$at_limit"
check "path one character over the limit is rejected"       1 "$WORKDIR" "$over_limit"
check "short path is accepted"                              0 "$WORKDIR" "$well_under"

# --- exit code reflects any failure in a batch ------------------------------
check "batch with one bad path fails" 1 "$WORKDIR" "$well_under" "$over_limit" "$at_limit"
check "batch of good paths passes"    0 "$WORKDIR" "$well_under" "$at_limit"

# --- arguments that do not resolve are skipped, as in the sibling checks -----
check "nonexistent path is skipped"      0 "$WORKDIR" "does/not/exist.cpp"
check "no arguments is a no-op"          0 "$WORKDIR"

# --- ./-prefixed and repository-relative spellings agree --------------------
check "leading ./ is stripped before measuring" 1 "$WORKDIR" "./$over_limit"

# --- paths given relative to a subdirectory are measured repo-relative ------
# From inside the deep directory, "f.cpp" is 5 characters, but its
# repository-relative path is the full 201 and must still be rejected.
deep_dir=$(dirname "$over_limit")
check "subdirectory-relative path is measured from the repository root" \
    1 "$WORKDIR/$deep_dir" "f.cpp"

# The reported path must be the repository-relative one, not the argument.
tests=$(( tests + 1 ))
report=$( cd "$WORKDIR/$deep_dir" && "$CHECKER" "f.cpp" 2>&1 | head -1 )
if [[ "$report" == "ERROR: $over_limit" ]]; then
    echo "PASS: error message reports the repository-relative path"
else
    echo "FAIL: error message reports the repository-relative path"
    echo "      expected: ERROR: $over_limit"
    echo "      actual:   $report"
    failures=$(( failures + 1 ))
fi

# --- limits are configurable ------------------------------------------------
tests=$(( tests + 1 ))
if ( cd "$WORKDIR" && CK_MAX_PATH_LEN=250 "$CHECKER" "$over_limit" >/dev/null 2>&1 ); then
    echo "PASS: CK_MAX_PATH_LEN raises the limit"
else
    echo "FAIL: CK_MAX_PATH_LEN raises the limit"
    failures=$(( failures + 1 ))
fi

tests=$(( tests + 1 ))
if ( cd "$WORKDIR" && CK_PREFIX_BUDGET=10 "$CHECKER" "$over_limit" >/dev/null 2>&1 ); then
    echo "PASS: CK_PREFIX_BUDGET widens the repo-relative budget"
else
    echo "FAIL: CK_PREFIX_BUDGET widens the repo-relative budget"
    failures=$(( failures + 1 ))
fi

# 90 - 60 = a 30-character budget, which the 40-character path exceeds.
tests=$(( tests + 1 ))
if ( cd "$WORKDIR" && CK_MAX_PATH_TOTAL=90 "$CHECKER" "$well_under" >/dev/null 2>&1 ); then
    echo "FAIL: CK_MAX_PATH_TOTAL tightens the limit"
    failures=$(( failures + 1 ))
else
    echo "PASS: CK_MAX_PATH_TOTAL tightens the limit"
fi

# --- the file-name limit is independent of the path limit -------------------
# A 20-character name under a 10-character name limit must be rejected even
# though the path itself is comfortably short.
tests=$(( tests + 1 ))
: > "$WORKDIR/nnnnnnnnnnnnnnnnn.cpp"
if ( cd "$WORKDIR" && CK_MAX_NAME_LEN=10 "$CHECKER" "nnnnnnnnnnnnnnnnn.cpp" >/dev/null 2>&1 ); then
    echo "FAIL: CK_MAX_NAME_LEN rejects an over-long file name"
    failures=$(( failures + 1 ))
else
    echo "PASS: CK_MAX_NAME_LEN rejects an over-long file name"
fi

tests=$(( tests + 1 ))
if ( cd "$WORKDIR" && "$CHECKER" "nnnnnnnnnnnnnnnnn.cpp" >/dev/null 2>&1 ); then
    echo "PASS: a normal file name passes the default 255 limit"
else
    echo "FAIL: a normal file name passes the default 255 limit"
    failures=$(( failures + 1 ))
fi

# --- outside a git work tree, arguments are measured as given ---------------
NONGIT=$(mktemp -d)
trap 'rm -rf "$WORKDIR" "$NONGIT"' EXIT
mkdir -p "$NONGIT/$(dirname "$over_limit")"
: > "$NONGIT/$over_limit"
check "works outside a git work tree" 1 "$NONGIT" "$over_limit"

echo
if [[ "$failures" -eq 0 ]]; then
    echo "All $tests checks passed."
    exit 0
fi
echo "$failures of $tests checks failed."
exit 1
