#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# RFC 0001 Phase 1 — public-API symbol equivalence check.
#
# Compares the set of MIOPEN_EXPORT function names declared in miopen.h
# against the set of miopen* symbols actually exported by a built
# libMIOpen.so. miopen.h is the source of truth — no separate baseline
# file is maintained.
#
# This catches:
#   - public symbols declared in the header but not defined (would also
#     fail at consumer link time, but this is a faster signal in CI),
#   - public symbols defined and exported by the library but not declared
#     in miopen.h (a private symbol that has accidentally escaped the
#     public surface).
#
# Used by the investigation_flagoff_equivalence CTest (RemainingWork item 9,
# RFC 0001 §1 byte-equivalence constraint promoted from "demonstrable" to
# "enforced") and as the baseline-extractor for the Q2 superset check
# (item 5) when the two flag-state libraries are compared via symbol_diff.sh.

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --header <miopen.h> --lib <libMIOpen.so>
      Extract MIOPEN_EXPORT names from the header and the miopen*
      exports from the library; fail on any symmetric difference.

  $0 --header <miopen.h> --extract-symbols
      Print the extracted MIOPEN_EXPORT name list, one per line, sorted.
      Used by symbol_diff.sh to create a header-derived baseline file.
EOF
}

extract_from_header() {
    local hdr="$1"
    [[ -f "$hdr" ]] || { echo "missing header: $hdr" >&2; exit 2; }
    # Same regex shape that gen_rename_header.py used at bootstrap (see RFC §6 Q1).
    # Matches "MIOPEN_EXPORT <return-type...> miopenFoo(" across line breaks.
    awk '
        /MIOPEN_EXPORT/    { capture=1 }
        capture            { buf = buf " " $0 }
        capture && /\(/    {
            if (match(buf, /(miopen[A-Z][A-Za-z0-9_]+)[ \t\n]*\(/, m)) {
                print m[1]
            }
            capture=0; buf=""
        }
    ' "$hdr" | sort -u
}

extract_from_lib() {
    local lib="$1"
    [[ -f "$lib" ]] || { echo "missing library: $lib" >&2; exit 2; }
    nm -D --defined-only --extern-only "$lib" \
        | awk '$2 ~ /^[TtWw]$/ && $3 ~ /^miopen[A-Z]/ { print $3 }' \
        | sort -u
}

mode_check() {
    local hdr="$1" lib="$2"
    local hdr_set lib_set tmp_hdr tmp_lib fail=0
    tmp_hdr=$(mktemp); tmp_lib=$(mktemp)
    trap 'rm -f "$tmp_hdr" "$tmp_lib"' EXIT

    extract_from_header "$hdr" > "$tmp_hdr"
    extract_from_lib    "$lib" > "$tmp_lib"

    local missing added
    missing=$(comm -23 "$tmp_hdr" "$tmp_lib")
    added=$(comm -13 "$tmp_hdr" "$tmp_lib")

    if [[ -n "$missing" ]]; then
        echo "FAIL: declared in $hdr but NOT exported by $lib:" >&2
        printf '  %s\n' $missing >&2
        fail=1
    fi
    if [[ -n "$added" ]]; then
        echo "FAIL: exported by $lib but NOT declared in $hdr:" >&2
        printf '  %s\n' $added >&2
        fail=1
    fi
    if [[ $fail -eq 0 ]]; then
        echo "OK: $(wc -l < "$tmp_hdr") public symbols match between header and library"
    fi
    return $fail
}

[[ $# -ge 1 ]] || { usage; exit 2; }

hdr=""
lib=""
mode="check"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --header)           hdr="$2"; shift 2 ;;
        --lib)              lib="$2"; shift 2 ;;
        --extract-symbols)  mode="extract"; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

case "$mode" in
    extract)
        [[ -n "$hdr" ]] || { usage; exit 2; }
        extract_from_header "$hdr"
        ;;
    check)
        [[ -n "$hdr" && -n "$lib" ]] || { usage; exit 2; }
        mode_check "$hdr" "$lib"
        ;;
esac
