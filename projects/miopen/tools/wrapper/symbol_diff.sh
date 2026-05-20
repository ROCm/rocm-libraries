#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Q2/Q6 harness for RFC 0001 (Phase 1 investigation):
#
#   - Dumps the public C symbol set, SONAME, and DT_NEEDED of a libMIOpen.so.
#   - With --baseline and --candidate, diffs the two and exits non-zero on
#     any divergence in the public symbol set or SONAME.
#
# Used to validate property #1 from §1 (flag-off byte-equivalence at the
# symbol level) and Q6 (DT_NEEDED includes libMIOpen_private.so.1 in flag-on
# builds).

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 dump <libMIOpen.so> [--out <prefix>]
      Emit <prefix>.symbols (sorted public miopen* exports)
           <prefix>.soname  (SONAME from readelf -d)
           <prefix>.needed  (DT_NEEDED entries from readelf -d)

  $0 diff --baseline <prefix> --candidate <prefix>
      Compare two dump prefixes. Exit non-zero on:
        - any miopen* public symbol present in baseline but missing in candidate
        - any change to SONAME

The candidate is allowed to export *additional* symbols (the wrapper itself
emits routing globals etc.); the requirement is that it remain a superset of
the baseline's public miopen* set.
EOF
}

dump_lib() {
    local lib="$1"
    local out="$2"
    [[ -f "$lib" ]] || { echo "missing: $lib" >&2; exit 2; }

    # Public C entry points: defined ('T' or 't' or 'W' for weak text)
    # symbols with exact prefix 'miopen', case-sensitive. We do not include
    # internal C++ symbols (those start with _Z) — they are out of scope per
    # §1 and Q3.
    nm -D --defined-only --extern-only "$lib" \
        | awk '$2 ~ /^[TtWw]$/ && $3 ~ /^miopen[A-Z]/ { print $3 }' \
        | sort -u > "${out}.symbols"

    readelf -d "$lib" | awk '/SONAME/ { gsub(/[\[\]]/,"",$NF); print $NF }' \
        > "${out}.soname"

    readelf -d "$lib" | awk '/NEEDED/  { gsub(/[\[\]]/,"",$NF); print $NF }' \
        | sort > "${out}.needed"

    echo "wrote ${out}.symbols ($(wc -l < "${out}.symbols") public symbols)"
    echo "wrote ${out}.soname  ($(cat "${out}.soname"))"
    echo "wrote ${out}.needed  ($(wc -l < "${out}.needed") entries)"
}

diff_dumps() {
    local base="$1"
    local cand="$2"
    local fail=0

    # Public symbol set: candidate must be a SUPERSET of baseline.
    local missing
    missing=$(comm -23 "${base}.symbols" "${cand}.symbols")
    if [[ -n "$missing" ]]; then
        echo "FAIL: public symbols missing in candidate:" >&2
        printf '  %s\n' $missing >&2
        fail=1
    fi
    local added
    added=$(comm -13 "${base}.symbols" "${cand}.symbols")
    if [[ -n "$added" ]]; then
        echo "INFO: public symbols added in candidate (allowed):"
        printf '  %s\n' $added | head -20
    fi

    # SONAME: must match exactly.
    if ! diff -u "${base}.soname" "${cand}.soname" >/dev/null; then
        echo "FAIL: SONAME differs:" >&2
        diff -u "${base}.soname" "${cand}.soname" >&2 || true
        fail=1
    fi

    # DT_NEEDED: not enforced as a hard match — flag-on adds libMIOpen_private
    # and libhipdnn. Just report the delta for the investigation log.
    if ! diff -q "${base}.needed" "${cand}.needed" >/dev/null; then
        echo "INFO: DT_NEEDED delta (expected in flag-on builds):"
        diff -u "${base}.needed" "${cand}.needed" || true
    fi

    if [[ $fail -eq 0 ]]; then
        echo "Q2/Q6 PASS: public symbol set is a superset and SONAME matches"
    fi
    return $fail
}

[[ $# -ge 1 ]] || { usage; exit 2; }
cmd="$1"; shift

case "$cmd" in
    dump)
        lib=""
        out="symbols"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --out) out="$2"; shift 2 ;;
                -*)    echo "unknown arg: $1" >&2; usage; exit 2 ;;
                *)     lib="$1"; shift ;;
            esac
        done
        [[ -n "$lib" ]] || { usage; exit 2; }
        dump_lib "$lib" "$out"
        ;;
    diff)
        base=""
        cand=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --baseline)  base="$2"; shift 2 ;;
                --candidate) cand="$2"; shift 2 ;;
                *)           echo "unknown arg: $1" >&2; usage; exit 2 ;;
            esac
        done
        [[ -n "$base" && -n "$cand" ]] || { usage; exit 2; }
        diff_dumps "$base" "$cand"
        ;;
    -h|--help) usage ;;
    *) usage; exit 2 ;;
esac
