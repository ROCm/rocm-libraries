#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Flag-off byte-equivalence gate for the MIOpen public/private split (RFC 0001).
#
# When MIOPEN_ENABLE_HIPDNN_WRAPPER is OFF (the default) the build must produce a
# single libMIOpen.so whose public ABI is identical to the pre-wrapper baseline.
# This script enforces that on a flag-off build:
#
#   1. SONAME is libMIOpen.so.1 (unchanged).
#   2. The exported public C API symbol set (miopen<Uppercase>...) exactly
#      matches the committed baseline (test/byte_equivalence/public_symbols.baseline).
#   3. No *_impl symbols are exported (the private rename must not leak into a
#      flag-off build).
#   4. libMIOpen_private is NOT in DT_NEEDED (flag-off is self-contained).
#
# The public C API symbol set and SONAME are build-config independent, so the
# committed baseline is a stable ABI contract. When the public API legitimately
# changes, regenerate the baseline with `dump-symbols` (reviewed alongside the
# rename header and wrapper updates).
#
# Full binary content-hash equivalence is build-path dependent and therefore not
# gated here; use `compare-pair` to diff two builds (e.g. a develop baseline and
# a flag-off build) from the same environment, where SHA256 is reported for
# information.
#
# Usage:
#   check_byte_equivalence.sh check <libMIOpen.so> <baseline_symbols_file>
#   check_byte_equivalence.sh dump-symbols <libMIOpen.so> [out_file]
#   check_byte_equivalence.sh dump <libMIOpen.so> <out_prefix>
#   check_byte_equivalence.sh compare-pair <baseline_prefix> <candidate_prefix>

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

command -v nm >/dev/null      || die "nm not found (binutils required)"
command -v readelf >/dev/null || die "readelf not found (binutils required)"

# Exported, defined, global-text symbols that follow the public MIOpen C API
# naming convention (miopen followed by an uppercase letter). This deliberately
# excludes internal exported shims such as miopen_sqlite3_memvfs_init.
public_api_symbols() {
    nm -D --defined-only "$1" | awk '$2 == "T" { print $3 }' | grep -E '^miopen[A-Z]' | sort -u
}

impl_symbols() {
    nm -D --defined-only "$1" | awk '$2 == "T" { print $3 }' | grep -E '_impl$' || true
}

soname() {
    readelf -d "$1" | awk '/\(SONAME\)/ { gsub(/[][]/, "", $NF); print $NF }'
}

dt_needed() {
    readelf -d "$1" | awk '/\(NEEDED\)/ { gsub(/[][]/, "", $NF); print $NF }' | sort
}

cmd_dump_symbols() {
    local so="$1" out="${2:-/dev/stdout}"
    [ -f "$so" ] || die "library not found: $so"
    public_api_symbols "$so" > "$out"
}

cmd_dump() {
    local so="$1" prefix="$2"
    [ -f "$so" ] || die "library not found: $so"
    public_api_symbols "$so" > "${prefix}.symbols"
    soname "$so"            > "${prefix}.soname"
    dt_needed "$so"         > "${prefix}.needed"
    sha256sum "$so" | awk '{print $1}' > "${prefix}.sha256"
    echo "wrote ${prefix}.{symbols,soname,needed,sha256}"
}

cmd_check() {
    local so="$1" baseline="$2"
    [ -f "$so" ]       || die "library not found: $so"
    [ -f "$baseline" ] || die "baseline symbols file not found: $baseline"

    local rc=0

    # 1. SONAME
    local sn
    sn="$(soname "$so")"
    if [ "$sn" = "libMIOpen.so.1" ]; then
        echo "PASS: SONAME is $sn"
    else
        echo "FAIL: SONAME is '$sn', expected 'libMIOpen.so.1'"; rc=1
    fi

    # 2. Exported public API symbol set == committed baseline
    local got
    got="$(mktemp)"
    public_api_symbols "$so" > "$got"
    if diff -u "$baseline" "$got" > /tmp/be_symdiff.txt 2>&1; then
        echo "PASS: exported public API symbol set matches baseline ($(wc -l < "$baseline") symbols)"
    else
        echo "FAIL: exported public API symbol set differs from baseline (< baseline, > built):"
        cat /tmp/be_symdiff.txt
        rc=1
    fi
    rm -f "$got"

    # 3. No _impl leakage
    local impl
    impl="$(impl_symbols "$so")"
    if [ -z "$impl" ]; then
        echo "PASS: no *_impl symbols exported"
    else
        echo "FAIL: flag-off build exported *_impl symbols (rename leaked into flag-off):"
        echo "$impl" | sed 's/^/  /'
        rc=1
    fi

    # 4. Self-contained: no dependency on the private library
    if dt_needed "$so" | grep -q '^libMIOpen_private'; then
        echo "FAIL: flag-off libMIOpen.so has DT_NEEDED on libMIOpen_private (not self-contained)"
        rc=1
    else
        echo "PASS: no DT_NEEDED on libMIOpen_private"
    fi

    if [ "$rc" -eq 0 ]; then
        echo "byte-equivalence: PASS"
    else
        echo "byte-equivalence: FAIL"
    fi
    return "$rc"
}

cmd_compare_pair() {
    local base="$1" cand="$2"
    local rc=0
    for ext in soname symbols needed; do
        if diff -u "${base}.${ext}" "${cand}.${ext}" > "/tmp/be_pair_${ext}.txt" 2>&1; then
            echo "PASS: ${ext} identical"
        else
            echo "FAIL: ${ext} differs:"; cat "/tmp/be_pair_${ext}.txt"; rc=1
        fi
    done
    if [ -f "${base}.sha256" ] && [ -f "${cand}.sha256" ]; then
        if [ "$(cat "${base}.sha256")" = "$(cat "${cand}.sha256")" ]; then
            echo "INFO: SHA256 identical"
        else
            echo "INFO: SHA256 differs (expected under build-path nondeterminism; not gated)"
        fi
    fi
    return "$rc"
}

[ $# -ge 1 ] || die "no subcommand; see header for usage"
sub="$1"; shift
case "$sub" in
    check)        [ $# -eq 2 ] || die "usage: check <so> <baseline_symbols_file>"; cmd_check "$@" ;;
    dump-symbols) [ $# -ge 1 ] || die "usage: dump-symbols <so> [out_file]";        cmd_dump_symbols "$@" ;;
    dump)         [ $# -eq 2 ] || die "usage: dump <so> <out_prefix>";               cmd_dump "$@" ;;
    compare-pair) [ $# -eq 2 ] || die "usage: compare-pair <base_prefix> <cand_prefix>"; cmd_compare_pair "$@" ;;
    *) die "unknown subcommand: $sub" ;;
esac
