#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Q6 harness for RFC 0001: configures and builds a tiny external CMake
# project that uses find_package(miopen) against an installed MIOpen prefix.
# Asserts both consumer paths link cleanly:
#   - consumer_public  → links MIOpen (wrapper or only target)
#   - consumer_private → links MIOpen_private (only present in flag-on installs)
#
# Also verifies the wrapper's DT_NEEDED references libMIOpen_private.so.1 in
# flag-on installs, completing RFC §6 Q6's empirical leg.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --prefix <install-prefix> [--workdir <dir>]
EOF
}

PREFIX=""
WORKDIR=""
CLEANUP_WORKDIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)  PREFIX="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$PREFIX" ]] || { usage; exit 2; }

if [[ -z "$WORKDIR" ]]; then
    WORKDIR=$(mktemp -d)
    CLEANUP_WORKDIR=1
fi
trap '[[ $CLEANUP_WORKDIR -eq 1 ]] && rm -rf "$WORKDIR"' EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRCDIR="$SCRIPT_DIR/find_package_smoke"

echo "Q6 [1/3] Configuring find_package(miopen) consumer against $PREFIX"
cmake -S "$SRCDIR" -B "$WORKDIR" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    >/dev/null

echo "Q6 [2/3] Building consumer(s)"
cmake --build "$WORKDIR" --parallel >/dev/null

# 3) If the install was flag-on, the wrapper DT_NEEDED must reference
#    libMIOpen_private.so.<SOVERSION>. With flag-off there is no MIOpen_private
#    and DT_NEEDED should not mention it.
echo "Q6 [3/3] Verifying DT_NEEDED on installed libMIOpen.so"
NEEDED=$(readelf -d "$PREFIX/lib/libMIOpen.so" | awk '/NEEDED/ { gsub(/[\[\]]/,"",$NF); print $NF }')
if [[ -e "$WORKDIR/consumer_private" ]]; then
    # flag-on install
    if ! grep -q '^libMIOpen_private\.so' <<< "$NEEDED"; then
        echo "FAIL: flag-on libMIOpen.so does not have libMIOpen_private.so in DT_NEEDED" >&2
        echo "DT_NEEDED:" >&2
        printf '  %s\n' $NEEDED >&2
        exit 1
    fi
    echo "Q6 PASS: find_package OK; both consumer paths built; wrapper depends on libMIOpen_private.so"
else
    # flag-off install
    if grep -q '^libMIOpen_private\.so' <<< "$NEEDED"; then
        echo "FAIL: flag-off libMIOpen.so unexpectedly references libMIOpen_private.so" >&2
        exit 1
    fi
    echo "Q6 PASS: find_package OK; wrapper consumer built (legacy install, no bypass)"
fi
