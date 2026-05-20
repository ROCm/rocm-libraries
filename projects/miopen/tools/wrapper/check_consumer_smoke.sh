#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Q5 harness for RFC 0001 (Phase 1 investigation):
#
#   - Compiles consumer_smoke.c against an installed MIOpen prefix.
#   - Asserts that no _impl symbol is referenced by the resulting binary
#     (i.e. the rename header did not leak into the installed headers).
#   - Asserts that the installed libMIOpen.so still exports the public
#     miopenCreate/miopenDestroy symbols by *unsuffixed* name.
#
# Designed to run in both flag-off and flag-on CI configurations. The exit
# criteria for Q5 is that this script returns 0 in both.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --prefix <install-prefix> [--cc <compiler>] [--workdir <dir>]

Required:
  --prefix     Path to a 'cmake --install' tree containing include/miopen/miopen.h
               and lib/libMIOpen.so.

Optional:
  --cc         Compiler to use (default: \$CC or cc).
  --workdir    Build dir (default: a tempdir; auto-cleaned).
EOF
}

PREFIX=""
CC_BIN="${CC:-cc}"
WORKDIR=""
CLEANUP_WORKDIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)  PREFIX="$2"; shift 2 ;;
        --cc)      CC_BIN="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$PREFIX" ]]; then
    echo "error: --prefix is required" >&2
    usage; exit 2
fi
if [[ ! -f "$PREFIX/include/miopen/miopen.h" ]]; then
    echo "error: $PREFIX/include/miopen/miopen.h not found" >&2
    exit 2
fi
if [[ ! -f "$PREFIX/lib/libMIOpen.so" ]]; then
    echo "error: $PREFIX/lib/libMIOpen.so not found" >&2
    exit 2
fi

if [[ -z "$WORKDIR" ]]; then
    WORKDIR=$(mktemp -d)
    CLEANUP_WORKDIR=1
fi
trap '[[ $CLEANUP_WORKDIR -eq 1 ]] && rm -rf "$WORKDIR"' EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC="$SCRIPT_DIR/consumer_smoke.c"
BIN="$WORKDIR/consumer_smoke"

echo "Q5 [1/3] Compiling $SRC against $PREFIX"
"$CC_BIN" -Wall -Wextra -Werror \
    "-I$PREFIX/include" \
    "$SRC" -o "$BIN" \
    "-L$PREFIX/lib" -lMIOpen \
    "-Wl,-rpath,$PREFIX/lib"

# 1) The installed header tree must not contain the rename header.
echo "Q5 [2/3] Asserting miopen_private_rename.h is not installed"
if find "$PREFIX/include" -name 'miopen_private_rename.h' | grep -q .; then
    echo "FAIL: miopen_private_rename.h leaked into installed headers" >&2
    exit 1
fi

# 2) The compiled binary must reference unsuffixed public symbols.
echo "Q5 [3/3] Asserting binary references miopenCreate (not miopenCreate_impl)"
if ! nm -u "$BIN" 2>/dev/null | grep -qE '\bmiopenCreate\b'; then
    echo "FAIL: binary does not reference miopenCreate as undefined symbol" >&2
    nm -u "$BIN" 2>/dev/null | grep -i miopen >&2 || true
    exit 1
fi
if nm -u "$BIN" 2>/dev/null | grep -qE 'miopen[A-Za-z0-9_]+_impl'; then
    echo "FAIL: binary references one or more miopen*_impl symbols" >&2
    nm -u "$BIN" 2>/dev/null | grep _impl >&2
    exit 1
fi

echo "Q5 PASS: consumer-build smoke is clean"
