#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Drift gate for the hipDNN miopen-provider's private-symbol rename header.
#
# When MIOpen is built with the public/private split (MIOPEN_ENABLE_HIPDNN_WRAPPER
# =ON), the provider links libMIOpen_private.so and force-includes its own copy of
# the `#define miopen<Name> miopen<Name>_impl` set so its public-name calls bind
# the private _impl entry points. That copy MUST match MIOpen's own force-included
# rename header; if the two diverge, a flag-on provider build fails to link (an
# undefined _impl symbol, or a public name that no longer resolves). This gate
# catches the divergence at commit/CI time instead of at build time.
#
# It compares only the set of `#define` mappings, ignoring comments, include-guard
# vs. #pragma once, blank lines, and line-continuation formatting.
#
# Usage: check_rename_header_sync.sh [<lib_rename_header> <provider_rename_header>]
# With no arguments the in-repo canonical locations are used (run from repo root).

set -euo pipefail

LIB_HEADER="${1:-projects/miopen/src/private/miopen_private_rename.h}"
PROV_HEADER="${2:-dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp}"

for f in "$LIB_HEADER" "$PROV_HEADER"; do
    [ -f "$f" ] || { echo "check_rename_header_sync: file not found: $f" >&2; exit 2; }
done

# Extract the set of `#define miopen<Name> miopen<Name>_impl` mappings: join
# backslash line-continuations, keep only #define lines whose macro name starts
# with lowercase `miopen` (this excludes the library header's include-guard
# `#define MIOPEN_PRIVATE_RENAME_H`), normalize whitespace, and sort.
extract_defines() {
    awk '/\\$/ { sub(/\\$/, ""); printf "%s", $0; next } { print }' "$1" \
        | grep -E '^[[:space:]]*#define[[:space:]]+miopen' \
        | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' \
        | sort -u
}

lib_defs="$(mktemp)"
prov_defs="$(mktemp)"
trap 'rm -f "$lib_defs" "$prov_defs"' EXIT

extract_defines "$LIB_HEADER"  > "$lib_defs"
extract_defines "$PROV_HEADER" > "$prov_defs"

if diff "$lib_defs" "$prov_defs" > /tmp/rename_sync_diff.txt 2>&1; then
    echo "check_rename_header_sync: OK ($(wc -l < "$lib_defs") renames in sync)"
    exit 0
fi

{
    echo "ERROR: the provider rename header is out of sync with MIOpen's private rename header."
    echo "  MIOpen:   $LIB_HEADER"
    echo "  provider: $PROV_HEADER"
    echo "  ('<' = only in MIOpen, '>' = only in provider):"
    sed 's/^/    /' /tmp/rename_sync_diff.txt
    echo "Fix: mirror the change into both headers so the provider binds the same _impl set."
} >&2
exit 1
