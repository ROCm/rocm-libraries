#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Every field of an operation's schema must be either consumed by the native
# pack or explicitly rejected. Prints one `UNCHECKED: <field>` line per field
# the pack never names, and exits 1 if there are any.
#
# A script rather than a snippet in two documents. The two transcribed copies
# disagreed on their working directory -- one rooted the pack at $PROVIDER, the
# other used a relative `packs/<Name>Native.cpp` that matches nothing when run
# from anywhere else, reporting every field as accounted for. A check that
# passes by looking at nothing is worse than no check.
#
# Usage: field_audit.sh <op_attributes.fbs> <Native.cpp>...
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $(basename "$0") <op_attributes.fbs> <Native.cpp>..." >&2
    exit 2
fi

fbs=$1
shift

[ -f "$fbs" ] || { echo "no such schema: $fbs" >&2; exit 2; }

# ONE schema. The quoted-glob guard below only catches a glob the shell did NOT
# expand; an unquoted `a*.fbs` arrives already expanded, so the extra schemas land
# in the pack-source list and are silently audited as if they were C++. Reject a
# .fbs in that position explicitly -- it is never a pack source.
case "$fbs" in
    *[*?]*) echo "pass ONE schema, not a glob: $fbs" >&2; exit 2 ;;
esac

for src in "$@"; do
    [ -f "$src" ] || { echo "no such pack source: $src" >&2; exit 2; }
    [ -r "$src" ] || { echo "pack source not readable: $src" >&2; exit 2; }
    case "$src" in
        *.fbs) echo "'$src' is a schema, not a pack source -- pass exactly one .fbs first, then the pack sources (an unquoted glob expands into this list)" >&2; exit 2 ;;
    esac
done

# `[a-z][a-z0-9_]*`, NOT `[a-z_]+`: a field containing a digit (in_0_tensor_uid,
# out_0_tensor_uid -- four of them in pointwise_attributes.fbs) never matched the
# letters-only pattern, so it never entered the audit set and could not be reported
# UNCHECKED. A pointwise pack ignoring every one of its I/O tensors scored 8/8.
fields=$(grep -hoP '^\s+\K[a-z][a-z0-9_]*(?=\s*:)' "$fbs" | sort -u)
[ -n "$fields" ] || { echo "parsed zero fields from $fbs -- wrong file?" >&2; exit 2; }

unchecked=0
total=0
for f in $fields; do
    total=$((total + 1))
    # Anchored on a non-identifier character, NOT a bare substring: `grep -F "$f("`
    # marks `k_tensor_uid` referenced when only `dk_tensor_uid(` appears. Twelve
    # shipped schemas carry such suffix pairs (k_/dk_, q_/dq_, o_/do_, value/
    # default_value...), so an SDPA-backward pack ignoring q, k and v audited clean.
    # $f is [a-z0-9_]+ by construction, so it needs no regex escaping.
    if ! grep -qE -- "(^|[^A-Za-z0-9_])${f}\(" "$@"; then
        echo "UNCHECKED: $f"
        unchecked=$((unchecked + 1))
    fi
done

echo "-- $((total - unchecked))/$total fields referenced in $# pack source(s)"
[ "$unchecked" -eq 0 ]
