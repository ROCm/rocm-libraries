#!/usr/bin/env bash
# Drift check for the parallel _impl-form public header (RFC 0001 §4.6,
# ALMIOPEN-2160). Asserts that <miopen/miopen_impl.h> stays in sync with the
# public <miopen/miopen.h>: every public C entry point declared with MIOPEN_EXPORT
# in miopen.h must have a matching <name>_impl declaration in miopen_impl.h, and
# the impl header must not declare an _impl for a name that no longer exists in the
# public header (stale declaration).
#
# Usage: check_impl_header_sync.sh <public_header> <impl_header>
# Exits non-zero (and prints the offending names) on any drift.

set -euo pipefail

PUBLIC_HEADER="${1:?usage: check_impl_header_sync.sh <public_header> <impl_header>}"
IMPL_HEADER="${2:?usage: check_impl_header_sync.sh <public_header> <impl_header>}"

for f in "$PUBLIC_HEADER" "$IMPL_HEADER"; do
    if [[ ! -f "$f" ]]; then
        echo "check_impl_header_sync: file not found: $f" >&2
        exit 2
    fi
done

# Public function entry points: identifiers of the form miopen<Name> immediately
# followed by '(', i.e. function declarations. Requiring the trailing '(' excludes
# opaque-handle typedefs (miopenHandle_t), enum/struct type names, and
# function-pointer typedefs (whose name is followed by ')'), which are not
# forwarded entry points. Doc-comment body lines (leading '*') are stripped first
# so that function names mentioned in prose (e.g. "call miopenFoo() before ...")
# are not mistaken for declarations.
grep -vE '^[[:space:]]*\*' "$PUBLIC_HEADER" \
    | grep -oE 'miopen[A-Za-z0-9_]+\(' | sed 's/(//' | sort -u \
    > /tmp/_miopen_public_fns.txt

# All miopen* tokens in the public header (used for the reverse staleness check,
# where a type name is an acceptable base too).
grep -oE 'miopen[A-Za-z0-9_]+' "$PUBLIC_HEADER" | sort -u > /tmp/_miopen_public_names.txt

# _impl declarations in the shim header.
grep -oE 'miopen[A-Za-z0-9_]+_impl' "$IMPL_HEADER" | sort -u \
    | sed 's/_impl$//' > /tmp/_miopen_impl_bases.txt

status=0

# (1) Staleness: every _impl must correspond to a name present in the public
# header. Bases that are intentionally private (not in miopen.h) are listed as
# allowed exceptions below.
#   miopenConvolution*GetWorkSpaceSizeRange -- exported but deliberately not in the
#   public miopen.h (ALMIOPEN-2246); declared _impl in the shim header only.
allowed_private_bases='^miopenConvolution(Forward|BackwardData|BackwardWeights)GetWorkSpaceSizeRange$'
while read -r base; do
    [[ -z "$base" ]] && continue
    if grep -qE "$allowed_private_bases" <<<"$base"; then
        continue
    fi
    if ! grep -qxF "$base" /tmp/_miopen_public_names.txt; then
        echo "DRIFT: ${base}_impl is declared in $(basename "$IMPL_HEADER") but $base is not present in $(basename "$PUBLIC_HEADER")" >&2
        status=1
    fi
done < /tmp/_miopen_impl_bases.txt

# (2) Completeness: every public function entry point must have an _impl.
while read -r name; do
    [[ -z "$name" ]] && continue
    if ! grep -qxF "$name" /tmp/_miopen_impl_bases.txt; then
        echo "DRIFT: public entry point $name has no ${name}_impl in $(basename "$IMPL_HEADER")" >&2
        status=1
    fi
done < /tmp/_miopen_public_fns.txt

if [[ "$status" -eq 0 ]]; then
    echo "check_impl_header_sync: OK ($(wc -l < /tmp/_miopen_impl_bases.txt) _impl declarations in sync)"
fi

exit "$status"
