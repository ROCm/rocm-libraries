#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fail when hipblaslt-rocroller-selection's compiler dependencies name
# rocRoller's headers or libfmt. The ranker describes a kernel with hipDataType
# and hipBLASLt's scaling format, so a green build that opened /opt/rocm is not
# enough. The sources under rocroller/ do not match rocRoller/.
#
# Makefile builds leave *.d files in the target directory. Ninja keeps the same
# edges in .ninja_deps, so this also reads `ninja -t deps` when no .d files
# were written. A path component is only a hit when it is a header under
# /rocRoller/ or /fmt/, not when the checkout or object path contains those
# letters.
# Usage: check_rocroller_selection_deps.sh <hipblaslt-rocroller-selection.dir>

set -euo pipefail

depdir=${1:?directory of hipblaslt-rocroller-selection.dir}
if [[ ! -d "$depdir" ]]; then
    echo "not a directory: $depdir" >&2
    exit 1
fi

is_header() {
    case "$1" in
        *.h|*.hh|*.hpp|*.hxx|*.inl|*.cuh) return 0 ;;
        *) return 1 ;;
    esac
}

# A header path whose directory component is rocRoller or fmt.
bad_header() {
    local path=$1
    is_header "$path" || return 1
    case "$path" in
        */rocRoller/*|*/fmt/*) return 0 ;;
        *) return 1 ;;
    esac
}

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
record=$tmpdir/deps.txt
: >"$record"

make_deps=()
while IFS= read -r -d '' f; do
    make_deps+=("$f")
done < <(find "$depdir" -type f \( -name '*.d' -o -name 'depend.make' -o -name 'compiler_depend.make' \) -print0)
if [[ ${#make_deps[@]} -gt 0 ]]; then
    # Prerequisites only. The rule target is the object path, which can sit
    # under a checkout whose name contains fmt or rocRoller.
    for dep in "${make_deps[@]}"; do
        [[ -z "$dep" ]] && continue
        awk '
            /^#/ { next }
            /^[^#].*:/ { sub(/^[^:]*:[[:space:]]*/, "") }
            {
                gsub(/\\$/, "")
                n = split($0, a, /[[:space:]]+/)
                for (i = 1; i <= n; i++) if (a[i] != "") print a[i]
            }
        ' "$dep"
    done | sort -u >>"$record"
fi

if [[ ! -s "$record" ]]; then
    builddir=$depdir
    ninjafile=
    while [[ "$builddir" != "/" ]]; do
        if [[ -f "$builddir/build.ninja" ]]; then
            ninjafile=$builddir
            break
        fi
        builddir=$(dirname "$builddir")
    done
    if [[ -z "$ninjafile" ]] || ! command -v ninja >/dev/null 2>&1; then
        echo "hipblaslt-rocroller-selection produced no dependency records under $depdir" >&2
        exit 1
    fi
    mapfile -t objs < <(find "$depdir" -type f \( -name '*.o' -o -name '*.obj' \) | sort)
    if [[ ${#objs[@]} -eq 0 ]]; then
        echo "hipblaslt-rocroller-selection produced no objects under $depdir" >&2
        exit 1
    fi
    # `ninja -t deps` prints a header per object, then one dependency per line.
    ninja -C "$ninjafile" -t deps "${objs[@]}" >"$tmpdir/ninja-deps.txt"
    awk 'NF && $0 ~ /^[[:space:]]/ { gsub(/^[[:space:]]+/, ""); print }' \
        "$tmpdir/ninja-deps.txt" >>"$record"
fi

if [[ ! -s "$record" ]]; then
    echo "hipblaslt-rocroller-selection produced no dependency records under $depdir" >&2
    exit 1
fi

bad=0
seen=0
while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    seen=$((seen + 1))
    if bad_header "$path"; then
        echo "names rocRoller/ or fmt/: $path" >&2
        bad=1
    fi
done <"$record"

if [[ $bad -ne 0 ]]; then
    exit 1
fi

echo "hipblaslt-rocroller-selection: $seen dependency path(s) name neither rocRoller/ nor fmt/ headers"
