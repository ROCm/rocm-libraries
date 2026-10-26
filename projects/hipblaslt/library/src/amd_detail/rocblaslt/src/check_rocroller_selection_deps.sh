#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fail when hipblaslt-rocroller-selection's compiler dependency files name
# rocRoller's headers or libfmt. The ranker compiles from rocroller_enums.hpp,
# so a green build that opened /opt/rocm is not enough. The sources under
# rocroller/ do not match rocRoller/.
# Usage: check_rocroller_selection_deps.sh <hipblaslt-rocroller-selection.dir>

set -euo pipefail

depdir=${1:?directory of hipblaslt-rocroller-selection.dir}
if [[ ! -d "$depdir" ]]; then
    echo "not a directory: $depdir" >&2
    exit 1
fi

mapfile -t deps < <(find "$depdir" -type f -name '*.d' | sort)
if [[ ${#deps[@]} -eq 0 ]]; then
    echo "hipblaslt-rocroller-selection produced no .d files under $depdir" >&2
    exit 1
fi

bad=0
for dep in "${deps[@]}"; do
    if grep -F -e 'rocRoller/' -e 'fmt/' -q "$dep"; then
        echo "names rocRoller/ or fmt/: $dep" >&2
        bad=1
    fi
done
if [[ $bad -ne 0 ]]; then
    exit 1
fi

echo "hipblaslt-rocroller-selection: ${#deps[@]} .d file(s) name neither rocRoller/ nor fmt/"
