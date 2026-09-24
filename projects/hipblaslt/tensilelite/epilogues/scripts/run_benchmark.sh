#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGIC_BASE="$(cd "$SCRIPT_DIR/../../../library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950" && pwd)"

export TENSILE_DISABLE_HELPER_CACHE=1

if [[ $# -ne 1 ]]; then
  echo "usage: $(basename "$0") <benchmark.yaml>"
  exit 1
fi

yaml="$1"

if [[ ! -f "$yaml" ]]; then
  echo "run_benchmark: file not found: $yaml"
  exit 1
fi

name="$(basename "$yaml" .yaml)"
tmp_dir="/tmp/benchmark-harness-$name"

library_type="$(grep -m1 'LibraryType:' "$yaml" | sed 's/.*LibraryType:[[:space:]]*"\([^"]*\)".*/\1/')"
case "$library_type" in
  Equality)   dest_dir="$LOGIC_BASE/Equality" ;;
  Prediction) dest_dir="$LOGIC_BASE/Origami" ;;
  *)
    echo "run_benchmark: unknown LibraryType '$library_type' in $yaml"
    exit 1
    ;;
esac

echo "--- running $name (LibraryType=$library_type) ---"
rm -rf "$tmp_dir"
cd "$TENSILE_DIR"
./Tensile/bin/Tensile "$yaml" "$tmp_dir"
rc=$?

if [[ $rc -ne 0 ]]; then
  echo "run_benchmark: Tensile FAILED for $name (rc=$rc)"
  exit $rc
fi

logic_dir="$tmp_dir/3_LibraryLogic"
mapfile -t logic_files < <(find "$logic_dir" -maxdepth 1 -name '*.yaml' 2>/dev/null)

if [[ ${#logic_files[@]} -eq 0 ]]; then
  echo "run_benchmark: no logic file produced for $name"
  exit 1
fi

for logic_file in "${logic_files[@]}"; do
  dest="$dest_dir/$(basename "$logic_file")"
  cp "$logic_file" "$dest"
  echo "copied $(basename "$logic_file") -> $dest_dir/"
done
