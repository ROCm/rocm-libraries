#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
YAML_DIR="$SCRIPT_DIR/../YAMLs"
LOGIC_BASE="$(cd "$SCRIPT_DIR/../../../library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950" && pwd)"

export TENSILE_DISABLE_HELPER_CACHE=1

overall_rc=0

for yaml in "$YAML_DIR"/benchmark_*.yaml; do
  name="$(basename "$yaml" .yaml)"
  tmp_dir="/tmp/benchmark-harness-$name"

  # Derive the destination subdirectory from the YAML's LibraryType field.
  library_type="$(grep -m1 'LibraryType:' "$yaml" | sed 's/.*LibraryType:[[:space:]]*"\([^"]*\)".*/\1/')"
  case "$library_type" in
    Equality)   dest_dir="$LOGIC_BASE/Equality" ;;
    Prediction) dest_dir="$LOGIC_BASE/Origami" ;;
    *)
      echo "benchmark_harness: unknown LibraryType '$library_type' in $yaml, skipping"
      overall_rc=1
      continue
      ;;
  esac

  echo "--- running $name (LibraryType=$library_type) ---"
  rm -rf "$tmp_dir"
  cd "$TENSILE_DIR"
  ./Tensile/bin/Tensile "$yaml" "$tmp_dir"
  rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "benchmark_harness: Tensile FAILED for $name (rc=$rc)"
    overall_rc=1
    continue
  fi

  logic_dir="$tmp_dir/3_LibraryLogic"
  mapfile -t logic_files < <(find "$logic_dir" -maxdepth 1 -name '*.yaml' 2>/dev/null)

  if [[ ${#logic_files[@]} -eq 0 ]]; then
    echo "benchmark_harness: no logic file produced for $name"
    overall_rc=1
    continue
  fi

  for logic_file in "${logic_files[@]}"; do
    dest="$dest_dir/$(basename "$logic_file")"
    cp "$logic_file" "$dest"
    echo "copied $(basename "$logic_file") -> $dest_dir/"
  done
done

if [[ $overall_rc -eq 0 ]]; then
  echo "benchmark_harness PASSED: all benchmarks succeeded"
else
  echo "benchmark_harness FAILED: one or more benchmarks did not complete"
fi
exit $overall_rc
