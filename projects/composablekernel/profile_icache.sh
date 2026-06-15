#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Compare L1 instruction-cache behaviour of the vanilla vs prefetch bwd_weight
# example using rocprofv3 SQC_ICACHE counters (AICK-1303).
#
# Hypothesis under test: the GEMM main loop body (~1.5 KB) fits in L1I and is
# reused across all ~3721 iterations, so the I-cache miss rate is ~0 and
# instruction prefetch has nothing to hide. If miss_rate is already near zero in
# BOTH runs, that confirms prefetch cannot help this kernel family.
#
# Runs the two binaries SEQUENTIALLY (never overlap GPU work). Counter values are
# deterministic ratios, so a single repeat is enough.
#
# Usage:
#   ./profile_icache.sh                       # default shape, default binaries
#   ./profile_icache.sh -g 2 -n 64 -c 256 ... # override shape args (passed to both binaries)
set -euo pipefail

VANILLA="${VANILLA_BIN:-./build/bin/tile_example_grouped_conv_bwd_weight}"
PREFETCH="${PREFETCH_BIN:-./build-prefetch/bin/tile_example_grouped_conv_bwd_weight}"

# Shape / run args passed to BOTH binaries. -v 0 disables validation so only the
# conv kernel dispatches; keep repeat small (counters are ratios).
SHAPE_ARGS=("$@")
if [[ ${#SHAPE_ARGS[@]} -eq 0 ]]; then
  SHAPE_ARGS=(-v 0 -warmup 0 -repeat 1)
fi

OUTROOT="icache_profile"
KERNEL_FILTER='%BackwardWeight%'   # matches mangled name GroupedConvolutionBackwardWeight

# rocprofv3 counter input. All four SQC_ICACHE counters in one pass; if rocprofv3
# reports it cannot collect them together, split into multiple "pmc:" lines.
COUNTERS_FILE="${OUTROOT}/icache_counters.txt"
mkdir -p "$OUTROOT"
printf 'pmc: SQC_ICACHE_REQ SQC_ICACHE_HITS SQC_ICACHE_MISSES SQC_ICACHE_MISSES_DUPLICATE\n' > "$COUNTERS_FILE"

run_one() {
  local label="$1" bin="$2"
  local outdir="${OUTROOT}/${label}"
  rm -rf "$outdir"; mkdir -p "$outdir"
  if [[ ! -x "$bin" ]]; then echo "ERROR: binary not found: $bin" >&2; exit 1; fi
  echo ">>> [$label] $bin ${SHAPE_ARGS[*]}"
  rocprofv3 -i "$COUNTERS_FILE" --output-format csv -d "$outdir" -- \
    "$bin" "${SHAPE_ARGS[@]}" > "${outdir}/stdout.log" 2>&1 || {
      echo "ERROR: rocprofv3 run failed for $label; see ${outdir}/stdout.log" >&2; exit 1; }
}

# --- run sequentially ---
run_one vanilla  "$VANILLA"
run_one prefetch "$PREFETCH"

# --- analyse with duckdb ---
analyse() {
  local label="$1"
  local glob="${OUTROOT}/${label}/**/*counter_collection.csv"
  duckdb -box -c "
    WITH c AS (
      SELECT Counter_Name, Counter_Value
      FROM read_csv_auto('${glob}', union_by_name=true)
      WHERE Kernel_Name LIKE '${KERNEL_FILTER}'
    )
    SELECT
      '${label}' AS run,
      sum(CASE WHEN Counter_Name='SQC_ICACHE_REQ'    THEN Counter_Value END) AS req,
      sum(CASE WHEN Counter_Name='SQC_ICACHE_HITS'   THEN Counter_Value END) AS hits,
      sum(CASE WHEN Counter_Name='SQC_ICACHE_MISSES' THEN Counter_Value END) AS misses,
      round(100.0 * sum(CASE WHEN Counter_Name='SQC_ICACHE_MISSES' THEN Counter_Value END)
            / nullif(sum(CASE WHEN Counter_Name='SQC_ICACHE_REQ' THEN Counter_Value END),0), 4)
            AS miss_rate_pct
    FROM c;
  "
}

echo
echo "=== Kernels matched by filter '${KERNEL_FILTER}' (sanity check) ==="
duckdb -box -c "
  SELECT DISTINCT Kernel_Name
  FROM read_csv_auto('${OUTROOT}/vanilla/**/*counter_collection.csv', union_by_name=true)
  WHERE Kernel_Name LIKE '${KERNEL_FILTER}';"

echo
echo "=== L1 instruction-cache miss rate: vanilla vs prefetch ==="
analyse vanilla
analyse prefetch
echo
echo "Interpretation: if miss_rate_pct is ~0 in both runs, the I-cache is not a"
echo "bottleneck and prefetch cannot move the needle (confirms the static analysis)."
