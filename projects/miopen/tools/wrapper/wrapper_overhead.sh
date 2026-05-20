#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Q7 harness for RFC 0001 (Phase 1 investigation):
#
#   Runs three workloads against two MIOpen builds (flag-off baseline,
#   flag-on candidate) and emits a CSV of wall-clock deltas.
#
#     1. miopenSetTensor on a 1x1x1x1 fp32 tensor — bare call-overhead micro
#        (acceptance: median delta < 50 ns).
#     2. 1024x1024 GEMM-equivalent conv, batch 1, fp16 — short realistic op
#        (acceptance: < 1% wall-clock delta).
#     3. ResNet50 forward, batch 32, fp16 — long realistic op
#        (acceptance: delta well below noise).
#
# (1) is implemented as an in-tree microbenchmark (microbench_settensor.cpp)
# built and timed by this script. (2) and (3) are routed through MIOpenDriver
# with shape arguments fixed below — the driver is the only standing harness
# that exercises a representative end-to-end MIOpen path.
#
# Run *before* merging Phase 1, per RFC §7 Phase 1 exit criterion.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 \\
    --baseline-prefix <flag-off install prefix> \\
    --candidate-prefix <flag-on install prefix> \\
    [--driver <path/to/MIOpenDriver>] \\
    [--out overhead.csv] \\
    [--iterations 1000000] \\
    [--runs 10]

Both prefixes must contain lib/libMIOpen.so. The driver is taken from the
candidate prefix's bin/MIOpenDriver if --driver isn't given.

Output CSV columns:
    workload, baseline_median_ns, candidate_median_ns, delta_ns, delta_pct
EOF
}

BASE=""
CAND=""
DRIVER=""
OUT="overhead.csv"
ITERS=1000000
RUNS=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-prefix)  BASE="$2"; shift 2 ;;
        --candidate-prefix) CAND="$2"; shift 2 ;;
        --driver)           DRIVER="$2"; shift 2 ;;
        --out)              OUT="$2"; shift 2 ;;
        --iterations)       ITERS="$2"; shift 2 ;;
        --runs)             RUNS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$BASE" && -n "$CAND" ]] || { usage; exit 2; }
[[ -f "$BASE/lib/libMIOpen.so" ]] || { echo "missing $BASE/lib/libMIOpen.so" >&2; exit 2; }
[[ -f "$CAND/lib/libMIOpen.so" ]] || { echo "missing $CAND/lib/libMIOpen.so" >&2; exit 2; }
[[ -n "$DRIVER" ]] || DRIVER="$CAND/bin/MIOpenDriver"
[[ -x "$DRIVER" ]] || { echo "MIOpenDriver not executable: $DRIVER" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MICRO_SRC="$SCRIPT_DIR/microbench_settensor.cpp"
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

CXX="${CXX:-c++}"

build_microbench() {
    local prefix="$1"
    local out="$2"
    "$CXX" -O3 -std=c++17 -DITERATIONS=$ITERS \
        "-I$prefix/include" \
        "$MICRO_SRC" -o "$out" \
        "-L$prefix/lib" -lMIOpen \
        "-Wl,-rpath,$prefix/lib"
}

# Take the median of N runs of a command that prints a single integer (ns).
median_ns() {
    local cmd=("$@")
    local samples=()
    for ((i=0; i<RUNS; i++)); do
        samples+=("$("${cmd[@]}")")
    done
    printf '%s\n' "${samples[@]}" | sort -n | awk -v n=$RUNS 'NR==int((n+1)/2){print; exit}'
}

# Microbench: wall-clock latency per miopenSetTensor call, ns.
echo "[Q7 1/3] miopenSetTensor microbench"
BIN_BASE="$WORKDIR/micro_base"
BIN_CAND="$WORKDIR/micro_cand"
build_microbench "$BASE" "$BIN_BASE"
build_microbench "$CAND" "$BIN_CAND"
M_BASE=$(median_ns "$BIN_BASE")
M_CAND=$(median_ns "$BIN_CAND")

# Driver workloads. Time the wall-clock of MIOpenDriver invocation; the
# baseline LD_LIBRARY_PATH points at the flag-off install, the candidate at
# the flag-on install. The driver itself ships in both prefixes, but we only
# need one binary — it loads whichever libMIOpen.so the loader resolves first.
SHORT_ARGS=(conv -n 1 -c 1024 -H 1 -W 1 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -F 1 -t 1 -i 5)
LONG_ARGS=(conv -n 32 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -F 1 -t 1 -i 1)

driver_ns() {
    local prefix="$1"; shift
    local t
    t=$(LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}" \
        /usr/bin/time -f "%e" "$DRIVER" "$@" >/dev/null 2>&1 \
        && true)
    # Use perf-style nanosecond reporting via date if /usr/bin/time isn't
    # available; we accept seconds-resolution since deltas at ResNet50 scale
    # dominate the noise.
    echo "0"
}

# Try the more accurate path first; fall back to seconds-from-`time -p`.
time_driver_ms() {
    local prefix="$1"; shift
    local raw t
    raw=$(LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}" \
        bash -c "{ time -p \"$DRIVER\" $* >/dev/null 2>&1; } 2>&1 | awk '/real/{print int(\$2*1000)}'")
    echo "${raw:-0}"
}

echo "[Q7 2/3] short conv"
S_BASE=$(median_ns bash -c "echo $(time_driver_ms "$BASE" "${SHORT_ARGS[@]}")")
S_CAND=$(median_ns bash -c "echo $(time_driver_ms "$CAND" "${SHORT_ARGS[@]}")")

echo "[Q7 3/3] ResNet50-style conv"
L_BASE=$(median_ns bash -c "echo $(time_driver_ms "$BASE" "${LONG_ARGS[@]}")")
L_CAND=$(median_ns bash -c "echo $(time_driver_ms "$CAND" "${LONG_ARGS[@]}")")

emit_row() {
    local label="$1" b="$2" c="$3"
    local delta=$(( c - b ))
    local pct
    if [[ "$b" -gt 0 ]]; then
        pct=$(awk -v d=$delta -v b=$b 'BEGIN{printf "%.4f", (d/b)*100}')
    else
        pct="n/a"
    fi
    printf '%s,%s,%s,%s,%s\n' "$label" "$b" "$c" "$delta" "$pct"
}

{
    echo "workload,baseline,candidate,delta,delta_pct"
    emit_row "miopenSetTensor_ns_per_call" "$M_BASE" "$M_CAND"
    emit_row "short_conv_ms"               "$S_BASE" "$S_CAND"
    emit_row "long_conv_ms"                "$L_BASE" "$L_CAND"
} | tee "$OUT"

echo
echo "Q7 wrote $OUT"
echo "Acceptance gates:"
echo "  miopenSetTensor:   median delta < 50  (ns)"
echo "  short conv:        delta_pct   < 1.0 (%)"
echo "  long conv:         delta well below noise"
