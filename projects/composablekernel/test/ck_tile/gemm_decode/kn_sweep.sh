#!/usr/bin/env bash
# gemm_decode FP8 M=1 effective-bandwidth vs B-matrix size (= N*K bytes).
# Tests whether the §15.H mid-N M=1 "valley" tracks B-size (cache residency)
# or N alone, by sweeping N at several K and tabulating eff-BW. Iso-B-size
# points (e.g. N16384/K7168 vs N8192/K14336, both 117 MB) sit at different N
# but the same B-size; if their eff-BW matches, the driver is cache residency.
set -u
BIN=./build/bin/bench_gemm_decode_msweep_fp8
printf "%6s %6s %8s %9s %9s\n" K N B_MB best_us effBW_TBs
for K in 2048 7168 14336; do
  for N in 2048 4096 7168 8192 16384; do
    row=$("$BIN" 25 200 "$N" "$K" 1 2>/dev/null \
          | awk -F, '/gemm_decode_fp8_best/{print $5","$7}')
    t_us=${row%,*}; gbps=${row#*,}
    python3 -c "
k=$K; n=$N; t=$t_us; g=$gbps
print(f'{k:6d} {n:6d} {n*k/1e6:8.1f} {t:9.2f} {g/1e3:9.2f}')"
  done
done
