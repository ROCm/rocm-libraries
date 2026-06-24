#!/usr/bin/env bash
# Does split-K help gemm_decode in the under-fill corner (small N x large K)?
# At M=1 the grid is ~N workgroups; N below ~2.5k under-fills 256 CUs, so eff-BW
# falls well below the ~7 TB/s large-B asymptote. This probes whether the bench's
# *atomic* k_batch sweep already recovers any of that headroom (if so, the cheaper
# LDS K-split / B1 would help more). Prints best cfg (kb,wpb) + base(kb1,wpb1).
set -u
BIN=./build/bin/bench_gemm_decode_msweep_fp8
printf "%5s %6s %7s %8s %8s %9s %4s %4s %7s\n" \
  N K B_MB base_us best_us effBW_TBs kb wpb spd
for K in 16384 32768; do
  for N in 256 512 1024 2048 4096; do
    out=$("$BIN" 25 200 "$N" "$K" 1 2>/tmp/sk_err)
    row=$(echo "$out" | awk -F, '/gemm_decode_fp8_best/{print $5","$7","$11","$14}')
    base=$(awk '/M= *1  base=/{for(i=1;i<=NF;i++) if($i ~ /base=/){gsub(/base=|us/,"",$(i+1)); print $(i+1)?$(i+1):$i}}' /tmp/sk_err | head -1)
    base=$(grep -oE 'base= *[0-9.]+us' /tmp/sk_err | head -1 | grep -oE '[0-9.]+')
    t=${row%%,*}; rest=${row#*,}; g=${rest%%,*}; rest=${rest#*,}; kb=${rest%%,*}; wpb=${rest#*,}
    python3 -c "
n=$N;k=$K;t=$t;g=$g;b=${base:-0}
print(f'{n:5d} {k:6d} {n*k/1e6:7.1f} {b:8.2f} {t:8.2f} {g/1e3:9.2f} {$kb:4d} {$wpb:4d} {(b/t if t else 0):7.3f}')"
  done
done