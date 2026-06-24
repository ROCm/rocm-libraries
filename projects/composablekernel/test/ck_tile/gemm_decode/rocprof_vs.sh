#!/usr/bin/env bash
# Side-by-side rocprofv3 counters: gemm_decode (wpb1, wpb4) vs AITER wvSplitKQ,
# all at FP8 M=1 N=7168 K=7168 on the same GPU. One counter per pass (multi-pmc
# hangs on this stack). Robust quoted-CSV parse + Kernel_Name filter (the
# wvSplitKQ template name contains commas, which shifts naive awk columns).
set -u
ITERS=${1:-40}
PROF=./build/bin/prof_gemm_decode_one
PY=/opt/venv/bin/python3
WVPY=test/ck_tile/gemm_decode/prof_wvsplitkq.py
GD_RE='kentry|GemmDecode'
WV_RE='wvSplitKQ'
COUNTERS=(FetchSize WriteSize VALUBusy SALUBusy MemUnitStalled MemUnitBusy \
          TCC_HIT_sum TCC_MISS_sum MeanOccupancyPerCU)
TMP=$(mktemp -d)

# $1=csv $2=kernel-name regex $3=column header -> mean over matching rows
mean_of() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import csv, re, sys
path, kre, col = sys.argv[1], sys.argv[2], sys.argv[3]
rx = re.compile(kre)
vals = []
with open(path) as f:
    for row in csv.DictReader(f):
        if not rx.search(row.get("Kernel_Name", "")):
            continue
        v = row.get(col) or row.get("Counter_Value")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
print(f"{sum(vals)/len(vals):.2f}" if vals else "n/a")
PY
}

run_case() {  # $1=label $2=outdir $3=counter ; sets global CSV
  local out="$TMP/$3_$1"; rm -rf "$out"
  case "$1" in
    gd1) timeout 90 rocprofv3 --pmc "$3" -f csv -d "$out" -- "$PROF" m1n7168 "$ITERS" >/dev/null 2>&1 ;;
    gd4) timeout 90 rocprofv3 --pmc "$3" -f csv -d "$out" -- "$PROF" m1n7168mw "$ITERS" >/dev/null 2>&1 ;;
    wv)  timeout 120 rocprofv3 --pmc "$3" -f csv -d "$out" -- "$PY" "$WVPY" "$ITERS" >/dev/null 2>&1 ;;
  esac
  find "$out" -name '*counter_collection.csv' 2>/dev/null | head -1
}

printf "%-18s %14s %14s %14s\n" counter gd_wpb1 gd_wpb4 wvSplitKQ
printf "%-18s %14s %14s %14s\n" "------" "------" "------" "---------"
declare -A CSV
for cnt in "${COUNTERS[@]}"; do
  printf "%-18s" "$cnt"
  for case in gd1 gd4 wv; do
    csv=$(run_case "$case" x "$cnt")
    re=$GD_RE; [[ "$case" == wv ]] && re=$WV_RE
    if [[ -n "$csv" ]]; then
      printf "%14s" "$(mean_of "$csv" "$re" "$cnt")"
      [[ "$cnt" == "MeanOccupancyPerCU" ]] && CSV[$case]="$csv"
    else printf "%14s" "n/a"; fi
  done
  echo
done

echo
echo "=== kernel launch geometry (from MeanOccupancy pass) ==="
printf "%-18s %14s %14s %14s\n" attr gd_wpb1 gd_wpb4 wvSplitKQ
for attr in Grid_Size Workgroup_Size VGPR_Count Accum_VGPR_Count SGPR_Count LDS_Block_Size Scratch_Size; do
  printf "%-18s" "$attr"
  for case in gd1 gd4 wv; do
    re=$GD_RE; [[ "$case" == wv ]] && re=$WV_RE
    printf "%14s" "$(mean_of "${CSV[$case]:-/dev/null}" "$re" "$attr")"
  done
  echo
done
rm -rf "$TMP"
