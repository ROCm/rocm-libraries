#!/usr/bin/env bash
# Profile the FP8 M=1/N=7168 single-warp baseline (m1n7168, wpb1) vs the §15.F
# multi-warp occupancy probe (m1n7168mw, wpb4) one counter per pass (multi-pmc
# hangs on this stack). Prints a compact mean-per-dispatch table per counter.
set -u
BIN=./build/bin/prof_gemm_decode_one
ITERS=${1:-40}
CASES=(m1n7168 m1n7168mw)
COUNTERS=(FetchSize WriteSize VALUBusy SALUBusy MemUnitStalled MemUnitBusy \
          TCC_HIT_sum TCC_MISS_sum MeanOccupancyPerCU)
TMP=$(mktemp -d)
printf "%-16s" "counter"
for c in "${CASES[@]}"; do printf "%18s" "$c"; done
echo
for cnt in "${COUNTERS[@]}"; do
  printf "%-16s" "$cnt"
  for case in "${CASES[@]}"; do
    out="$TMP/${cnt}_${case}"
    rm -rf "$out"
    rocprofv3 --pmc "$cnt" -f csv -d "$out" -- "$BIN" "$case" "$ITERS" \
      >/dev/null 2>&1
    csv=$(find "$out" -name '*counter_collection.csv' 2>/dev/null | head -1)
    if [[ -z "$csv" ]]; then printf "%18s" "n/a"; continue; fi
    # Average the counter value over all dispatches (last CSV column named
    # "Counter_Value"); fall back to the last numeric column.
    val=$(python3 - "$csv" "$cnt" <<'PY'
import csv,sys
path,cnt=sys.argv[1],sys.argv[2]
vals=[]
with open(path) as f:
    r=csv.DictReader(f)
    for row in r:
        v=row.get('Counter_Value') or row.get(cnt)
        if v is None:
            # find a column whose header contains the counter name
            for k in row:
                if cnt in k:
                    v=row[k]; break
        try: vals.append(float(v))
        except (TypeError,ValueError): pass
print(f"{sum(vals)/len(vals):.3f}" if vals else "n/a")
PY
)
    printf "%18s" "$val"
  done
  echo
done
rm -rf "$TMP"
