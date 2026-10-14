#!/usr/bin/env bash
# usage: cycles.sh name=logdir [name=logdir ...]
# GPU-active cycles (GRBM_GUI_ACTIVE) and kernel duration for each build's ClientParameters.ini on
# physical GPU $GPU_ID (default 3). The part is power-throttled and its clock depends on the data
# (zero A/B runs ~20% faster in the same cycles), so variants that leave stale LDS data must be
# compared in cycles, not microseconds. MHz is GUI cycles per microsecond / 8 (one count per XCD).
set -u
cd "$(dirname "$0")"
ROCM_PATH=/opt/rocm-10.1.0a20260908+bkc.20260917
export HIP_VISIBLE_DEVICES=${GPU_ID:-3} LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib/x86_64-unknown-linux-gnu
unset ROCR_VISIBLE_DEVICES
C=../../build_tmp/tensilelite/client/tensilelite-client
stamp=$(date +%Y%m%d-%H%M%S)
names=()
for nd in "$@"; do
  name=${nd%%=*} dir=${nd#*=}
  names+=("$name")
  out=logs/$stamp-cyc-$name
  mkdir -p "$out"
  timeout -s KILL 600 $ROCM_PATH/bin/rocprofv3 --pmc GRBM_GUI_ACTIVE --kernel-trace -E configs/extra_counters.yaml \
    --kernel-include-regex Cijk -f csv --output-directory "$out/rocprof" --output-file pmc \
    -- $C --config-file "$dir/ClientParameters.ini" > "$out/run.log" 2>&1 || echo "$out failed"
done
../../.venv/bin/python - "$stamp" "${names[@]}" <<'PY' 2>&1 | grep -v frozen
import csv, glob, collections, statistics as st, sys
stamp, names = sys.argv[1], sys.argv[2:]
print(f"{'build':14s} {'Mcycles':>8s} {'vs first':>9s} {'us':>7s} {'MHz':>6s}")
ref = None
for name in names:
    cyc, dur = {}, {}
    for f in glob.glob(f"logs/{stamp}-cyc-{name}/rocprof/**/*counter_collection.csv", recursive=True):
        for r in csv.DictReader(open(f)):
            if r["Counter_Name"] == "GRBM_GUI_ACTIVE":
                cyc[r["Dispatch_Id"]] = cyc.get(r["Dispatch_Id"], 0) + float(r["Counter_Value"])
    for f in glob.glob(f"logs/{stamp}-cyc-{name}/rocprof/**/*kernel_trace.csv", recursive=True):
        for r in csv.DictReader(open(f)):
            dur[r["Dispatch_Id"]] = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
    ids = [i for i in cyc if i in dur]
    if not ids:
        print(f"{name:14s} no data"); continue
    c, t = st.median(cyc[i] for i in ids) / 1e6, st.median(dur[i] for i in ids)
    ref = ref or c
    print(f"{name:14s} {c:8.3f} {100 * (c / ref - 1):+8.1f}% {t:7.1f} {c * 1e6 / t / 8:6.0f}")
PY
