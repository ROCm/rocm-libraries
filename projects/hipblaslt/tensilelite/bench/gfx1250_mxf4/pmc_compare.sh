#!/usr/bin/env bash
# usage: pmc_compare.sh ROUNDS name=logdir [name=logdir ...]
# LDS bank / address / cross-port segment conflict counters (configs/rocprof_pmc.yaml) for each
# build's ClientParameters.ini on physical GPU $GPU_ID (default 3), then one summary row per build:
# medians over dispatches and rounds, plus stall share of CU cycles (see KNOWN_ISSUES.md).
set -u
cd "$(dirname "$0")"
rounds=$1; shift
ROCM_PATH=/opt/rocm-10.1.0a20260908+bkc.20260917
export HIP_VISIBLE_DEVICES=${GPU_ID:-3} LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib/x86_64-unknown-linux-gnu
unset ROCR_VISIBLE_DEVICES
C=../../build_tmp/tensilelite/client/tensilelite-client
stamp=$(date +%Y%m%d-%H%M%S)
names=()
for i in $(seq "$rounds"); do
  for nd in "$@"; do
    name=${nd%%=*} dir=${nd#*=}
    [ "$i" = 1 ] && names+=("$name")
    ini=$(find "$dir" -maxdepth 1 -name ClientParameters.ini | head -1)
    out=logs/$stamp-pmcc-$name-r$i
    mkdir -p "$out"
    timeout -s KILL 600 $ROCM_PATH/bin/rocprofv3 --input configs/rocprof_pmc.yaml -E configs/extra_counters.yaml \
      --output-directory "$out/rocprof" --output-file pmc -- $C --config-file "$ini" > "$out/run.log" 2>&1 \
      || echo "$out failed"
  done
done
../../.venv/bin/python - "$stamp" "${names[@]}" <<'PY' 2>&1 | grep -v frozen
import csv, glob, collections, statistics as st, sys
stamp, names = sys.argv[1], sys.argv[2:]
P = "TX_PERF_SEL_VMW_"
keys = [("bank", P + "LDS_BANK_CONFLICT"), ("addr", P + "LDS_ADDR_CONFLICT"),
        ("seg", P + "CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES"), ("ldsActive", P + "LDS_INPUT_ACTIVE"),
        ("gui", "GRBM_GUI_ACTIVE")]
print(f"{'build':12s} {'bank':>10s} {'addr':>10s} {'seg stall':>10s} {'seg %CU':>8s} {'LDS active':>11s} {'GUI active':>11s}")
for name in names:
    vals = collections.defaultdict(list)
    for f in glob.glob(f"logs/{stamp}-pmcc-{name}-r*/rocprof/pass_*/pmc_counter_collection.csv"):
        d = collections.defaultdict(dict)
        for r in csv.DictReader(open(f)):
            d[r["Dispatch_Id"]][r["Counter_Name"]] = float(r["Counter_Value"])
        for short, k in keys:
            xs = [v[k] for v in d.values() if k in v]
            if xs: vals[short].append(st.median(xs))
    m = {k: st.median(v) for k, v in vals.items()}
    if not m:
        print(f"{name:12s} no data"); continue
    segPct = 100 * m["seg"] / 256 / (m["gui"] / 8) if m.get("gui") else float("nan")
    print(f"{name:12s} {m['bank']:10.0f} {m['addr']:10.0f} {m['seg']:10.0f} {segPct:7.3f}% "
          f"{m['ldsActive']:11.0f} {m['gui']:11.0f}")
PY
