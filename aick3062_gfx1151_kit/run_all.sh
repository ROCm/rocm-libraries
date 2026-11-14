#!/bin/bash
# AICK-3062 gfx1151 kit: the whole tuned-table selection on this Strix Halo, end to end.
#   1. sweep    geometry -> runtime, primary general slices, d64 and d128   (~5-6 h)
#   2. calibrate the bench's gate constants from this device's own A/A noise
#   3. time     rocKE shipped plan + 2 finalists (each with an A/A twin) + PyTorch
#               on every dense census cell                                  (~most of a day)
#   4. judge    apply the RDNA entry rule -> entries_gfx1151.json + judge_gfx1151.json
#   5. pack     gfx1151_results.tgz (send this back)
# Resumable: every step skips work already recorded; rerun the same command after a stop.
# Progress: $AICK_WORK/progress.txt.
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
ROCKE=$(cd "$KIT/../dnn-providers/hip-kernel-provider/rocke" && pwd)
export AICK_WORK=${AICK_WORK:-$HOME/aick3062_gfx1151_work}
W=$AICK_WORK; PY=$W/venv/bin/python; ARCH=gfx1151
[ -x "$PY" ] || { echo "run setup.sh first"; exit 1; }
cd "$ROCKE/library"
export PYTHONPATH=$ROCKE/platform/python:$ROCKE/library PYTHONFAULTHANDLER=1
log() { echo "$*" | tee -a "$W/progress.txt"; }

timeout 120 rocminfo 2>/dev/null | grep -qE "Name: +gfx1151" || { log "PROBE_FAIL no gfx1151"; exit 3; }
COMGR=$($PY -c "from benchmarks.common.attention_bwd_bench import _toolchain_record as t; print(t()['comgr'])") \
  || { log "PROBE_FAIL toolchain"; exit 3; }
export EXPECT_COMGR=$COMGR
log "start $(date -Is) host=$(hostname) commit=$(git -C "$KIT" rev-parse --short HEAD) comgr=$COMGR"

# 1. sweep (no gate constants needed: these stages rank on raw medians)
OUT=$W/sweep/$ARCH
COMMON="--arch $ARCH --family general --out-dir $OUT --compile-cache $OUT/compile_cache --cpus $(nproc)
  --gate tests._attention_bwd_perf_gate:gate --expect-flavor llvm22 --expect-comgr $COMGR"
for D in 64 128; do
  for ST in geometry structure schedule codegen swizzle alternate alternate_schedule runtime; do
    t0=$(date +%s)
    $PY -m benchmarks.common.attention_bwd_sweep $COMMON --d $D --stage $ST > "$W/sweep_d${D}_$ST.log" 2>&1
    rc=$?
    log "SWEEP d=$D $ST rc=$rc secs=$(( $(date +%s)-t0 )) $(grep -E '^stage=' "$W/sweep_d${D}_$ST.log" | tail -1 | cut -d' ' -f3-)"
    [ $rc -eq 0 ] || { log "STOP sweep d=$D $ST rc=$rc"; exit 4; }
  done
done

# 2. gate constants from this device (kept in the work dir; private, never commit)
[ -f "$W/gate_constants_$ARCH.json" ] || $PY "$KIT/calibrate_gate_constants.py" $ARCH > "$W/calibrate.log" 2>&1 \
  || { log "STOP calibrate (see calibrate.log)"; exit 5; }
log "CALIBRATED $(date -Is)"

# 3. selection timing
for D in 64 128; do
  t0=$(date +%s); $PY "$KIT/rdna_best_rocke.py" time $ARCH $D > "$W/time_d$D.log" 2>&1; rc=$?
  log "TIME d=$D rc=$rc secs=$(( $(date +%s)-t0 )) cells=$(grep -c '^cell ' "$W/time_d$D.log")"
  [ $rc -eq 0 ] || { log "STOP time d=$D rc=$rc"; exit 6; }
done

# 4. judge (applies the rule, checks entries with gfx1151_tuning.general_entries)
$PY "$KIT/rdna_best_rocke.py" judge $ARCH 2>&1 | grep -v Warning | tee -a "$W/progress.txt"

# 5. results to send back (timing records + entries; NOT the constants)
cd "$W" || exit 7
rocminfo > env.txt 2>&1
tar czf gfx1151_results.tgz --exclude=compile_cache best_rocke "sweep/$ARCH" \
  "entries_$ARCH.json" "judge_$ARCH.json" progress.txt env.txt time_d*.log
log "DONE $(date -Is): send $W/gfx1151_results.tgz"
