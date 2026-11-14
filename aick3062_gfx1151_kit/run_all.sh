#!/bin/bash
# AICK-3062 gfx1151 kit: the whole tuned-table selection on this Strix Halo, end to end.
#   1. sweep    geometry -> runtime, primary general slices, d64 and d128   (~5-6 h)
#   2. calibrate the bench's gate constants from this device's own A/A noise
#   3. time     rocKE shipped plan + 2 finalists (each with an A/A twin) + PyTorch
#               on every dense census cell                                  (~most of a day)
#   4. judge    apply the RDNA entry rule -> entries_gfx1151.json + judge_gfx1151.json
#   5. pack     gfx1151_results.tgz (send this back)
# Resumable: every step skips work already recorded; rerun the same command after a stop.
# Crash-safe: half-written JSONL tails are repaired on start; results are snapshotted to
# $AICK_WORK/checkpoints every 20 min and after every step (+ $CHECKPOINT_DIR if set).
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

# --- crash safety -------------------------------------------------------------------
# A hard stop can leave a half-written last JSONL line; the next append would glue onto
# it and both records would be lost. Cut every such tail back to its last newline.
repair_jsonl() {
  "$PY" - "$W" <<'PYREPAIR'
import os, sys
from pathlib import Path
fixed = 0
for p in Path(sys.argv[1]).rglob("*.jsonl"):
    if "venv" in p.parts or "compile_cache" in p.parts:
        continue
    b = p.read_bytes()
    if b and not b.endswith(b"\n"):
        cut = b.rfind(b"\n") + 1
        with open(p, "r+b") as fh:
            fh.truncate(cut)
            fh.flush(); os.fsync(fh.fileno())
        fixed += 1
print(f"repair_jsonl: {fixed} file(s) had a partial last line")
PYREPAIR
}
# Snapshot every result (not the venv / compile cache) into $W/checkpoints, keep the
# last 6; also copy to $CHECKPOINT_DIR when set (another disk or a network share).
checkpoint() {
  mkdir -p "$W/checkpoints"
  local f="$W/checkpoints/ckpt-$(date +%Y%m%d-%H%M%S).tgz"
  ( cd "$W" && tar czf "$f.part" --exclude=venv --exclude=compile_cache --exclude=checkpoints \
      --exclude='*.tgz' --exclude='*.part' . 2>/dev/null ) && mv "$f.part" "$f" && sync
  ls -1t "$W"/checkpoints/ckpt-*.tgz 2>/dev/null | tail -n +7 | xargs -r rm -f
  [ -n "${CHECKPOINT_DIR:-}" ] && mkdir -p "$CHECKPOINT_DIR" && cp -f "$f" "$CHECKPOINT_DIR/" 2>/dev/null
  echo "checkpoint $(basename "$f") $(date -Is)" >> "$W/checkpoints/index.txt"
}
checkpoint_loop() { while sleep "${CHECKPOINT_EVERY_S:-1200}"; do checkpoint; done; }

timeout 120 rocminfo 2>/dev/null | grep -qE "Name: +gfx1151" || { log "PROBE_FAIL no gfx1151"; exit 3; }
COMGR=$($PY -c "from benchmarks.common.attention_bwd_bench import _toolchain_record as t; print(t()['comgr'])") \
  || { log "PROBE_FAIL toolchain"; exit 3; }
export EXPECT_COMGR=$COMGR
log "start $(date -Is) host=$(hostname) commit=$(git -C "$KIT" rev-parse --short HEAD) comgr=$COMGR"
repair_jsonl | tee -a "$W/progress.txt"
checkpoint_loop & CKPT_PID=$!
trap 'kill $CKPT_PID 2>/dev/null; checkpoint' EXIT

# 1. sweep (no gate constants needed: these stages rank on raw medians)
OUT=$W/sweep/$ARCH
COMMON="--arch $ARCH --family general --out-dir $OUT --compile-cache $OUT/compile_cache --cpus $(nproc)
  --gate tests._attention_bwd_perf_gate:gate --expect-flavor llvm22 --expect-comgr $COMGR"
for D in 64 128; do
  for ST in geometry structure schedule codegen swizzle alternate alternate_schedule runtime; do
    t0=$(date +%s)
    $PY -m benchmarks.common.attention_bwd_sweep $COMMON --d $D --stage $ST >> "$W/sweep_d${D}_$ST.log" 2>&1
    rc=$?
    log "SWEEP d=$D $ST rc=$rc secs=$(( $(date +%s)-t0 )) $(grep -E '^stage=' "$W/sweep_d${D}_$ST.log" | tail -1 | cut -d' ' -f3-)"
    [ $rc -eq 0 ] || { log "STOP sweep d=$D $ST rc=$rc"; exit 4; }
    checkpoint
  done
done

# 2. gate constants from this device (kept in the work dir; private, never commit)
[ -f "$W/gate_constants_$ARCH.json" ] || $PY "$KIT/calibrate_gate_constants.py" $ARCH > "$W/calibrate.log" 2>&1 \
  || { log "STOP calibrate (see calibrate.log)"; exit 5; }
log "CALIBRATED $(date -Is)"; checkpoint

# 3. selection timing
for D in 64 128; do
  t0=$(date +%s); $PY "$KIT/rdna_best_rocke.py" time $ARCH $D >> "$W/time_d$D.log" 2>&1; rc=$?
  log "TIME d=$D rc=$rc secs=$(( $(date +%s)-t0 )) cells=$(grep -c '^cell ' "$W/time_d$D.log")"
  [ $rc -eq 0 ] || { log "STOP time d=$D rc=$rc"; exit 6; }
  checkpoint
done

# 4. judge (applies the rule, checks entries with gfx1151_tuning.general_entries)
$PY "$KIT/rdna_best_rocke.py" judge $ARCH 2>&1 | grep -v Warning | tee -a "$W/progress.txt"

# 5. results to send back (timing records + entries; NOT the constants)
cd "$W" || exit 7
rocminfo > env.txt 2>&1
tar czf gfx1151_results.tgz --exclude=compile_cache best_rocke "sweep/$ARCH" \
  "entries_$ARCH.json" "judge_$ARCH.json" progress.txt env.txt time_d*.log
log "DONE $(date -Is): send $W/gfx1151_results.tgz"
