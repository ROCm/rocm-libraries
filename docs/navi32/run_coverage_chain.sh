#!/bin/bash
# Phase 2-3 of the coverage campaign: wait for the full-grid cold sweep, then re-map over the
# completed matrix, build the arm, and benchmark it TWICE.
#
# WAITING IS STRUCTURAL, NOT pgrep. A previous script in this campaign deadlocked forever
# because it was created by a heredoc and launched in the same shell command, so the launcher's
# own command line contained the string the script was pgrep-waiting on. The rule that came out
# of it is broader than "pgrep matches itself": pgrep -f on ANY string that appears in your
# tooling is unsafe. So this waits on the matrix row count, which is a fact about the work.
set -u
cd /home/vmijovic/navi32
log(){ echo "[$(date +%F\ %H:%M:%S)] $*"; }

MATRIX=results/P1_cold_matrix.jsonl
TARGET=9680
# RE-MAP ON TOP OF THE SHIPPED CATALOG, NOT ON TOP OF LEAN.
# This was a real bug, caught by predicting where the eval queries would change kernel before
# spending 20h measuring: with --logic lean, a held-out row keeps the LEAN pick, but the
# comparison baseline is SHIPPED -- which is itself a re-map of lean. So on held-out rows the
# "control" arm was silently REVERTING shipped's re-map rather than holding it constant. The
# power analysis showed it plainly: 43 of 44 changed queries were in the control group and only
# 1 in the treated group, i.e. exactly inverted. Re-mapping on top of shipped makes an untouched
# row byte-identical to shipped, which is what a negative control has to mean.
BASE=arms/hhs_remap_gated/x.yaml
SRC=src/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/navi31/GridBased/navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml

log "waiting for sweep: need $TARGET rows in $MATRIX"
STALL=0; LAST=0
while :; do
  N=$(wc -l < $MATRIX)
  [ "$N" -ge "$TARGET" ] && { log "sweep complete: $N rows"; break; }
  if [ "$N" -eq "$LAST" ]; then
    STALL=$((STALL+1))
    # 30 min with no new row means the sweep died or is wedged; proceed with what exists
    # rather than block forever. Some rows legitimately fail to measure.
    [ "$STALL" -ge 30 ] && { log "STALLED at $N rows for 30m -- proceeding with partial matrix"; break; }
  else
    STALL=0; LAST=$N
  fi
  sleep 60
done
log "matrix has $(wc -l < $MATRIX) measured rows"

# ---- Phase 2: re-map over the full matrix -------------------------------------------------
# Same operation and same guards as the shipped re-map, only with more measurements behind it:
#   --skip-strata tiny,gemv : ungated these measured 64.8% and 94.9% (up to a 35% regression),
#                             invisible in a time-weighted mean because they are too short to
#                             move it. Their time sits at the ~25us dispatch floor, so the
#                             "best kernel" at a key is drawn from near-noise.
#   --min-gain 0.02         : the cold noise floor is ~0.7% median / ~2% p90, so repointing for
#                             a 0.5% difference is churn, not improvement.
# TWO arms, because the arm that can be MEASURED honestly is not the arm worth SHIPPING:
#   hhs_remap_ext      -- holds out 49% of rows. Those rows keep the SHIPPED pick, so queries
#                         served by them run a byte-identical catalog and form a clean,
#                         stratum-balanced negative control. This is what makes any gain
#                         attributable rather than merely observed.
#   hhs_remap_extship  -- no holdout: every measured row re-mapped. This is the ship candidate.
#                         Shipping the holdout arm would leave half the grid un-re-mapped, and
#                         validating only the ship arm would leave no control at all -- with
#                         full coverage there are no unmeasured rows left, so its control group
#                         would collapse to tiny/gemv and be confounded with stratum.
log "re-mapping over the matrix (holdout arm + ship arm)"
mkdir -p arms/hhs_remap_ext arms/hhs_remap_extship arms/hhs_remap_nogate logs
python3 remap_grid.py --logic $BASE --matrix $MATRIX --src-pool "$SRC" \
  --holdout state/holdout_fullcov.json --skip-strata tiny,gemv --min-gain 0.02 \
  --out arms/hhs_remap_ext/x.yaml 2>&1 | sed 's/^/    [ext] /'
python3 remap_grid.py --logic $BASE --matrix $MATRIX --src-pool "$SRC" \
  --skip-strata tiny,gemv --min-gain 0.02 \
  --out arms/hhs_remap_extship/x.yaml 2>&1 | sed 's/^/    [ship] /'
# UNGATED variant: the tiny/gemv gate is now a live question, not a settled one.
#   - it was adopted from the `argmax` catalog measuring tiny at 64.80%, but the `full` catalog
#     measures tiny at 99.79% -- and those are different catalogs, so the gate was justified
#     against something other than what is now on the table;
#   - measured offline from the matrix, tiny/gemv have the BEST kernel-choice transfer of any
#     stratum: a row's winner delivers 99.5% / 99.4% of achievable at the neighbouring row below,
#     vs med's 95.2%.
# CORRECTION: I first wrote that this is "because they are dispatch-bound and every kernel is
# nearly equivalent there". That is FALSE and measuring it refuted it -- on tiny/gemv rows the
# median kernel reaches only ~73% of the best, p10 49.5%, worst 27% (a 3.7x spread), which is
# comparable to every other stratum. The real result is stronger than the one I assumed: the
# winner is STABLE ACROSS NEIGHBOURING SHAPES despite kernel choice mattering a great deal.
# Both readings still point the same way, so the gate gets tested rather than assumed.
python3 remap_grid.py --logic $BASE --matrix $MATRIX --src-pool "$SRC" \
  --min-gain 0.02 \
  --out arms/hhs_remap_nogate/x.yaml 2>&1 | sed 's/^/    [nogate] /'

for A in hhs_remap_ext hhs_remap_extship hhs_remap_nogate; do
  # exactly one YAML per arm dir: TensileCreateLibrary RECURSES, and a stale second file once
  # made a gate report "298 kernels, 0 errors" while silently merging three catalogs (471).
  NY=$(ls arms/$A/*.yaml | wc -l)
  [ "$NY" -eq 1 ] || { log "FATAL: $NY yaml files in arms/$A, expected 1"; exit 1; }
  log "building arm $A"
  timeout 7200 ./build_arm.sh $A > logs/build_$A.log 2>&1
  CO=$(ls libs/$A/library/gfx1100/*.co 2>/dev/null | head -1)
  K=$(llvm-readelf --notes "$CO" 2>/dev/null | grep -c '\.symbol:')
  E=$(grep -ciE 'overflowedResources|not a valid operand' logs/build_$A.log)
  log "  $A: $K kernels, $E assembler/resource errors"
  [ "$E" -eq 0 ] || { log "FATAL: build errors in $A"; exit 1; }
  [ "$K" -gt 0 ] || { log "FATAL: no kernels in built library $A"; exit 1; }
done

# ---- Phase 3: benchmark, twice -------------------------------------------------------------
# Cold, because the re-map was DERIVED from cold measurements; validating warm would decide in
# a different regime than it ships in (warm vs cold is 3.1x on small shapes).
# Iterations are time-derived: a fixed count puts the small end of this eval set back under the
# 8-10% cold noise floor. shipped_aa is the same library as shipped -- it measures the floor
# rather than assuming it.
#
# `full` is here because the right ship question is "does the extension beat the BEST existing
# candidate", not "does it beat what happens to be shipped". Re-reading the finished P4 run
# (207 shapes, 4 arms, A/A floor 100.03%) showed the ungated `full` arm at 103.98% vs lean
# against the shipped gated arm's 101.00% -- with NO tiny/gemv regression (tiny 99.79%, gemv
# 101.96%). That is one run and so not yet a conclusion, but comparing the extension only
# against `shipped` would have quietly assumed the gating decision was still correct.
#
# `lean` is carried purely as a CROSS-EXPERIMENT REFERENCE. It is the common baseline of the
# earlier P4 run, so shipped/full reproducing their P4 values (101.00% / 103.98%) on a brand new
# eval set is a consistency check on the whole measurement chain -- and if they do NOT reproduce,
# that is a finding about the harness rather than about the catalogs.
#
# --timeout 45: a legitimate run is well under 10s here (time-derived iterations cap accumulated
# kernel time at ~13ms; the rest is library init and the 512MB rotating buffer). The masked
# stream hangs ~2.7% of runs at 60 CU, so with 3600 runs per pass the timeout value is a real
# wall-clock term: 120s would have risked hours of pure waiting for nothing.
for RUN in 1 2; do
  log "benchmark run $RUN (7 arms x 600 shapes, cold, 60 CU, time-derived iters)"
  timeout 40000 python3 bench_arms.py \
    --arms shipped=$HOME/navi32/libs/hhs_remap_gated/library/gfx1100 \
           extended=$HOME/navi32/libs/hhs_remap_ext/library/gfx1100 \
           extended_ship=$HOME/navi32/libs/hhs_remap_extship/library/gfx1100 \
           nogate=$HOME/navi32/libs/hhs_remap_nogate/library/gfx1100 \
           full=$HOME/navi32/libs/hhs_remap_full/library/gfx1100 \
           lean=$HOME/navi32/libs/hhs_lean100/library/gfx1100 \
           shipped_aa=$HOME/navi32/libs/hhs_remap_gated/library/gfx1100 \
    --shapes state/eval_fullcov.json --out results/P6_cov_run$RUN.csv \
    --reps 1 --cus 60 --target-us 10000 --timeout 45 \
    --extra-args "--flush --rotating 512" >> logs/P6_cov_run$RUN.log 2>&1
  log "  run $RUN done: $(wc -l < results/P6_cov_run$RUN.csv) rows"
done

# ---- Phase 3b: the ORACLE on the eval queries ----------------------------------------------
# Benchmarking tells me which catalog is faster. It cannot tell me how much was AVAILABLE.
# `--algo_method all` measures every solution on a shape in one call, so sweeping the 600 eval
# queries yields the per-query ceiling -- and with it, what fraction of the achievable gain each
# catalog actually captures, plus the ability to price any future catalog on this eval set with
# no further GPU time. Deliberately LAST: ground truth first, ceiling afterwards.
log "oracle sweep of the 600 eval queries (--algo_method all)"
timeout 20000 python3 matrix_sweep.py --shapes state/eval_fullcov_flat.json \
  --out results/P7_eval_oracle.jsonl --cus 60 --cold --target-us 10000 --timeout 120 \
  >> logs/P7_oracle.log 2>&1
log "  oracle rows: $(wc -l < results/P7_eval_oracle.jsonl 2>/dev/null || echo 0)"

log "CHAIN COMPLETE"
