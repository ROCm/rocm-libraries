#!/usr/bin/env bash
# Measure a set of descriptor arms against each other. One node, one session, one job.
#
# GENERALISED FROM the two one-off sweeps that produced the gfx942 attention_dense
# comparison. The diff between those two generations was 83 lines and every one of them
# was a corpus, an arm list, an install tree or an expected descriptor count -- the
# structure was already right. This is the same harness with those five things lifted
# into a config, so the next comparison is a config rather than a fork.
#
#   SWEEP_CONFIG=my-sweep.env sweep.sh
#
# The config declares:
#   OUT             where results land
#   CORPORA         space-separated corpus names; graphs live in $CORPUS_DIR/<name>
#   ARMS            space-separated arm labels, in FIXED ORDER (see the sign check)
#   tree_for ARM    -> the install tree for that arm
#   expect_for ARM  -> the attention-descriptor count that tree MUST have
#   MIN_SERVED      the ingestor-served floor below which a phase FAILS
#   ROUNDS          how many times to repeat the whole cycle (3 is the default, and
#                   the reason is below)
#
# ---------------------------------------------------------------------------
# WHY THE METHODOLOGY LOOKS LIKE THIS
#
# Clocks often CANNOT be pinned: `rocm-smi --setperflevel` needs root, which a shared or
# containerised machine usually will not give you, so the performance level stays `auto`
# and clocks sag under sustained load. Check whether you can pin them -- if you can, do,
# and the controls below become cheap insurance rather than load-bearing. Every
# structural choice here exists for the case where you cannot:
#
#   * ONE NODE, ONE SESSION, ONE JOB -- cross-node absolute numbers are meaningless.
#   * A WARMUP PASS, DISCARDED, so phase 1 does not measure a cool device.
#   * SEVERAL ROUNDS, so round 1 vs round N is a direct readout of drift rather than an
#     assumption that there was none.
#   * FIXED ARM ORDER, deliberately not rotated. Drift then penalises whichever arm runs
#     later, so if an effect moves the OPPOSITE way to the confound, position cannot
#     explain it. That sign check is worth more than the symmetry rotation would buy.
#   * A KNOWN-IDENTICAL CONTROL. Shapes whose arms can only pick a byte-identical binary
#     must read 1.000x; whatever they actually read is the noise floor. Build that set
#     from descriptor sha256, NEVER from timing -- an earlier version selected graphs
#     *because* they timed alike, which is circular.
#
# Report geomean-of-ratios and time-weighted (sum baseline / sum arm) SIDE BY SIDE. On
# the run this harness comes from they differed by more than an order of magnitude in
# how much of a win they described, on the same data. Both are true; publishing only
# the geomean implies wall-clock savings the data does not support, because the shapes
# driving it held a tiny fraction of total GPU time. And always split by provenance:
# the corpus carries its source in the graph name for exactly this reason.
# ---------------------------------------------------------------------------
set -uo pipefail

CONFIG="${SWEEP_CONFIG:-}"
[ -n "$CONFIG" ] && [ -f "$CONFIG" ] || {
    echo "FATAL: set SWEEP_CONFIG to a sweep config file" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONFIG"

: "${OUT:?config must set OUT}"
: "${CORPORA:?config must set CORPORA}"
: "${ARMS:?config must set ARMS}"
: "${CORPUS_DIR:?config must set CORPUS_DIR}"
: "${ROUNDS:=3}"
: "${MIN_SERVED:=20}"
: "${WARMUP_ARM:=}"
: "${BENCH:=dnn-benchmark}"
declare -F tree_for   >/dev/null || { echo "FATAL: config must define tree_for()" >&2; exit 2; }
declare -F expect_for >/dev/null || { echo "FATAL: config must define expect_for()" >&2; exit 2; }

mkdir -p "$OUT"
L=$OUT/progress.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }
say "=== sweep start === host=$(hostname) job=${SLURM_JOB_ID:-none} config=$CONFIG"
rocminfo 2>/dev/null | grep -m1 -E 'Name:\s+gfx' >> "$L"

# One sweep per output directory at a time. Two jobs sharing $OUT interleave phases and
# each one's resume logic then sees the other's files as its own completed work.
LOCK=$OUT/.running.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    say "FATAL: $LOCK exists -- job $(cat "$LOCK/job" 2>/dev/null) may be running"
    exit 92
fi
echo "${SLURM_JOB_ID:-local}" > "$LOCK/job"
trap 'rm -rf "$LOCK"' EXIT

command -v "$BENCH" >/dev/null || { say "FATAL: $BENCH not on PATH"; exit 93; }
[ -n "${PROBE_ENV:-}" ] && [ -x "$PROBE_ENV" ] && {
    say "--- environment provenance ---"; "$PROBE_ENV" 2>&1 | tee -a "$L"; }

# --- corpus staging and gates ------------------------------------------------
# rm -rf FIRST: /tmp is per-NODE and survives between jobs, so a plain copy MERGES with
# whatever the last run left behind. One job measured 514 graphs from a corpus of 499
# because the excluded ones were still sitting in the node's /tmp.
for CORPUS in $CORPORA; do
    STAGE=/tmp/sweep-g-$CORPUS
    rm -rf "$STAGE" && mkdir -p "$STAGE" && cp "$CORPUS_DIR/$CORPUS"/*.json "$STAGE/"
    N=$(find "$STAGE" -name '*.json' | wc -l)
    EXPECT_VAR="EXPECT_GRAPHS_${CORPUS}"
    EXPECT_N="${!EXPECT_VAR:-}"
    if [ -n "$EXPECT_N" ] && [ "$N" -ne "$EXPECT_N" ]; then
        say "FATAL: corpus $CORPUS is $N graphs, expected $EXPECT_N"; exit 91
    fi
    # A graph the kernel under test cannot serve falls through to whichever engine
    # will take it -- and one such path raised an HSA memory fault that took the
    # DEVICE down two minutes into warmup, with every later phase then measuring a
    # dead GPU. For attention that class is BACKWARD graphs, marked structurally by
    # their gradient tensors; the filename is not authoritative.
    #
    # EXCLUDE_TENSORS is the marker set, and it is REQUIRED rather than defaulted.
    # Defaulting it to attention's gradient names would make this gate silently
    # no-op for any other op -- the check would run, find nothing, print "0
    # excluded", and look like protection it was not providing. An op with no such
    # class says so explicitly with EXCLUDE_TENSORS=none.
    : "${EXCLUDE_TENSORS:?config must set EXCLUDE_TENSORS (comma-separated tensor \
names whose presence makes a graph unservable and dangerous), or EXCLUDE_TENSORS=none \
if this op has no such class -- defaulting it would make this gate silently no-op}"
    if [ "$EXCLUDE_TENSORS" = "none" ]; then
        say "corpus $CORPUS: $N graphs, exclusion gate DISABLED by config"
    else
        NBAD=$(EXCLUDE_TENSORS="$EXCLUDE_TENSORS" python3 -c "
import json, glob, os
bad = {t.strip().lower() for t in os.environ['EXCLUDE_TENSORS'].split(',') if t.strip()}
n = 0
for f in glob.glob('$STAGE/*.json'):
    try: d = json.load(open(f))
    except Exception: continue
    if {str(t.get('name','')).lower() for t in d.get('tensors',[])} & bad: n += 1
print(n)")
        [ "$NBAD" -eq 0 ] || {
            say "FATAL: $NBAD graphs in $CORPUS carry [$EXCLUDE_TENSORS] -- unservable, and one such class faults the device"
            exit 91; }
        say "corpus $CORPUS: $N graphs, 0 carrying [$EXCLUDE_TENSORS]"
    fi
done

run_arm() {  # corpus arm round
    local CORPUS=$1 ARM=$2 ROUND=$3
    local INSTALL; INSTALL=$(tree_for "$ARM")
    local EXPECT;  EXPECT=$(expect_for "$ARM")
    local TAG=${CORPUS}_${ARM}_r${ROUND}
    local STAGE=/tmp/sweep-g-$CORPUS

    # Resume: a completed phase is skipped, so a re-submit continues rather than restarts.
    if [ -s "$OUT/$TAG.json" ] && python3 -c "
import json,sys
d=json.load(open('$OUT/$TAG.json'))
n=sum(1 for g in d.get('graphs',[]) for r in g.get('results',[])
      if r.get('status')=='success' and (r.get('gpu_kernel_stats') or {}).get('mean_ms'))
sys.exit(0 if n>0 else 1)" 2>/dev/null; then
        say "$TAG SKIP (already complete)"; return 0
    fi

    # Gate 1: descriptor count. NECESSARY, NOT SUFFICIENT -- it is a property of files on
    # disk and says nothing about what the loader accepted. Summed over every bundle in
    # the tree: counting one would pass a tree missing another.
    local N
    N=$(python3 -c "
import json,glob
t=0
for f in glob.glob('$INSTALL/**/*.kdp.json', recursive=True):
    t += len(json.load(open(f))['kernelDescriptors'])
print(t)" 2>/dev/null || echo -1)
    if [ "$N" != "$EXPECT" ]; then
        say "FATAL $TAG: $INSTALL has $N descriptors, expected $EXPECT"; return 90
    fi

    export ROCM_PATH=$INSTALL
    export LD_LIBRARY_PATH=$INSTALL/lib:${LD_LIBRARY_PATH:-}
    # Per-phase cache, cleared before AND after: the winner record is consulted BEFORE
    # the benchmarking flag, so a warm cache replays a previous phase's picks instead of
    # sampling this arm.
    export HIPDNN_CACHE_DIR=$OUT/cache-$TAG
    rm -rf "$HIPDNN_CACHE_DIR"; mkdir -p "$HIPDNN_CACHE_DIR"

    say "$TAG begin ($N descriptors)"
    local T0; T0=$(date +%s)
    HIPDNN_FORCE_BENCHMARKING=1 HIPDNN_LOG_LEVEL=info \
    HIPDNN_LOG_FILE="$OUT/$TAG.hipdnn.log" \
    "$BENCH" --graph "$STAGE/*.json" \
        --plugin-path "$INSTALL/lib/hipdnn_plugins/engines" \
        --warmup 10 --iters 50 -o "$OUT/$TAG.json" > "$OUT/$TAG.log" 2>&1
    local RC=$?
    say "$TAG rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
    [ "$RC" -eq 0 ] || { say "$TAG FAILED"; tail -3 "$OUT/$TAG.log" >> "$L"; return 94; }

    # Gate 2: provenance. Matched ANYWHERE in the log, not head -1 -- hipDNN loads every
    # plugin it can find, and a first-line check once failed a run whose arm plugin had
    # served every graph correctly.
    if grep -q "load plugin from \[$INSTALL/lib/hipdnn_plugins/engines/" "$OUT/$TAG.hipdnn.log" 2>/dev/null; then
        say "  provenance OK ($INSTALL)"
    else
        say "  FATAL: $INSTALL plugin never loaded"; return 96
    fi

    # Gate 3: what the arm actually SERVED. This is the one that catches a dropped
    # engine, and the count gate cannot: a duplicate metadata tuple makes the loader
    # reject the whole engine, the phase then runs to completion, exits 0, passes gates 1
    # and 2 -- and serves every graph from a different engine. That shipped once and cost
    # a full sweep.
    local SERVED
    SERVED=$(python3 -c "
import json
d=json.load(open('$OUT/$TAG.json'))
n={g.get('graph_name') for g in d.get('graphs',[]) for r in g.get('results',[])
   if r.get('status')=='success' and (r.get('engine_name') or '').startswith('engine_')}
print(len(n))" 2>/dev/null || echo 0)
    if [ "${SERVED:-0}" -lt "$MIN_SERVED" ]; then
        say "  FATAL: $ARM served only ${SERVED:-0} graphs through the ingestor (min $MIN_SERVED)"
        say "         engine likely rejected at load -- check for duplicate metadata tuples"
        return 97
    fi
    say "  ingestor-served graphs: $SERVED"
    cp -a "$HIPDNN_CACHE_DIR"/ingestor-winners/*/*/*/winners.jsonl "$OUT/$TAG.winners.jsonl" 2>/dev/null
    rm -rf "$HIPDNN_CACHE_DIR"
}

FAILED=""

# Warmup, discarded, so phase 1 does not get a cool device.
if [ -n "$WARMUP_ARM" ]; then
    for CORPUS in $CORPORA; do
        say "WARMUP $CORPUS (discarded)"
        run_arm "$CORPUS" "$WARMUP_ARM" 0 || true
        rm -f "$OUT/${CORPUS}"_*_r0.*
    done
    say "warmup complete -- device at steady state"
fi

for ROUND in $(seq 1 "$ROUNDS"); do
    say "=== ROUND $ROUND of $ROUNDS ==="
    for CORPUS in $CORPORA; do
        # Fixed arm order, every round. See the sign check in the header.
        for ARM in $ARMS; do
            run_arm "$CORPUS" "$ARM" "$ROUND" || FAILED="$FAILED ${CORPUS}_${ARM}_r$ROUND"
        done
    done
done

# Phase inventory. Counting rc=0 lines is not equivalent: it says N phases succeeded,
# not that they were the RIGHT N, so a run that repeated one arm and skipped another
# still counts correctly. This enumerates the expected grid and names absences.
say "--- phase inventory ---"
for ROUND in $(seq 1 "$ROUNDS"); do
    for CORPUS in $CORPORA; do
        for ARM in $ARMS; do
            F="$OUT/${CORPUS}_${ARM}_r${ROUND}.json"
            if [ -s "$F" ]; then
                say "  ${CORPUS}_${ARM}_r${ROUND}: $(python3 -c "
import json
d=json.load(open('$F'))
g=d.get('graphs',[])
ing=sum(1 for x in g for r in x.get('results',[])
        if r.get('status')=='success' and (r.get('gpu_kernel_stats') or {}).get('mean_ms')
        and str(r.get('engine_name','')).startswith('engine_'))
tot=sum(1 for x in g for r in x.get('results',[])
        if r.get('status')=='success' and (r.get('gpu_kernel_stats') or {}).get('mean_ms'))
print(f'{tot} timings ({ing} ingestor)')" 2>/dev/null || echo unreadable)"
            else
                say "  ${CORPUS}_${ARM}_r${ROUND}: MISSING"
            fi
        done
    done
done

if [ -n "$FAILED" ]; then
    say "SWEEP_INCOMPLETE failed:$FAILED"
    say "Re-submit the same script to resume: completed phases are skipped."
    exit 1
fi

# Correctness is a SEPARATE pass. A reference execution per graph would distort the very
# timings the drift gate checks, and an early sweep shipped with correctness.passed=false
# on every row because --validate was never passed at all.
if [ "${RUN_CORRECTNESS:-1}" = "1" ]; then
    say "--- correctness pass (untimed, --validate pytorch) ---"
    for CORPUS in $CORPORA; do
        for ARM in $ARMS; do
            TAG=validate_${CORPUS}_${ARM}
            [ -s "$OUT/$TAG.json" ] && { say "$TAG SKIP"; continue; }
            INSTALL=$(tree_for "$ARM")
            export ROCM_PATH=$INSTALL LD_LIBRARY_PATH=$INSTALL/lib:${LD_LIBRARY_PATH:-}
            export HIPDNN_CACHE_DIR=$OUT/cache-$TAG
            rm -rf "$HIPDNN_CACHE_DIR"; mkdir -p "$HIPDNN_CACHE_DIR"
            say "$TAG begin"
            HIPDNN_LOG_LEVEL=info "$BENCH" --graph "/tmp/sweep-g-$CORPUS/*.json" \
                --plugin-path "$INSTALL/lib/hipdnn_plugins/engines" \
                --validate pytorch --warmup 1 --iters 3 \
                -o "$OUT/$TAG.json" > "$OUT/$TAG.log" 2>&1
            # A reference-provider row carries correctness.passed=false and no
            # comparison; counting it as a failure produced alarming-looking
            # "pass=120 fail=101" lines in an earlier run's log that meant nothing.
            say "$TAG rc=$? $(python3 -c "
import json
try:
    d=json.load(open('$OUT/$TAG.json'))
    p=f=0
    for g in d.get('graphs',[]):
        for r in g.get('results',[]):
            c=r.get('correctness') or {}
            if c.get('passed') is True: p+=1
            elif r.get('status')=='success' and c.get('tolerance_match') is not None: f+=1
    print(f'correctness pass={p} fail={f}')
except Exception as e: print('unreadable', e)" 2>/dev/null)"
            rm -rf "$HIPDNN_CACHE_DIR"
        done
    done
fi

NPHASE=$(( ROUNDS * $(echo "$CORPORA" | wc -w) * $(echo "$ARMS" | wc -w) ))
say "SWEEP_DONE $NPHASE timed phases ($ROUNDS rounds x corpora[$CORPORA] x arms[$ARMS])"
