#!/bin/bash
# Drive the wrapper microbenchmark across both builds, two modes, and
# cold/warm OS-cache states.
#
# Output columns (TSV, header on first line of out.tsv):
#   build  mode  state  rep  wall_s  cpu_s  per_call_ns  rss_kb  minflt  majflt
set -u

BENCH_DIR=/data/nhanna/repos/rocm-libraries/projects/miopen/perf-results/wrapper-bench
OUT=$BENCH_DIR/out.tsv
RAW=$BENCH_DIR/raw.log
WARM_REPS=10
COLD_REPS=5
GETVER_ITERS=100000000     # 100M -> ~0.5-1s timed loop
CREATE_ITERS=1000          # ~1-2s timed loop
NOOP_ITERS=100000000       # match getver iters to subtract harness cost

REPO=/data/nhanna/repos/rocm-libraries/projects/miopen
FLAGON_LIB=$REPO/build-flagon/lib
FLAGOFF_LIB=$REPO/build-flagoff/lib

echo -e "build\tmode\tstate\trep\twall_s\tcpu_s\tper_call_ns\trss_kb\tminflt\tmajflt" > "$OUT"
: > "$RAW"

drop_caches() {
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
    # Also evict the two MIOpen .so files explicitly in case drop_caches was
    # rate-limited or not effective for some reason.
    for f in "$FLAGON_LIB"/libMIOpen.so.1 \
             "$FLAGON_LIB"/libMIOpen_private.so.1 \
             "$FLAGOFF_LIB"/libMIOpen.so.1; do
        [ -f "$f" ] && sudo -n dd if="$f" iflag=nocache count=0 of=/dev/null 2>/dev/null || true
    done
}

parse_into_tsv() {
    # $1=build $2=mode $3=state $4=rep; reads stdin (one bench output line)
    local build=$1 mode=$2 state=$3 rep=$4
    awk -v b="$build" -v m="$mode" -v s="$state" -v r="$rep" '
        { for (i=1; i<=NF; i++) {
              split($i, kv, "=")
              v[kv[1]] = kv[2]
          }
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                 b, m, s, r,
                 v["wall_s"], v["cpu_s"], v["per_call_ns"],
                 v["max_rss_kb"], v["minflt"], v["majflt"]
        }' >> "$OUT"
}

run_one() {
    local build=$1 mode=$2 state=$3 rep=$4 iters=$5
    local bin=$BENCH_DIR/wrapper_bench_${build}
    local line
    line=$("$bin" "$mode" "$iters")
    echo "[$build $mode $state rep=$rep] $line" >> "$RAW"
    printf '%s\n' "$line" | parse_into_tsv "$build" "$mode" "$state" "$rep"
}

# ---- warm runs ----
echo "=== warm: priming page cache ==="
# Prime by running each binary once and discarding the result.
for b in flagoff flagon; do
    "$BENCH_DIR/wrapper_bench_${b}" noop 1 > /dev/null
done

echo "=== warm: getversion ($WARM_REPS reps each) ==="
for rep in $(seq 1 $WARM_REPS); do
    run_one flagoff getversion warm "$rep" "$GETVER_ITERS"
    run_one flagon  getversion warm "$rep" "$GETVER_ITERS"
done

echo "=== warm: createdestroy ($WARM_REPS reps each) ==="
for rep in $(seq 1 $WARM_REPS); do
    run_one flagoff createdestroy warm "$rep" "$CREATE_ITERS"
    run_one flagon  createdestroy warm "$rep" "$CREATE_ITERS"
done

echo "=== warm: noop harness baseline ($WARM_REPS reps each) ==="
for rep in $(seq 1 $WARM_REPS); do
    run_one flagoff noop warm "$rep" "$NOOP_ITERS"
    run_one flagon  noop warm "$rep" "$NOOP_ITERS"
done

# ---- cold runs ----
# "cold" = drop page cache before each invocation, so the dynamic loader
# has to read the .so files from disk. We use a small iter count so the
# timed loop doesn't dominate the wall time -- we want the *process startup*
# cost to dominate.
echo "=== cold: getversion 1-iter, drop_caches before each ==="
for rep in $(seq 1 $COLD_REPS); do
    drop_caches
    run_one flagoff getversion cold "$rep" 1
    drop_caches
    run_one flagon  getversion cold "$rep" 1
done

echo "=== cold: createdestroy 1-iter, drop_caches before each ==="
for rep in $(seq 1 $COLD_REPS); do
    drop_caches
    run_one flagoff createdestroy cold "$rep" 1
    drop_caches
    run_one flagon  createdestroy cold "$rep" 1
done

echo "done -> $OUT"
