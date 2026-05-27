#!/bin/bash
# v2 runner: drops --gtest_brief=1 (so per-test lines exist), but parses the
# summary lines anyway so counts are robust either way.
set -u
BINDIR=/data/nhanna/repos/rocm-libraries/projects/miopen/build-flagon/bin
OUTDIR=/data/nhanna/repos/rocm-libraries/projects/miopen/perf-results/gtest-flagon
LIST=${1:-/tmp/remaining.txt}
SUMMARY=$OUTDIR/_summary_v2.tsv
PROGRESS=$OUTDIR/_progress_v2.log
[ -f "$SUMMARY" ] || printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" name exit_code duration_s ran pass fail skip > "$SUMMARY"
echo "started $(date)" >> "$PROGRESS"

total=$(wc -l < "$LIST")
i=0
while read -r t; do
  i=$((i+1))
  log="$OUTDIR/${t}.log"
  start=$(date +%s)
  timeout 1200 "$BINDIR/$t" > "$log" 2>&1
  ec=$?
  end=$(date +%s); dur=$((end-start))
  ran=$(grep -oE '\[==========\] [0-9]+ tests? from' "$log" | head -1 | grep -oE '[0-9]+')
  pass=$(grep -oE '\[  PASSED  \] [0-9]+ tests?' "$log" | head -1 | grep -oE '[0-9]+')
  skip=$(grep -oE '\[  SKIPPED \] [0-9]+ tests?' "$log" | head -1 | grep -oE '[0-9]+')
  fail=$(grep -oE '\[  FAILED  \] [0-9]+ tests?' "$log" | head -1 | grep -oE '[0-9]+')
  ran=${ran:-0}; pass=${pass:-0}; skip=${skip:-0}; fail=${fail:-0}
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$t" "$ec" "$dur" "$ran" "$pass" "$fail" "$skip" >> "$SUMMARY"
  printf "[%d/%d] %s ec=%d dur=%ds ran=%d pass=%d fail=%d skip=%d\n" \
    "$i" "$total" "$t" "$ec" "$dur" "$ran" "$pass" "$fail" "$skip" >> "$PROGRESS"
done < "$LIST"
echo "finished $(date)" >> "$PROGRESS"
