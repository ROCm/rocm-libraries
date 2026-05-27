#!/bin/bash
BINDIR=/data/nhanna/repos/rocm-libraries/projects/miopen/build-flagon/bin
OUTDIR=/data/nhanna/repos/rocm-libraries/projects/miopen/perf-results/gtest-flagon
SUMMARY=$OUTDIR/_summary.tsv
PROGRESS=$OUTDIR/_progress.log
printf "%s\t%s\t%s\t%s\t%s\t%s\n" name exit_code duration_s pass fail skip > $SUMMARY
echo "started $(date)" > $PROGRESS

total=$(wc -l < /tmp/gtest_bins.txt)
i=0
while read t; do
  i=$((i+1))
  log=$OUTDIR/${t}.log
  start=$(date +%s)
  timeout 600 $BINDIR/$t --gtest_brief=1 > $log 2>&1
  ec=$?
  end=$(date +%s); dur=$((end-start))
  pass=$(grep -cE "^\[       OK \]" $log)
  fail=$(grep -cE "^\[  FAILED  \] " $log)
  skip=$(grep -cE "^\[  SKIPPED \]" $log)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$t" "$ec" "$dur" "$pass" "$fail" "$skip" >> $SUMMARY
  printf "[%d/%d] %s ec=%d dur=%ds pass=%d fail=%d skip=%d\n" "$i" "$total" "$t" "$ec" "$dur" "$pass" "$fail" "$skip" >> $PROGRESS
done < /tmp/gtest_bins.txt
echo "finished $(date)" >> $PROGRESS
