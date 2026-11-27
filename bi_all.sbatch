#!/bin/bash
#SBATCH -p defq
#SBATCH -c 64
#SBATCH -t 06:00:00
#SBATCH -J pr5_bi
#SBATCH -o /home/AMD/japiasec/workspace/gfx950_prs/pr5/bi_p5/slurm_%j.out
# PR 4 P2: byte identity at llvm20/22/23/24 (fresh build root per flavor, two
# flavors at a time on 32 cores each) + host ctest of a full build, on the
# committed tree snapshot in $S (git archive; /data is not mounted here).
set -u
O=/home/AMD/japiasec/workspace/gfx950_prs/pr5/bi_p5
S=$O/src
R=$S/dnn-providers/hip-kernel-provider/rocke
W=/tmp/pr5_bi_$SLURM_JOB_ID
mkdir -p $W
export PATH=/home/AMD/japiasec/workspace/gfx950_prs/p2tools/bin:/usr/bin:/bin CMAKE_GENERATOR=Ninja
echo "node $(hostname) start $(date)" > $O/summary.txt
run_bi() {
  fl=$1
  ( cd $R/platform && export ROCKE=$(pwd) PYTHONPATH=$R/platform/python:$R/library \
      ROCKE_LLVM_FLAVOR=$fl TMPDIR=$W/tmp_$fl CMAKE_BUILD_PARALLEL_LEVEL=32; mkdir -p $TMPDIR
    rm -rf $W/build_$fl
    taskset -c $2 timeout 10800 python3 tools/check_byte_identity.py --build-root $W/build_$fl > $O/bi_$fl.log 2>&1
    echo "$fl rc=$? $(grep RESULT $O/bi_$fl.log)" >> $O/summary.txt )
}
A=$(python3 -c "import os;c=sorted(os.sched_getaffinity(0));h=len(c)//2;print(','.join(map(str,c[:h])))")
B=$(python3 -c "import os;c=sorted(os.sched_getaffinity(0));h=len(c)//2;print(','.join(map(str,c[h:])))")
run_bi llvm20 $A & run_bi llvm22 $B & wait
run_bi llvm23 $A & run_bi llvm24 $B & wait
# host ctest of a full build (BUILD_TESTING on), flavor pinned (no COMGR here)
( export ROCKE_LLVM_FLAVOR=llvm22
  cmake -S $R/platform -B $W/full -G Ninja -DCMAKE_BUILD_TYPE=Release > $O/full_cfg.log 2>&1 &&
  cmake --build $W/full -j 64 > $O/full_build.log 2>&1
  echo "full build rc=$?" >> $O/summary.txt
  cd $W/full && ctest -L host -j 32 --timeout 900 > $O/ctest_host.log 2>&1
  echo "ctest host rc=$? $(grep -E 'tests passed' $O/ctest_host.log)" >> $O/summary.txt
  grep -E '\(Failed\)|\(Timeout\)|Not Run|\*\*\*Exception' $O/ctest_host.log >> $O/summary.txt
  cd $W/full && ctest -R 'conv_implicit' --output-on-failure > $O/ctest_new.log 2>&1
  echo "ctest new rc=$? $(grep -E 'tests passed' $O/ctest_new.log)" >> $O/summary.txt )
rm -rf $W
echo "end $(date)" >> $O/summary.txt
