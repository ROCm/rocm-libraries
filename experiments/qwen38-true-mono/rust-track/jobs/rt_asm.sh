#!/bin/bash
# RT asm-kernel job (q38_job.sh --mode venv; stages rocke-rt-<v>, rt-bin-<v>, rt-probes-<v>): the kernels that carry
# general inline asm (probes P4 / P5, the CCL collectives), built by COMGR and by the direct compiler (strict: a
# refusal is an error), through their own suites: P4 / P5 numerics + negative controls + mutants (short timing),
# the CCL multi-process GPU test suite on RT_NGPU GPUs.  Env: RT_ROCKE, RT_BIN, RT_PROBES, RT_PARTS (p4 p5 ccl).
set -uo pipefail
RK=$Q38_STAGES/${RT_ROCKE:?}
BIN=$Q38_STAGES/${RT_BIN:?}
PB=$Q38_STAGES/${RT_PROBES:?}
O=$Q38_OUT
TL=$(python -c 'import importlib.util,os; print(os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin),"lib"))')
export ROCKE_HIP_LIB=$TL/libamdhip64.so ROCKE_LLVM_FLAVOR=${ROCKE_LLVM_FLAVOR:-llvm22} ROCKE_BACKEND=python
export ROCKE_CODEOBJ_DIRECT_LIB=$BIN/so/librocke_codeobj_ffi.so
export AMD_COMGR_CACHE_DIR=$Q38_JOB/comgr-cache TMPDIR=$Q38_JOB/tmp
mkdir -p $TMPDIR $AMD_COMGR_CACHE_DIR
export PYTHONPATH=$RK/platform/python:$RK/library:$PB/P4:$PB/P5
echo "== node $(hostname) direct lib $(sha256sum $ROCKE_CODEOBJ_DIRECT_LIB | cut -c1-16) $(date -u +%T)"
rc=0
for part in ${RT_PARTS:-p4 p5 ccl}; do
  for be in ${RT_BACKENDS:-comgr direct}; do
    d=$O/${part}_$be
    mkdir -p $d
    strict=0
    [ $be = direct ] && strict=1
    if [ $be = direct ] && [ -n "${RT_DIRECT_ENV:-}" ]; then
      for kv in ${RT_DIRECT_ENV//,/ }; do export "$kv"; done
    fi
    echo "=== $part $be $(date -u +%T)"
    case $part in
      p4) ROCKE_CODEOBJ_BACKEND=$be ROCKE_DIRECT_STRICT=$strict timeout -s KILL 1800 python $PB/P4/p4_bench.py --out $d \
            --rounds-a 3 --rounds-b 3 --rounds-c 3 --phases-a 20 > $d/log.txt 2>&1 ;;
      p5) ROCKE_CODEOBJ_BACKEND=$be ROCKE_DIRECT_STRICT=$strict timeout -s KILL 1800 python $PB/P5/p5_bench.py --out $d \
            --rounds 3 --jobs 4 > $d/log.txt 2>&1 ;;
      ccl) ROCKE_CODEOBJ_BACKEND=$be ROCKE_DIRECT_STRICT=$strict timeout -s KILL ${RT_CCL_TIMEOUT:-2400} python -m torch.distributed.run --standalone \
            --nproc_per_node ${RT_NGPU:-2} -m rocke.examples.gfx950.ccl.bench_ccl test --out $d ${RT_CCL_ONLY:+--only $RT_CCL_ONLY} > $d/log.txt 2>&1 ;;
    esac
    prc=$?
    echo "   rc=$prc"
    grep -aE "PASS|FAIL|ok |Traceback|Error|error|refus|DIRECT" $d/log.txt | tail -25 | cut -c1-220
    [ $prc = 0 ] || rc=1
  done
done
echo "== RTASM rc=$rc $(date -u +%T)"
exit $rc
