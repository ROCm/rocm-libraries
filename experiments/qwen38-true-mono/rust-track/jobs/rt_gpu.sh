#!/bin/bash
# RT GPU job (rocke node, q38_job.sh --mode container, stages: atom-rkrepro, rocke-rt-<v>, rt-bin-<v>, rt-h-<v>):
# numerics then paired interleaved timing of direct vs COMGR code objects of the true-mono stage kernels.
#   RT_STEPS (space-separated, run in order), each one of:
#     b3:<variant>:<which>[:<arms>]   rt_b3.py (arms: comma list, D and NAME=LEVERS; '+'-joined levers)
#     fmoe:<rows>:<kind>[:<arms>]     rt_fmoe.py (the fused MoE suffix after a FlyDSL a1..m2 prefix)
#     fst:<rows>:<kind>[:<arms>]      rt_fmoe.py --stamps RT_STAMP_ITERS (phase-stamped builds, per-phase ratios vs C)
#     r2                               the R2 head / toy harness (r2_gpu.py) with this .so (regression + levers)
#     head:<epi>:<rows>[:<arms>]      rt_head.py (lm_head GEMV: epi fused|logits, rows 4|16)
#     q2:<variant>[:<arms>]            rt_q2.py (Q2 in the framework layer kernel)
#     probe:<p4|p5>:<kind>[:<arms>]    rt_probe_ab.py (P4 / P5 probe kernels; stage rt-probes-<v>, RT_PROBES)
#     tune:<family>                    rt_tune.py (the lever lookback; families b3 fmoe head q2 probe)
#     mut:<name>                       an rt_mutants.py mutant (gpu: RT_MUT stage's mut_<name>.so; py: patched tree copy;
#                                      a target's 4th field = levers of the arm that must fail)
#   RT_ROUNDS (default 24), RT_NOTIME=1 numerics only, Q38_ROCKE (set by q38_job.sh) the tree stage.
set -uo pipefail
H=$Q38_STAGES/${RT_H:?}
BIN=$Q38_STAGES/${RT_BIN:?}
O=$Q38_OUT
source $H/common/caches.sh
export ROCKE_BACKEND=python ROCKE_CODEOBJ_DIRECT_LIB=$BIN/so/librocke_codeobj_ffi.so PYTHONDONTWRITEBYTECODE=1
unset ROCKE_CODEOBJ_BACKEND ROCKE_DIRECT_LEVERS
EXTRA=$Q38_JOB/pysite
python -c 'import msgpack' 2> /dev/null || python -m pip install -q --target "$EXTRA" msgpack > /dev/null 2>&1
export PYTHONPATH=$H/rust-track:$H/harness:$H/gdn-b3${PYTHONPATH:+:$PYTHONPATH}:$EXTRA
LLVM=/work/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin
echo "== node $(hostname) direct lib $(sha256sum "$ROCKE_CODEOBJ_DIRECT_LIB" | cut -c1-16) $(date -u +%T)"
echo "== neighbours on $(hostname -s): $(squeue -h -w "$(hostname -s)" -o '%u:%j:%b' 2> /dev/null | tr '\n' ' ' | cut -c1-400)"
(while true; do amd-smi metric -g 0 --clock --power --json 2> /dev/null | tr -d '\n'; echo; sleep 2; done) > "$O/amdsmi.jsonl" &
smi=$!
rc=0
cd /atom || exit 126
# Preflight: a wedged GPU (a hung kernel left from elsewhere) shows as a timeout here, not as a hung harness step.
timeout -s KILL 120 python -c 'import torch; x = torch.ones(1 << 20, device="cuda"); torch.cuda.synchronize(); print("== preflight ok", float(x.sum()))'   || { echo "== preflight FAILED (GPU not responding)"; exit 3; }
for step in ${RT_STEPS:?}; do
  IFS=: read -r kind a b c <<< "$step"
  echo "=== $step $(date -u +%T)"
  case $kind in
    b3)
      d=$O/b3_${a}_${b}
      extra=(); [ "${RT_NOTIME:-0}" = 1 ] && extra=(--no-time)
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-track/rt_b3.py --variant "$a" --which "$b" --arms "${c:-D}" \
        --rounds "${RT_ROUNDS:-24}" --out "$d" "${extra[@]}" 2>&1 | grep -aE '^(CHECK|TIME|NUMERICS|DIRECT|ALIAS|DROPPED|STAMPS)|Traceback|Error|error' | cut -c1-600
      [ ${PIPESTATUS[0]} = 0 ] || rc=1
      for h in "$d"/*.hsaco; do [ -f "$h" ] && "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "${h%.hsaco}.s" 2> /dev/null; done ;;
    fmoe | fst)
      d=$O/${kind}_r${a}_${b}
      extra=(); [ "${RT_NOTIME:-0}" = 1 ] && extra=(--no-time)
      [ "$kind" = fst ] && extra+=(--stamps "${RT_STAMP_ITERS:-48}")
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-track/rt_fmoe.py --rows "$a" --kind "$b" --arms "${c:-D}"         --rounds "${RT_ROUNDS:-20}" --out "$d" "${extra[@]}" 2>&1 | grep -aE '^(CHECK|TIME|NUMERICS|DIRECT|ALIAS|DROPPED|STAMPS)|Traceback|Error|error' | cut -c1-900
      [ ${PIPESTATUS[0]} = 0 ] || rc=1
      for h in "$d"/*.hsaco; do [ -f "$h" ] && "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "${h%.hsaco}.s" 2> /dev/null; done ;;
    head)
      d=$O/head_${a}_r${b}
      extra=(); [ "${RT_NOTIME:-0}" = 1 ] && extra=(--no-time)
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-track/rt_head.py --epi "$a" --rows "$b" --arms "${c:-D}"         --rounds "${RT_ROUNDS:-24}" --out "$d" "${extra[@]}" 2>&1 | grep -aE '^(CHECK|TIME|NUMERICS|DIRECT|ALIAS|DROPPED|STAMPS)|Traceback|Error|error' | cut -c1-600
      [ ${PIPESTATUS[0]} = 0 ] || rc=1
      for h in "$d"/*.hsaco; do [ -f "$h" ] && "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "${h%.hsaco}.s" 2> /dev/null; done ;;
    q2)
      # q2:<variant>[:<arms>]: rt_q2.py (the Q2 stage in the framework layer kernel after the FlyDSL a1..q1 prefix)
      d=$O/q2_${a}
      extra=(); [ "${RT_NOTIME:-0}" = 1 ] && extra=(--no-time)
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-track/rt_q2.py --variant "$a" --arms "${b:-D}" \
        --rounds "${RT_ROUNDS:-20}" --out "$d" "${extra[@]}" 2>&1 | grep -aE '^(CHECK|TIME|NUMERICS|DIRECT|ALIAS|DROPPED)|Traceback|Error|error' | cut -c1-700
      [ ${PIPESTATUS[0]} = 0 ] || rc=1
      for h in "$d"/*.hsaco; do [ -f "$h" ] && "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "${h%.hsaco}.s" 2> /dev/null; done ;;
    probe)
      # probe:<p4|p5>:<kind>[:<arms>]: rt_probe_ab.py (the probe bench's own cells; needs the rt-probes stage: RT_PROBES)
      d=$O/probe_${a}_${b}
      extra=(); [ "${RT_NOTIME:-0}" = 1 ] && extra=(--no-time)
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-track/rt_probe_ab.py --probe "$a" --kind "$b" --arms "${c:-D}" \
        --bench-dir "$Q38_STAGES/${RT_PROBES:?}/${a^^}" --rounds "${RT_ROUNDS:-24}" --out "$d" "${extra[@]}" 2>&1 \
        | grep -aE '^(CHECK|TIME|NUMERICS|DIRECT|ALIAS|DROPPED)|Traceback|Error|error' | cut -c1-500
      [ ${PIPESTATUS[0]} = 0 ] || rc=1 ;;
    tune)
      # tune:<family> (b3:<variant>:<which> | fmoe:<rows> | head:<epi>:<rows>): rt_tune.py, the combination lookback
      # over every direct lever (RT_TUNE_START the start set, RT_TUNE_POOL the levers (default all), RT_TUNE_BUDGET,
      # RT_TUNE_BATCH, RT_TUNE_SCOPES fmoe phase_levers, RT_ROUNDS per batch, RT_CONFIRM_ROUNDS)
      fam=${step#tune:}
      d=$O/tune_${fam//:/_}
      timeout -s KILL ${RT_STEP_TIMEOUT:-14400} python $H/rust-track/rt_tune.py --family "$fam" --start "${RT_TUNE_START:-}" \
        --pool "${RT_TUNE_POOL:-}" --scopes "${RT_TUNE_SCOPES:-}" --budget "${RT_TUNE_BUDGET:-96}" --batch "${RT_TUNE_BATCH:-8}" \
        --rounds "${RT_ROUNDS:-12}" --confirm-rounds "${RT_CONFIRM_ROUNDS:-24}" --out "$d" 2>&1 \
        | grep -aE '^(BATCH|EVAL|CONFIRM)|Traceback|Error' | cut -c1-400
      [ ${PIPESTATUS[0]} = 0 ] || rc=1 ;;
    r2)
      # r2:<what>[:<arms>]: the R2 head / toy timing (r2_gpu.py time; arms WHAT:NAME=LABEL/BACKEND[/LEVERS], comma-joined)
      d=$O/r2_${a}
      mkdir -p "$d"
      arms=()
      [ -n "${b:-}" ] && [ -n "${c:-}" ] && IFS=, read -r -a arms <<< "$b:$c"
      [ -n "${b:-}" ] && [ -z "${c:-}" ] && IFS=, read -r -a arms <<< "$b"
      timeout -s KILL ${RT_STEP_TIMEOUT:-900} python $H/rust-pilot/r2_gpu.py time --out "$d" --what "$a" --rounds "${RT_ROUNDS:-20}"         ${arms[@]:+--arm "${arms[@]}"} 2>&1 | grep -aE "ratio|median|WIN|LOSS|neutral|verdict|Traceback|Error" | cut -c1-400 | tail -40
      [ ${PIPESTATUS[0]} = 0 ] || rc=1 ;;
    mut)
      # mut:<name>: the mutant's harness (rt_mutants.py list) runs numerics only, arm D on the mutant; it must fail.
      read -r mkind target <<< "$(python $H/rust-track/rt_mutants.py list | awk -v n="$a" '$1 == n { print $2, $3 }')"
      IFS=: read -r hk h1 h2 h3 <<< "$target"
      am=D; [ -n "${h3:-}" ] && am="Dm=$h3"   # a lever mutant: the lever arm carries the mutation
      env_lib=$ROCKE_CODEOBJ_DIRECT_LIB; pp=$PYTHONPATH
      if [ "$mkind" = gpu ]; then
        env_lib=$Q38_STAGES/${RT_MUT:?}/mut_$a.so
      elif [ "$mkind" = py ]; then
        mt=$Q38_JOB/mt-$a; mkdir -p "$mt"; cp -r "$Q38_STAGES/$Q38_ROCKE/." "$mt/"
        python $H/rust-track/rt_mutants.py py "$mt" "$a" || { echo "MUTANT $a: patch failed"; rc=1; continue; }
        pp=$mt/platform/python:$mt/library:$PYTHONPATH
      fi
      d=$O/mut_$a
      if [ "$hk" = b3 ]; then
        cmdm=(python $H/rust-track/rt_b3.py --variant "$h1" --which "$h2" --arms "$am" --no-time --out "$d")
      elif [ "$hk" = toy ]; then
        mkdir -p "$d"
        cmdm=(python $H/rust-pilot/r2_gpu.py numerics --what toy --out "$d" --arm "toy_$h1:Dm=toy_${h1}_irops/direct/$h2")
      else
        cmdm=(python $H/rust-track/rt_fmoe.py --rows "$h1" --kind "$h2" --arms "$am" --no-time --check-cases 2 --out "$d")
      fi
      ROCKE_CODEOBJ_DIRECT_LIB=$env_lib PYTHONPATH=$pp timeout -s KILL 900 "${cmdm[@]}" > "$d.log" 2>&1
      mrc=$?
      grep -aE '^CHECK.* D |"D": \{|NUMERICS|DIRECT' "$d.log" | cut -c1-400 | tail -3
      if [ $mrc != 0 ]; then echo "MUTANT $a ($mkind, $target): KILLED (rc $mrc)"; else echo "MUTANT $a ($mkind, $target): SURVIVED"; rc=1; fi ;;
    *) echo "unknown step $step"; rc=1 ;;
  esac
done
kill $smi 2> /dev/null
echo "== RTGPU rc=$rc $(date -u +%T)"
exit $rc
