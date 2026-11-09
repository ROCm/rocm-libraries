#!/usr/bin/env bash
# Runs INSIDE the pinned container (rocm/pytorch@sha256:bbdaba66..., the image of ATOM's
# scripts/qwen38_flash_mono/repro.sh), started as REPRO.md "Reproduce on your own MI355X" shows. Expected mounts:
#   /atom                      ATOM-truemono tree   (ATOM 352229d6 + patch 01 + patch 02)
#   /job/stages/hb-atom-sa     hybrid ATOM tree     (ATOM 352229d6 + patch 01)
#   /job/stages/rocke-qk-sa    rocKE of the stack   (rocm-libraries 4478c174a13 + patch 03, dir dnn-providers/hip-kernel-provider/rocke)
#   /job/stages/rocke-tmbase   rocKE of the hybrid  (rocm-libraries 89209ac7665, same dir, unpatched)
#   /job/stages/h-qk-sa        kit/harness          (the frozen harness subset of this kit)
#   /kit                       kit/                 (this directory)
#   /model                     Qwen3.8-Flash-Next-MXFP4 (read-only)
#   /work                      ATOM repro.sh's $WORK, with the venv `repro.sh setup` built
#   /job                       a new, empty run directory (results land in /job/out)
#
#   bash /kit/in_container.sh gen     regenerate the stage functions (gen_c2Ab + the hybrid's gen_e2ecomp2) and compare
#                                     them with the measured manifests (kit/expected; *.bc excluded: they embed the path)
#   bash /kit/in_container.sh bench   gen, then the quick harness unchanged: hybrid / hybrid_b / c2s3b / c2s3b_b at K=3
#   bash /kit/in_container.sh k0      gen, then hybrid / hybrid_b / c2s0b / c2s0b_b at K=0
# Env: PASSES (4), SEED (0), ARMS_K3 / ARMS_K0 (override the arm lists), PROMPTS (32), TOKENS (2048).
set -uo pipefail
mode=${1:-bench}
die() { echo "in_container: $*" >&2; exit 1; }
for p in /atom/atom/models/qwen4_exp_mono /job/stages/hb-atom-sa/atom/models/qwen4_exp_mono \
  /job/stages/rocke-qk-sa/library/kernels/gfx950/qwen38_mono/qsa2 /job/stages/rocke-tmbase/library/kernels/gfx950/qwen38_mono \
  /job/stages/h-qk-sa/truemono-quick/quick.sh /kit/expected/manifest-c2Ab.sha256 /model/config.json /work/venv/bin/activate; do
  [ -e "$p" ] || die "missing $p (see the mounts at the top of this file)"
done
[ -f /atom/atom/models/qwen4_exp_mono/truemono/tail.py ] || die "/atom lacks truemono/tail.py: patch 02 not applied"

# the environment of the measured jobs' container mode (hackathon-prep tools/q38/q38_job.sh)
export HOME=/work/home TMPDIR=/work/tmp PIP_CACHE_DIR=/work/cache/pip XDG_CACHE_HOME=/work/cache/xdg HF_HOME=/work/cache/hf
export AITER_LOG_LEVEL=WARNING HSA_COREDUMP_PATTERN=/dev/null PYTHONDONTWRITEBYTECODE=1
export Q38_JOB=/job Q38_OUT=/job/out Q38_STAGES=/job/stages
mkdir -p "$HOME" "$TMPDIR" /job/out
# shellcheck disable=SC1091
source /work/venv/bin/activate
SP=/work/venv/lib/python3.12/site-packages
export ROCM_PATH=$SP/_rocm_sdk_devel HIP_PATH=$SP/_rocm_sdk_devel PATH=$SP/_rocm_sdk_devel/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/host-math/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/job/stages/rocke-tmbase/platform/python:/job/stages/rocke-tmbase/library  # = q38_job.sh's first rocke* stage
arch=$(python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2> /dev/null) \
  || die "no GPU visible in the container (check --device=/dev/kfd --device=/dev/dri and HIP_VISIBLE_DEVICES)"
[ "$arch" = gfx950 ] || die "GPU is $arch; this stack needs gfx950 (MI355X)"
echo "== in_container $mode: $(python -V) gpu $(python -c 'import torch; print(torch.cuda.get_device_name(0))') $arch $(date -u)"

# the env every measured c2 job passed (REPRO.md "The knobs, per arm"; alola rules 236 / COORD-consolidated)
export QK_SFX=sa CS_GEN_PP=1 WP2B_KNOBS_JSON=picks_all.json WP2B_JSON_ONLY=1
H=/job/stages/h-qk-sa
export ARM_FILES="$H/truemono/RB/arms_rb.sh $H/truemono/WP2a-fix/arms_wp2a.sh $H/truemono/WP2b/arms_wp2b.sh"

gen() { # gen_c2Ab + gen_e2ecomp2, compared with the measured job's manifests (qk-c2-e4-n01)
  local O=/job/out/gen rc=0
  mkdir -p "$O"
  (
    HB=/job/stages/hb-atom-sa TM=/atom RK_BASE=rocke-tmbase RK_TM=rocke-qk-sa
    source "$H/common/server.sh"
    source "$H/truemono-quick/arms.sh"
    for spec in "c2Ab rocke-qk-sa manifest-c2Ab.sha256" "e2ecomp2 rocke-tmbase manifest-e2ecomp2.sha256"; do
      read -r R RK EXP <<< "$spec"
      FN=/job/fns/$R-$RK
      "gen_$R" "$RK" "$FN" > "$O/gen-$R.log" 2>&1 || { tail -20 "$O/gen-$R.log"; echo "GEN $R: generator FAILED"; exit 3; }
      [ -f "$FN/manifest.sha256" ] || (cd "$FN" && find . -type f ! -name manifest.sha256 | sed 's|^\./||' | sort | xargs sha256sum > manifest.sha256)
      grep -v '\.bc$' "/kit/expected/$EXP" | sort -k2 > "$O/exp-$R.txt"
      grep -v '\.bc$' "$FN/manifest.sha256" | sort -k2 > "$O/got-$R.txt"
      n=$(wc -l < "$O/exp-$R.txt"); d=$(diff "$O/exp-$R.txt" "$O/got-$R.txt" | grep -c '^[<>]')
      if [ "$d" = 0 ]; then echo "GEN $R: PASS ($n files byte-identical to the measured functions, .bc excluded)"
      else echo "GEN $R: FAIL ($d differing manifest lines of $n)"; diff "$O/exp-$R.txt" "$O/got-$R.txt" | head -20; exit 4; fi
    done
  ) || rc=$?
  [ $rc = 0 ] && echo "GEN PASS" || echo "GEN FAIL (rc $rc): the trees, the kit or the container differ from the measured ones"
  return $rc
}

case $mode in
  gen) gen ;;
  bench | k0)
    gen || exit 1
    if [ "$mode" = bench ]; then export KS=3 ARMS_K3=${ARMS_K3:-hybrid hybrid_b c2s3b c2s3b_b}
    else export KS=0 ARMS_K0=${ARMS_K0:-hybrid hybrid_b c2s0b c2s0b_b}; fi
    export PASSES=${PASSES:-4} SEED=${SEED:-0}
    cd /job && bash "$H/truemono-quick/quick.sh" 2>&1 | tee /job/out/quick.log
    rc=${PIPESTATUS[0]}
    python /kit/verdict.py /job/out/q/ratios.json || rc=1
    exit $rc
    ;;
  *) die "mode gen | bench | k0" ;;
esac
