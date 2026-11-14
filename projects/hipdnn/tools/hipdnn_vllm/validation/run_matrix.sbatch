#!/bin/bash
#SBATCH -J hv-validate
# Model-matrix validation job. Submit it with submit.sh, which adds the
# per-arch partition, GPU request, node excludes and log path from the
# validation config and exports HARNESS_DIR and VALIDATION_CONFIG (absolute).
#
#   run_matrix.sbatch <arch> <models> <candidates> <tag>
#     models:     comma list of models.json names, or "matrix" (every entry
#                 with "matrix" not false)
#     candidates: comma list for e2e_matrix.py --candidates
#     tag:        results go to $HV_RESULTS/<tag>/<arch>-<job id>/
# Optional environment: HV_WHEEL (absolute wheel file; default newest in the
# arch's family dir), E2E_ARGS (extra e2e_matrix.py run arguments).
#
# Exit status: 0 only when the doctor check, every model run and the
# aggregation succeed and at least one model ran; 1 when any of them failed;
# 2 when the job could not be set up.
set -uo pipefail
: "${HARNESS_DIR:?set by submit.sh}" "${VALIDATION_CONFIG:?set by submit.sh}"
[ -f "$VALIDATION_CONFIG" ] || { echo "no validation config at $VALIDATION_CONFIG"; exit 2; }
# shellcheck source=site.env.example
source "$VALIDATION_CONFIG"
ARCH=${1:?arch}
MODELS=${2:?models}
CANDIDATES=${3:?candidates}
TAG=${4:?tag}
FAMILY_VAR=HV_FAMILY_$ARCH
FAMILY=${!FAMILY_VAR:?no HV_FAMILY_$ARCH in $VALIDATION_CONFIG}
WHEEL=${HV_WHEEL:-$(ls -t "$HV_WHEEL_ROOT/$FAMILY"/hipdnn_vllm-*.whl 2>/dev/null | head -1)}
OUT=$HV_RESULTS/$TAG/$ARCH-${SLURM_JOB_ID:-manual}
O=$HV_SCRATCH/validate-overlay-${SLURM_JOB_ID:-$$}
mkdir -p "$OUT" || exit 2
echo "host=$(hostname) job=${SLURM_JOB_ID:-} arch=$ARCH family=$FAMILY date=$(date -Is)"
echo "models=$MODELS candidates=$CANDIDATES out=$OUT"
echo "wheel=$WHEEL"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
[ -f "$WHEEL" ] || { echo "no wheel for $FAMILY under $HV_WHEEL_ROOT"; exit 2; }

# Overlay venv: the wheel is installed here; a .pth exposes the shared pinned
# venv's site-packages, which stays untouched. The trap is set first so a
# failed setup does not leave the overlay behind.
trap 'rm -rf "$O"' EXIT
SP=$HV_VENV/lib/python3.12/site-packages
"$HV_VENV/bin/python" -m venv "$O" || exit 2
echo "$SP" > "$O/lib/python3.12/site-packages/zz-hipdnn-vllm-shared.pth" || exit 2
"$O/bin/python" -m pip install -q --no-deps "$WHEEL" || exit 2
export PATH=$O/bin:$HV_VENV/bin:$PATH HF_HOME=$HV_HF_HOME HF_HUB_OFFLINE=1 HV_ROOT
# Point the loader at the pip ROCm first: on nodes with a system ROCm in the
# ldcache, an empty LD_LIBRARY_PATH lets those libraries shadow the pip ones,
# and a node-level one into a system ROCm does the same.
echo "LD_LIBRARY_PATH(node)=${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib
cd "$OUT" || exit 2

RC=0
python - "$OUT/job.json" "$WHEEL" <<'EOF' || { echo "JOB_INFO_RC=$?"; RC=1; }
import hashlib, json, os, sys, hipdnn_vllm, torch, vllm
info = os.path.join(os.path.dirname(hipdnn_vllm.__file__), "_build_info.json")
json.dump({"job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "wheel": sys.argv[2],
           "wheel_sha256": hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(),
           "wheel_mtime": os.path.getmtime(sys.argv[2]),
           "build_info": json.load(open(info)) if os.path.exists(info) else None,
           "hipdnn_vllm": hipdnn_vllm.__file__, "vllm": vllm.__version__, "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0), "arch": torch.cuda.get_device_properties(0).gcnArchName},
          open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
EOF
python -m hipdnn_vllm doctor --json > doctor.json
DOCTOR_RC=$?
echo "DOCTOR_RC=$DOCTOR_RC"
[ "$DOCTOR_RC" -eq 0 ] || RC=1

if [ "$MODELS" = matrix ]; then
  if ! MODELS=$(python -c "import json,sys; print(','.join(m['name'] for m in json.load(open(sys.argv[1]))['models'] if m.get('matrix', True)))" "$HARNESS_DIR/models.json"); then
    echo "cannot read the model list from $HARNESS_DIR/models.json"
    MODELS=
    RC=1
  fi
fi
RAN=0
for M in ${MODELS//,/ }; do
  S=$(date +%s)
  # shellcheck disable=SC2086
  python "$HARNESS_DIR/e2e_matrix.py" run --model-name "$M" --models "$HARNESS_DIR/models.json" \
    --root "$HV_ROOT" --out "$OUT/$M" --candidates "$CANDIDATES" ${E2E_ARGS:-}
  MODEL_RC=$?
  echo "MODEL_RC model=$M rc=$MODEL_RC seconds=$(( $(date +%s) - S ))"
  [ "$MODEL_RC" -eq 0 ] || RC=1
  RAN=$((RAN + 1))
done
if [ "$RAN" -eq 0 ]; then
  echo "no model ran"
  RC=1
fi
python "$HARNESS_DIR/aggregate.py" "$OUT" --out "$OUT/tables"
AGG_RC=$?
echo "AGGREGATE_RC=$AGG_RC"
[ "$AGG_RC" -eq 0 ] || RC=1
echo "DONE rc=$RC $(date -Is) results=$OUT"
exit $RC
