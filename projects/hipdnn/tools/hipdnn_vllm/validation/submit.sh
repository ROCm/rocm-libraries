#!/bin/bash
# Submit the validation matrix for one or more archs (submit host; sbatch only).
#
#   submit.sh <arch[,arch...]> [models] [candidates] [tag] [-- extra sbatch args]
#     arch:       gfx942 gfx950 gfx1100 gfx1201 gfx1151 (keys in the config)
#     models:     comma list of models.json names, or "matrix" (default)
#     candidates: e2e_matrix.py --candidates (default "vllm_c,default,default+linear")
#     tag:        results subdirectory (default matrix-<date>)
# Environment: VALIDATION_CONFIG (site settings; default site.env beside this
# script, see site.env.example), HV_WHEEL (wheel override, e.g. a
# bundled-backend wheel), E2E_ARGS.
# This script must live on a compute-visible path: jobs run the harness from
# the same directory.
set -euo pipefail
HARNESS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
VALIDATION_CONFIG=${VALIDATION_CONFIG:-$HARNESS_DIR/site.env}
if [ ! -f "$VALIDATION_CONFIG" ]; then
  echo "submit.sh: no validation config at $VALIDATION_CONFIG; copy $HARNESS_DIR/site.env.example" \
    "there (or set VALIDATION_CONFIG) and fill in the site values" >&2
  exit 2
fi
# Jobs run with --chdir=$HV_ROOT, so every path handed to them must be absolute.
VALIDATION_CONFIG=$(cd "$(dirname "$VALIDATION_CONFIG")" && pwd -P)/$(basename "$VALIDATION_CONFIG")
if [ -n "${HV_WHEEL:-}" ]; then
  [ -f "$HV_WHEEL" ] || { echo "submit.sh: HV_WHEEL=$HV_WHEEL is not a file" >&2; exit 2; }
  HV_WHEEL=$(cd "$(dirname "$HV_WHEEL")" && pwd -P)/$(basename "$HV_WHEEL")
  export HV_WHEEL
fi
# shellcheck source=site.env.example
source "$VALIDATION_CONFIG"
ARCHS=${1:?arch list}
MODELS=${2:-matrix}
CANDIDATES=${3:-vllm_c,default,default+linear}
TAG=${4:-matrix-$(date +%Y%m%d)}
shift $(( $# < 4 ? $# : 4 ))
[ "${1:-}" = "--" ] && shift
EXCLUDE=()
[ -n "${HV_EXCLUDE:-}" ] && EXCLUDE=(--exclude="$HV_EXCLUDE")
mkdir -p "$HV_LOG_DIR"
for ARCH in ${ARCHS//,/ }; do
  GRES_VAR=HV_GRES_$ARCH
  EXTRA_VAR=HV_SBATCH_$ARCH
  MEM_VAR=HV_MEM_$ARCH
  GRES=${!GRES_VAR:?no HV_GRES_$ARCH in $VALIDATION_CONFIG}
  # Per-arch flags come after the defaults so they override them.
  # shellcheck disable=SC2086
  sbatch --parsable -J "hv-val-$ARCH" -p "$HV_PARTITION" --gres="$GRES" \
    -t "$HV_TIME" -c "$HV_CPUS" --mem="${!MEM_VAR:-$HV_MEM}" ${EXCLUDE[@]+"${EXCLUDE[@]}"} ${!EXTRA_VAR:-} \
    --chdir="$HV_ROOT" -o "$HV_LOG_DIR/validate-$ARCH-%j.log" \
    --export=ALL,HARNESS_DIR="$HARNESS_DIR",VALIDATION_CONFIG="$VALIDATION_CONFIG" "$@" \
    "$HARNESS_DIR/run_matrix.sbatch" "$ARCH" "$MODELS" "$CANDIDATES" "$TAG" |
    sed "s/^/$ARCH job /"
done
