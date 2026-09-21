#!/usr/bin/env bash
# Run Tensile tuning for every config in configs/ and extract results.
#
# Usage:
#   ./run_tuning.sh <output_dir>
#   ./run_tuning.sh tuning_run1
#   ./run_tuning.sh tuning_run1 --keep-outputs
#   ./run_tuning.sh tuning_run1 --configs=my_configs/
#
# Each configs/*.yaml is run into <output_dir>/<yaml-stem>/ with log.txt.
# At the end, extract_tuning_perf.py is called to produce a summary CSV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPBLASLT_DIR="${HIPBLASLT_DIR:-$HOME/rocm-libraries/projects/hipblaslt}"
TENSILELITE_DIR="${TENSILELITE_DIR:-$HIPBLASLT_DIR/tensilelite}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
VENV_DIR="${VENV_DIR:-$HOME/env1}"
CONFIGS_DIR="${CONFIGS_DIR:-$ROOT/configs}"
KEEP_OUTPUTS=0
OUT_ARG=""

usage() {
    awk 'NR==1 && /^#!/ {next}
         /^#/ {sub(/^# ?/, ""); print; next}
         {exit}' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --keep-outputs) KEEP_OUTPUTS=1 ;;
        --rocm-path=*) ROCM_PATH="${1#--rocm-path=}" ;;
        --rocm-path) shift; ROCM_PATH="$1" ;;
        --venv=*) VENV_DIR="${1#--venv=}" ;;
        --venv) shift; VENV_DIR="$1" ;;
        --configs=*) CONFIGS_DIR="${1#--configs=}" ;;
        --configs) shift; CONFIGS_DIR="$1" ;;
        --) shift; break ;;
        -*)
            printf 'ERROR: unknown flag %s\n' "$1" >&2
            exit 2
            ;;
        *)
            if [[ -n "$OUT_ARG" ]]; then
                printf 'ERROR: extra positional argument %s\n' "$1" >&2
                exit 2
            fi
            OUT_ARG="$1"
            ;;
    esac
    shift
done

log() { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -n "$OUT_ARG" ]] || die "output folder required (e.g. tuning_run1). See --help."
if [[ "$OUT_ARG" = /* ]]; then
    OUT_DIR="$OUT_ARG"
else
    OUT_DIR="$ROOT/$OUT_ARG"
fi

TENSILE_BIN="$TENSILELITE_DIR/Tensile/bin/Tensile"
PYTHON_BIN="$VENV_DIR/bin/python3"

[[ -d "$TENSILELITE_DIR" ]] || die "tensilelite not found at $TENSILELITE_DIR"
[[ -x "$TENSILE_BIN" ]]     || die "Tensile launcher not found at $TENSILE_BIN"
[[ -d "$CONFIGS_DIR" ]]      || die "configs dir not found at $CONFIGS_DIR"
[[ -x "$ROCM_PATH/bin/hipcc" ]] || die "hipcc not found under $ROCM_PATH (set ROCM_PATH or --rocm-path)"
[[ -x "$PYTHON_BIN" ]]       || die "python not found at $PYTHON_BIN (set VENV_DIR or --venv)"

shopt -s nullglob
CONFIGS=("$CONFIGS_DIR"/*.yaml)
(( ${#CONFIGS[@]} )) || die "no *.yaml files in $CONFIGS_DIR"

# ---------------------------------------------------------------------------
# ROCm env
# ---------------------------------------------------------------------------
export ROCM_PATH
export PATH="$ROCM_PATH/bin${PATH:+:$PATH}"
TENSILE_LIB_DIR="$TENSILELITE_DIR/build_tmp/tensilelite"
export LD_LIBRARY_PATH="$TENSILE_LIB_DIR:$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH:-$ROCM_PATH/lib/llvm/amdgcn/bitcode}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export HSA_USE_SVM="${HSA_USE_SVM:-1}"
export HSA_XNACK="${HSA_XNACK:-1}"

# ---------------------------------------------------------------------------
# Run every config YAML
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
log "output directory: $OUT_DIR"
log "configs (${#CONFIGS[@]}): ${CONFIGS[*]}"

FAILED=()
for yaml in "${CONFIGS[@]}"; do
    name="$(basename "$yaml" .yaml)"
    dest="$OUT_DIR/$name"
    if (( KEEP_OUTPUTS )); then
        mkdir -p "$dest"
    else
        log "wiping stale output $dest"
        rm -rf "$dest"
        mkdir -p "$dest"
    fi
    log "Tensile $name -> $dest"
    set +e
    "$PYTHON_BIN" "$TENSILE_BIN" \
        "$yaml" \
        "$dest" \
        2>&1 | tee "$dest/log.txt"
    rc="${PIPESTATUS[0]}"
    set -e
    if (( rc != 0 )); then
        log "FAILED $name (exit $rc)"
        FAILED+=("$name")
    else
        log "ok $name"
    fi
done

# ---------------------------------------------------------------------------
# Report failures
# ---------------------------------------------------------------------------
if (( ${#FAILED[@]} )); then
    printf '\033[1;31mFailed configs: %s\033[0m\n' "${FAILED[*]}" >&2
fi

# ---------------------------------------------------------------------------
# Extract tuning perf summary
# ---------------------------------------------------------------------------
if [[ -f "$ROOT/extract_tuning_perf.py" ]]; then
    log "extracting tuning results from $OUT_DIR"
    "$PYTHON_BIN" "$ROOT/extract_tuning_perf.py" "$OUT_DIR" \
        -o "$OUT_DIR/tuning_results.csv" || true
fi

log "all configs finished under $OUT_DIR"
if (( ${#FAILED[@]} )); then
    exit 1
fi
