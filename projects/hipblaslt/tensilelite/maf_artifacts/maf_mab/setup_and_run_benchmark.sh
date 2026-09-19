#!/usr/bin/env bash
# Rebuild tensilelite-client from the current hipBLASLt tree (needed after
# develop picks up new library YAML fields such as FusedGemmA2A) and run
# every config in maf/ and mab/ into an output folder.
#
# Usage:
#   ./setup_and_run_benchmark.sh bkc_26.9.5
#   ./setup_and_run_benchmark.sh bkc_26.9.5 --skip-rebuild
#   ./setup_and_run_benchmark.sh /abs/path/to/out --gpu-targets=gfx1250
#   ./setup_and_run_benchmark.sh bkc_26.9.5 --configs=maf   # run only MAF
#   ./setup_and_run_benchmark.sh bkc_26.9.5 --configs=mab   # run only MAB
#   ./setup_and_run_benchmark.sh bkc_26.9.5 --num-runs=5    # 5 iterations per config
#
# The first argument is the output directory (relative to this script, or
# absolute). Each {maf,mab}/*.yaml is run into <out>/<yaml-stem>/ with log.txt.
# Use --configs=<dir> to run only one subset (maf or mab).
# Use --num-runs=N to repeat each config N times (default: 3).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPBLASLT_DIR="${HIPBLASLT_DIR:-$HOME/rocm-libraries/projects/hipblaslt}"
TENSILELITE_DIR="${TENSILELITE_DIR:-$HIPBLASLT_DIR/tensilelite}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
VENV_DIR="${VENV_DIR:-}"
CONFIGS_DIRS=()
GPU_TARGETS="${GPU_TARGETS:-gfx1250}"
SKIP_REBUILD=0
KEEP_OUTPUTS=0
SKIP_EXTRACT=0
NUM_RUNS="${NUM_RUNS:-3}"
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
        --skip-rebuild) SKIP_REBUILD=1 ;;
        --keep-outputs) KEEP_OUTPUTS=1 ;;
        --skip-extract) SKIP_EXTRACT=1 ;;
        --gpu-targets=*) GPU_TARGETS="${1#--gpu-targets=}" ;;
        --gpu-targets) shift; GPU_TARGETS="$1" ;;
        --rocm-path=*) ROCM_PATH="${1#--rocm-path=}" ;;
        --rocm-path) shift; ROCM_PATH="$1" ;;
        --venv=*) VENV_DIR="${1#--venv=}" ;;
        --venv) shift; VENV_DIR="$1" ;;
        --num-runs=*) NUM_RUNS="${1#--num-runs=}" ;;
        --num-runs) shift; NUM_RUNS="$1" ;;
        --configs=*) CONFIGS_DIRS+=("${1#--configs=}") ;;
        --configs) shift; CONFIGS_DIRS+=("$1") ;;
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

[[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]] || die "--num-runs must be a positive integer (got '$NUM_RUNS')"
[[ -n "$OUT_ARG" ]] || die "output folder required (e.g. bkc_26.9.5). See --help."
if [[ "$OUT_ARG" = /* ]]; then
    OUT_DIR="$OUT_ARG"
else
    OUT_DIR="$ROOT/$OUT_ARG"
fi
mkdir -p "$OUT_DIR"

# Default venv location: inside the output directory
[[ -n "$VENV_DIR" ]] || VENV_DIR="$OUT_DIR/.venv"

TENSILE_BIN="$TENSILELITE_DIR/Tensile/bin/Tensile"
TENSILE_CLIENT="$TENSILELITE_DIR/build_tmp/tensilelite/client/tensilelite-client"
INVOKE_BIN="$VENV_DIR/bin/invoke"
PYTHON_BIN="$VENV_DIR/bin/python3"

[[ -d "$TENSILELITE_DIR" ]] || die "tensilelite not found at $TENSILELITE_DIR"
[[ -x "$TENSILE_BIN" ]] || die "Tensile launcher not found at $TENSILE_BIN"
[[ -x "$ROCM_PATH/bin/hipcc" ]] || die "hipcc not found under $ROCM_PATH (set ROCM_PATH or --rocm-path)"

# ---------------------------------------------------------------------------
# Create venv and install rocisa + dependencies
# ---------------------------------------------------------------------------
if [[ ! -x "$PYTHON_BIN" ]]; then
    log "creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    log "installing requirements-dev.txt (includes rocisa)"
    "$VENV_DIR/bin/pip" install --upgrade pip
    (cd "$TENSILELITE_DIR" && "$VENV_DIR/bin/pip" install -r requirements-dev.txt)
else
    log "reusing existing venv at $VENV_DIR"
fi

[[ -x "$INVOKE_BIN" ]] || die "invoke not found at $INVOKE_BIN (venv may be incomplete)"
[[ -x "$PYTHON_BIN" ]] || die "python not found at $PYTHON_BIN (venv creation failed)"

# Default: scan both maf/ and mab/ under ROOT
if (( ${#CONFIGS_DIRS[@]} == 0 )); then
    CONFIGS_DIRS=("$ROOT/maf" "$ROOT/mab")
fi

# Resolve relative dirs against ROOT; validate they exist
for i in "${!CONFIGS_DIRS[@]}"; do
    d="${CONFIGS_DIRS[$i]}"
    [[ "$d" = /* ]] || d="$ROOT/$d"
    CONFIGS_DIRS[$i]="$d"
    [[ -d "$d" ]] || die "configs dir not found: $d"
done

shopt -s nullglob
CONFIGS=()
for d in "${CONFIGS_DIRS[@]}"; do
    CONFIGS+=("$d"/*.yaml)
done
(( ${#CONFIGS[@]} )) || die "no *.yaml files in ${CONFIGS_DIRS[*]}"

# ---------------------------------------------------------------------------
# ROCm env (must be set before invoke and Tensile; Tensile.sh wipes PATH)
# ---------------------------------------------------------------------------
export ROCM_PATH
export PATH="$ROCM_PATH/bin${PATH:+:$PATH}"
# build_tmp's libtensilelite-host.so must come FIRST so the client doesn't
# pick up a stale copy from hipblaslt-install/lib/ or the system.
TENSILE_LIB_DIR="$TENSILELITE_DIR/build_tmp/tensilelite"
export LD_LIBRARY_PATH="$TENSILE_LIB_DIR:$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH:-$ROCM_PATH/lib/llvm/amdgcn/bitcode}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"
export HSA_USE_SVM="${HSA_USE_SVM:-1}"
export HSA_XNACK="${HSA_XNACK:-1}"

# ---------------------------------------------------------------------------
# Rebuild tensilelite-client so it matches the current develop YAML schema
# ---------------------------------------------------------------------------
if (( SKIP_REBUILD )); then
    [[ -x "$TENSILE_CLIENT" ]] || die "--skip-rebuild but client missing: $TENSILE_CLIENT"
    log "skipping client rebuild; using $TENSILE_CLIENT"
else
    log "invoke build-client --clean --gpu-targets=$GPU_TARGETS --rocm-path=$ROCM_PATH"
    (
        cd "$TENSILELITE_DIR"
        "$INVOKE_BIN" build-client \
            --clean \
            --gpu-targets="$GPU_TARGETS" \
            --rocm-path="$ROCM_PATH"
    )
    [[ -x "$TENSILE_CLIENT" ]] || die "client not produced at $TENSILE_CLIENT"
    log "client rebuilt: $TENSILE_CLIENT"
fi

# Record the hipBLASLt tree that produced this run
mkdir -p "$OUT_DIR"
{
    git -C "$HIPBLASLT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown-branch"
    git -C "$HIPBLASLT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown-commit"
} > "$OUT_DIR/hipblaslt.txt"
log "wrote $OUT_DIR/hipblaslt.txt ($(tr '\n' ' ' < "$OUT_DIR/hipblaslt.txt"))"

# ---------------------------------------------------------------------------
# Run every config YAML (repeated $NUM_RUNS times)
# ---------------------------------------------------------------------------
log "each config will be run $NUM_RUNS time(s)"
FAILED=()
for (( run=1; run<=NUM_RUNS; run++ )); do
    log "--- run $run / $NUM_RUNS ---"
    for yaml in "${CONFIGS[@]}"; do
        name="$(basename "$yaml" .yaml)"
        if (( NUM_RUNS > 1 )); then
            dest="$OUT_DIR/${name}/run_${run}"
        else
            dest="$OUT_DIR/${name}"
        fi
        if (( KEEP_OUTPUTS )); then
            mkdir -p "$dest"
        else
            log "wiping stale output $dest"
            rm -rf "$dest"
            mkdir -p "$dest"
        fi
        log "Tensile $name (run $run/$NUM_RUNS) -> $dest"
        set +e
        "$PYTHON_BIN" "$TENSILE_BIN" \
            "$yaml" \
            --prebuilt-client "$TENSILE_CLIENT" \
            "$dest" \
            2>&1 | tee "$dest/log.txt"
        rc="${PIPESTATUS[0]}"
        set -e
        if (( rc != 0 )); then
            log "FAILED $name run $run (exit $rc)"
            FAILED+=("${name}:run${run}")
        else
            log "ok $name run $run"
        fi
    done
done

if (( ${#FAILED[@]} )); then
    printf '\033[1;31mERROR: failed configs: %s\033[0m\n' "${FAILED[*]}" >&2
    exit 1
fi

if (( ! SKIP_EXTRACT )) && [[ -x "$ROOT/extract_perf.py" || -f "$ROOT/extract_perf.py" ]]; then
    log "extract_perf.py $ROOT -s $(basename "$OUT_DIR") -o $OUT_DIR/perf_summary.csv"
    "$PYTHON_BIN" "$ROOT/extract_perf.py" "$ROOT" \
        -s "$(basename "$OUT_DIR")" \
        -o "$OUT_DIR/perf_summary.csv" || true
fi

log "all configs finished under $OUT_DIR"
