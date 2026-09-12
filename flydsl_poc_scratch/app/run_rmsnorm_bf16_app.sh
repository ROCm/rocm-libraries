#!/usr/bin/env bash
# Runs the M2-SWAP proof: the REAL FlyDSL bf16 RMSNorm family through hipDNN.
#
# Same local-libs-win trick as run_rmsnorm_app.sh, but points at the SEPARATE bf16
# descriptor root so only the real family loads. The dispatch raw-loads
# rmsnorm_real_n<N>_bf16_gfx950.hsaco from FLYDSL_RMSNORM_HSACO_DIR (real2d ABI).
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
APP="$REPO/flydsl_poc_scratch/app/build/flydsl_rmsnorm_bf16_app"

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors_rmsnorm_bf16"
export HIPDNN_PLUGIN_DIR="$REPO/build/lib/hipdnn_plugins/engines"
export FLYDSL_RMSNORM_HSACO_DIR="$REPO/flydsl_poc_scratch"
export HIPDNN_LOG_LEVEL="${HIPDNN_LOG_LEVEL:-info}"

echo "[run] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[run] HIPDNN_DESCRIPTOR_DIR=$HIPDNN_DESCRIPTOR_DIR"
echo "[run] FLYDSL_RMSNORM_HSACO_DIR=$FLYDSL_RMSNORM_HSACO_DIR"
echo "[run] resolving libhipdnn_backend.so ->"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$APP" | grep -i hipdnn || true
echo "[run] ----------------------------------------------------------------"
exec "$APP"
