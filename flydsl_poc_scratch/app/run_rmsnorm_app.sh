#!/usr/bin/env bash
# Runs the M2 flyDSL-RMSNorm-family-through-hipDNN proof app.
#
# Same local-libs-win trick as run_app.sh: prepend build/lib so the fresh
# libhipdnn_backend.so beats the stale SDK one. Descriptors come from the SEPARATE
# rmsnorm descriptor root (so only the RMSNorm pack's descriptors load). The dispatch
# raw-loads rmsnorm_toy_n<N>_gfx950.hsaco from FLYDSL_RMSNORM_HSACO_DIR.
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
APP="$REPO/flydsl_poc_scratch/app/build/flydsl_rmsnorm_app"

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors_rmsnorm"
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
