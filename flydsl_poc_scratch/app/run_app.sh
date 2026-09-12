#!/usr/bin/env bash
# Runs the standalone flyDSL-through-hipDNN proof app.
#
# CRITICAL: prepend the freshly-built build/lib to LD_LIBRARY_PATH so the app loads
# the current libhipdnn_backend.so (which knows attr 609 =
# HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT). The ROCm SDK ships a
# stale libhipdnn_backend.so in rocm-sdk/current/lib that predates that attribute and
# would otherwise win via LD_LIBRARY_PATH, causing HIPDNN_ATTR_UNKNOWN at graph build.
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
APP="$REPO/flydsl_poc_scratch/app/build/flydsl_hipdnn_app"

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors"
export HIPDNN_PLUGIN_DIR="$REPO/build/lib/hipdnn_plugins/engines"
export FLYDSL_HSACO_PATH="$REPO/flydsl_poc_scratch/vadd_gfx950.hsaco"
export HIPDNN_LOG_LEVEL="${HIPDNN_LOG_LEVEL:-info}"

echo "[run] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[run] resolving libhipdnn_backend.so ->"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$APP" | grep -i hipdnn || true
echo "[run] ----------------------------------------------------------------"
exec "$APP"
