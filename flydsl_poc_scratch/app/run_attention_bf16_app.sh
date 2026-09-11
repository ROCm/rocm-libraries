#!/usr/bin/env bash
# Runs the M3 proof: the REAL FlyDSL bf16 flash-attention family through hipDNN.
#
# Same local-libs-win trick as the RMSNorm apps, but points at the attention descriptor
# root so the attention family loads. The dispatch raw-loads
# flash_attn_real_h<H>_d128_causal_bf16_gfx950.hsaco from FLYDSL_ATTENTION_HSACO_DIR.
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
APP="$REPO/flydsl_poc_scratch/app/build/flydsl_attention_bf16_app"

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors_attention"
export HIPDNN_PLUGIN_DIR="$REPO/build/lib/hipdnn_plugins/engines"
export FLYDSL_ATTENTION_HSACO_DIR="$REPO/flydsl_poc_scratch"
export HIPDNN_LOG_LEVEL="${HIPDNN_LOG_LEVEL:-info}"

echo "[run] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[run] HIPDNN_DESCRIPTOR_DIR=$HIPDNN_DESCRIPTOR_DIR"
echo "[run] FLYDSL_ATTENTION_HSACO_DIR=$FLYDSL_ATTENTION_HSACO_DIR"
echo "[run] resolving libhipdnn_backend.so ->"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$APP" | grep -i hipdnn || true
echo "[run] ----------------------------------------------------------------"
exec "$APP"
