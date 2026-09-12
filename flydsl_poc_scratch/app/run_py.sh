#!/usr/bin/env bash
# Reusable env wrapper for the hipdnn_torch Python harnesses (mistral_*, llama3_*, qwen_*).
# Mirrors run_attention_bakeoff_app.sh's provider/descriptor env but for the torch injection:
#   PROVIDER_SO   -> our built hip_kernel_provider
#   FRONTEND_DIR  -> hipDNN python frontend bindings build (hipdnn_frontend_python.abi3.so)
#   DESCRIPTOR_DIR + DESCRIPTOR_RUNTIME_DIR -> flyDSL descriptors + packed rocKE tree
#   AITER_ASM_DIR -> asm_sdpa .co catalog (else it looks under /opt/rocm and finds nothing)
# Usage: bash run_py.sh <harness.py> [args...]   (extra HIPDNN_* env may be pre-set by caller)
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
PY=/home/AMD/brpepers/flydsl-venv/bin/python
HARNESS="${1:?usage: run_py.sh <harness.py> [args...]}"
shift || true

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_TORCH_PROVIDER_SO="$REPO/build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so"
export HIPDNN_TORCH_FRONTEND_DIR="$REPO/projects/hipdnn/python/frontend_bindings/build"
export HIPDNN_PLUGIN_DIR="$REPO/build/lib/hipdnn_plugins/engines"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors_attention"
# Default to the FULL shipped rocKE set (1694 variants incl. persistent+wide_lds_dma
# fast path) so we test best-vs-best. The old 5-instance slow-path POC pack
# (rocke_pack_out/gfx950) is kept for reference; override RUNTIME_DIR to use it.
export HIPDNN_DESCRIPTOR_RUNTIME_DIR="${HIPDNN_DESCRIPTOR_RUNTIME_DIR:-$REPO/flydsl_poc_scratch/rocke_pack_out_all/gfx950}"
export FLYDSL_ATTENTION_HSACO_DIR="$REPO/flydsl_poc_scratch"
export HIPDNN_AITER_ASM_DIR="$REPO/build/hip_kernel_provider/asm_kernels"
export HIPDNN_TORCH_SELECT="${HIPDNN_TORCH_SELECT:-default}"
export HIPDNN_LOG_LEVEL="${HIPDNN_LOG_LEVEL:-warn}"
export PYTHONPATH="$REPO/projects/hipdnn/tools/hipdnn_torch:${PYTHONPATH:-}"

echo "[run_py] $HARNESS $*"
echo "[run_py] SELECT=$HIPDNN_TORCH_SELECT TUNE=${HIPDNN_TORCH_TUNE:-<unset>} RUNTIME_DIR=$HIPDNN_DESCRIPTOR_RUNTIME_DIR"
echo "[run_py] --------------------------------------------------------------"
exec "$PY" "$HARNESS" "$@"
