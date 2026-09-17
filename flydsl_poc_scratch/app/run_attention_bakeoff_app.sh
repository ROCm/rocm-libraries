#!/usr/bin/env bash
# M4 SDPA bake-off runner. Two apps share this env:
#   flydsl_attention_bakeoff_app <causal>       -> just enumerate applicable engines (no exec)
#   flydsl_attention_bakeoff_exec_app           -> enumerate + execute EACH engine + numerics
#
# The bake-off shape is NON-CAUSAL bf16 head_dim=128 (the flyDSL ∩ asm_sdpa common shape on
# gfx950 — the gfx950 asm_sdpa catalog has no causal kernel). asm_sdpa loads its aiter .co from
# HIPDNN_AITER_ASM_DIR (else it falls back to a hardcoded /opt/rocm path that doesn't exist here).
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
APP="${1:-$REPO/flydsl_poc_scratch/app/build/flydsl_attention_bakeoff_exec_app}"
shift || true

export LD_LIBRARY_PATH="$REPO/build/lib:${LD_LIBRARY_PATH:-}"
export HIPDNN_DESCRIPTOR_DIR="$REPO/flydsl_poc_scratch/flydsl_descriptors_attention"
# rocKE (3rd engine): additive descriptor root holding the packed Gfx950AttentionDense
# kpack for (1,8,256,128 non-causal bf16). Built by rocke_build/build_rocke_attention_pack.sh.
# Set to empty to run the 2-engine (flyDSL + asm_sdpa) bake-off.
export HIPDNN_DESCRIPTOR_RUNTIME_DIR="${HIPDNN_DESCRIPTOR_RUNTIME_DIR:-$REPO/flydsl_poc_scratch/rocke_pack_out/gfx950}"
export HIPDNN_PLUGIN_DIR="$REPO/build/lib/hipdnn_plugins/engines"
export FLYDSL_ATTENTION_HSACO_DIR="$REPO/flydsl_poc_scratch"
export HIPDNN_AITER_ASM_DIR="$REPO/build/hip_kernel_provider/asm_kernels"
export HIPDNN_LOG_LEVEL="${HIPDNN_LOG_LEVEL:-warn}"

echo "[run] APP=$APP $*"
echo "[run] HIPDNN_DESCRIPTOR_DIR=$HIPDNN_DESCRIPTOR_DIR"
echo "[run] HIPDNN_DESCRIPTOR_RUNTIME_DIR=$HIPDNN_DESCRIPTOR_RUNTIME_DIR (rocKE)"
echo "[run] HIPDNN_AITER_ASM_DIR=$HIPDNN_AITER_ASM_DIR"
echo "[run] FLYDSL_ATTENTION_HSACO_DIR=$FLYDSL_ATTENTION_HSACO_DIR"
echo "[run] ----------------------------------------------------------------"
exec "$APP" "$@"
