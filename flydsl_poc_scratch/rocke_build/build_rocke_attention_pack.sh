#!/usr/bin/env bash
# Reproduce the rocKE gfx950 dense-attention kpack for the bake-off shape
# (B=1, H=8, S=256, D=128, non-causal, bf16). Reuses Brian Harrison's in-tree
# rocke authoring SDK + descriptor specs; packs ONE variant offline via hkp_pack.
# No provider rebuild, no C++ changes — just produces a runtime kpack tree that
# HIPDNN_DESCRIPTOR_RUNTIME_DIR points at so hipkernel:Gfx950AttentionDense enumerates.
#
# One-time prerequisite (already done in this worktree): the rocm_kpack PYTHON packer,
# blobless-sparse-fetched to $KP below from ROCm/rocm-systems@a022846 shared/kpack/python,
# with `msgpack` + `zstandard` pip-installed into the flydsl venv.
set -euo pipefail

REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
PROV=$REPO/dnn-providers/hip-kernel-provider
BASE=/home/AMD/brpepers/rocm-dev-work/base/rocm
KP=/home/AMD/brpepers/rocm-dev-work/kpack-src/shared/kpack/python   # rocm_kpack python packer
PY=/home/AMD/brpepers/flydsl-venv/bin/python                        # has numpy+msgpack+zstandard
SRC=$REPO/flydsl_poc_scratch/rocke_pack_src                         # one-UKD authored source root
OUT=${1:-$REPO/flydsl_poc_scratch/rocke_pack_out}                   # packed runtime tree

mkdir -p /tmp/comgr-cache
export AMD_COMGR_CACHE_DIR=/tmp/comgr-cache          # local disk: network home is ~10x slower
export ROCKE_COMGR_LIB=$BASE/lib/libamd_comgr.so.3   # rocke lowers kernels through comgr
export PATH=$BASE/bin:$PATH
export PYTHONPATH=$PROV/descriptor-packaging/python:$PROV/rocke/library:$PROV/rocke/platform/python:$KP

echo "[pack] source-root=$SRC"
echo "[pack] out-root=$OUT"
rm -rf "$OUT"
cd "$PROV"
"$PY" descriptor-packaging/tools/hkp_pack.py \
  --source-root "$SRC" \
  --out-root "$OUT" \
  --arches gfx950 \
  --hipcc "$BASE/bin/hipcc" \
  --kpack-python-dir "$KP"

echo "[pack] done. kpack + kind:kpack descriptors under $OUT/gfx950/"
find "$OUT" -type f
