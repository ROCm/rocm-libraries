#!/usr/bin/env bash
# Pack the WORKLOAD-RELEVANT subset (hd128, dense, batch=1, BF16+FP16 -> 1694 specs)
# of the shipped rocKE gfx950_attention_dense descriptor set into a runtime kpack,
# so HIPDNN_DESCRIPTOR_RUNTIME_DIR can point at the REAL fast-path variants
# (persistent + wide_lds_dma) for best-vs-best. Mirrors build_rocke_attention_pack.sh
# but with the full-set staged source root + a separate out root.
set -euo pipefail
REPO=/home/AMD/brpepers/wt/hipdnn-flydsl-poc
PROV=$REPO/dnn-providers/hip-kernel-provider
BASE=/home/AMD/brpepers/rocm-dev-work/base/rocm
KP=/home/AMD/brpepers/rocm-dev-work/kpack-src/shared/kpack/python
PY=/home/AMD/brpepers/flydsl-venv/bin/python
SRC=$REPO/flydsl_poc_scratch/rocke_pack_src_all
OUT=${1:-$REPO/flydsl_poc_scratch/rocke_pack_out_all}
mkdir -p /tmp/comgr-cache
export AMD_COMGR_CACHE_DIR=/tmp/comgr-cache
export ROCKE_COMGR_LIB=$BASE/lib/libamd_comgr.so.3
export PATH=$BASE/bin:$PATH
export PYTHONPATH=$PROV/descriptor-packaging/python:$PROV/rocke/library:$PROV/rocke/platform/python:$KP
export HKP_PACK_JOBS=${HKP_PACK_JOBS:-14}
echo "[pack] source-root=$SRC out-root=$OUT jobs=$HKP_PACK_JOBS"
rm -rf "$OUT"
cd "$PROV"
time "$PY" descriptor-packaging/tools/hkp_pack.py \
  --source-root "$SRC" --out-root "$OUT" --arches gfx950 \
  --hipcc "$BASE/bin/hipcc" --kpack-python-dir "$KP"
echo "[pack] done."
find "$OUT" -type f | head
