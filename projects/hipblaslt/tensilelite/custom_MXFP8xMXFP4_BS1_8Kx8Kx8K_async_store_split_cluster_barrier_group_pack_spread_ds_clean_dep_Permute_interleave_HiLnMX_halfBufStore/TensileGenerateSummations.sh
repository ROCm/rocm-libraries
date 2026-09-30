#!/bin/bash
set -euo pipefail
cmake --build /home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/custom_MXFP8xMXFP4_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore -j 128
if [[ "${DEBUGPY_ENABLE:-}" == "1" ]]; then
    echo "===DEBUGPY_READY==="
    PYTHONPATH=/home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/custom_MXFP8xMXFP4_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore/lib /home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/venv/bin/python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client /home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/TensileGenerateSummations "$@"
else
    PYTHONPATH=/home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/custom_MXFP8xMXFP4_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep_Permute_interleave_HiLnMX_halfBufStore/lib /home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/venv/bin/python3 /home/chicyang/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/TensileGenerateSummations "$@"
fi
