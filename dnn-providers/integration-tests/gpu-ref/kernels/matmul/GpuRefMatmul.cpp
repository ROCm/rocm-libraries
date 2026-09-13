// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference matrix multiplication kernel.
// Compiled via HipRTC with -DA_TYPE=<type> -DB_TYPE=<type> -DC_TYPE=<type> -DCOMPUTE_TYPE=<type> -DMATMUL_K=<value> -DMATMUL_M=<value> -DMATMUL_N=<value> -DMATMUL_BATCH_DIM_COUNT=<value> -DTILE_SIZE=<value>. For each batch of matrices, each thread block computes a TILE_SIZE by TILE_SIZE tile of matrix C, which is the matrix multiplication of matrix A and matrix B.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void MatmulRef(MatmulArgs args)
{
    auto* a = static_cast<const A_TYPE*>(args.a);
    auto* b = static_cast<const B_TYPE*>(args.b);
    auto* c = static_cast<C_TYPE*>(args.c);

    long long gidM = blockIdx.x;
    long long gidN = blockIdx.y;
    long long lidM = threadIdx.x;
    long long lidN = threadIdx.y;
    long long idxM = TILE_SIZE * gidM + lidM;
    long long idxN = TILE_SIZE * gidN + lidN;
    long long idx = MATMUL_N * idxM + idxN;
    long long lid = TILE_SIZE * lidM + lidN;

    __shared__ COMPUTE_TYPE aTile[TILE_SIZE * TILE_SIZE];
    __shared__ COMPUTE_TYPE bTile[TILE_SIZE * TILE_SIZE];

    long long batchDimSize = 1;
    for(int d = 0; d < MATMUL_BATCH_DIM_COUNT; ++d)
    {
        batchDimSize *= args.cDims[d];
    }

    for(long long cBatch = 0; cBatch < batchDimSize; ++cBatch)
    {
        long long aBatch = 0;
        long long bBatch = 0;
        long long remainingBatch = cBatch;
        for(int d = 0; d < MATMUL_BATCH_DIM_COUNT; ++d)
        {
            long long aStride = 1;
            long long bStride = 1;
            long long cStride = 1;
            for(int s = d + 1; s < MATMUL_BATCH_DIM_COUNT; ++s)
            {
                aStride *= args.aDims[s];
                bStride *= args.bDims[s];
                cStride *= args.cDims[s];
            }
            long long idxD = remainingBatch / cStride;
            aBatch += aStride * (idxD * args.aDims[d] / args.cDims[d]);
            bBatch += bStride * (idxD * args.bDims[d] / args.cDims[d]);
            remainingBatch -= idxD * cStride;
        }

        auto value = static_cast<COMPUTE_TYPE>(0.0f);
        for(long long k = 0; k < (MATMUL_K + TILE_SIZE - 1) / TILE_SIZE; ++k)
        {
            if(TILE_SIZE * k + lidN < MATMUL_K && idxM < MATMUL_M)
            {
                long long idxA = MATMUL_K * idxM + TILE_SIZE * k + lidN;
                aTile[lid] = static_cast<COMPUTE_TYPE>(a[MATMUL_M * MATMUL_K * aBatch + idxA]);
            }
            else
            {
                aTile[lid] = static_cast<COMPUTE_TYPE>(0.0f);
            }
            if(TILE_SIZE * k + lidM < MATMUL_K && idxN < MATMUL_N)
            {
                long long idxB = MATMUL_N * TILE_SIZE * k + MATMUL_N * lidM + idxN;
                bTile[lid] = static_cast<COMPUTE_TYPE>(b[MATMUL_K * MATMUL_N * bBatch + idxB]);
            }
            else
            {
                bTile[lid] = static_cast<COMPUTE_TYPE>(0.0f);
            }
            __syncthreads();

            for(long long i = 0; i < TILE_SIZE; ++i)
            {
                long long idxATile = TILE_SIZE * lidM + i;
                long long idxBTile = TILE_SIZE * i + lidN;
                value += aTile[idxATile] * bTile[idxBTile];
            }
            __syncthreads();
        }

        if(idxM < MATMUL_M && idxN < MATMUL_N)
        {
            c[MATMUL_M * MATMUL_N * cBatch + idx] = static_cast<C_TYPE>(value);
        }
    }
}
