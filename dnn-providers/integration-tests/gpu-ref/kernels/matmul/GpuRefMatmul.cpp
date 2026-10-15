// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference matrix multiplication kernel.
// Compiled via HipRTC with -DA_TYPE=<type> -DB_TYPE=<type> -DC_TYPE=<type> -DCOMPUTE_TYPE=<type> -DMATMUL_BATCH_DIM_COUNT=<value> -DTILE_SIZE=<value>. For each batch of matrices, each thread block computes a TILE_SIZE by TILE_SIZE tile of matrix C, which is the matrix multiplication of matrix A and matrix B.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void MatmulRef(MatmulArgs args)
{
    auto* a = static_cast<const A_TYPE*>(args.a);
    auto* b = static_cast<const B_TYPE*>(args.b);
    auto* c = static_cast<C_TYPE*>(args.c);

    const long long MATMUL_K = args.aDims[MATMUL_BATCH_DIM_COUNT + 1];
    const long long MATMUL_M = args.aDims[MATMUL_BATCH_DIM_COUNT];
    const long long MATMUL_N = args.bDims[MATMUL_BATCH_DIM_COUNT + 1];

    long long gidM = blockIdx.x;
    long long gidN = blockIdx.y;
    long long lidM = threadIdx.x;
    long long lidN = threadIdx.y;
    long long idxM = TILE_SIZE * gidM + lidM;
    long long idxN = TILE_SIZE * gidN + lidN;
    long long lid = TILE_SIZE * lidM + lidN;

    __shared__ COMPUTE_TYPE aTile[TILE_SIZE * TILE_SIZE];
    __shared__ COMPUTE_TYPE bTile[TILE_SIZE * TILE_SIZE];

    long long batchDimSize = 1;
    for(int d = 0; d < MATMUL_BATCH_DIM_COUNT; ++d)
    {
        batchDimSize *= args.cDims[d];
    }

    long long aBatchIdx = 0;
    long long bBatchIdx = 0;
    long long cBatchIdx = 0;
    long long remainingBatch = blockIdx.z;
    for(int d = 0; d < MATMUL_BATCH_DIM_COUNT; ++d)
    {
        long long cBatchStride = 1;
        for(int s = d + 1; s < MATMUL_BATCH_DIM_COUNT; ++s)
        {
            cBatchStride *= args.cDims[s];
        }
        long long idxCD = remainingBatch / cBatchStride;
        aBatchIdx += args.aStrides[d] * (idxCD * args.aDims[d] / args.cDims[d]);
        bBatchIdx += args.bStrides[d] * (idxCD * args.bDims[d] / args.cDims[d]);
        cBatchIdx += args.cStrides[d] * idxCD;
        remainingBatch -= idxCD * cBatchStride;
    }

    auto value = toAccum(0.0f);
    for(long long k = 0; k < (MATMUL_K + TILE_SIZE - 1) / TILE_SIZE; ++k)
    {
        if(TILE_SIZE * k + lidN < MATMUL_K && idxM < MATMUL_M)
        {
            long long idxA = aBatchIdx + idxM * args.aStrides[MATMUL_BATCH_DIM_COUNT]
                             + (TILE_SIZE * k + lidN) * args.aStrides[MATMUL_BATCH_DIM_COUNT + 1];
            aTile[lid] = toAccum(a[idxA]);
        }
        else
        {
            aTile[lid] = toAccum(0.0f);
        }
        if(TILE_SIZE * k + lidM < MATMUL_K && idxN < MATMUL_N)
        {
            long long idxB = bBatchIdx
                             + (TILE_SIZE * k + lidM) * args.bStrides[MATMUL_BATCH_DIM_COUNT]
                             + idxN * args.bStrides[MATMUL_BATCH_DIM_COUNT + 1];
            bTile[lid] = toAccum(b[idxB]);
        }
        else
        {
            bTile[lid] = toAccum(0.0f);
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
        long long idxC = cBatchIdx + idxM * args.cStrides[MATMUL_BATCH_DIM_COUNT]
                         + idxN * args.cStrides[MATMUL_BATCH_DIM_COUNT + 1];
        c[idxC] = fromAccum<C_TYPE>(value);
    }
}
