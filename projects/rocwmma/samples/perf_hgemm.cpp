/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

/* Motivation
*
* For this particular GEMM kernel, high performance can be
* achieved through two general principles:
* 1) Data re-use
* 2) Latency hiding
*
* From the simple_gemm implementation, we know that the GEMM
* equation takes the form:
*
* D = alpha * AxB + beta * C, where
*
* A, B = input tiles of MxK and KxN, respectively
* C = input tile of MxN and
* D = final output tile, MxN
* alpha, beta are scalar factors
* (M, N and K are block dimensions)
*
* In the simple_gemm sample, each warp is responsible for computing
* one output D tile of the final result. In the current sample, each
* warp is now responsible for computing multiple D tiles, what we
* might call a Warp Tile. Because Warp Tile blocks share data locality
* in either the same row or column direction, warps can re-use input
* data from A and B as they step through the K dimension for each block.
*
* Moreover, Warp Tiles processed by warps in a thread block
* have common locality in the larger Macro Tile. In the Global D layout
* shown below, data re-use opportunities await in D tiles aligned in the
* same rows / columns. These will pass over the same input A/B values as
* they march through the K dimension.
*
* Block size:      (BlockM x BlockN)
* Warp tile size:  (BlocksM * BlockSize.x) x (BlocksN * BlockSize.y)
* Macro Tile size: (TBlock.x / WarpSize * WarpTileSize.x) x (TBlock.y * WarpTileSize.y)
*
* Wave data share input A: same row
* Wave data share input B: same col
*
* Global D layout & warp assignment for BlocksM = BlocksN = 2, 2x2 Warps
*
* W (X, Y) = wave row X, col Y
*                                     |--------- Macro Tile N-------------|
*                                     |-- Wave Tile N --|
*                                     |-BlockN-|
*
*                                       BlockN x BlocksN   BlockN x BlocksN
*                                      |<--------------->|<--------------->|
*      _ _   _ _      _ _          ___  ________ ________ ________ ________
*       |     |        |            ^  |        |        |        |        |
*       | Wave| BlockM |   BlockM   |  |        W        |        W        |
*       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
*       |  M  |            BlocksM  |  |                 |                 |
* Macro |     |                     |  |                 |                 |
*  Tile |    _|_                   _v_ |________|________|________|________|
*   M   |                           ^  |        |        |        |        |
*       |                  BlockM   |  |        W        |        W        |
*       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
*       |                  BlocksM  |  |                 |                 |
*       |                           |  |                 |                 |
*      _|_                         _v_ |________|________|________|________|
*
*
* From the above diagram, we can see that input A/B data can be shared within warps,
* as well as between warps in the same threadblock. This means that warps in the same
* thread block can share the input loading responsibilities if they synchronize stepping
* through the K dimension for tiles at the same time.
*
* rocWMMA Cooperative API allows thread blocks to collaboratively move data from
* one location to another. In this case, we will move data from global memory space to
* local storage such that inter-warp data sharing is possible. Maximizing data re-use
* in this way reduces costly access to global memory and improves performance.
*
* To maximize efficiency, we can structure the kernel to maximize bandwidth usage and keep
* the compute resources as busy as possible at the same time. Using a pre-fetch technique,
* we can fetch A/B inputs for the next K-step while keeping the compute resources busy
* processing the current K-step. This helps to hide memory fetching latency.
*
* In general, the process would flow like the following:
*
*       Start
*         |
*   Pre-Fetch Global A/B for K0
*         |
*   Store LDS buffer0
*         |
*         v
*   Loop: i = 1:K-1
*   ^         |
*   |    Fetch Global A/B i+1; store LDS Buffer 1
*   |         |
*   |    Load LDS buffer0; Accum A x B
*   |         |
*   |    Swap buffer0, buffer1
*   |         |
*   |         |
*   end_loop <-
*         |
*   Load LDS buffer0; Accum A x B
*         |
*   Load Global C Tile
*         |
*   D = alpha * AccumAB + beta * C
*         |
*   Write D Tile
*         |
*         v
*        End
*
* Lds Mapping
* Buffer Width = LDS Width = BlockK
* Matrix geometry for inputs A and B have a common dimension (BlockK).
* We can fix one of the LDS dimensions to BlockK (in this case the width),
* and insert blocks of different heights (BlockM, BlockN) to use the space
* without the need of extra padding.
*
* Fragments of B must be transposed to fit this geometry,
* and both fragments from A and B must accomodate LDS data layout.
*
* Local Layout (LDS):
*
* Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
* of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
*
* Transposed B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
* Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
*
*
*                        _____________BlockK_____________
*                       |                                |
*                       v                                v
*                  (0,0) ----------------------------------->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockM |   | C0 | C1 | C2               | Ck-1 |   A0
*  Tile M  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...          AX-1
*          -->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
*  Tile N  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...        BY-1 (T)
*          -->                                           (MacroTileM + MacroTileN - 1, BlockK -1)
*
* Depending on the locality of the block being processed, warps load the corresponding
* A and B inputs from LDS buffer and use them for the accumulation of AxB calculations.
*/

using namespace rocwmma;

/* Depending on the GPU architecture this sample is run on, the following kernel parameters need to
*  be modified in order to obtain high performance.
* _________________________________________________________________________________________
*|         |           |           |           |          |          |          |          |
*|         | ROCWMMA_M | ROCWMMA_N | ROCWMMA_K | BLOCKS_X | BLOCKS_Y | TBLOCK_X | TBLOCK_Y |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_9  |    32     |    32     |    16     |    2     |    2     |   128    |    2     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_11 |    16     |    16     |    16     |    4     |    2     |    64    |    4     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*
* __________________________________________
*|         |                                |
*|         |           WARP_SIZE            |
*|_________|________________________________|
*|         |                                |
*|  GFX_9  | Constants::AMDGCN_WAVE_SIZE_64 |
*|_________|________________________________|
*|         |                                |
*|  GFX_11 | Constants::AMDGCN_WAVE_SIZE_32 |
*|_________|________________________________|
*/

namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 32u,
        ROCWMMA_N = 32u,
        ROCWMMA_K = 16u,
        BLOCKS_M  = 2u,
        BLOCKS_N  = 2u,
        TBLOCK_X  = 128u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_M  = 4u,
        BLOCKS_N  = 2u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif // defined(ROCWMMA_ARCH_GFX9)

// Types and Data Layouts
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

// Warp tile: computed by each warp
constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;
constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;
constexpr uint32_t WARP_TILE_K = ROCWMMA_K;

// Macro Tile: computed by each thread block (workgroup)
// Note: TBLOCK_M must be multiple of WARP_SIZE.
constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_N      = TBLOCK_Y;
constexpr uint32_t WARP_COUNT   = WARPS_M * WARPS_N;
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M;
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

// Mma frags (warp tile), non-cooperative
using MmaFragA = fragment<matrix_a, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutA>;
using MmaFragB = fragment<matrix_b, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, InputT, DataLayoutB>;
using MmaFragC = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, OutputT, DataLayoutC>;
using MmaFragD = MmaFragC;
using MmaFragAcc = fragment<accumulator, WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, ComputeT>;

// Global read (macro tile), cooperative
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
using GRFragA       = fragment<matrix_a,
                         MACRO_TILE_M,
                         MACRO_TILE_N,
                         MACRO_TILE_K,
                         InputT,
                         DataLayoutA,
                         CoopScheduler>;
using GRFragB       = fragment<matrix_b,
                         MACRO_TILE_M,
                         MACRO_TILE_N,
                         MACRO_TILE_K,
                         InputT,
                         DataLayoutB,
                         CoopScheduler>;

// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

// Transform helpers
ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& grFragA) {
    return apply_data_layout<DataLayoutLds>(grFragA);
}

ROCWMMA_DEVICE auto transformGRFragBToLWFragB(GRFragB const& grFragB) {
    return apply_data_layout<DataLayoutLds>(apply_transpose(grFragB));
}

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, DataLayoutLds>;

// Transform helpers
ROCWMMA_DEVICE auto transformLRFragAToMmaFragA(LRFragA const& lrFragA) {
    return apply_data_layout<DataLayoutA>(lrFragA);
}

ROCWMMA_DEVICE auto transformLRFragBToMmaFragB(LRFragB const& lrFragB) {
    return apply_data_layout<DataLayoutB>(apply_transpose(lrFragB));
}

ROCWMMA_KERNEL void __launch_bounds__(256) gemm_rocwmma_d(uint32_t       m,
                                                          uint32_t       n,
                                                          uint32_t       k,
                                                          InputT const*  a,
                                                          InputT const*  b,
                                                          OutputT const* c,
                                                          OutputT*       d,
                                                          uint32_t       lda,
                                                          uint32_t       ldb,
                                                          uint32_t       ldc,
                                                          uint32_t       ldd,
                                                          ComputeT       alpha,
                                                          ComputeT       beta)
{
    // Tile Sizes
    constexpr auto warpTileSize  = make_coord2d(WARP_TILE_M, WARP_TILE_N);
    constexpr auto macroTileSize = make_coord2d(MACRO_TILE_M, MACRO_TILE_N);

    // Local warp coordinate relative to current threadblock (wg).
    constexpr auto warpDims        = make_coord2d(WARPS_M, WARPS_N);
    auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
    auto           localWarpOffset = localWarpCoord * warpTileSize;

    // Global matrix coordinates for C/D
    auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
    auto warpTileCoord  = macroTileCoord + localWarpOffset;

    // Bounds check
    auto warpTileBound = warpTileCoord + warpTileSize;
    if(get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
    {
        return;
    }

    // 1D global read coordinate transform
    using GRFragAMap1d = GetDataLayout_t<GRFragA>;
    using GRFragBMap1d = GetDataLayout_t<GRFragB>;

    // Initial global read address offsets
    auto globalReadOffsetA
        = GRFragAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
    auto globalReadOffsetB
        = GRFragBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

    // Incremental global read address offsets
    auto kStepOffsetA = GRFragAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepOffsetB = GRFragBMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    // Perform initial global pre-fetch
    GRFragA grFragA;
    GRFragB grFragB;

    load_matrix_sync(grFragA, a + globalReadOffsetA, lda);
    load_matrix_sync(grFragB, b + globalReadOffsetB, ldb);

    globalReadOffsetA += kStepOffsetA;
    globalReadOffsetB += kStepOffsetB;

    // Setup LDS addressing
    // This kernel will use 2 separate LDS blocks for pipelining
    // the input prefetching during the accumulation loop
    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    using LWFragAShape = GetIOShape_t<LWFragA>;
    using LWFragBShape = GetIOShape_t<LWFragB>;
    using LWFragAMap1d = GetDataLayout_t<LWFragA>;
    using LWFragBMap1d = GetDataLayout_t<LWFragB>;

    constexpr uint32_t ldsWidth  = MACRO_TILE_K;
    constexpr uint32_t ldsHeight = LWFragAShape::BlockHeight + LWFragBShape::BlockHeight;
    constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
    constexpr uint32_t ldsld     = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    // Local write offsets to start of A / B data
    auto ldsWriteOffsetA = 0u;
    auto ldsWriteOffsetB
        = LWFragAMap1d::fromMatrixCoord(make_coord2d(LWFragAShape::BlockHeight, 0u), ldsld);

    // Local read offsets for mma frags
    auto ldsReadOffsetA
        = ldsWriteOffsetA
          + LWFragAMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
    auto ldsReadOffsetB
        = ldsWriteOffsetB
          + LWFragBMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

    // Write prefetch to local
    store_matrix_sync(ldsPtrLo + ldsWriteOffsetA, transformGRFragAToLWFragA(grFragA), ldsld);
    store_matrix_sync(ldsPtrLo + ldsWriteOffsetB, transformGRFragBToLWFragB(grFragB), ldsld);

    // Initialize accumulation frags
    MmaFragAcc mmaFragAcc;
    fill_fragment(mmaFragAcc, 0.0f);

    // Synchronize states
    synchronize_workgroup();

    // Accumulate A * B for all mfma frags in warp tile
    for(uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
    {
        // Local read mma frags from first LDS buffer.
        LRFragA lrFragA;
        LRFragB lrFragB;
        load_matrix_sync(lrFragA, ldsPtrLo + ldsReadOffsetA, ldsld);
        load_matrix_sync(lrFragB, ldsPtrLo + ldsReadOffsetB, ldsld);

        // Prefetch next round of global frags
        load_matrix_sync(grFragA, a + globalReadOffsetA, lda);
        load_matrix_sync(grFragB, b + globalReadOffsetB, ldb);

        // Advance offsets to next k step
        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        // Matrix mult-accum(A * B)
        mma_sync(mmaFragAcc,
                 transformLRFragAToMmaFragA(lrFragA),
                 transformLRFragBToMmaFragB(lrFragB),
                 mmaFragAcc);

        // Write prefetch to second LDS buffer
        store_matrix_sync(ldsPtrHi + ldsWriteOffsetA, transformGRFragAToLWFragA(grFragA), ldsld);
        store_matrix_sync(ldsPtrHi + ldsWriteOffsetB, transformGRFragBToLWFragB(grFragB), ldsld);

        // Make sure that all waves have finished reading / writing to lds for currentK.
        synchronize_workgroup();

        // Swap Lds buffers
        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }

    // Start loading C
    using MmaFragCMap1d = GetDataLayout_t<MmaFragC>;
    using MmaFragDMap1d = GetDataLayout_t<MmaFragD>;

    MmaFragC mmaFragC;
    load_matrix_sync(mmaFragC, c + MmaFragCMap1d::fromMatrixCoord(warpTileCoord, ldc), ldc);

    // Clean up tail A * B
    LRFragA lrFragA;
    LRFragB lrFragB;
    load_matrix_sync(lrFragA, ldsPtrLo + ldsReadOffsetA, ldsld);
    load_matrix_sync(lrFragB, ldsPtrLo + ldsReadOffsetB, ldsld);
    mma_sync(mmaFragAcc,
             transformLRFragAToMmaFragA(lrFragA),
             transformLRFragBToMmaFragB(lrFragB),
             mmaFragAcc);

    // D = alpha * accum + beta * C

    // MmaFragD is a wave tile that can be large.
    // Unroll the operation in chunks to save registers.
    MmaFragD           mmaFragD;
    constexpr uint32_t chunkSize = 8u;
    constexpr uint32_t chunks    = mmaFragD.num_elements / chunkSize;
    constexpr uint32_t remain    = mmaFragD.num_elements % chunkSize;

    auto uniform_fma_unroll_chunk = [&](uint32_t start_idx, uint32_t size) {
        for(int i = start_idx; i < start_idx + size; i++)
        {
            mmaFragD.x[i] = static_cast<OutputT>(alpha * mmaFragAcc.x[i]
                                                 + beta * static_cast<ComputeT>(mmaFragC.x[i]));
        }
    };

    for(uint32_t c = 0u; c < chunks; c++)
    {
        uniform_fma_unroll_chunk(c * chunkSize, chunkSize);
    }
    uniform_fma_unroll_chunk(chunks * chunkSize, remain);

    // Final store of completed warp tile
    store_matrix_sync(d + MmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd), mmaFragD, ldd);
}

ROCWMMA_HOST void gemm_test(uint32_t m, uint32_t n, uint32_t k, ComputeT alpha, ComputeT beta)
{
    // Runtime checks for host parameters
    uint32_t hTBLOCK_X    = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
    uint32_t hTBLOCK_Y    = isGfx9() ? gfx9Params::TBLOCK_Y : gfx11Params::TBLOCK_Y;
    uint32_t hBLOCKS_M    = isGfx9() ? gfx9Params::BLOCKS_M : gfx11Params::BLOCKS_M;
    uint32_t hBLOCKS_N    = isGfx9() ? gfx9Params::BLOCKS_N : gfx11Params::BLOCKS_N;
    uint32_t hROCWMMA_M   = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
    uint32_t hROCWMMA_N   = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
    uint32_t hROCWMMA_K   = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
    uint32_t hWARP_TILE_M = hBLOCKS_M * hROCWMMA_M;
    uint32_t hWARP_TILE_N = hBLOCKS_N * hROCWMMA_N;

    // Runtime warp calculation (host code needs to query warpsize dynamically)
    auto warpSize = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_M, hTBLOCK_Y * hWARP_TILE_N);

    // Device check for supported block and wave sizes
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if((isGfx11() || isGfx12()) && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if(isGfx9() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    // Bounds check
    if((m < get<0>(macroTileSize) || n < get<1>(macroTileSize) || k < hROCWMMA_K)
       || (m % hROCWMMA_M || n % hROCWMMA_N || k % hROCWMMA_K))
    {
        std::cout << "Unsupported matrix size!\n";
        return;
    }

    // Layouts leading dims
    int lda = std::is_same_v<DataLayoutA, row_major> ? k : m;
    int ldb = std::is_same_v<DataLayoutB, row_major> ? n : k;
    int ldc = std::is_same_v<DataLayoutC, row_major> ? n : m;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n);

    // Fill outputs with NaN to catch contamination
    std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_c;
    OutputT* d_d;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesC = matrixC.size() * sizeof(OutputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, get<0>(macroTileSize)),
                        rocwmma::ceil_div(n, get<1>(macroTileSize)));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "gridDim (" << gridDim.x << " " << gridDim.y << ")"
              << " blockdim (" << blockDim.x << " " << blockDim.y << ")" << std::endl;

    // Uses 2 lds blocks for prefetch loop (A and B)
    int ldsusage
        = 2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * hROCWMMA_K;

    // Kernel caller
    auto rocwmmaKernel = [&]() {
        hipExtLaunchKernelGGL(gemm_rocwmma_d,
                              gridDim,
                              blockDim,
                              ldsusage,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              m,
                              n,
                              k,
                              d_a,
                              d_b,
                              d_c,
                              d_d,
                              lda,
                              ldb,
                              ldc,
                              ldd,
                              alpha,
                              beta);
    };

    constexpr uint32_t warmups    = 2u;
    constexpr uint32_t recordRuns = 5u;

    // Warm-up runs, not recorded
    for(uint32_t i = 0; i < warmups; ++i)
    {
        rocwmmaKernel();
    }

    // Actual recorded runs
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < recordRuns; ++i)
    {
        rocwmmaKernel();
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    auto gFlops = calculateGFlops(m, n, k);
    auto tFlopsPerSec
        = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs), recordRuns);

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << std::left << std::setw(8) << "TBlockX" << std::setw(8) << "TBlockY" << std::setw(8)
              << "BlocksM" << std::setw(8) << "BlocksN" << std::setw(6) << "BlkM" << std::setw(6)
              << "BlkN" << std::setw(6) << "BlkK" << std::setw(8) << "MatM" << std::setw(8)
              << "MatN" << std::setw(8) << "MatK" << std::setw(8) << "alpha" << std::setw(8)
              << "lda" << std::setw(8) << "ldb" << std::setw(8) << "beta" << std::setw(8) << "ldc"
              << std::setw(8) << "ldd" << std::setw(13) << "elapsedMs" << std::setw(23)
              << "Problem Size(GFlops)" << std::setw(10) << "TFlops/s" << std::endl;

    std::cout << std::left << std::setw(8) << hTBLOCK_X << std::setw(8) << hTBLOCK_Y << std::setw(8)
              << hBLOCKS_M << std::setw(8) << hBLOCKS_N << std::setw(6) << hROCWMMA_M
              << std::setw(6) << hROCWMMA_N << std::setw(6) << hROCWMMA_K << std::setw(8) << m
              << std::setw(8) << n << std::setw(8) << k << std::setw(8) << alpha << std::setw(8)
              << lda << std::setw(8) << ldb << std::setw(8) << beta << std::setw(8) << ldc
              << std::setw(8) << ldd << std::setw(13) << elapsedTimeMs << std::setw(23) << gFlops
              << std::setw(10) << tFlopsPerSec << std::endl;

#if !NDEBUG

    std::cout << "Validating result with reference..." << std::endl;

    if((uint64_t)m * (uint64_t)n * (uint64_t)k > (2048ull * 2048ull * 2048ull))
    {
        std::cout << "Please wait. Large sizes can take a while!" << std::endl;
    }

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<OutputT> matrixD_ref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
    gemm_cpu_h<InputT, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutC>(m,
                                                                                 n,
                                                                                 k,
                                                                                 matrixA.data(),
                                                                                 matrixB.data(),
                                                                                 matrixC.data(),
                                                                                 matrixD_ref.data(),
                                                                                 lda,
                                                                                 ldb,
                                                                                 ldc,
                                                                                 ldd,
                                                                                 alpha,
                                                                                 beta);

    auto res = compareEqual(matrixD.data(), matrixD_ref.data(), m * n);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED\n";
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    gemm_test(7168, 7168, 7168, 2, 2);
    return 0;
}
