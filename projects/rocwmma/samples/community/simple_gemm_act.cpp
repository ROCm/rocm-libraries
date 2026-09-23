/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
* This GEMM kernel implements a high-performance matrix multiplication with SiLU activation function:
* D = silu(A * B)
*
* Where:
* A, B = input tiles of MxK and KxN, respectively
* D = output matrix of dimension MxN with SiLU activation applied, MxN
* (M, N and K are block dimensions)
* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

* High performance is achieved through:
* 1) Data re-use
* 2) Latency hiding
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
* Key kernel features:
* - Uses rocWMMA for efficient matrix operations on AMD GPUs
* - Implements double-buffering to overlap computation with memory operations
* - Supports int8 inputs with int32 accumulation and int8 outputs
* - Applies SiLU activation with proper saturation handling
*
* Block size:      (BlockM x BlockN)
* Warp tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
* Macro Tile size: (TBlock.x * WarpTileSize.x) x (TBlock.y * WarpTileSize.y)
*
* Wave data share input A: same row
* Wave data share input B: same col
*
* Global D layout & warp assignment for BlocksX = BlocksY = 2, 2x2 Warps
*
* W (X, Y) = wave row X, col Y
*                                     |--------- Macro Tile Y-------------|
*                                     |-- Wave Tile Y --|
*                                     |-BlockN-|
*
*                                      BlockN x BlocksY   BlockN x BlocksY
*                                     |<--------------->|<--------------->|
*      _ _   _ _      _ _          ___  ________ ________ ________ ________
*       |     |        |            ^  |        |        |        |        |
*       | Wave| BlockM |   BlockM   |  |        W        |        W        |
*       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
*       |  X  |            BlocksX  |  |                 |                 |
* Macro |     |                     |  |                 |                 |
*  Tile |    _|_                   _v_ |________|________|________|________|
*   X   |                           ^  |        |        |        |        |
*       |                  BlockM   |  |        W        |        W        |
*       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
*       |                  BlocksX  |  |                 |                 |
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
*   D = Accum(A*B)
*         |
*   Do Silu Operation on Acc
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
*  Tile X  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...          AX-1
*          -->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
*  Tile Y  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...        BY-1 (T)
*          -->                                           (MacroTileX + MacroTileY - 1, BlockK -1)
*
* Depending on the locality of the block being processed, warps load the corresponding
* A and B inputs from LDS buffer and use them for the accumulation of AxB calculations.
*/

using namespace rocwmma;

///
/// Parameter configuration
///

/* Depending on the GPU architecture this sample is run on, the following kernel parameters need to
*  be modified in order to obtain high performance.
* _________________________________________________________________________________________
*|         |           |           |           |          |          |          |          |
*|         | ROCWMMA_M | ROCWMMA_N | ROCWMMA_K | BLOCKS_X | BLOCKS_Y | TBLOCK_X | TBLOCK_Y |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_9  |    16     |    16     |    32     |    2     |    2     |   128    |    2     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*
* __________________________________________
*|         |                                |
*|         |           WARP_SIZE            |
*|_________|________________________________|
*|         |                                |
*|  GFX_9  | Constants::AMDGCN_WAVE_SIZE_64 |
*|_________|________________________________|
*/

namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 32u,
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 128u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

using namespace gfx9Params;

///
/// Types and Data Layouts
///

using InputT      = int8_t;
using OutputT     = int8_t;
using ComputeT    = int32_t;
using ActComputeT = float;

using DataLayoutA   = row_major;
using DataLayoutB   = col_major;
using DataLayoutD   = row_major;
using DataLayoutLds = col_major;

///
/// Fragment types
///

// Warp tile: computed by each warp
constexpr uint32_t WARP_TILE_X = BLOCKS_X * ROCWMMA_M;
constexpr uint32_t WARP_TILE_Y = BLOCKS_Y * ROCWMMA_N;

constexpr uint32_t els_per_thread  = ROCWMMA_M * ROCWMMA_N / WARP_SIZE;
constexpr uint32_t threads_per_row = ROCWMMA_N / els_per_thread;

// Macro Tile: computed by each thread block (workgroup)
// Note: TBLOCK_X must be multiple of WARP_SIZE.
constexpr uint32_t WARPS_X      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WARPS_X * WARP_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WARPS_Y * WARP_TILE_Y;

// Mfma frags
using MfmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
using MfmaFragD   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutD>;
using MfmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT, DataLayoutD>;
using MfmaFragAct = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutD>;

// Global read (macro tile)
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffA
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;
using GRBuffB
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB, CoopScheduler>;

// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWBuffA = apply_data_layout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = apply_data_layout_t<apply_transpose_t<GRBuffB>, DataLayoutLds>;

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragA = apply_data_layout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MfmaFragB>, DataLayoutLds>;

///
/// Wrapper functions: repeat mfma tile operations across entire warp tile.
///

// Cooperative global read / local write (Macro tile data movement)
// Loads / stores a global data fragment cooperatively across warps. Each participating warp is
// responsible for only a portion of the whole fragment.
//
// The cooperative operation is split into work items (SplitCount). Work items are consumed in
// a round robin fashion by warps in the range of [0, WaveCount). The wave index determines the
// order of the current wave in the collaboration pool.
//
// WaveCount, SplitCount and waveIndex parameters must match successive coop load / store calls
// to ensure the entire fragment remains coherent.

// Global A reads in cooperative mode (macro tile)
ROCWMMA_DEVICE static inline void
    globalReadCoopA(GRBuffA& grBuffA, InputT const* gAddrA, uint32_t lda)
{
    load_matrix_sync(grBuffA, gAddrA, lda);
}

// Global B reads in cooperative mode (macro tile)

ROCWMMA_DEVICE static inline void
    globalReadCoopB(GRBuffB& grBuffB, InputT const* gAddrB, uint32_t ldb)
{
    load_matrix_sync(grBuffB, gAddrB, ldb);
}

// Local A writes in cooperative mode (macro tile)
ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& grBuffA, uint32_t ldsld)
{
    // No transpose, but apply the lds data layout
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(grBuffA), ldsld);
}

// Local B writes in cooperative mode (macro tile)
ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& grBuffB, uint32_t ldsld)
{
    // Transpose B and then apply lds data layout
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(grBuffB)), ldsld);
}

// Local A reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadA(MfmaFragA (&fragsA)[BLOCKS_X], InputT const* ldsAddrA, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragA>;
    using Mapper1d  = GetDataLayout_t<LRFragA>;

    // Each A block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragA tmp;
        load_matrix_sync(tmp, ldsAddrA, ldsld);
        fragsA[i] = apply_data_layout<DataLayoutA>(tmp);

        ldsAddrA += blockStep;
    }
}

// Local B reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadB(MfmaFragB (&fragsB)[BLOCKS_Y], InputT const* ldsAddrB, uint32_t ldsld)
{
    using FragShape = GetIOShape_t<LRFragB>;
    using Mapper1d  = GetDataLayout_t<LRFragB>;

    // Each B block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);

#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragB tmp;
        load_matrix_sync(tmp, ldsAddrB, ldsld);

        // Transform back to MFMA tile
        fragsB[i] = apply_data_layout<DataLayoutB>(apply_transpose(tmp));

        ldsAddrB += blockStep;
    }
}

// Global D reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    globalWriteD(OutputT* gAddrD, MfmaFragD const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using FragShape = GetIOShape_t<MfmaFragD>;
    using Mapper1d  = GetDataLayout_t<MfmaFragD>;

    // Iterative offsets for each D block in the warp tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrD + offsetY, fragsD[i][j], ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

// Broadcast value to fragments in warp tile
template <typename FragT>
ROCWMMA_DEVICE static inline void fill(FragT (&frags)[BLOCKS_X][BLOCKS_Y],
                                       GetDataType_t<FragT> value)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            fill_fragment(frags[i][j], value);
        }
    }
}

// Performs warp tile mfma
ROCWMMA_DEVICE static inline void mfma(MfmaFragAcc (&fragsAccOut)[BLOCKS_X][BLOCKS_Y],
                                       MfmaFragA const (&fragsA)[BLOCKS_X],
                                       MfmaFragB const (&fragsB)[BLOCKS_Y],
                                       MfmaFragAcc const (&fragsAccIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            mma_sync(fragsAccOut[i][j], fragsA[i], fragsB[j], fragsAccIn[i][j]);
        }
    }
}

ROCWMMA_DEVICE static inline void DoSilu(MfmaFragAcc (&frags_acc)[BLOCKS_X][BLOCKS_Y],
                                         MfmaFragAct (&frags_act)[BLOCKS_X][BLOCKS_Y],
                                         bool apply_silu)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; ++i)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; ++j)
        {
#pragma unroll
            for(int k = 0; k < frags_acc[i][j].num_elements; ++k)
            {
                ComputeT    data  = frags_acc[i][j].x[k];
                ActComputeT value = static_cast<ActComputeT>(data);

                if(apply_silu)
                {
                    if(data >= 0)
                    {
                        ActComputeT exp_negx = std::expf(-data);
                        value                = data * (ComputeT(1) / (ComputeT(1) + exp_negx));
                    }
                    else
                    {
                        ActComputeT exp_x = std::expf(data);
                        value             = data * (exp_x / (ComputeT(1) + exp_x));
                    }
                }

                if(value > INT8_MAX)
                {
                    frags_act[i][j].x[k] = INT8_MAX;
                }
                else if(value < INT8_MIN)
                {
                    frags_act[i][j].x[k] = INT8_MIN;
                }
                else
                {
                    frags_act[i][j].x[k] = static_cast<OutputT>(value);
                }
            }
        }
    }
}

ROCWMMA_KERNEL void __launch_bounds__(256) gemm_rocwmma_d(uint32_t      m,
                                                          uint32_t      n,
                                                          uint32_t      k,
                                                          InputT const* a,
                                                          InputT const* b,
                                                          OutputT*      d,
                                                          uint32_t      lda,
                                                          uint32_t      ldb,
                                                          uint32_t      ldd,
                                                          bool          apply_silu)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        ///
        /// 2D matrix coordinate setup
        ///

        // Tile Sizes
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y); //(64,64)
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y); //(128,128)

        // Local warp coordinate relative to current threadblock (wg).
        constexpr auto warpDims        = make_coord2d(WARPS_X, WARPS_Y); //(2,2)
        auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto           localWarpOffset = localWarpCoord * warpTileSize;

        const int iterations = (n + MACRO_TILE_Y - 1) / MACRO_TILE_Y;
        for(int iter = 0; iter < iterations; iter++)
        {
            // Global matrix coordinates for C/D
            auto macroTileCoord = make_coord2d(blockIdx.x, iter) * macroTileSize;
            auto warpTileCoord  = macroTileCoord + localWarpOffset;

            // Bounds check
            auto warpTileBound = warpTileCoord + warpTileSize;
            if(get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
            {
                return;
            }

            ///
            /// 1D global read coordinate setup
            ///
            using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
            using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

            // Initial global read address offsets
            auto globalReadOffsetA
                = GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
            auto globalReadOffsetB
                = GRBuffBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

            // Incremental global read address offsets
            auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, ROCWMMA_K), lda);
            auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(ROCWMMA_K, 0u), ldb);

            ///
            /// Cooperative config for global read A / B
            ///

            // WorkItems will be split up by minimum IOCount to perform either global read or local write.
            // These are inputs to cooperative functions.
            constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);

            // Scheduling warp order is analogous to row major priority.
            // E.g. Wg = (128, 2) = 2x2 warps
            // (0, 0)   (0, 1)   Share Schedule: w0 = (0, 0), w1 = (0, 1),
            // (1, 0)   (1, 1)                   w2 = (1, 0), w3 = (1, 1), count = 4
            const auto warpIndex
                = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

            ///
            /// Perform initial global pre-fetch
            ///

            GRBuffA grBuffA;
            GRBuffB grBuffB;

            globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
            globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);

            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            ///
            /// Setup LDS addressing
            /// This kernel will use 2 separate LDS blocks for pipelining
            /// the input prefetching during the accumulation loop
            ///

            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            using LWBuffAShape = GetIOShape_t<LWBuffA>;
            using LWBuffBShape = GetIOShape_t<LWBuffB>;
            using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
            using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;

            constexpr uint32_t ldsWidth  = ROCWMMA_K;
            constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
            constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
            constexpr uint32_t ldsld
                = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

            auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + sizeLds;

            // Local write offsets to start of A / B data
            auto ldsWriteOffsetA = 0u;
            auto ldsWriteOffsetB
                = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

            // Local read offsets for mfma frags
            auto ldsReadOffsetA
                = ldsWriteOffsetA
                  + LWBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
            auto ldsReadOffsetB
                = ldsWriteOffsetB
                  + LWBuffBMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

            ///
            /// Write prefetch to local
            ///
            localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld);
            localWriteCoopB(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld);

            ///
            /// Initialize accumulation frags
            ///
            MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
            fill(fragsAcc, 0.0f);

            ///
            /// Synchronize warps and memory
            ///
            synchronize_workgroup();

            ///
            /// Accumulate A * B for all mfma frags in warp tile
            ///
            for(uint32_t currentK = ROCWMMA_K; currentK < k; currentK += ROCWMMA_K)
            {
                MfmaFragA fragsA[BLOCKS_X];
                MfmaFragB fragsB[BLOCKS_Y];

                // Local read mfma frags from first LDS buffer
                localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
                localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

                // Prefetch next round of global frags
                globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
                globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);

                // Advance offsets to next k step
                globalReadOffsetA += kStepOffsetA;
                globalReadOffsetB += kStepOffsetB;

                // accum(A * B)
                mfma(fragsAcc, fragsA, fragsB, fragsAcc);

                // Write prefetch to second LDS buffer
                localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld);
                localWriteCoopB(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld);

                // Make sure that all waves have finished reading / writing to lds for currentK.
                synchronize_workgroup();

                // Swap Lds buffers
                auto* tmp = ldsPtrLo;
                ldsPtrLo  = ldsPtrHi;
                ldsPtrHi  = tmp;
            }

            using MfmaFragDMap1d = GetDataLayout_t<MfmaFragD>;

            ///
            /// Clean up tail A * B
            ///
            MfmaFragA fragsA[BLOCKS_X];
            MfmaFragB fragsB[BLOCKS_Y];

            // Local read mfma frags
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
            mfma(fragsAcc, fragsA, fragsB, fragsAcc);

            // Do Silu on fragsAcc Result and processing datatype convertion
            MfmaFragAct fragsAct[BLOCKS_X][BLOCKS_Y];
            DoSilu(fragsAcc, fragsAct, apply_silu);

            globalWriteD(d + MfmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd), fragsAct, ldd);
        }
    }
}

template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void gemm_act_cpu(uint32_t      m,
                           uint32_t      n,
                           uint32_t      k,
                           InputT const* a,
                           InputT const* b,
                           OutputT*      d,
                           uint32_t      lda,
                           uint32_t      ldb,
                           uint32_t      ldd,
                           bool          apply_silu = false)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, rocwmma::row_major>::value ? rowMjr : colMjr;

    auto silu = [](ComputeT x) -> ActComputeT {
        if(x >= 0)
        {
            ActComputeT exp_negx = std::expf(-x);
            return x * (ComputeT(1) / (ComputeT(1) + exp_negx));
        }
        else
        {
            ActComputeT exp_x = std::expf(x);
            return x * (exp_x / (ComputeT(1) + exp_x));
        }
    };

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j)
        {
            ComputeT    accum     = static_cast<ComputeT>(0);
            for(int h = 0; h < k; ++h)
            {
                accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                         * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
            }
            ActComputeT act_value = static_cast<ActComputeT>(accum);

            if(apply_silu)
            {
                act_value = silu(accum);
            }

            if(act_value > INT8_MAX)
            {
                d[dIndex(i, j, ldd)] = INT8_MAX;
            }
            else if(act_value < INT8_MIN)
            {
                d[dIndex(i, j, ldd)] = INT8_MIN;
            }
            else
            {
                d[dIndex(i, j, ldd)] = static_cast<OutputT>(act_value);
            }
        }
    }
}

ROCWMMA_HOST void gemm_test(uint32_t m, uint32_t n, uint32_t k)
{
    // Runtime checks for host parameters
    uint32_t hTBLOCK_X    = gfx9Params::TBLOCK_X;
    uint32_t hTBLOCK_Y    = gfx9Params::TBLOCK_Y;
    uint32_t hBLOCKS_X    = gfx9Params::BLOCKS_X;
    uint32_t hBLOCKS_Y    = gfx9Params::BLOCKS_Y;
    uint32_t hROCWMMA_M   = gfx9Params::ROCWMMA_M;
    uint32_t hROCWMMA_N   = gfx9Params::ROCWMMA_N;
    uint32_t hROCWMMA_K   = gfx9Params::ROCWMMA_K;
    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    // Runtime warp calculation (host code needs to query warpsize dynamically)
    auto warpSize = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

    // Device check for supported block and wave sizes
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && ((hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32)))
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
    int  lda        = std::is_same_v<DataLayoutA, row_major> ? k : m;
    int  ldb        = std::is_same_v<DataLayoutB, row_major> ? n : k;
    int  ldd        = std::is_same_v<DataLayoutD, row_major> ? n : m;
    bool apply_silu = true;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<InputT> matrixA(m * k);
    std::vector<InputT> matrixB(k * n);

    // Fill outputs with NaN to catch contamination
    std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_d;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, get<0>(macroTileSize)));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "gridDim (" << gridDim.x << " " << gridDim.y << ")" << " blockDim (" << blockDim.x
              << " " << blockDim.y << ")" << std::endl;

    // Uses 2 lds blocks for prefetch loop (A and B)
    int ldsusage
        = max(2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * hROCWMMA_K,
              sizeof(ComputeT) * get<0>(macroTileSize) * get<1>(macroTileSize));

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
                              d_d,
                              lda,
                              ldb,
                              ldd,
                              apply_silu);
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
    std::cout << "TBlockX, TBlockY, " << "BlocksX, BlocksY, " << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, " << "lda, ldb, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << "," << "sizeof(InputT)"
              << std::endl;

    std::cout << hTBLOCK_X << ", " << hTBLOCK_Y << ", " << hBLOCKS_X << ", " << hBLOCKS_Y << ", "
              << hROCWMMA_M << ", " << hROCWMMA_N << ", " << hROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << lda << ", " << ldb << ", " << ldd << ", "
              << elapsedTimeMs << ", " << gFlops << ", " << tFlopsPerSec << "," << sizeof(InputT)
              << std::endl;

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
    gemm_act_cpu<InputT, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutD>(
        m, n, k, matrixA.data(), matrixB.data(), matrixD_ref.data(), lda, ldb, ldd, apply_silu);

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
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    if(isGfx9())
        gemm_test(128, 128, 64);
    else
        std::cout << "This sample not tested on gfx11/gfx12 yet!" << std::endl;
    return 0;
}
