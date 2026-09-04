/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/* COMMUNITY SAMPLE: Mixture-of-Experts Grouped GEMM
 *
 * ============================================================================
 * 1. WHAT IS MoE GROUPED GEMM?
 * ============================================================================
 *
 * Mixture-of-Experts (MoE) is a sparse architecture used in Mixtral-8x7B
 * (Jiang et al., 2024) and DeepSeek-MoE (Dai et al., 2024).  A gating
 * network routes each token to its top-K experts, producing a set of
 * variable-size GEMMs -- one per expert -- that must all be executed.
 *
 * References:
 *   [1] Jiang et al., "Mixtral of Experts", 2024
 *       https://arxiv.org/abs/2401.04088
 *   [2] Dai et al., "DeepSeekMoE", 2024
 *       https://arxiv.org/abs/2401.06066
 *
 * ============================================================================
 * 2. MATHEMATICAL FORMULATION
 * ============================================================================
 *
 *   Tokens are pre-sorted by expert assignment.  Expert e owns tokens
 *   in the range [expert_offsets[e], expert_offsets[e+1]).
 *
 *   For expert e with M_e tokens:
 *     Y_e = X_e * W_e     [M_e x N] = [M_e x K] x [K x N]
 *
 *   where X_e is the token sub-matrix, W_e is expert e's weight matrix.
 *
 * ============================================================================
 * 3. WHY GROUP INTO A SINGLE KERNEL?
 * ============================================================================
 *
 * A naive implementation launches E separate GEMMs.  This sample groups
 * them into ONE kernel launch using a work-list scheduler:
 *
 *   a) Single launch -- eliminates E-1 extra launch barriers.
 *   b) Better occupancy -- many small tiles from different experts fill
 *      the GPU better than sequential large launches.
 *   c) Work-list scheduling -- a host-built table of (expert_id, tile_m,
 *      tile_n) triples lets each CTA pick its work independently.
 *
 * ============================================================================
 * 4. WHAT YOU WILL LEARN FROM THIS SAMPLE
 * ============================================================================
 *
 *   - How to implement work-list-based tile scheduling for grouped GEMMs
 *   - How to use CSR-style expert_offsets for variable-size batching
 *   - Standard rocWMMA double-buffered K-loop (same as SwiGLU sample)
 *   - How to reuse a single kernel for heterogeneous matrix sizes
 *
 * ============================================================================
 * 5. KERNEL DATA-FLOW OVERVIEW
 * ============================================================================
 *
 *   Host: build work_list[(expert_id, tile_m, tile_n)]
 *         |
 *         v
 *   Kernel: CTA reads work_list[blockIdx.x]
 *         |
 *         +-- expert_offsets[eid] -> token start row
 *         +-- expert_weights + eid*K*N -> weight base
 *         |
 *         v
 *   Global Memory --coop load--> LDS (2 segments, double-buffered)
 *        |                            |
 *      X [tokens]              [A   segment]
 *      W [weights]             [B^T segment]
 *                                     |
 *                               local read
 *                                     |
 *                    fragsA  ----+---- fragsB
 *                                mma
 *                                 |
 *                              accOut
 *                                 |
 *                            cast to f16
 *                                 |
 *                          Y [output] --> Global Memory
 *
 * ============================================================================
 * 6. DATA LAYOUTS
 * ============================================================================
 *
 *   tokens        : row_major [total_tokens x K], ld = K
 *   expert_weights: row_major [num_experts x K x N], per-expert ld = N
 *   output        : row_major [total_tokens x N], ld = N
 *
 * ============================================================================
 * 7. LDS LAYOUT (col_major, one ping-pong buffer)
 * ============================================================================
 *
 *   Two segments stacked vertically in col_major order:
 *
 *     Segment | Height         | Content
 *     --------+----------------+---------
 *        A    | MACRO_TILE_X   | token tile
 *        B    | MACRO_TILE_Y   | weight^T tile
 *
 *     Width  = MACRO_TILE_K (16)
 *     Height = MACRO_TILE_X + MACRO_TILE_Y = 128
 *     ldsld  = Height
 *     1 buffer = 128 * 16 = 2048 elems = 4 KiB
 *     Total (Lo + Hi) = 8 KiB
 *
 * ============================================================================
 * 8. REQUIREMENTS
 * ============================================================================
 *
 *   - Minimum ROCm version: ROCm 6.0+
 *   - GPU architectures: gfx9 / gfx11 / gfx12
 *       (tested on RDNA4 gfx1201 / RX9070)
 *   - Data types: float16 input, float32 compute, float16 output
 *   - Per-expert M must be a multiple of MACRO_TILE_X (64)
 *   - N must be a multiple of MACRO_TILE_Y (64)
 *   - K must be a multiple of ROCWMMA_K (16)
 *
 * ============================================================================
 * 9. KNOWN LIMITATIONS
 * ============================================================================
 *
 *   - No boundary handling (see Requirements above)
 *   - Uniform token distribution only (for simplicity)
 *   - No gating score multiplication (pure grouped GEMM)
 *   - Only row_major layouts
 *   - Educational sample, not production-optimised
 *
 * Note: This is a community-contributed sample provided as-is.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

using namespace rocwmma;

// ---------------------------------------------------------------------------
// Compile-time kernel parameters per architecture
// ---------------------------------------------------------------------------
namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
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
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

#if (ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif

// ---------------------------------------------------------------------------
// Work item: each CTA picks one from the host-built work list
// ---------------------------------------------------------------------------
struct WorkItem
{
    uint32_t expert_id;
    uint32_t tile_m_idx;
    uint32_t tile_n_idx;
};

// ---------------------------------------------------------------------------
// Types and Data Layouts
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = row_major;
using DataLayoutB   = row_major; // W : [K x N] row_major, ld = N
using DataLayoutD   = row_major;
using DataLayoutLds = col_major;

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------
constexpr uint32_t WARP_TILE_X  = BLOCKS_X * ROCWMMA_M;
constexpr uint32_t WARP_TILE_Y  = BLOCKS_Y * ROCWMMA_N;
constexpr uint32_t WARPS_X      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WARPS_X * WARP_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WARPS_Y * WARP_TILE_Y;
constexpr uint32_t MACRO_TILE_K = ROCWMMA_K;

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------

// Per-block MFMA fragments
using MfmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
using MfmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;
using MfmaFragStoreOut
    = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutD>;

// Cooperative global read (macro tile)
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffA
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;
using GRBuffB
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB, CoopScheduler>;

// Local write (macro tile) -- col_major LDS; B must be transposed
using LWBuffA = apply_data_layout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = apply_data_layout_t<apply_transpose_t<GRBuffB>, DataLayoutLds>;

// Local read (MFMA fragment-level) -- matches LDS col_major layout
using LRFragA = apply_data_layout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MfmaFragB>, DataLayoutLds>;

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

ROCWMMA_DEVICE static inline void globalReadCoopA(GRBuffA& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void globalReadCoopB(GRBuffB& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(gr), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(gr)), ldsld);
}

// Read BLOCKS_X A-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadA(MfmaFragA (&fragsA)[BLOCKS_X], InputT const* ldsAddrA, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragA>;
    using FragShape = GetIOShape_t<LRFragA>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragA tmp;
        load_matrix_sync(tmp, ldsAddrA, ldsld);
        fragsA[i] = apply_data_layout<DataLayoutA>(tmp);
        ldsAddrA += blockStep;
    }
}

// Read BLOCKS_Y B-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadB(MfmaFragB (&fragsB)[BLOCKS_Y], InputT const* ldsAddrB, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragB>;
    using FragShape = GetIOShape_t<LRFragB>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragB tmp;
        load_matrix_sync(tmp, ldsAddrB, ldsld);
        fragsB[i] = apply_data_layout<DataLayoutB>(apply_transpose(tmp));
        ldsAddrB += blockStep;
    }
}

// Fill all BLOCKS_X x BLOCKS_Y accumulator fragments with a scalar
ROCWMMA_DEVICE static inline void clear_acc_fragments(MfmaFragAcc (&frags)[BLOCKS_X][BLOCKS_Y],
                                                      ComputeT val)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            fill_fragment(frags[i][j], val);
}

// MFMA accumulation across the warp tile: acc += fragsA * fragsB
ROCWMMA_DEVICE static inline void mfma_warp_tile(MfmaFragAcc (&accOut)[BLOCKS_X][BLOCKS_Y],
                                                 MfmaFragA const (&fragsA)[BLOCKS_X],
                                                 MfmaFragB const (&fragsB)[BLOCKS_Y],
                                                 MfmaFragAcc const (&accIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            mma_sync(accOut[i][j], fragsA[i], fragsB[j], accIn[i][j]);
}

// Write the D warp tile to global memory (ComputeT -> OutputT)
ROCWMMA_DEVICE static inline void
    globalWriteD(OutputT* gAddrD, MfmaFragAcc const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using Mapper1d  = GetDataLayout_t<MfmaFragStoreOut>;
    using FragShape = GetIOShape_t<MfmaFragStoreOut>;
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            MfmaFragStoreOut fragOut;
#pragma unroll
            for(int e = 0; e < (int)fragsD[i][j].num_elements; e++)
                fragOut.x[e] = static_cast<OutputT>(fragsD[i][j].x[e]);
            store_matrix_sync(gAddrD + offsetY, fragOut, ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

// ---------------------------------------------------------------------------
// Main Grouped GEMM MoE kernel
//
//   Y_e = X_e * W_e   for each expert e
//
//   tokens         [total_tokens x K] row_major
//   expert_weights [num_experts x K x N] row_major (stacked)
//   expert_offsets [num_experts + 1] CSR-style
//   output         [total_tokens x N] row_major
//   work_list      [gridDim.x] WorkItem array
// ---------------------------------------------------------------------------
constexpr uint32_t  kBlockThreads = TBLOCK_X * TBLOCK_Y;
ROCWMMA_KERNEL void __launch_bounds__(kBlockThreads)
    grouped_gemm_moe(uint32_t        k,
                     uint32_t        n,
                     InputT const*   tokens,
                     InputT const*   expert_weights,
                     uint32_t const* expert_offsets,
                     OutputT*        output,
                     WorkItem const* work_list,
                     uint32_t        lda,
                     uint32_t        ldb,
                     uint32_t        ldd)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // ------------------------------------------------------------------
        // Read this CTA's work assignment
        // ------------------------------------------------------------------
        WorkItem const wi = work_list[blockIdx.x];

        uint32_t expert_start = expert_offsets[wi.expert_id];

        // Base pointers for this expert.  The per-expert weight stride is
        // k * ldb (rows x leading dimension), not k * n: if the weight
        // leading dimension is ever padded (ldb != n) the base pointer must
        // still advance by ldb to land on the correct expert.
        InputT const* a_base = tokens + expert_start * lda;
        InputT const* b_base = expert_weights + wi.expert_id * k * ldb;
        OutputT*      d_base = output + expert_start * ldd;

        // ------------------------------------------------------------------
        // Warp / tile coordinate setup (same pattern as SwiGLU)
        // ------------------------------------------------------------------
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y);
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);

        auto macroTileCoord = make_coord2d(wi.tile_m_idx, wi.tile_n_idx) * macroTileSize;

        auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto localWarpOffset = localWarpCoord * warpTileSize;
        auto warpTileCoord   = macroTileCoord + localWarpOffset;

        // No per-warp boundary guard here.  This sample requires
        // tokens_per_expert % MACRO_TILE_X == 0 and n % MACRO_TILE_Y == 0
        // (see Requirements / Known limitations), and the host work list only
        // emits full macro tiles, so every warp tile is guaranteed to lie
        // fully inside the expert's output (warpTileBound <= {expert_m, n}).
        // A per-warp early return would be non-uniform across the CTA and
        // could deadlock the synchronize_workgroup() barriers below, so it is
        // intentionally omitted rather than made conditional.

        // ------------------------------------------------------------------
        // Global read address setup
        // ------------------------------------------------------------------
        using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
        using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

        auto globalReadOffsetA
            = GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
        auto globalReadOffsetB
            = GRBuffBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

        auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
        auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

        // ------------------------------------------------------------------
        // Initial global pre-fetch
        // ------------------------------------------------------------------
        GRBuffA grBuffA;
        GRBuffB grBuffB;

        globalReadCoopA(grBuffA, a_base + globalReadOffsetA, lda);
        globalReadCoopB(grBuffB, b_base + globalReadOffsetB, ldb);

        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        // ------------------------------------------------------------------
        // LDS layout (col_major, 2 segments stacked vertically)
        //
        //   Segment | rows                  | height       | content
        //   --------+-----------------------+--------------+---------
        //      A    | [0 .. MtX)            | MACRO_TILE_X | token tile
        //      B    | [MtX .. MtX+MtY)      | MACRO_TILE_Y | weight^T tile
        // ------------------------------------------------------------------
        HIP_DYNAMIC_SHARED(InputT, ldsBase);

        using LWBuffAShape = GetIOShape_t<LWBuffA>;
        using LWBuffBShape = GetIOShape_t<LWBuffB>;
        using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
        using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;

        constexpr uint32_t ldsWidth  = MACRO_TILE_K;
        constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
        constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
        constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

        auto* ldsPtrLo = ldsBase;
        auto* ldsPtrHi = ldsPtrLo + sizeLds;

        // Segment write offsets
        auto ldsWriteOffsetA = 0u;
        auto ldsWriteOffsetB
            = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

        // Per-warp read offsets
        auto ldsReadOffsetA
            = ldsWriteOffsetA
              + LWBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetB
            = ldsWriteOffsetB
              + LWBuffBMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

        // ------------------------------------------------------------------
        // Write initial prefetch to Lo buffer
        // ------------------------------------------------------------------
        localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld);
        localWriteCoopB(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld);

        // ------------------------------------------------------------------
        // Initialize accumulator
        // ------------------------------------------------------------------
        MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
        clear_acc_fragments(fragsAcc, ComputeT(0));

        synchronize_workgroup();

        // ------------------------------------------------------------------
        // K-loop with double-buffer prefetch
        // ------------------------------------------------------------------
        for(uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
        {
            MfmaFragA fragsA[BLOCKS_X];
            MfmaFragB fragsB[BLOCKS_Y];

            // Read current tile from Lo buffer
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

            // Prefetch next K-step from global memory
            globalReadCoopA(grBuffA, a_base + globalReadOffsetA, lda);
            globalReadCoopB(grBuffB, b_base + globalReadOffsetB, ldb);

            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            // Accumulate
            mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);

            // Write prefetch to Hi buffer
            localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld);
            localWriteCoopB(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld);

            synchronize_workgroup();

            // Swap ping-pong buffers
            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;
        }

        // ------------------------------------------------------------------
        // Tail: accumulate the last K-step still in Lo
        // ------------------------------------------------------------------
        {
            MfmaFragA fragsA[BLOCKS_X];
            MfmaFragB fragsB[BLOCKS_Y];

            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

            mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);
        }

        // ------------------------------------------------------------------
        // Store output to global memory
        // ------------------------------------------------------------------
        using MfmaFragStoreOutMap1d = GetDataLayout_t<MfmaFragStoreOut>;
        globalWriteD(
            d_base + MfmaFragStoreOutMap1d::fromMatrixCoord(warpTileCoord, ldd), fragsAcc, ldd);
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static void grouped_gemm_moe_cpu_ref(uint32_t        num_experts,
                                     uint32_t        k,
                                     uint32_t        n,
                                     InputT const*   tokens,
                                     InputT const*   expert_weights,
                                     uint32_t const* expert_offsets,
                                     OutputT*        output,
                                     uint32_t        lda,
                                     uint32_t        ldb,
                                     uint32_t        ldd)
{
    auto rowMjr = [](uint32_t r, uint32_t c, uint32_t ld) { return r * ld + c; };

    for(uint32_t e = 0; e < num_experts; e++)
    {
        uint32_t      m_e   = expert_offsets[e + 1] - expert_offsets[e];
        InputT const* x_e   = tokens + expert_offsets[e] * lda;
        InputT const* w_e   = expert_weights + e * k * ldb; // per-expert stride = k * ldb
        OutputT*      out_e = output + expert_offsets[e] * ldd;

        for(uint32_t i = 0; i < m_e; i++)
        {
            for(uint32_t j = 0; j < n; j++)
            {
                float acc = 0.0f;
                for(uint32_t h = 0; h < k; h++)
                {
                    float a_val = static_cast<float>(x_e[rowMjr(i, h, lda)]);
                    float b_val = static_cast<float>(w_e[rowMjr(h, j, ldb)]);
                    acc += a_val * b_val;
                }
                out_e[rowMjr(i, j, ldd)] = static_cast<OutputT>(acc);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Print GPU hardware info
// ---------------------------------------------------------------------------
static void printDeviceInfo()
{
    hipDevice_t     dev;
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDevice(&dev));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, dev));

    std::cout << "\n=== GPU Hardware Info ===\n"
              << "  Device name     : " << props.name << "\n"
              << "  GCN arch        : " << props.gcnArchName << "\n"
              << "  Compute units   : " << props.multiProcessorCount << "\n"
              << "  Warp size       : " << props.warpSize << "\n"
              << "  Global memory   : " << (props.totalGlobalMem >> 20) << " MiB\n"
              << "  Shared mem/blk  : " << (props.sharedMemPerBlock >> 10) << " KiB\n"
              << "  Max clock (MHz) : " << (props.clockRate / 1000) << "\n"
              << "  Memory bw (GB/s): "
              << (static_cast<double>(props.memoryBusWidth) / 8.0
                  * static_cast<double>(props.memoryClockRate) * 2.0)
                     / 1.0e6
              << "\n"
              << "========================\n\n";
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
ROCWMMA_HOST void run_grouped_gemm_moe_sample(uint32_t num_experts,
                                              uint32_t tokens_per_expert,
                                              uint32_t k,
                                              uint32_t n,
                                              bool     printInfo      = false,
                                              bool     skipValidation = false)
{
    if(printInfo)
        printDeviceInfo();

    uint32_t hTBLOCK_X  = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
    uint32_t hTBLOCK_Y  = isGfx9() ? gfx9Params::TBLOCK_Y : gfx11Params::TBLOCK_Y;
    uint32_t hROCWMMA_M = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
    uint32_t hROCWMMA_N = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
    uint32_t hROCWMMA_K = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
    uint32_t hBLOCKS_X  = isGfx9() ? gfx9Params::BLOCKS_X : gfx11Params::BLOCKS_X;
    uint32_t hBLOCKS_Y  = isGfx9() ? gfx9Params::BLOCKS_Y : gfx11Params::BLOCKS_Y;

    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    auto warpSize = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

    uint32_t hMACRO_TILE_X = get<0>(macroTileSize);
    uint32_t hMACRO_TILE_Y = get<1>(macroTileSize);

    // Architecture checks (same as SwiGLU)
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }
    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N || (hROCWMMA_M != 16 && hROCWMMA_M != 32)))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }
    if((isGfx11() || isGfx12()) && warpSize != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }
    if(isGfx9() && warpSize != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    // Dimension checks
    if(tokens_per_expert % hMACRO_TILE_X || n % hMACRO_TILE_Y || k % hROCWMMA_K)
    {
        std::cout << "Unsupported dimensions!"
                  << " tokens_per_expert must be a multiple of " << hMACRO_TILE_X
                  << ", N must be a multiple of " << hMACRO_TILE_Y << ", K must be a multiple of "
                  << hROCWMMA_K << "\n";
        return;
    }

    uint32_t total_tokens = num_experts * tokens_per_expert;
    uint32_t lda          = k;
    uint32_t ldb          = n;
    uint32_t ldd          = n;

    // Build expert_offsets (uniform distribution)
    std::vector<uint32_t> expertOffsets(num_experts + 1);
    for(uint32_t e = 0; e <= num_experts; e++)
        expertOffsets[e] = e * tokens_per_expert;

    // Build work list
    uint32_t tilesM       = tokens_per_expert / hMACRO_TILE_X;
    uint32_t tilesN       = n / hMACRO_TILE_Y;
    uint32_t numWorkItems = num_experts * tilesM * tilesN;

    std::vector<WorkItem> workList(numWorkItems);
    uint32_t              idx = 0;
    for(uint32_t e = 0; e < num_experts; e++)
        for(uint32_t tm = 0; tm < tilesM; tm++)
            for(uint32_t tn = 0; tn < tilesN; tn++)
                workList[idx++] = {e, tm, tn};

    std::cout << "Initializing host data (experts=" << num_experts
              << " tokens_per_expert=" << tokens_per_expert << " k=" << k << " n=" << n << ")...\n";

    std::vector<InputT>  matTokens(total_tokens * k);
    std::vector<InputT>  matWeights(num_experts * k * n);
    std::vector<OutputT> matOutput(total_tokens * n, std::numeric_limits<OutputT>::signaling_NaN());

    constexpr float kScale = 1.0f / 16.0f;
    fillRand(matTokens.data(), total_tokens, k);
    fillRand(matWeights.data(), num_experts * k, n);
    for(auto& x : matTokens)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matWeights)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);

    std::cout << "Allocating device memory...\n";

    InputT*   d_tokens;
    InputT*   d_weights;
    uint32_t* d_offsets;
    OutputT*  d_output;
    WorkItem* d_workList;

    CHECK_HIP_ERROR(hipMalloc(&d_tokens, total_tokens * k * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_weights, num_experts * k * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_offsets, (num_experts + 1) * sizeof(uint32_t)));
    CHECK_HIP_ERROR(hipMalloc(&d_output, total_tokens * n * sizeof(OutputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_workList, numWorkItems * sizeof(WorkItem)));

    CHECK_HIP_ERROR(hipMemcpy(
        d_tokens, matTokens.data(), total_tokens * k * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_weights, matWeights.data(), num_experts * k * n * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_offsets,
                              expertOffsets.data(),
                              (num_experts + 1) * sizeof(uint32_t),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_output, matOutput.data(), total_tokens * n * sizeof(OutputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_workList, workList.data(), numWorkItems * sizeof(WorkItem), hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(numWorkItems);

    std::cout << "gridDim (" << gridDim.x << ")"
              << "  blockDim (" << blockDim.x << " " << blockDim.y << ")\n";

    // LDS: 2 ping-pong buffers
    using LWBuffAShape              = GetIOShape_t<LWBuffA>;
    using LWBuffBShape              = GetIOShape_t<LWBuffB>;
    constexpr uint32_t ldsSegHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
    int                ldsUsage     = 2 * sizeof(InputT) * ldsSegHeight * MACRO_TILE_K;
    std::cout << "LDS usage: " << ldsUsage << " bytes (" << ldsUsage / 1024 << " KiB)\n";

    auto kernelLambda = [&]() {
        hipExtLaunchKernelGGL(grouped_gemm_moe,
                              gridDim,
                              blockDim,
                              ldsUsage,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              k,
                              n,
                              d_tokens,
                              d_weights,
                              d_offsets,
                              d_output,
                              d_workList,
                              lda,
                              ldb,
                              ldd);
        CHECK_HIP_ERROR(hipGetLastError());
    };

    constexpr uint32_t warmups    = 2u;
    constexpr uint32_t recordRuns = 5u;

    std::cout << "Warming up...\n";
    for(uint32_t i = 0; i < warmups; ++i)
        kernelLambda();
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    std::cout << "Benchmarking...\n";
    hipEvent_t evStart, evStop;
    CHECK_HIP_ERROR(hipEventCreate(&evStart));
    CHECK_HIP_ERROR(hipEventCreate(&evStop));

    CHECK_HIP_ERROR(hipEventRecord(evStart));
    for(uint32_t i = 0; i < recordRuns; ++i)
        kernelLambda();
    CHECK_HIP_ERROR(hipEventRecord(evStop));
    CHECK_HIP_ERROR(hipEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, evStart, evStop));
    CHECK_HIP_ERROR(hipEventDestroy(evStart));
    CHECK_HIP_ERROR(hipEventDestroy(evStop));

    // Performance: E GEMMs, each 2*M_e*N*K FLOPs = 2*total_tokens*N*K total
    double gFlopsPerRun = 2.0 * static_cast<double>(total_tokens) * static_cast<double>(n)
                          * static_cast<double>(k) * 1e-9;
    double tFlopsPerSec
        = gFlopsPerRun * recordRuns / (static_cast<double>(elapsedMs) * 1e-3) * 1e-3;

    // The time and GFlops columns below are totals across all `recordRuns`
    // timed launches (TFlops/s is the sustained rate).  Label the time column
    // with the run count so it is not misread as a single-launch latency.
    std::string elapsedHdr = "totalMs(" + std::to_string(recordRuns) + "x)";

    std::cout << std::left << std::setw(10) << "Experts" << std::setw(10) << "TokPerExp"
              << std::setw(8) << "MatK" << std::setw(8) << "MatN" << std::setw(12) << "WorkItems"
              << std::setw(14) << elapsedHdr << std::setw(14) << "GFlops" << std::setw(12)
              << "TFlops/s"
              << "\n";

    std::cout << std::left << std::setw(10) << num_experts << std::setw(10) << tokens_per_expert
              << std::setw(8) << k << std::setw(8) << n << std::setw(12) << numWorkItems
              << std::setw(14) << elapsedMs << std::setw(14) << (gFlopsPerRun * recordRuns)
              << std::setw(12) << tFlopsPerSec << "\n";

#if !NDEBUG
    if(skipValidation)
    {
        std::cout << "Skipping validation as requested.\n";
        CHECK_HIP_ERROR(hipFree(d_tokens));
        CHECK_HIP_ERROR(hipFree(d_weights));
        CHECK_HIP_ERROR(hipFree(d_offsets));
        CHECK_HIP_ERROR(hipFree(d_output));
        CHECK_HIP_ERROR(hipFree(d_workList));
        std::cout << "Finished!\n";
        return;
    }

    std::cout << "\nValidating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(
        matOutput.data(), d_output, total_tokens * n * sizeof(OutputT), hipMemcpyDeviceToHost));

    uint64_t refOps
        = static_cast<uint64_t>(total_tokens) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
    if(refOps > (512ULL * 1024ULL * 1024ULL))
    {
        std::cout << "[Note] Running CPU reference validation for large grouped GEMM ("
                  << total_tokens << "x" << n << "x" << k
                  << "). Please wait. This may take a while...\n";
    }

    std::vector<OutputT> matOutputRef(total_tokens * n,
                                      std::numeric_limits<OutputT>::signaling_NaN());
    grouped_gemm_moe_cpu_ref(num_experts,
                             k,
                             n,
                             matTokens.data(),
                             matWeights.data(),
                             expertOffsets.data(),
                             matOutputRef.data(),
                             lda,
                             ldb,
                             ldd);

    auto res = compareEqual(matOutput.data(), matOutputRef.data(), total_tokens * n);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_tokens));
    CHECK_HIP_ERROR(hipFree(d_weights));
    CHECK_HIP_ERROR(hipFree(d_offsets));
    CHECK_HIP_ERROR(hipFree(d_output));
    CHECK_HIP_ERROR(hipFree(d_workList));

    std::cout << "Finished!\n";
}

// ---------------------------------------------------------------------------
// Usage:
//   ./grouped_gemm_moe          # Quick validation (4 experts, 64 tok/exp)
//   ./grouped_gemm_moe --all    # Also run Mixtral-8x7B and DeepSeek sizes
//   ./grouped_gemm_moe --skip-validation # Skip Debug CPU validation
// ---------------------------------------------------------------------------
static void printUsage(char const* exe)
{
    std::cout << "Usage: " << exe << " [--all] [--skip-validation] [--help]\n"
              << "  --all              Run additional Mixtral / DeepSeek-style shapes\n"
              << "  --skip-validation  Skip Debug CPU reference validation\n"
              << "  --help             Print this message\n";
}

int main(int argc, char** argv)
{
    std::cout << "Community Sample: Mixture-of-Experts Grouped GEMM" << std::endl;
    std::cout << "This sample demonstrates: work-list-scheduled grouped GEMM"
              << " for MoE inference using rocWMMA" << std::endl;

    bool runAll         = false;
    bool skipValidation = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--all")
        {
            runAll = true;
        }
        else if(arg == "--skip-validation")
        {
            skipValidation = true;
        }
        else if(arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cout << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Quick validation (small, any GPU)
    // run_grouped_gemm_moe_sample(num_experts, tokens_per_expert, K, N)
    run_grouped_gemm_moe_sample(4, 64, 128, 256, /*printInfo=*/true, skipValidation);

    if(runAll)
    {
        // Mixtral-8x7B style: 8 experts, intermediate_size=14336, hidden=4096
        run_grouped_gemm_moe_sample(8, 64, 4096, 14336, /*printInfo=*/false, skipValidation);

        // DeepSeek-MoE-16B style: 64 experts, intermediate=1408, hidden=2048
        run_grouped_gemm_moe_sample(64, 64, 2048, 1408, /*printInfo=*/false, skipValidation);
    }
    else
    {
        std::cout << "\nTip: pass --all to also run Mixtral-8x7B / DeepSeek sizes.\n";
        std::cout << "Tip: pass --skip-validation to skip Debug CPU validation.\n";
    }

    std::cout << "Sample completed successfully!" << std::endl;
    return 0;
}
