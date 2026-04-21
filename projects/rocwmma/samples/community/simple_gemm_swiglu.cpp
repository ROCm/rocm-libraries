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

/* COMMUNITY SAMPLE: SwiGLU Fused Dual GEMM
 *
 * ============================================================================
 * 1. WHAT IS SwiGLU?
 * ============================================================================
 *
 * SwiGLU is a gated activation function proposed by Shazeer (2020) as a
 * drop-in replacement for the standard ReLU or GELU feed-forward layer in
 * Transformers.  It was adopted by Meta's LLaMA (Touvron et al., 2023) and
 * subsequently by Mistral, Qwen and other major LLM families for their
 * feed-forward network (FFN) gate layer.
 *
 * References:
 *   [1] Shazeer, "GLU Variants Improve Transformer", 2020
 *       https://arxiv.org/abs/2002.05202
 *   [2] Touvron et al. (Meta AI), "LLaMA: Open and Efficient Foundation
 *       Language Models", 2023
 *       https://arxiv.org/abs/2302.13971
 *
 * ============================================================================
 * 2. MATHEMATICAL FORMULATION
 * ============================================================================
 *
 *   gate = A x B_gate                     [M x N] = [M x K] x [K x N]
 *   up   = A x B_up                       [M x N] = [M x K] x [K x N]
 *   D    = silu(gate) (*) up              element-wise (Hadamard) product
 *
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
 *
 * In a typical LLM FFN block:
 *   A      = hidden states              (sequence tokens)
 *   B_gate = gate projection weights     (learned parameters)
 *   B_up   = up   projection weights     (learned parameters)
 *   D      = activated hidden states      (fed to down-projection next)
 *
 * ============================================================================
 * 3. WHY FUSE INTO A SINGLE KERNEL?
 * ============================================================================
 *
 * A naive implementation launches 3 separate kernels:
 *   Kernel 1:  gate = A x B_gate          (GEMM)
 *   Kernel 2:  up   = A x B_up            (GEMM)
 *   Kernel 3:  D    = silu(gate) (*) up   (element-wise)
 *
 * This sample fuses all three into ONE kernel.  The benefits are:
 *
 *   a) A-tile reuse -- A is loaded from global memory ONCE and used for
 *      both the gate and up GEMMs.  This cuts A's global memory traffic
 *      by 50% compared to two separate GEMM launches.
 *
 *   b) Zero temporary buffers -- The gate and up accumulators live in
 *      registers.  The SiLU activation and Hadamard product are applied
 *      directly in registers before writing D to global memory, so no
 *      intermediate [M x N] buffers are allocated.
 *
 *   c) Single launch overhead -- One kernel launch instead of three
 *      eliminates two launch + synchronisation barriers.
 *
 * ============================================================================
 * 4. WHAT YOU WILL LEARN FROM THIS SAMPLE
 * ============================================================================
 *
 *   - How to declare and use rocWMMA fragment types for dual-GEMM fusion
 *   - How cooperative global reads (fragment_scheduler::coop_row_major_2d)
 *     distribute work across all threads in a block
 *   - How to design a 3-segment LDS layout (A | B_gate^T | B_up^T) so that
 *     one matrix tile can feed two independent GEMM streams
 *   - How to implement K-loop double buffering (ping-pong) to overlap
 *     global memory latency with MFMA compute
 *   - How to apply an activation function (SiLU) and element-wise product
 *     (Hadamard) directly on accumulator fragments in registers
 *   - The ComputeT -> OutputT cast pattern (MfmaFragAcc -> MfmaFragStoreOut)
 *
 * ============================================================================
 * 5. KERNEL DATA-FLOW OVERVIEW
 * ============================================================================
 *
 *   Global Memory --coop load--> LDS (3 segments, double-buffered)
 *        |                            |
 *      A [MxK]                  [A    segment]
 *      Bg[KxN]                  [Bg^T segment]
 *      Bu[KxN]                  [Bu^T segment]
 *                                     |
 *                               local read
 *                                     |
 *                    +-------fragsA-------+
 *                    |                    |
 *              fragsBGate            fragsBUp
 *                    |                    |
 *                   mma                  mma
 *                    |                    |
 *                accGate              accUp
 *                    |                    |
 *              silu(accGate)              |
 *                    \                   /
 *                     *--- Hadamard ---*
 *                             |
 *                        fragsD (f32)
 *                             |
 *                        cast to f16
 *                             |
 *                          D [MxN]  --store--> Global Memory
 *
 * ============================================================================
 * 6. DATA LAYOUTS
 * ============================================================================
 *
 *   A       : row_major  [M x K],  lda     = K
 *   B_gate  : row_major  [K x N],  ldBGate = N
 *   B_up    : row_major  [K x N],  ldBUp   = N
 *   D       : row_major  [M x N],  ldd     = N
 *
 * ============================================================================
 * 7. LDS LAYOUT (col_major, one ping-pong buffer)
 * ============================================================================
 *
 *   Three segments stacked vertically in col_major order:
 *
 *     Segment  | Height         | Content
 *     ---------+----------------+---------
 *        A     | MACRO_TILE_X   | A tile
 *       Bg     | MACRO_TILE_Y   | B_gate^T tile
 *       Bu     | MACRO_TILE_Y   | B_up^T tile
 *
 *     Width  = MACRO_TILE_K
 *     Height = MACRO_TILE_X + 2 * MACRO_TILE_Y
 *     ldsld  = Height  (col_major leading dimension = number of rows)
 *
 *   Per-architecture derivation (both share BLOCKS_X=2, BLOCKS_Y=2):
 *
 *     Param            | gfx9 (wave64)        | gfx11/12 (wave32)
 *     -----------------+----------------------+---------------------
 *     TBLOCK_X         | 128                  | 64
 *     WARP_SIZE        | 64                   | 32
 *     WARPS_X          | 128 / 64 = 2         | 64 / 32 = 2
 *     WARP_TILE_X      | 2 * 16 = 32          | 2 * 16 = 32
 *     MACRO_TILE_X     | 2 * 32 = 64          | 2 * 32 = 64
 *     MACRO_TILE_Y     | 2 * 32 = 64          | 2 * 32 = 64
 *     MACRO_TILE_K     | 16                   | 16
 *     LDS Height       | 64 + 64 + 64 = 192   | 64 + 64 + 64 = 192
 *     LDS Width        | 16                   | 16
 *     1 buffer (elems) | 192 * 16 = 3072      | 192 * 16 = 3072
 *     1 buffer (bytes) | 3072 * 2 = 6144 (6K) | 3072 * 2 = 6144 (6K)
 *     Total (Lo + Hi)  | 12288 bytes (12K)    | 12288 bytes (12K)
 *
 *   Note: Both architectures produce identical macro tile sizes because
 *   the halved TBLOCK_X on gfx11/12 is exactly compensated by the
 *   halved WARP_SIZE, yielding the same WARPS_X = 2.
 *
 *   Two such buffers (Lo / Hi) are allocated back-to-back for double
 *   buffering.
 *
 * ============================================================================
 * 8. REQUIREMENTS
 * ============================================================================
 *
 *   - Minimum ROCm version: ROCm 6.0+
 *   - GPU architectures: CDNA3 gfx942 (MI300x), RDNA4 gfx1201 (RX9070)
 *   - Data types: float16 input, float32 compute, float16 output
 *   - Matrix dimensions: M must be a multiple of MACRO_TILE_X (64),
 *     N must be a multiple of MACRO_TILE_Y (64),
 *     K must be a multiple of ROCWMMA_K (16)
 *
 * ============================================================================
 * 9. KNOWN LIMITATIONS
 * ============================================================================
 *
 *   - No boundary handling: M must be a multiple of MACRO_TILE_X (64),
 *     N must be a multiple of MACRO_TILE_Y (64),
 *     K must be a multiple of ROCWMMA_K (16)
 *   - Only supports row_major layout for all inputs and output
 *   - Performance is not optimized for production use (educational sample)
 *   - LDS usage: ~12 KiB per block
 *   - Input values are scaled by 1/16 to prevent FP16 overflow;
 *     real workloads may need different scaling strategies
 *
 * Note: This is a community-contributed sample provided as-is. It may not be
 * maintained with the same rigor as official samples.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
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

#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
#else
using namespace gfx11Params;
#endif

// ---------------------------------------------------------------------------
// Types and Data Layouts
// NOTE: Both B_gate and B_up use row_major to match fillRand's row-major fill
//       convention (fillRand writes mat[i*n+j]).
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA     = row_major;
using DataLayoutBGate = row_major; // B_gate : [K x N] row_major, ldb = N
using DataLayoutBUp   = row_major; // B_up   : [K x N] row_major, ldb = N
using DataLayoutD     = row_major;
using DataLayoutLds   = col_major;

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

// Per-block MFMA fragments (warp tile element)
using MfmaFragA     = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragBGate = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutBGate>;
using MfmaFragBUp   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutBUp>;
// MfmaFragComputeOut: keep as ComputeT to avoid cross-type fragment register mapping issues.
// We use a separate MfmaFragStoreOut (OutputT) only at the final store step, matching
// the pattern used in perf_hgemm.cpp (MmaFragAcc -> MmaFragD cast).
using MfmaFragComputeOut = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;
using MfmaFragStoreOut
    = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutD>;
using MfmaFragAcc = MfmaFragComputeOut; // gate/up accumulators are the same type as D

// Cooperative global read (macro tile)
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffA
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;
using GRBuffBGate = fragment<matrix_b,
                             ROCWMMA_M,
                             MACRO_TILE_Y,
                             ROCWMMA_K,
                             InputT,
                             DataLayoutBGate,
                             CoopScheduler>;
using GRBuffBUp
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutBUp, CoopScheduler>;

// Local write (macro tile) -- apply col_major LDS layout; B must be transposed
using LWBuffA     = apply_data_layout_t<GRBuffA, DataLayoutLds>;
using LWBuffBGate = apply_data_layout_t<apply_transpose_t<GRBuffBGate>, DataLayoutLds>;
using LWBuffBUp   = apply_data_layout_t<apply_transpose_t<GRBuffBUp>, DataLayoutLds>;

// Local read (MFMA fragment-level) -- matches LDS col_major layout
using LRFragA     = apply_data_layout_t<MfmaFragA, DataLayoutLds>;
using LRFragBGate = apply_data_layout_t<apply_transpose_t<MfmaFragBGate>, DataLayoutLds>;
using LRFragBUp   = apply_data_layout_t<apply_transpose_t<MfmaFragBUp>, DataLayoutLds>;

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

ROCWMMA_DEVICE static inline void globalReadCoopA(GRBuffA& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    globalReadCoopBGate(GRBuffBGate& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void globalReadCoopBUp(GRBuffBUp& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(gr), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopBGate(InputT* ldsAddr, GRBuffBGate const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(gr)), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopBUp(InputT* ldsAddr, GRBuffBUp const& gr, uint32_t ldsld)
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

// Read BLOCKS_Y B_gate-blocks from LDS
ROCWMMA_DEVICE static inline void localReadBGate(MfmaFragBGate (&fragsBGate)[BLOCKS_Y],
                                                 InputT const* ldsAddrBGate,
                                                 uint32_t      ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragBGate>;
    using FragShape = GetIOShape_t<LRFragBGate>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragBGate tmp;
        load_matrix_sync(tmp, ldsAddrBGate, ldsld);
        fragsBGate[i] = apply_data_layout<DataLayoutBGate>(apply_transpose(tmp));
        ldsAddrBGate += blockStep;
    }
}

// Read BLOCKS_Y B_up-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadBUp(MfmaFragBUp (&fragsBUp)[BLOCKS_Y], InputT const* ldsAddrBUp, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragBUp>;
    using FragShape = GetIOShape_t<LRFragBUp>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragBUp tmp;
        load_matrix_sync(tmp, ldsAddrBUp, ldsld);
        fragsBUp[i] = apply_data_layout<DataLayoutBUp>(apply_transpose(tmp));
        ldsAddrBUp += blockStep;
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
template <typename FragB>
ROCWMMA_DEVICE static inline void mfma_warp_tile(MfmaFragAcc (&accOut)[BLOCKS_X][BLOCKS_Y],
                                                 MfmaFragA const (&fragsA)[BLOCKS_X],
                                                 FragB const (&fragsB)[BLOCKS_Y],
                                                 MfmaFragAcc const (&accIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            mma_sync(accOut[i][j], fragsA[i], fragsB[j], accIn[i][j]);
}

// SiLU (numerically stable two-branch form)
ROCWMMA_DEVICE static inline ComputeT silu(ComputeT x)
{
    if(x >= ComputeT(0))
        return x / (ComputeT(1) + __expf(-x));
    else
    {
        ComputeT ex = __expf(x);
        return x * ex / (ComputeT(1) + ex);
    }
}

// Fused SwiGLU: D[i][j] = silu(gate[i][j]) * up[i][j]  (in ComputeT, no cast)
// fragsD has the SAME type as fragsGate/fragsUp (ComputeT = MfmaFragAcc),
// so element indices are guaranteed to correspond to the same output positions.
ROCWMMA_DEVICE static inline void apply_swiglu(MfmaFragComputeOut (&fragsD)[BLOCKS_X][BLOCKS_Y],
                                               MfmaFragAcc const (&fragsGate)[BLOCKS_X][BLOCKS_Y],
                                               MfmaFragAcc const (&fragsUp)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
#pragma unroll
            for(int e = 0; e < (int)fragsGate[i][j].num_elements; e++)
                fragsD[i][j].x[e] = silu(fragsGate[i][j].x[e]) * fragsUp[i][j].x[e];
    // No cast here: stays in ComputeT (float32) to preserve precision.
    // The OutputT cast happens in globalWriteD via MfmaFragStoreOut.
}

// Write the D warp tile to global memory (ComputeT -> OutputT via intermediate fragment)
// Pattern: fragsD (ComputeT) -> fragsOut (OutputT) -> store_matrix_sync with OutputT*
// This matches the perf_hgemm.cpp convention and ensures type-safe store.
ROCWMMA_DEVICE static inline void globalWriteD(
    OutputT* gAddrD, MfmaFragComputeOut const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
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
            // Convert ComputeT element-wise to OutputT (same element geometry on same arch)
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
// Main SwiGLU kernel
//
//   D = silu(A * B_gate)  *  (A * B_up)
//
//   A      [M x K] row_major
//   B_gate [K x N] row_major
//   B_up   [K x N] row_major
//   D      [M x N] row_major
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(256) gemm_swiglu_fused(uint32_t      m,
                                                             uint32_t      n,
                                                             uint32_t      k,
                                                             InputT const* a,
                                                             InputT const* b_gate,
                                                             InputT const* b_up,
                                                             OutputT*      d,
                                                             uint32_t      lda,
                                                             uint32_t      ldBGate,
                                                             uint32_t      ldBUp,
                                                             uint32_t      ldd)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // ------------------------------------------------------------------
        // Warp / tile coordinate setup
        // Each block covers one macro tile of the output matrix.  Within a
        // block, each warp is responsible for a WARP_TILE_X x WARP_TILE_Y
        // sub-tile.  We compute the global (row, col) origin of this warp's
        // sub-tile so that all subsequent address calculations can be
        // expressed as offsets from this origin.
        // ------------------------------------------------------------------
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y);
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);

        auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto localWarpOffset = localWarpCoord * warpTileSize;

        auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
        auto warpTileCoord  = macroTileCoord + localWarpOffset;

        // Skip warps that fall outside the output matrix
        auto warpTileBound = warpTileCoord + warpTileSize;
        if(get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
            return;

        // ------------------------------------------------------------------
        // Global read address setup
        // ------------------------------------------------------------------
        using GRBuffAMap1d     = GetDataLayout_t<GRBuffA>;
        using GRBuffBGateMap1d = GetDataLayout_t<GRBuffBGate>;
        using GRBuffBUpMap1d   = GetDataLayout_t<GRBuffBUp>;

        auto globalReadOffsetA
            = GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
        auto globalReadOffsetBGate
            = GRBuffBGateMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldBGate);
        auto globalReadOffsetBUp
            = GRBuffBUpMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldBUp);

        auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
        auto kStepOffsetBGate
            = GRBuffBGateMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldBGate);
        auto kStepOffsetBUp
            = GRBuffBUpMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldBUp);

        // ------------------------------------------------------------------
        // Initial global pre-fetch
        // Load the very first K-tile of A, B_gate and B_up from global
        // memory into register buffers.  These will be written to LDS
        // before the K-loop begins, establishing the first "Lo" buffer
        // for the double-buffering scheme.
        // ------------------------------------------------------------------
        GRBuffA     grBuffA;
        GRBuffBGate grBuffBGate;
        GRBuffBUp   grBuffBUp;

        globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
        globalReadCoopBGate(grBuffBGate, b_gate + globalReadOffsetBGate, ldBGate);
        globalReadCoopBUp(grBuffBUp, b_up + globalReadOffsetBUp, ldBUp);

        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetBGate += kStepOffsetBGate;
        globalReadOffsetBUp += kStepOffsetBUp;

        // ------------------------------------------------------------------
        // LDS layout  (col_major, 3 segments stacked vertically per buffer)
        //
        // WHY 3 segments?  We need A, B_gate and B_up simultaneously for
        // the dual-GEMM.  Stacking them vertically in one contiguous LDS
        // allocation means a single HIP_DYNAMIC_SHARED pointer services
        // all three, and the col_major layout lets each warp read its
        // own row-slice with a simple offset.
        //
        //   Segment  | rows                  | height       | content
        //   ---------+-----------------------+--------------+---------
        //      A     | [0 .. MtX)            | MACRO_TILE_X | A tile
        //     B_gate | [MtX .. MtX+MtY)      | MACRO_TILE_Y | Bg^T tile
        //     B_up   | [MtX+MtY .. MtX+2*MtY)| MACRO_TILE_Y | Bu^T tile
        //
        //   width  = MACRO_TILE_K
        //   ldsld  = ldsHeight  (col_major leading dim = number of rows)
        // ------------------------------------------------------------------
        HIP_DYNAMIC_SHARED(void*, localMemPtr);

        using LWBuffAShape     = GetIOShape_t<LWBuffA>;
        using LWBuffBGateShape = GetIOShape_t<LWBuffBGate>;
        using LWBuffBUpShape   = GetIOShape_t<LWBuffBUp>;
        using LWBuffAMap1d     = GetDataLayout_t<LWBuffA>;
        using LWBuffBGateMap1d = GetDataLayout_t<LWBuffBGate>;

        constexpr uint32_t ldsWidth  = MACRO_TILE_K;
        constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBGateShape::BlockHeight
                                       + LWBuffBUpShape::BlockHeight;
        constexpr uint32_t sizeLds = ldsHeight * ldsWidth;
        constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

        auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
        auto* ldsPtrHi = ldsPtrLo + sizeLds;

        // Absolute write offsets for each segment (from the buffer base ptr)
        auto ldsWriteOffsetA = 0u;
        auto ldsWriteOffsetBGate
            = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);
        auto ldsWriteOffsetBUp = ldsWriteOffsetBGate
                                 + LWBuffBGateMap1d::fromMatrixCoord(
                                     make_coord2d(LWBuffBGateShape::BlockHeight, 0u), ldsld);

        // Per-warp read offsets (warp selects its row/col slice within the segment)
        using LWBuffAMap1d2     = GetDataLayout_t<LWBuffA>;
        using LWBuffBGateMap1d2 = GetDataLayout_t<LWBuffBGate>;
        using LWBuffBUpMap1d2   = GetDataLayout_t<LWBuffBUp>;

        auto ldsReadOffsetA
            = ldsWriteOffsetA
              + LWBuffAMap1d2::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetBGate = ldsWriteOffsetBGate
                                  + LWBuffBGateMap1d2::fromMatrixCoord(
                                      make_coord2d(get<1>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetBUp
            = ldsWriteOffsetBUp
              + LWBuffBUpMap1d2::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

        // ------------------------------------------------------------------
        // Write initial prefetch to Lo buffer
        // ------------------------------------------------------------------
        localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld);
        localWriteCoopBGate(ldsPtrLo + ldsWriteOffsetBGate, grBuffBGate, ldsld);
        localWriteCoopBUp(ldsPtrLo + ldsWriteOffsetBUp, grBuffBUp, ldsld);

        // ------------------------------------------------------------------
        // Initialize gate and up accumulators
        // ------------------------------------------------------------------
        MfmaFragAcc fragsAccGate[BLOCKS_X][BLOCKS_Y];
        MfmaFragAcc fragsAccUp[BLOCKS_X][BLOCKS_Y];
        clear_acc_fragments(fragsAccGate, ComputeT(0));
        clear_acc_fragments(fragsAccUp, ComputeT(0));

        synchronize_workgroup();

        // ------------------------------------------------------------------
        // K-loop with double-buffer prefetch
        //
        // WHY double-buffer?
        //   Global memory reads have high latency.  By issuing the NEXT
        //   tile's global read while the CURRENT tile's MFMA compute is
        //   in flight, we overlap memory latency with arithmetic --
        //   keeping both the memory bus and the MFMA units busy
        //   simultaneously.
        //
        // HOW it works:
        //   - Two LDS buffers (Lo and Hi) alternate each iteration.
        //   - Iter i : compute from Lo,  prefetch into Hi.
        //   - Iter i+1 : compute from Hi, prefetch into Lo.
        //   - The swap avoids write-after-read (WAR) hazards because
        //     the producer and consumer never touch the same buffer
        //     in the same iteration.
        // ------------------------------------------------------------------
        for(uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
        {
            MfmaFragA     fragsA[BLOCKS_X];
            MfmaFragBGate fragsBGate[BLOCKS_Y];
            MfmaFragBUp   fragsBUp[BLOCKS_Y];

            // Read current tile from Lo buffer
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadBGate(fragsBGate, ldsPtrLo + ldsReadOffsetBGate, ldsld);
            localReadBUp(fragsBUp, ldsPtrLo + ldsReadOffsetBUp, ldsld);

            // Prefetch next K-step from global memory
            globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
            globalReadCoopBGate(grBuffBGate, b_gate + globalReadOffsetBGate, ldBGate);
            globalReadCoopBUp(grBuffBUp, b_up + globalReadOffsetBUp, ldBUp);

            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetBGate += kStepOffsetBGate;
            globalReadOffsetBUp += kStepOffsetBUp;

            // Accumulate
            mfma_warp_tile(fragsAccGate, fragsA, fragsBGate, fragsAccGate);
            mfma_warp_tile(fragsAccUp, fragsA, fragsBUp, fragsAccUp);

            // Write prefetch to Hi buffer
            localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld);
            localWriteCoopBGate(ldsPtrHi + ldsWriteOffsetBGate, grBuffBGate, ldsld);
            localWriteCoopBUp(ldsPtrHi + ldsWriteOffsetBUp, grBuffBUp, ldsld);

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
            MfmaFragA     fragsA[BLOCKS_X];
            MfmaFragBGate fragsBGate[BLOCKS_Y];
            MfmaFragBUp   fragsBUp[BLOCKS_Y];

            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadBGate(fragsBGate, ldsPtrLo + ldsReadOffsetBGate, ldsld);
            localReadBUp(fragsBUp, ldsPtrLo + ldsReadOffsetBUp, ldsld);

            mfma_warp_tile(fragsAccGate, fragsA, fragsBGate, fragsAccGate);
            mfma_warp_tile(fragsAccUp, fragsA, fragsBUp, fragsAccUp);
        }

        // ------------------------------------------------------------------
        // Fused SwiGLU: D = silu(gate) * up  (in registers)
        //
        // WHY fuse here?  Both accGate and accUp are already in register
        // fragments after the K-loop.  Applying silu() and the Hadamard
        // product element-wise on registers avoids writing two [M x N]
        // intermediate matrices to global memory and reading them back --
        // saving 4 * M * N * sizeof(ComputeT) bytes of global traffic.
        // ------------------------------------------------------------------
        MfmaFragComputeOut fragsD[BLOCKS_X][BLOCKS_Y];
        apply_swiglu(fragsD, fragsAccGate, fragsAccUp);

        // ------------------------------------------------------------------
        // Store D to global memory
        // ------------------------------------------------------------------
        using MfmaFragStoreOutMap1d = GetDataLayout_t<MfmaFragStoreOut>;
        globalWriteD(d + MfmaFragStoreOutMap1d::fromMatrixCoord(warpTileCoord, ldd), fragsD, ldd);
    }
}

// ---------------------------------------------------------------------------
// CPU reference: D[i,j] = silu(sum_k A[i,k]*BGate[k,j]) * sum_k A[i,k]*BUp[k,j]
// Both A, BGate, BUp are row_major; D is row_major.
// ---------------------------------------------------------------------------
static void swiglu_cpu_ref(uint32_t      m,
                           uint32_t      n,
                           uint32_t      k,
                           InputT const* a,
                           InputT const* b_gate,
                           InputT const* b_up,
                           OutputT*      d,
                           uint32_t      lda,
                           uint32_t      ldBGate,
                           uint32_t      ldBUp,
                           uint32_t      ldd)
{
    // row_major index: element(row, col) = ptr[row * ld + col]
    auto rowMjr = [](uint32_t r, uint32_t c, uint32_t ld) { return r * ld + c; };

    auto silu_f = [](float x) -> float {
        if(x >= 0.0f)
            return x / (1.0f + std::expf(-x));
        else
        {
            float ex = std::expf(x);
            return x * ex / (1.0f + ex);
        }
    };

    // Note: single-threaded for simplicity; this is only for debug validation.
    for(int i = 0; i < (int)m; i++)
    {
        for(int j = 0; j < (int)n; j++)
        {
            float gate_acc = 0.0f;
            float up_acc   = 0.0f;
            for(int h = 0; h < (int)k; h++)
            {
                float a_val = static_cast<float>(a[rowMjr(i, h, lda)]);
                gate_acc += a_val * static_cast<float>(b_gate[rowMjr(h, j, ldBGate)]);
                up_acc += a_val * static_cast<float>(b_up[rowMjr(h, j, ldBUp)]);
            }
            d[rowMjr(i, j, ldd)] = static_cast<OutputT>(silu_f(gate_acc) * up_acc);
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
ROCWMMA_HOST void run_swiglu_sample(uint32_t m, uint32_t n, uint32_t k)
{
    printDeviceInfo();

    // Runtime arch selection mirrors compile-time namespace
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

    // Architecture checks
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

    if(m % get<0>(macroTileSize) || n % get<1>(macroTileSize) || k % hROCWMMA_K)
    {
        std::cout << "Unsupported matrix size!" << " M must be a multiple of "
                  << get<0>(macroTileSize) << ", N must be a multiple of " << get<1>(macroTileSize)
                  << ", K must be a multiple of " << hROCWMMA_K << "\n";
        return;
    }

    // Leading dimensions -- all row_major
    uint32_t lda     = k; // A  [M x K] row_major
    uint32_t ldBGate = n; // Bg [K x N] row_major
    uint32_t ldBUp   = n; // Bu [K x N] row_major
    uint32_t ldd     = n; // D  [M x N] row_major

    std::cout << "Initializing host data (m=" << m << " n=" << n << " k=" << k << ")...\n";

    std::vector<InputT>  matA(m * k);
    std::vector<InputT>  matBgate(k * n);
    std::vector<InputT>  matBup(k * n);
    std::vector<OutputT> matD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    // fillRand fills mat[i*cols+j] (row_major), values are small integers 0..4.
    // Scale by 1/16: prevents silu(gate)*up from overflowing FP16 (max 65504).
    // Without scaling: K=128, max|gate| ~ 128*4*4 = 2048, silu(2048)*2048 = 4M >> 65504.
    // With 1/16 scale: max|gate| ~ 128*(4/16)^2 = 2, silu(2)*2 ~ 3.8, safe in FP16.
    constexpr float kScale = 1.0f / 16.0f;
    fillRand(matA.data(), m, k);
    fillRand(matBgate.data(), k, n);
    fillRand(matBup.data(), k, n);
    for(auto& x : matA)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matBgate)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matBup)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);

    std::cout << "Allocating device memory...\n";

    InputT*  d_a;
    InputT*  d_bgate;
    InputT*  d_bup;
    OutputT* d_d;

    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_bgate, k * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_bup, k * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * sizeof(OutputT)));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matA.data(), m * k * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_bgate, matBgate.data(), k * n * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_bup, matBup.data(), k * n * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matD.data(), m * n * sizeof(OutputT), hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, get<0>(macroTileSize)),
                        rocwmma::ceil_div(n, get<1>(macroTileSize)));

    std::cout << "gridDim  (" << gridDim.x << " " << gridDim.y << ")" << "  blockDim ("
              << blockDim.x << " " << blockDim.y << ")\n";

    // LDS: 2 ping-pong buffers, each = (3 segments) * MACRO_TILE_K columns
    using LWBuffAShape     = GetIOShape_t<LWBuffA>;
    using LWBuffBGateShape = GetIOShape_t<LWBuffBGate>;
    using LWBuffBUpShape   = GetIOShape_t<LWBuffBUp>;
    constexpr uint32_t ldsSegHeight
        = LWBuffAShape::BlockHeight + LWBuffBGateShape::BlockHeight + LWBuffBUpShape::BlockHeight;
    int ldsUsage = 2 * sizeof(InputT) * ldsSegHeight * MACRO_TILE_K;
    std::cout << "LDS usage: " << ldsUsage << " bytes (" << ldsUsage / 1024 << " KiB)\n";

    auto kernelLambda = [&]() {
        hipExtLaunchKernelGGL(gemm_swiglu_fused,
                              gridDim,
                              blockDim,
                              ldsUsage,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              m,
                              n,
                              k,
                              d_a,
                              d_bgate,
                              d_bup,
                              d_d,
                              lda,
                              ldBGate,
                              ldBUp,
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

    // Performance: count 2 GEMMs (each 2MNK FLOPs) + MN elementwise (negligible)
    double gFlopsPerRun = 2.0 * 2.0 * static_cast<double>(m) * static_cast<double>(n)
                          * static_cast<double>(k) * 1e-9;
    double tFlopsPerSec
        = gFlopsPerRun * recordRuns / (static_cast<double>(elapsedMs) * 1e-3) * 1e-3;

    std::cout << std::left << std::setw(8) << "TBlkX" << std::setw(8) << "TBlkY" << std::setw(6)
              << "BlkM" << std::setw(6) << "BlkN" << std::setw(6) << "BlkK" << std::setw(8)
              << "MatM" << std::setw(8) << "MatN" << std::setw(8) << "MatK" << std::setw(8) << "lda"
              << std::setw(8) << "ldBGate" << std::setw(8) << "ldBUp" << std::setw(8) << "ldd"
              << std::setw(14) << "elapsedMs" << std::setw(22) << "GFlops(2xGEMM)" << std::setw(12)
              << "TFlops/s" << std::setw(10) << "Warmups" << std::setw(10) << "Runs" << "\n";

    std::cout << std::left << std::setw(8) << hTBLOCK_X << std::setw(8) << hTBLOCK_Y << std::setw(6)
              << hROCWMMA_M << std::setw(6) << hROCWMMA_N << std::setw(6) << hROCWMMA_K
              << std::setw(8) << m << std::setw(8) << n << std::setw(8) << k << std::setw(8) << lda
              << std::setw(8) << ldBGate << std::setw(8) << ldBUp << std::setw(8) << ldd
              << std::setw(14) << elapsedMs << std::setw(22) << (gFlopsPerRun * recordRuns)
              << std::setw(12) << tFlopsPerSec << std::setw(10) << warmups << std::setw(10)
              << recordRuns << "\n";

#if !NDEBUG
    std::cout << "\nValidating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(matD.data(), d_d, m * n * sizeof(OutputT), hipMemcpyDeviceToHost));

    std::vector<OutputT> matDref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
    swiglu_cpu_ref(m,
                   n,
                   k,
                   matA.data(),
                   matBgate.data(),
                   matBup.data(),
                   matDref.data(),
                   lda,
                   ldBGate,
                   ldBUp,
                   ldd);

    auto res = compareEqual(matD.data(), matDref.data(), m * n);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_bgate));
    CHECK_HIP_ERROR(hipFree(d_bup));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!\n";
}

// ---------------------------------------------------------------------------
// Usage:
//   ./simple_gemm_swiglu          # Quick validation only (64 x 256 x 128)
//   ./simple_gemm_swiglu --all    # Also run full LLaMA-2 7b/13b/70b sizes
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << "Community Sample: SwiGLU Fused Dual GEMM" << std::endl;
    std::cout << "This sample demonstrates: fused dual-GEMM + SiLU activation"
              << " (LLaMA/Mistral FFN gate layer) using rocWMMA" << std::endl;

    // Check for --all flag
    bool runAll = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--all")
            runAll = true;
    }

    // Quick validation case (small enough for any GPU, all multiples of tile size 16)
    // run_swiglu_sample(seq_len, intermediate, hidden_dim);
    run_swiglu_sample(64, 256, 128);

    // Full LLaMA-2 configurations (require significant GPU memory)
    if(runAll)
    {
        // Llama-2-7b
        // https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
        run_swiglu_sample(64, 11008, 4096);

        // Llama-2-13b
        // https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/blob/main/config.json
        run_swiglu_sample(64, 13824, 5120);

        // Llama-2-70b
        // https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/blob/main/config.json
        run_swiglu_sample(64, 28672, 8192);
    }
    else
    {
        std::cout << "\nTip: pass --all to also run full LLaMA-2 7b/13b/70b sizes.\n";
    }

    std::cout << "Sample completed successfully!" << std::endl;
    return 0;
}
