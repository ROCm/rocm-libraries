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

/* COMMUNITY SAMPLE: Stream-K GEMM (work-centric persistent scheduler)
 *
 * ============================================================================
 * 1. WHAT IS STREAM-K?
 * ============================================================================
 *
 * Stream-K (Osama et al., 2023) is a *grid-level scheduling* strategy for
 * dense GEMM that re-formulates the problem from "output-tile-centric"
 * (one CTA per Cij tile) to "work-centric" (one CTA per equal slice of the
 * total inner-loop work).  It targets the wave-quantization / last-wave
 * under-utilization that classic tiled GEMM suffers from when the number
 * of output tiles is not a multiple of the number of compute units (CUs).
 *
 * Reference:
 *   [1] Osama, Cecka et al., "Stream-K: Work-centric Parallel Decomposition
 *       for Dense Matrix-Matrix Multiplication on the GPU", PPoPP'23
 *       https://arxiv.org/abs/2301.03598
 *   [2] Colfax Research, "CUTLASS Tutorial: Persistent Kernels and Stream-K"
 *       https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
 *
 * ============================================================================
 * 2. THE WAVE QUANTIZATION PROBLEM
 * ============================================================================
 *
 * Classic tiled GEMM launches one CTA per output tile and lets each CTA do
 * its own full K-loop.  If a GPU has 4 CUs and the problem produces 9
 * output tiles, the schedule looks like:
 *
 *      Wave 0:  CU0 CU1 CU2 CU3   (4 / 4 busy)
 *      Wave 1:  CU0 CU1 CU2 CU3   (4 / 4 busy)
 *      Wave 2:  CU0 .   .   .     (1 / 4 busy) <-- 75% utilization
 *
 * Stream-K instead views the GEMM as a single linear "work stream" of
 * total_iters = num_output_tiles * iters_per_tile inner-loop K-steps, and
 * gives every persistent CTA an equal contiguous slice of that stream.
 * A CTA's slice may span multiple output tiles, and a tile may be split
 * across several CTAs (sharing the K-dimension).  Partial accumulators
 * are then reduced across CTAs.
 *
 * ============================================================================
 * 3. DESIGN OF THIS SAMPLE
 * ============================================================================
 *
 * The sample provides TWO kernels that share the SAME rocWMMA inner GEMM:
 *
 *   (A)  gemm_dp_kernel        -- baseline data-parallel ("classic tiled")
 *                                 grid = (num_M_tiles, num_N_tiles)
 *                                 each CTA owns one full Cij tile and writes
 *                                 D[Cij] directly (cast fp32 -> fp16).
 *
 *   (B)  gemm_streamk_kernel   -- Stream-K persistent worker
 *                                 grid = (num_workers,)  ~= multiProcessorCount
 *                                 each CTA processes a contiguous range of
 *                                 [tile_id, k_iter] iterations and atomicAdds
 *                                 partial accumulators into a fp32 workspace.
 *
 *   (C)  gemm_streamk_finish_kernel
 *                                 1D kernel that converts the fp32 workspace
 *                                 to fp16 D[i,j] = (OutputT)workspace[i,j].
 *
 * Both (A) and (B) reuse identical rocWMMA building blocks:
 *   - Cooperative global reads (fragment_scheduler::coop_row_major_2d)
 *   - LDS double-buffered K-loop (Lo / Hi ping-pong)
 *   - rocWMMA mma_sync warp-tile MFMA accumulation
 *
 * The ONLY difference is the OUTER schedule.  This makes the sample useful
 * for studying how grid-level scheduling alone can reshape utilization
 * without changing the inner compute path.
 *
 * ============================================================================
 * 4. WHAT YOU WILL LEARN FROM THIS SAMPLE
 * ============================================================================
 *
 *   - How rocWMMA fragment / mma_sync / load/store_matrix_sync compose with
 *     two different grid schedulers without changing the warp-tile compute
 *   - How to assign a contiguous slice of total GEMM inner-loop work to a
 *     persistent CTA (work-centric decomposition)
 *   - How to reuse the same LDS allocation for the K-loop ping-pong AND for
 *     a fp32 commit-staging tile (via reinterpret_cast)
 *   - How to commit a partial-K accumulator via LDS staging + cooperative
 *     atomicAdd into a global fp32 reduction workspace
 *   - How to choose benchmark shapes that expose wave quantization so that
 *     the Stream-K benefit is visible
 *
 * ============================================================================
 * 5. STREAM-K WORK ASSIGNMENT (math)
 * ============================================================================
 *
 *   iters_per_tile = K / MACRO_TILE_K
 *   total_iters    = num_M_tiles * num_N_tiles * iters_per_tile
 *
 *   For worker w in [0, num_workers):
 *     base   = total_iters / num_workers
 *     extra  = total_iters % num_workers
 *     start  = w * base + min(w, extra)
 *     end    = start + base + (w < extra ? 1 : 0)
 *
 *   Each iter index i decomposes as:
 *     tile_id = i / iters_per_tile
 *     k_iter  = i % iters_per_tile
 *     tile_x  = tile_id / num_N_tiles
 *     tile_y  = tile_id % num_N_tiles
 *
 *   The worker walks tiles (start_tile .. end_tile).  For each tile the
 *   "mini K-loop" range is:
 *     k_lo = (tile == start_tile) ? (start % iters_per_tile) : 0
 *     k_hi = (tile == end_tile  ) ? ((end-1) % iters_per_tile + 1)
 *                                 : iters_per_tile
 *
 * ============================================================================
 * 6. KERNEL DATA-FLOW (Stream-K mode)
 * ============================================================================
 *
 *   Global Memory --coop load--> LDS (2 segments, double-buffered)
 *        |                            |
 *      A [MxK]                  [A    segment]
 *      B [KxN]                  [B^T  segment]
 *                                     |
 *                               local read
 *                                     |
 *                    +--- fragsA ---+--- fragsB ---+
 *                    |                             |
 *                    +-----  mfma warp tile  ------+
 *                    |
 *               fragsAcc (fp32, BLOCKS_X x BLOCKS_Y)
 *                    |                             (per-tile mini K-loop)
 *                    v
 *               commit_partial:
 *                    |
 *                    +--> store_matrix_sync to LDS_fp32 staging
 *                    +--> cooperative atomicAdd LDS -> workspace[tx,ty]
 *
 *   After kernel:  finish kernel converts fp32 workspace -> fp16 D (cast).
 *
 * ============================================================================
 * 7. LDS LAYOUT (per block)
 * ============================================================================
 *
 *   K-loop phase  (col_major ping-pong, two buffers Lo / Hi)
 *     One buffer = (MACRO_TILE_X + MACRO_TILE_Y) * MACRO_TILE_K * sizeof(fp16)
 *                = (64 + 64) * 16 * 2 = 4096 B
 *     Total K-loop LDS (Lo + Hi)         = 8 KiB
 *
 *   Commit phase  (row_major fp32 staging, REUSES same memory region)
 *     One staging tile = MACRO_TILE_X * MACRO_TILE_Y * sizeof(fp32)
 *                      = 64 * 64 * 4 = 16384 B = 16 KiB
 *
 *   Allocate per block = max(8 KiB, 16 KiB) = 16 KiB   (well under 64 KiB)
 *
 *   Note: the two phases never overlap in time (K-loop fully completes
 *   before commit starts), so reinterpret_cast the same base pointer.
 *
 * ============================================================================
 * 8. REQUIREMENTS
 * ============================================================================
 *
 *   - Minimum ROCm version: ROCm 6.0+
 *   - GPU architectures   : gfx9 / gfx11 / gfx12
 *                           (tested on RDNA4 gfx1201 / RX9070; other archs are
 *                           compile-time parameter paths and have not been validated)
 *   - Data types          : float16 input, float32 compute / workspace,
 *                           float16 output
 *   - Matrix dimensions   : M multiple of MACRO_TILE_X (64),
 *                           N multiple of MACRO_TILE_Y (64),
 *                           K multiple of MACRO_TILE_K (16)
 *
 * ============================================================================
 * 9. KNOWN LIMITATIONS
 * ============================================================================
 *
 *   - No boundary handling: tile-aligned shapes only
 *   - Only row_major layout supported for A, B, D
 *   - Stream-K commit always uses atomicAdd; production Stream-K skips the
 *     atomic when only one worker contributes to a tile (the
 *     "fully-contained" fast path).  This sample stays uniform for clarity.
 *   - The number of persistent workers defaults to multiProcessorCount; pass
 *     --workers N on the command line to override.
 *   - fp32 reduction workspace is M*N*sizeof(float); large for huge shapes.
 *
 * Note: This is a community-contributed sample provided as-is. It may not be
 * maintained with the same rigor as official samples.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
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
//
// NOTE: identical macro-tile geometry to simple_gemm_swiglu.cpp so readers
// can compare the two samples line-by-line and see that ONLY the scheduler
// differs.  WARPS_X = TBLOCK_X / WARP_SIZE = 2 on both CDNA3 (wave64) and
// RDNA4 (wave32), giving MACRO_TILE_X = MACRO_TILE_Y = 64 universally.
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
// Types and Data Layouts
// ---------------------------------------------------------------------------
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = row_major;
using DataLayoutB   = row_major; // [K x N] row_major, ldb = N
using DataLayoutD   = row_major;
using DataLayoutLds = col_major;

// Stream-K reduction workspace is row_major fp32 [M x N]
using WorkspaceT       = float32_t;
using DataLayoutAccLds = row_major; // commit-stage LDS layout

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
constexpr uint32_t TBLOCK_TOTAL = TBLOCK_X * TBLOCK_Y;

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------

// Per-block MFMA fragments (warp tile element)
using MfmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
using MfmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;

// Final output store fragment (used by the data-parallel kernel)
using MfmaFragStoreOut
    = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutD>;

// Cooperative global read (macro tile) -- inner pieces shared by both kernels
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffA
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;
using GRBuffB
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB, CoopScheduler>;

// Local write (macro tile) -- col_major LDS; B is transposed
using LWBuffA = apply_data_layout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = apply_data_layout_t<apply_transpose_t<GRBuffB>, DataLayoutLds>;

// Local read (MFMA fragment-level)
using LRFragA = apply_data_layout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<apply_transpose_t<MfmaFragB>, DataLayoutLds>;

// Commit-stage staging fragment (fp32, row_major, accumulator semantics)
using LWFragAcc = apply_data_layout_t<MfmaFragAcc, DataLayoutAccLds>;

// ---------------------------------------------------------------------------
// LDS sizing (per workgroup)
//
// The K-loop allocates two ping-pong buffers stacked vertically.  The commit
// phase later reinterprets the same memory as a single fp32 staging tile.
// We need to allocate the maximum of the two requirements.
// ---------------------------------------------------------------------------
constexpr uint32_t LDS_KLOOP_HEIGHT_PER_BUF
    = GetIOShape_t<LWBuffA>::BlockHeight + GetIOShape_t<LWBuffB>::BlockHeight;
constexpr uint32_t LDS_KLOOP_BYTES  = 2u * LDS_KLOOP_HEIGHT_PER_BUF * MACRO_TILE_K * sizeof(InputT);
constexpr uint32_t LDS_COMMIT_BYTES = MACRO_TILE_X * MACRO_TILE_Y * sizeof(WorkspaceT);
constexpr uint32_t LDS_TOTAL_BYTES
    = (LDS_KLOOP_BYTES > LDS_COMMIT_BYTES) ? LDS_KLOOP_BYTES : LDS_COMMIT_BYTES;

// =============================================================================
// SECTION I: rocWMMA inner pieces (shared by data-parallel and Stream-K)
// =============================================================================

// --- Cooperative global reads --------------------------------------------------
ROCWMMA_DEVICE static inline void globalReadCoopA(GRBuffA& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void globalReadCoopB(GRBuffB& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

// --- Cooperative local writes (LDS) --------------------------------------------
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

// --- Local reads (per-warp, K-step granularity) --------------------------------
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

// --- Accumulator helpers -------------------------------------------------------
ROCWMMA_DEVICE static inline void clear_acc_fragments(MfmaFragAcc (&frags)[BLOCKS_X][BLOCKS_Y],
                                                      ComputeT val)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            fill_fragment(frags[i][j], val);
}

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

// --- Direct cast-and-store of accumulator fragments to row_major fp16 D --------
//
// Used by the data-parallel baseline kernel (no reduction needed since each
// CTA owns its full Cij tile).  Mirrors the fp32 -> fp16 store pattern from
// simple_gemm_swiglu.cpp.
ROCWMMA_DEVICE static inline void globalWriteD_direct(
    OutputT* gAddrD, MfmaFragAcc const (&fragsAcc)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
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
            for(int e = 0; e < (int)fragsAcc[i][j].num_elements; e++)
                fragOut.x[e] = static_cast<OutputT>(fragsAcc[i][j].x[e]);
            store_matrix_sync(gAddrD + offsetY, fragOut, ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

// =============================================================================
// SECTION II: Shared K-loop building blocks
//
// Both kernels run a K-loop over a contiguous range [k_iter_lo, k_iter_hi) of
// MACRO_TILE_K-sized K-steps for one (tileX, tileY) output tile.  Wrapping
// this in a device function keeps both kernels short and lets the reader see
// that the rocWMMA inner work is identical.
// =============================================================================

// Run the LDS double-buffered K-loop for a given output tile, accumulating
// from k_iter = k_lo to k_iter = k_hi - 1 inclusive into fragsAcc.
// Caller supplies a zeroed-or-preloaded fragsAcc and the LDS base ptr.
ROCWMMA_DEVICE static inline void run_kloop(MfmaFragAcc (&fragsAcc)[BLOCKS_X][BLOCKS_Y],
                                            InputT const* a,
                                            InputT const* b,
                                            uint32_t      lda,
                                            uint32_t      ldb,
                                            uint32_t      tileX, // macro-tile row index
                                            uint32_t      tileY, // macro-tile col index
                                            uint32_t      k_lo, // first K-iter (inclusive)
                                            uint32_t      k_hi, // last  K-iter (exclusive)
                                            void*         localMemPtr,
                                            uint32_t      localWarpRowOffset,
                                            uint32_t      localWarpColOffset)
{
    if(k_lo >= k_hi)
        return; // empty range -- nothing to do

    // -------------------------------------------------------------------
    // Macro-tile origin in the (M, K) and (K, N) global addressing
    // -------------------------------------------------------------------
    using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
    using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

    // Initial K-step start position (absolute K offset into A and B)
    uint32_t kStart_elements = k_lo * MACRO_TILE_K;

    auto globalReadOffsetA
        = GRBuffAMap1d::fromMatrixCoord(make_coord2d(tileX * MACRO_TILE_X, kStart_elements), lda);
    auto globalReadOffsetB
        = GRBuffBMap1d::fromMatrixCoord(make_coord2d(kStart_elements, tileY * MACRO_TILE_Y), ldb);

    auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), lda);
    auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldb);

    // -------------------------------------------------------------------
    // LDS layout (col_major, A on top of B^T)
    // -------------------------------------------------------------------
    using LWBuffAShape = GetIOShape_t<LWBuffA>;
    using LWBuffBShape = GetIOShape_t<LWBuffB>;
    using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
    using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;

    constexpr uint32_t ldsWidth  = MACRO_TILE_K;
    constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
    constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
    constexpr uint32_t ldsld     = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

    auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto* ldsPtrHi = ldsPtrLo + sizeLds;

    // Absolute write offsets for each segment (from buffer base)
    auto ldsWriteOffsetA = 0u;
    auto ldsWriteOffsetB
        = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

    // Per-warp read offsets (warp selects its row/col slice within the segment)
    auto ldsReadOffsetA
        = ldsWriteOffsetA
          + LWBuffAMap1d::fromMatrixCoord(make_coord2d(localWarpRowOffset, 0u), ldsld);
    auto ldsReadOffsetB
        = ldsWriteOffsetB
          + LWBuffBMap1d::fromMatrixCoord(make_coord2d(localWarpColOffset, 0u), ldsld);

    // -------------------------------------------------------------------
    // Initial prefetch (k = k_lo) into Lo
    // -------------------------------------------------------------------
    GRBuffA grBuffA;
    GRBuffB grBuffB;

    globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
    globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);

    globalReadOffsetA += kStepOffsetA;
    globalReadOffsetB += kStepOffsetB;

    localWriteCoopA(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld);
    localWriteCoopB(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld);

    synchronize_workgroup();

    // -------------------------------------------------------------------
    // Main K-loop with double-buffer prefetch (Lo / Hi)
    // -------------------------------------------------------------------
    for(uint32_t k_iter = k_lo + 1u; k_iter < k_hi; ++k_iter)
    {
        MfmaFragA fragsA[BLOCKS_X];
        MfmaFragB fragsB[BLOCKS_Y];

        // Read current tile from Lo
        localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
        localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

        // Prefetch next K-step into Hi
        globalReadCoopA(grBuffA, a + globalReadOffsetA, lda);
        globalReadCoopB(grBuffB, b + globalReadOffsetB, ldb);

        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);

        localWriteCoopA(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld);
        localWriteCoopB(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld);

        synchronize_workgroup();

        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }

    // -------------------------------------------------------------------
    // Tail: consume the last K-step still in Lo
    // -------------------------------------------------------------------
    {
        MfmaFragA fragsA[BLOCKS_X];
        MfmaFragB fragsB[BLOCKS_Y];

        localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
        localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
        mfma_warp_tile(fragsAcc, fragsA, fragsB, fragsAcc);
    }

    // Make sure all warps are done with LDS before the caller potentially
    // reinterprets it as the fp32 commit-staging tile.
    synchronize_workgroup();
}

// =============================================================================
// SECTION III: Stream-K commit (partial-K -> fp32 workspace via atomicAdd)
//
// 1. Each warp writes its accumulator block to the fp32 LDS staging tile via
//    rocWMMA store_matrix_sync (accumulator semantics, row_major).
// 2. After workgroup sync, ALL threads in the workgroup cooperatively walk
//    the staging tile (one fp32 each, strided by TBLOCK_TOTAL) and atomic-add
//    each element into workspace[(tx*MACRO_TILE_X + r) * ldd + (ty*MACRO_TILE_Y + c)].
// 3. After workgroup sync the LDS region can be reused for the next tile.
// =============================================================================
ROCWMMA_DEVICE static inline void commit_partial(WorkspaceT* workspace,
                                                 uint32_t    ldd_ws,
                                                 MfmaFragAcc const (&fragsAcc)[BLOCKS_X][BLOCKS_Y],
                                                 uint32_t tileX,
                                                 uint32_t tileY,
                                                 void*    localMemPtr,
                                                 uint32_t localWarpRowOffset,
                                                 uint32_t localWarpColOffset)
{
    // ---- Stage 1: store fragments to row_major fp32 LDS ----
    auto* lds_acc = reinterpret_cast<WorkspaceT*>(localMemPtr);

    constexpr uint32_t ldsld_acc = MACRO_TILE_Y; // row_major leading dim = #cols

    using FragShape = GetIOShape_t<LWFragAcc>;

    auto* warpStagingPtr = lds_acc + localWarpRowOffset * ldsld_acc + localWarpColOffset;

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            auto* blockPtr = warpStagingPtr + (i * FragShape::BlockHeight) * ldsld_acc
                             + (j * FragShape::BlockWidth);
            // Cast accumulator fragment to row_major LDS layout, then store.
            store_matrix_sync(
                blockPtr, apply_data_layout<DataLayoutAccLds>(fragsAcc[i][j]), ldsld_acc);
        }
    }

    synchronize_workgroup();

    // ---- Stage 2: cooperative atomicAdd LDS -> global workspace ----
    constexpr uint32_t TILE_ELEMS = MACRO_TILE_X * MACRO_TILE_Y;

    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t tile_row_origin = tileX * MACRO_TILE_X;
    const uint32_t tile_col_origin = tileY * MACRO_TILE_Y;

    for(uint32_t e = tid; e < TILE_ELEMS; e += TBLOCK_TOTAL)
    {
        uint32_t r = e / MACRO_TILE_Y;
        uint32_t c = e - r * MACRO_TILE_Y;

        ComputeT val = lds_acc[e]; // row_major linear -> (r, c)
        // Use `unsafeAtomicAdd` semantics by way of standard atomicAdd on float;
        // fp32 atomicAdd is widely supported on AMD GPUs through HIP.
        atomicAdd(workspace + (tile_row_origin + r) * ldd_ws + (tile_col_origin + c), val);
    }

    // Make sure all threads in the workgroup have finished using LDS before
    // the next tile reuses it (could be K-loop ping-pong again).
    synchronize_workgroup();
}

// =============================================================================
// SECTION IV: Kernel A -- baseline data-parallel ("classic tiled GEMM")
//
//   gridDim  = (num_M_tiles, num_N_tiles)
//   blockDim = (TBLOCK_X, TBLOCK_Y)
//
//   Each CTA owns one full output tile and writes D[Cij] = A[i,:] * B[:,j]
//   directly (cast fp32 -> fp16).  Zero workspace, zero atomics.
//
//   Suffers from wave quantization when num_output_tiles is not a multiple
//   of multiProcessorCount: the last wave runs at fractional utilization.
// =============================================================================
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_TOTAL) gemm_dp_kernel(uint32_t      m,
                                                                   uint32_t      n,
                                                                   uint32_t      k,
                                                                   InputT const* a,
                                                                   InputT const* b,
                                                                   OutputT*      d,
                                                                   uint32_t      lda,
                                                                   uint32_t      ldb,
                                                                   uint32_t      ldd)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // ----- Per-warp coordinates in the macro tile owned by this CTA -----
        constexpr auto warpTileSize = make_coord2d(WARP_TILE_X, WARP_TILE_Y);

        auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto localWarpOffset = localWarpCoord * warpTileSize;

        const uint32_t tileX = blockIdx.x;
        const uint32_t tileY = blockIdx.y;

        // Skip warps that fall outside the output matrix (defensive; M/N
        // are required to be tile-aligned by the host driver).
        const uint32_t warp_row_end = tileX * MACRO_TILE_X + get<0>(localWarpOffset) + WARP_TILE_X;
        const uint32_t warp_col_end = tileY * MACRO_TILE_Y + get<1>(localWarpOffset) + WARP_TILE_Y;
        if(warp_row_end > m || warp_col_end > n)
            return;

        // ----- LDS base ptr (used only for the K-loop double buffer here) -----
        HIP_DYNAMIC_SHARED(void*, localMemPtr);

        // ----- Run the full K-loop into a fresh accumulator -----
        MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
        clear_acc_fragments(fragsAcc, ComputeT(0));

        const uint32_t iters_per_tile = k / MACRO_TILE_K;
        run_kloop(fragsAcc,
                  a,
                  b,
                  lda,
                  ldb,
                  tileX,
                  tileY,
                  /*k_lo=*/0u,
                  /*k_hi=*/iters_per_tile,
                  localMemPtr,
                  get<0>(localWarpOffset),
                  get<1>(localWarpOffset));

        // ----- Direct cast-and-store fp32 -> fp16 D -----
        using MfmaFragStoreOutMap1d = GetDataLayout_t<MfmaFragStoreOut>;
        const auto warpTileCoord    = make_coord2d(tileX * MACRO_TILE_X + get<0>(localWarpOffset),
                                                tileY * MACRO_TILE_Y + get<1>(localWarpOffset));
        globalWriteD_direct(
            d + MfmaFragStoreOutMap1d::fromMatrixCoord(warpTileCoord, ldd), fragsAcc, ldd);
    }
}

// =============================================================================
// SECTION V: Kernel B -- Stream-K persistent worker
//
//   gridDim  = (num_workers,)            # ~= multiProcessorCount
//   blockDim = (TBLOCK_X, TBLOCK_Y)
//
//   Each CTA processes a contiguous range [start_iter, end_iter) of the total
//   GEMM work stream, where total_iters = num_M_tiles*num_N_tiles*iters_per_tile.
//   Partial accumulators are committed to a fp32 reduction workspace via
//   LDS staging + atomicAdd.
// =============================================================================
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_TOTAL) gemm_streamk_kernel(uint32_t      m,
                                                                        uint32_t      n,
                                                                        uint32_t      k,
                                                                        InputT const* a,
                                                                        InputT const* b,
                                                                        WorkspaceT*   workspace,
                                                                        uint32_t      lda,
                                                                        uint32_t      ldb,
                                                                        uint32_t      ldd_ws,
                                                                        uint32_t      num_workers)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        const uint32_t worker_id = blockIdx.x;
        if(worker_id >= num_workers)
            return;

        // ----- Per-warp coordinates within the macro tile -----
        constexpr auto warpTileSize = make_coord2d(WARP_TILE_X, WARP_TILE_Y);

        auto localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto localWarpOffset = localWarpCoord * warpTileSize;

        // ----- Stream-K work assignment -----
        const uint32_t num_M_tiles    = m / MACRO_TILE_X;
        const uint32_t num_N_tiles    = n / MACRO_TILE_Y;
        const uint32_t iters_per_tile = k / MACRO_TILE_K;
        const uint32_t num_tiles      = num_M_tiles * num_N_tiles;
        const uint32_t total_iters    = num_tiles * iters_per_tile;

        const uint32_t base       = total_iters / num_workers;
        const uint32_t extra      = total_iters % num_workers;
        const uint32_t start_iter = worker_id * base + (worker_id < extra ? worker_id : extra);
        const uint32_t end_iter   = start_iter + base + (worker_id < extra ? 1u : 0u);
        if(start_iter >= end_iter)
            return; // worker has no work (can happen if num_workers > total_iters)

        const uint32_t start_tile = start_iter / iters_per_tile;
        const uint32_t end_tile   = (end_iter - 1u) / iters_per_tile;

        HIP_DYNAMIC_SHARED(void*, localMemPtr);

        // ----- Walk every tile this worker touches -----
        for(uint32_t tile = start_tile; tile <= end_tile; ++tile)
        {
            const uint32_t k_lo = (tile == start_tile) ? (start_iter - tile * iters_per_tile) : 0u;
            const uint32_t k_hi
                = (tile == end_tile) ? (end_iter - tile * iters_per_tile) : iters_per_tile;

            const uint32_t tileX = tile / num_N_tiles;
            const uint32_t tileY = tile - tileX * num_N_tiles;

            // ----- Run mini K-loop for [k_lo, k_hi) -----
            MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
            clear_acc_fragments(fragsAcc, ComputeT(0));

            run_kloop(fragsAcc,
                      a,
                      b,
                      lda,
                      ldb,
                      tileX,
                      tileY,
                      k_lo,
                      k_hi,
                      localMemPtr,
                      get<0>(localWarpOffset),
                      get<1>(localWarpOffset));

            // ----- Commit partial accumulator to fp32 reduction workspace -----
            //
            // NOTE: For pedagogical clarity we ALWAYS use atomicAdd, even when
            // (k_lo == 0 && k_hi == iters_per_tile) implies sole ownership and
            // a non-atomic store would suffice.  Production Stream-K would add
            // a fast-path here and skip the global atomic.
            commit_partial(workspace,
                           ldd_ws,
                           fragsAcc,
                           tileX,
                           tileY,
                           localMemPtr,
                           get<0>(localWarpOffset),
                           get<1>(localWarpOffset));
        }
    }
}

// =============================================================================
// SECTION VI: Stream-K finish kernel
//
// 1D kernel: each thread reads one fp32 workspace element and writes the
// corresponding fp16 D element with a static_cast.  Trivially launchable as
// (M*N + 255) / 256 blocks of 256 threads.
// =============================================================================
ROCWMMA_KERNEL void
    gemm_streamk_finish_kernel(WorkspaceT const* workspace, OutputT* d, uint32_t total_elems)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= total_elems)
        return;
    d[tid] = static_cast<OutputT>(workspace[tid]);
}

// =============================================================================
// SECTION VII: CPU reference
// =============================================================================
static void gemm_cpu_ref(uint32_t      m,
                         uint32_t      n,
                         uint32_t      k,
                         InputT const* a,
                         InputT const* b,
                         OutputT*      d,
                         uint32_t      lda,
                         uint32_t      ldb,
                         uint32_t      ldd)
{
    auto rowMjr = [](uint32_t r, uint32_t c, uint32_t ld) { return r * ld + c; };
#pragma omp parallel for
    for(int i = 0; i < (int)m; i++)
    {
        for(int j = 0; j < (int)n; j++)
        {
            float acc = 0.0f;
            for(int h = 0; h < (int)k; h++)
            {
                acc += static_cast<float>(a[rowMjr(i, h, lda)])
                       * static_cast<float>(b[rowMjr(h, j, ldb)]);
            }
            d[rowMjr(i, j, ldd)] = static_cast<OutputT>(acc);
        }
    }
}

// =============================================================================
// SECTION VIII: Device info / argument parsing
// =============================================================================
struct DeviceCounters
{
    uint32_t multiProcessorCount;
    uint32_t warpSize;
};

static DeviceCounters getDeviceCounters()
{
    hipDevice_t     dev;
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDevice(&dev));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, dev));
    return {static_cast<uint32_t>(props.multiProcessorCount),
            static_cast<uint32_t>(props.warpSize)};
}

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

// =============================================================================
// SECTION IX: Host driver
// =============================================================================
enum class GemmSchedule
{
    DataParallel,
    StreamK
};

static const char* scheduleName(GemmSchedule s)
{
    switch(s)
    {
    case GemmSchedule::DataParallel:
        return "DataParallel";
    case GemmSchedule::StreamK:
        return "StreamK     ";
    }
    return "?";
}

ROCWMMA_HOST void run_gemm_sample(uint32_t     m,
                                  uint32_t     n,
                                  uint32_t     k,
                                  GemmSchedule sched,
                                  uint32_t     userWorkers, // 0 = auto (multiProcessorCount)
                                  bool         skipValidation = false)
{
    // Runtime arch-dependent params (mirror compile-time selection)
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
    uint32_t hMACRO_X = get<0>(macroTileSize);
    uint32_t hMACRO_Y = get<1>(macroTileSize);

    // ----- Architecture / shape sanity checks -----
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
    if(m % hMACRO_X || n % hMACRO_Y || k % hROCWMMA_K)
    {
        std::cout << "Unsupported matrix size! M must be a multiple of " << hMACRO_X
                  << ", N must be a multiple of " << hMACRO_Y << ", K must be a multiple of "
                  << hROCWMMA_K << "\n";
        return;
    }

    // Leading dims (all row_major)
    uint32_t lda = k;
    uint32_t ldb = n;
    uint32_t ldd = n;

    // ----- Worker count -----
    auto     dc          = getDeviceCounters();
    uint32_t num_workers = userWorkers ? userWorkers : dc.multiProcessorCount;
    if(num_workers == 0u)
        num_workers = 1u;

    // ----- Wave-quantization analysis (data-parallel baseline) -----
    uint32_t num_M_tiles = m / hMACRO_X;
    uint32_t num_N_tiles = n / hMACRO_Y;
    uint32_t num_tiles   = num_M_tiles * num_N_tiles;

    uint32_t dp_total_blocks = num_tiles;
    double   dp_waves
        = static_cast<double>(dp_total_blocks) / static_cast<double>(dc.multiProcessorCount);
    uint32_t dp_full_waves  = dp_total_blocks / dc.multiProcessorCount;
    uint32_t dp_tail_blocks = dp_total_blocks - dp_full_waves * dc.multiProcessorCount;
    double   dp_tail_util   = dp_tail_blocks ? static_cast<double>(dp_tail_blocks)
                                               / static_cast<double>(dc.multiProcessorCount)
                                             : 1.0;

    std::cout << "------------------------------------------------------------\n"
              << " run_gemm_sample(M=" << m << ", N=" << n << ", K=" << k
              << ", sched=" << scheduleName(sched) << ", workers=" << num_workers << ")\n"
              << "   macro tile : " << hMACRO_X << " x " << hMACRO_Y << " x " << hROCWMMA_K << "\n"
              << "   tiles      : " << num_M_tiles << " x " << num_N_tiles << " = " << num_tiles
              << "\n"
              << "   #CUs       : " << dc.multiProcessorCount << "\n"
              << "   DP waves   : " << dp_waves << "  (full=" << dp_full_waves
              << ", tail blocks=" << dp_tail_blocks << ", tail util=" << dp_tail_util * 100.0
              << "%)\n"
              << "------------------------------------------------------------\n";

    // ----- Allocate and initialise host data -----
    std::vector<InputT>  matA(static_cast<size_t>(m) * k);
    std::vector<InputT>  matB(static_cast<size_t>(k) * n);
    std::vector<OutputT> matD(static_cast<size_t>(m) * n,
                              std::numeric_limits<OutputT>::signaling_NaN());

    // Scale by 1/16 to keep A*B in fp16 dynamic range (same trick as
    // simple_gemm_swiglu).
    constexpr float kScale = 1.0f / 16.0f;
    fillRand(matA.data(), m, k);
    fillRand(matB.data(), k, n);
    for(auto& x : matA)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);
    for(auto& x : matB)
        x = static_cast<InputT>(static_cast<float>(x) * kScale);

    // ----- Allocate device buffers -----
    InputT*     d_a         = nullptr;
    InputT*     d_b         = nullptr;
    OutputT*    d_d         = nullptr;
    WorkspaceT* d_workspace = nullptr;

    CHECK_HIP_ERROR(hipMalloc(&d_a, static_cast<size_t>(m) * k * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, static_cast<size_t>(k) * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, static_cast<size_t>(m) * n * sizeof(OutputT)));
    if(sched == GemmSchedule::StreamK)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, static_cast<size_t>(m) * n * sizeof(WorkspaceT)));

    CHECK_HIP_ERROR(hipMemcpy(
        d_a, matA.data(), static_cast<size_t>(m) * k * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_b, matB.data(), static_cast<size_t>(k) * n * sizeof(InputT), hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);

    // ----- Per-mode launcher -----
    auto launch = [&]() {
        if(sched == GemmSchedule::DataParallel)
        {
            dim3 gridDim(num_M_tiles, num_N_tiles);
            hipExtLaunchKernelGGL(gemm_dp_kernel,
                                  gridDim,
                                  blockDim,
                                  LDS_TOTAL_BYTES,
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
                                  ldd);
            CHECK_HIP_ERROR(hipGetLastError());
        }
        else
        {
            // Zero workspace, run Stream-K, then cast to fp16 D.
            CHECK_HIP_ERROR(
                hipMemsetAsync(d_workspace, 0, static_cast<size_t>(m) * n * sizeof(WorkspaceT), 0));

            dim3 gridDimSK(num_workers);
            hipExtLaunchKernelGGL(gemm_streamk_kernel,
                                  gridDimSK,
                                  blockDim,
                                  LDS_TOTAL_BYTES,
                                  0,
                                  nullptr,
                                  nullptr,
                                  0,
                                  m,
                                  n,
                                  k,
                                  d_a,
                                  d_b,
                                  d_workspace,
                                  lda,
                                  ldb,
                                  /*ldd_ws=*/n,
                                  num_workers);
            CHECK_HIP_ERROR(hipGetLastError());

            const uint32_t total_elems = m * n;
            const uint32_t finishBlk   = 256u;
            const uint32_t finishGrid  = (total_elems + finishBlk - 1u) / finishBlk;
            hipLaunchKernelGGL(gemm_streamk_finish_kernel,
                               dim3(finishGrid),
                               dim3(finishBlk),
                               0,
                               0,
                               d_workspace,
                               d_d,
                               total_elems);
            CHECK_HIP_ERROR(hipGetLastError());
        }
    };

    constexpr uint32_t warmups    = 2u;
    constexpr uint32_t recordRuns = 5u;

    std::cout << "Warming up...\n";
    for(uint32_t i = 0; i < warmups; ++i)
        launch();
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    std::cout << "Benchmarking...\n";
    hipEvent_t evStart, evStop;
    CHECK_HIP_ERROR(hipEventCreate(&evStart));
    CHECK_HIP_ERROR(hipEventCreate(&evStop));
    CHECK_HIP_ERROR(hipEventRecord(evStart));
    for(uint32_t i = 0; i < recordRuns; ++i)
        launch();
    CHECK_HIP_ERROR(hipEventRecord(evStop));
    CHECK_HIP_ERROR(hipEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, evStart, evStop));
    CHECK_HIP_ERROR(hipEventDestroy(evStart));
    CHECK_HIP_ERROR(hipEventDestroy(evStop));

    double gFlopsPerRun
        = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1e-9;
    double tFlopsPerSec
        = gFlopsPerRun * recordRuns / (static_cast<double>(elapsedMs) * 1e-3) * 1e-3;

    std::cout << std::left << std::setw(14) << "Schedule" << std::setw(8) << "MatM" << std::setw(8)
              << "MatN" << std::setw(8) << "MatK" << std::setw(10) << "Tiles" << std::setw(10)
              << "Workers" << std::setw(14) << "elapsedMs(5x)" << std::setw(14) << "TFlops/s"
              << "\n";
    std::cout << std::left << std::setw(14) << scheduleName(sched) << std::setw(8) << m
              << std::setw(8) << n << std::setw(8) << k << std::setw(10) << num_tiles
              << std::setw(10) << (sched == GemmSchedule::StreamK ? num_workers : num_tiles)
              << std::setw(14) << elapsedMs << std::setw(14) << tFlopsPerSec << "\n";

#if !NDEBUG
    if(skipValidation)
    {
        std::cout << "Skipping validation as requested.\n";
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_d));
        if(d_workspace)
            CHECK_HIP_ERROR(hipFree(d_workspace));
        std::cout << "Finished!\n";
        return;
    }

    std::cout << "\nValidating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(
        matD.data(), d_d, static_cast<size_t>(m) * n * sizeof(OutputT), hipMemcpyDeviceToHost));

    uint64_t refFlops
        = static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
    // threshold:512M flops
    if(refFlops > (512ULL * 1024ULL * 1024ULL))
    {
        std::cout << "[Note] Running CPU reference validation for large problem (" << m << "x" << n
                  << "x" << k << ").Please wait.  This may take a while...\n";
    }

    std::vector<OutputT> matDref(static_cast<size_t>(m) * n,
                                 std::numeric_limits<OutputT>::signaling_NaN());
    gemm_cpu_ref(m, n, k, matA.data(), matB.data(), matDref.data(), lda, ldb, ldd);

    auto res = compareEqual(matD.data(), matDref.data(), m * n);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED")
              << "  max relative error = " << std::get<1>(res) << "\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_d));
    if(d_workspace)
        CHECK_HIP_ERROR(hipFree(d_workspace));

    std::cout << "Finished!\n";
}

// =============================================================================
// Wave-quantization shape generator
//
// Returns (M, N, K) tuned to expose wave quantization for the data-parallel
// baseline so the Stream-K benefit is visible.  We pick num_tiles = num_CU+1
// so that DataParallel needs 2 waves but the second wave only fills 1/num_CU
// of the GPU.
// =============================================================================
static std::tuple<uint32_t, uint32_t, uint32_t> waveQuantShape()
{
    auto dc = getDeviceCounters();
    // Place num_CU+1 tiles along M with N fixed to one tile.
    uint32_t M = (dc.multiProcessorCount + 1u) * MACRO_TILE_X;
    uint32_t N = MACRO_TILE_Y;
    uint32_t K = 4096; // multiple of MACRO_TILE_K = 16
    return {M, N, K};
}

static bool parseUint32Arg(char const* text, uint32_t& out)
{
    char*         end   = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if(end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max())
        return false;
    out = static_cast<uint32_t>(value);
    return true;
}

static void printUsage(char const* argv0)
{
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " [--all] [--workers N] [--skip-validation]\n"
        << "\nOptions:\n"
        << "  --all              Also run wave-quantization, LLaMA-style, and square GEMM shapes.\n"
        << "  --workers N        Override Stream-K persistent worker count; default is #CUs.\n"
        << "  --skip-validation  Skip CPU reference validation.\n";
}

// =============================================================================
// SECTION X: main()
// =============================================================================
int main(int argc, char** argv)
{
    std::cout << "Community Sample: Stream-K GEMM (work-centric persistent scheduler)\n";
    std::cout << "This sample demonstrates a Stream-K-style grid scheduler wrapped around\n"
              << "the same rocWMMA inner GEMM kernel as the data-parallel baseline.\n";

    bool     runAll         = false;
    bool     skipValidation = false;
    uint32_t userWorkers    = 0u;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--all")
            runAll = true;
        else if(arg == "--skip-validation")
            skipValidation = true;
        else if(arg == "--workers")
        {
            if(i + 1 >= argc || !parseUint32Arg(argv[++i], userWorkers))
            {
                std::cout << "Invalid --workers value.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if(arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cout << "Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    printDeviceInfo();

    // ----- 1) Quick sanity check on a tile-aligned shape -----
    std::cout << "\n[1] Quick sanity (small shape, both schedulers, validate)\n";
    run_gemm_sample(
        /*M=*/256, /*N=*/256, /*K=*/512, GemmSchedule::DataParallel, userWorkers, skipValidation);
    run_gemm_sample(
        /*M=*/256, /*N=*/256, /*K=*/512, GemmSchedule::StreamK, userWorkers, skipValidation);

    if(runAll)
    {
        // ----- 2) Wave-quantization shape (the headline Stream-K benefit) -----
        auto [Mq, Nq, Kq] = waveQuantShape();
        std::cout << "\n[2] Wave-quantization shape: (num_CU+1) tiles along M\n";
        run_gemm_sample(Mq, Nq, Kq, GemmSchedule::DataParallel, userWorkers, skipValidation);
        run_gemm_sample(Mq, Nq, Kq, GemmSchedule::StreamK, userWorkers, skipValidation);

        // ----- 3) LLaMA-style FFN shape (large K, narrow M) -----
        std::cout << "\n[3] LLaMA-2 7b FFN-projection style shape\n";
        run_gemm_sample(64, 11008, 4096, GemmSchedule::DataParallel, userWorkers, skipValidation);
        run_gemm_sample(64, 11008, 4096, GemmSchedule::StreamK, userWorkers, skipValidation);

        // ----- 4) Square shape (Stream-K typically a wash here) -----
        std::cout << "\n[4] Square shape (no wave quantization)\n";
        run_gemm_sample(2048, 2048, 1024, GemmSchedule::DataParallel, userWorkers, skipValidation);
        run_gemm_sample(2048, 2048, 1024, GemmSchedule::StreamK, userWorkers, skipValidation);
    }
    else
    {
        std::cout
            << "\nTip: pass --all to also run wave-quantization, LLaMA-FFN, and square shapes.\n";
        std::cout << "Tip: pass --workers N to override the persistent worker count "
                     "(default = #CUs).\n";
        std::cout << "Tip: pass --skip-validation to skip validation.\n";
    }

    std::cout << "Sample completed successfully!" << std::endl;
    return 0;
}
