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

/* COMMUNITY SAMPLE: LoRA Adapter Fusion (base GEMM + low-rank delta)
 *
 * ============================================================================
 * 1. WHAT IS LoRA?
 * ============================================================================
 *
 * LoRA (Low-Rank Adaptation, Hu et al. 2021) is a parameter-efficient
 * fine-tuning technique for large language models.  Instead of updating
 * the full weight matrix W, a small low-rank "delta" is learned and added
 * at inference time:
 *
 *     Y = X * W  +  alpha * (X * A_lora) * B_lora
 *
 * where R << min(K, N) is the LoRA rank (typical: 8, 16, 32, 64).
 * The frozen base W is shared across deployments, and only the adapters
 * A_lora / B_lora are swapped per task.  This sample shows how rocWMMA can
 * fuse the three GEMM stages and the scale-add epilogue into a single kernel.
 *
 * Reference:
 *   Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
 *   https://arxiv.org/abs/2106.09685
 *
 * ============================================================================
 * 2. MATHEMATICAL FORMULATION
 * ============================================================================
 *
 *   W_acc   = X * W                         [M x N] = [M x K] * [K x N]
 *   S       = X * A_lora                    [M x R] = [M x K] * [K x R]
 *   P       = S * B_lora                    [M x N] = [M x R] * [R x N]
 *   Y       = W_acc + alpha * P             element-wise (FMA)
 *
 * Tensor roles in a typical LLM (e.g. attention / FFN projection):
 *   X       = activation tensor              (sequence tokens)
 *   W       = frozen base projection         (learned during pre-training)
 *   A_lora  = LoRA down-projection           [K x R], small
 *   B_lora  = LoRA up-projection             [R x N], small
 *   Y       = adapted output activations
 *
 * ============================================================================
 * 3. WHY FUSE INTO A SINGLE KERNEL?
 * ============================================================================
 *
 * A naive implementation launches 4 separate kernels:
 *   Kernel 1:  W_acc = X * W                 (heavy GEMM)
 *   Kernel 2:  S     = X * A_lora            (skinny GEMM, [MxR])
 *   Kernel 3:  P     = S * B_lora            (skinny GEMM, [RxN])
 *   Kernel 4:  Y     = W_acc + alpha * P     (element-wise)
 *
 * This sample fuses the three GEMM stages and the scale-add epilogue into
 * ONE kernel. Benefits:
 *
 *   a) X-tile reuse -- X is loaded from global memory ONCE per K-tile and
 *      consumed by BOTH the base GEMM (X*W) and the LoRA-down GEMM (X*A_lora)
 *      simultaneously.  This is the same trick used by simple_gemm_swiglu
 *      for its gate / up projections.
 *
 *   b) Zero global temporaries -- W_acc, S, and P never touch global
 *      memory.  W_acc and S live in registers throughout the K-loop;
 *      S is then briefly staged through LDS so it can be re-read as a
 *      matrix_a fragment for the inner S * B_lora GEMM.  P is
 *      accumulated directly into W_acc via fused-multiply-add (FMA),
 *      so no [M x N] intermediate buffer is ever materialised.
 *
 *   c) Single launch overhead -- One kernel instead of four eliminates
 *      three launch + synchronisation barriers.
 *
 *   Note: this sample demonstrates the fusion pattern rather than
 *   establishing a production-tuned LoRA implementation; see KNOWN
 *   LIMITATIONS below for shape constraints.
 *
 * ============================================================================
 * 4. WHAT YOU WILL LEARN FROM THIS SAMPLE
 * ============================================================================
 *
 *   - How to fuse a "wide" GEMM (X*W, K-reduce) with a "skinny" GEMM
 *     (X*A_lora, K-reduce, narrow N=R) inside the same K-loop.
 *   - How to bridge an accumulator fragment (ComputeT) into a matrix_a
 *     fragment (InputT) for a downstream GEMM, by staging through LDS.
 *     This is the key technique for any back-to-back GEMM (B2B-GEMM)
 *     pattern such as attention (QK^T then softmax * V) or LoRA.
 *   - How to size LDS for FOUR live tiles (X, W^T, A_lora^T, S-stage).
 *   - How to apply a scalar scale (alpha) and FMA epilogue directly on
 *     accumulator fragments in registers (ComputeT precision preserved).
 *   - The MfmaFragAcc -> MfmaFragStoreOut cast pattern for fp32 -> fp16
 *     output, identical to perf_hgemm.cpp / simple_gemm_swiglu.cpp.
 *   - How to specialize compute along the warp_y axis when an intermediate
 *     tile (here S = X*A_lora) is narrower than the warp grid, and broadcast
 *     the result through LDS instead of recomputing it on every warp.
 *
 * ============================================================================
 * 5. KERNEL DATA-FLOW OVERVIEW
 * ============================================================================
 *
 *   Global Memory --coop load--> LDS (3 K-loop segments, double-buffered)
 *        |                            |
 *      X [MxK]                  [X        segment]
 *      W [KxN]                  [W^T      segment]
 *      A_lora[KxR]              [A_lora^T segment]
 *                                     |
 *                               local read
 *                                     |
 *                    +-------fragsX-------+
 *                    |                    |
 *                fragsW              fragsA_lora
 *                    |                    |
 *                   mma                  mma
 *                    |                    |
 *                  acc_W               acc_S
 *                    |                    |
 *                    |          (cast f32 -> f16, store to LDS)
 *                    |                    |
 *                    |             +--LDS S-stage--+
 *                    |                    |
 *                    |         (load as matrix_a fragment)
 *                    |                    |
 *                    |                  fragsS    + fragsBlora (global)
 *                    |                    |             |
 *                    |                    +---- mma ----+
 *                    |                          |
 *                    |                       acc_lora
 *                    |                          |
 *                    +--- acc_W += alpha * acc_lora (FMA) ---+
 *                                 |
 *                            fragsY (f32)
 *                                 |
 *                            cast to f16
 *                                 |
 *                              Y [MxN]  --store--> Global Memory
 *
 * ============================================================================
 * 6. DATA LAYOUTS
 * ============================================================================
 *
 *   X        : row_major  [M x K],   ldx     = K
 *   W        : row_major  [K x N],   ldw     = N
 *   A_lora   : row_major  [K x R],   ldAlora = R
 *   B_lora   : row_major  [R x N],   ldBlora = N
 *   Y        : row_major  [M x N],   ldy     = N
 *
 * ============================================================================
 * 7. LDS LAYOUT
 * ============================================================================
 *
 *   Two regions are reserved in dynamic shared memory:
 *
 *   (a) K-loop ping-pong (col_major), 3 segments per buffer:
 *
 *     Segment   | Height                | Content
 *     ----------+-----------------------+----------------------------
 *        X      | MACRO_TILE_X          | X tile          [MxK_tile]
 *        W      | MACRO_TILE_Y          | W^T tile        [KxN_tile]
 *      A_lora   | MACRO_TILE_R          | A_lora^T tile   [KxR_tile]
 *
 *      Width = MACRO_TILE_K     (K of the outer GEMM)
 *
 *   (b) S-stage (col_major), single buffer used after the K-loop:
 *
 *     Segment   | Height        | Width        | Content
 *     ----------+---------------+--------------+----------------
 *        S      | MACRO_TILE_X  | LORA_RANK    | S = X*A_lora
 *
 *   Per-architecture derivation (BLOCKS_X = BLOCKS_Y = 2, LORA_RANK = 16):
 *
 *     Param            | gfx9 (wave64)        | gfx11/12 (wave32)
 *     -----------------+----------------------+---------------------
 *     TBLOCK_X         | 128                  | 64
 *     WARP_SIZE        | 64                   | 32
 *     WARPS_X          | 2                    | 2
 *     WARP_TILE_X      | 32                   | 32
 *     MACRO_TILE_X     | 64                   | 64
 *     MACRO_TILE_Y     | 64                   | 64
 *     MACRO_TILE_R     | 16                   | 16
 *     MACRO_TILE_K     | 16                   | 16
 *     K-loop seg height| 64+64+16 = 144       | 64+64+16 = 144
 *     K-loop seg bytes | 144*16*2 = 4608      | 144*16*2 = 4608
 *     2x ping-pong     | 9216                 | 9216
 *     S-stage bytes    | 64*16*2 = 2048       | 64*16*2 = 2048
 *     Total LDS        | ~11 KiB              | ~11 KiB
 *
 *   Note: Both architectures produce identical macro tile sizes because
 *   the halved TBLOCK_X on gfx11/12 is exactly compensated by the
 *   halved WARP_SIZE, yielding the same WARPS_X = 2.
 *
 * ============================================================================
 * 8. WARP TILING FOR THE LoRA-DOWN GEMM (X * A_lora)
 * ============================================================================
 *
 *   The LoRA-down output S has shape [MACRO_TILE_X x LORA_RANK] = [64 x 16].
 *   In warp-tile units this is [WARPS_X x 1] warps because the rank
 *   direction (16 = ROCWMMA_N) fits inside a single warp's N-tile.
 *
 *   Solution: only warp_y == 0 computes S for each warp_x row slice and
 *   stores it to LDS. Other warp_y values skip the LoRA-down MMA. After
 *   the workgroup synchronization, all warp_y values load the shared S
 *   tile from LDS and combine it with their own N-slice of B_lora to
 *   produce acc_lora[BLOCKS_X][BLOCKS_Y].
 *
 *   This avoids redundant LoRA-down compute while still allowing every
 *   warp_y to participate in the final S * B_lora GEMM.
 *
 * ============================================================================
 * 9. REQUIREMENTS
 * ============================================================================
 *
 *   - Minimum ROCm version: ROCm 6.0+
 *   - GPU architectures: gfx9 / gfx11 / gfx12
 *       (tested on RDNA4 gfx1201 / RX9070)
 *   - Data types: float16 input, float32 compute, float16 output
 *   - Matrix dimensions:
 *       M must be a multiple of MACRO_TILE_X (64)
 *       N must be a multiple of MACRO_TILE_Y (64)
 *       K must be a multiple of ROCWMMA_K    (16)
 *   - LoRA rank R is fixed at compile time to ROCWMMA_K (16) so that the
 *     inner S * B_lora GEMM is exactly one mma_sync per output fragment.
 *
 * ============================================================================
 * 10. KNOWN LIMITATIONS
 * ============================================================================
 *
 *   - LoRA rank R is fixed at 16 (ROCWMMA_K).  Larger ranks would require
 *     a small K-loop inside the inner GEMM and additional LDS for staging.
 *   - No boundary handling: dimensions must be tile-aligned (see above).
 *   - Only row_major layout for all inputs and output.
 *   - alpha is passed as a runtime float scalar (typical LoRA ratio
 *     alpha / r is folded by the caller).
 *   - Performance is not tuned for production use (educational sample).
 *   - LDS usage: ~11 KiB per block.
 *   - Input values are scaled by 1/16 to prevent FP16 overflow in the
 *     compounded base + LoRA accumulation; real workloads may need
 *     different scaling strategies.
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

using DataLayoutX     = row_major;
using DataLayoutW     = row_major; // W       : [K x N] row_major
using DataLayoutAlora = row_major; // A_lora  : [K x R] row_major
using DataLayoutBlora = row_major; // B_lora  : [R x N] row_major
using DataLayoutY     = row_major;
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

// LoRA rank: fixed to ROCWMMA_K so the inner GEMM is a single fragment-MMA.
constexpr uint32_t LORA_RANK    = ROCWMMA_K;
constexpr uint32_t MACRO_TILE_R = LORA_RANK; // only one "warp tile" wide in R

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------

// Per-block MFMA fragments (warp tile element)
using MfmaFragX     = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutX>;
using MfmaFragW     = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutW>;
using MfmaFragAlora = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutAlora>;
using MfmaFragBlora = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutBlora>;

// Accumulator (compute-precision) fragments.
// MfmaFragAccCompute keeps ComputeT (fp32) for the K-loop and the FMA
// epilogue.  MfmaFragStoreOut is the OutputT-typed accumulator we cast
// to right before store_matrix_sync, mirroring perf_hgemm.cpp.
using MfmaFragAccCompute = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;
using MfmaFragStoreOut
    = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutY>;
using MfmaFragAcc = MfmaFragAccCompute;

// MfmaFragSStage is an accumulator-shaped fragment in InputT (fp16) used
// to write S to LDS in the SAME element layout as MfmaFragAccCompute, so
// the per-element cast (ComputeT -> InputT) preserves index correspondence.
using MfmaFragSStage
    = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutLds>;

// Matrix-a fragment in InputT used to RE-READ S from LDS for the inner GEMM.
using MfmaFragS = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutX>;

// Cooperative global read (macro tile)
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffX
    = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutX, CoopScheduler>;
using GRBuffW
    = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutW, CoopScheduler>;
using GRBuffAlora = fragment<matrix_b,
                             ROCWMMA_M,
                             MACRO_TILE_R,
                             ROCWMMA_K,
                             InputT,
                             DataLayoutAlora,
                             CoopScheduler>;

// Local write (macro tile) -- col_major LDS; B-side fragments transposed.
using LWBuffX     = apply_data_layout_t<GRBuffX, DataLayoutLds>;
using LWBuffW     = apply_data_layout_t<apply_transpose_t<GRBuffW>, DataLayoutLds>;
using LWBuffAlora = apply_data_layout_t<apply_transpose_t<GRBuffAlora>, DataLayoutLds>;

// Local read (MFMA fragment-level) -- matches LDS col_major layout
using LRFragX     = apply_data_layout_t<MfmaFragX, DataLayoutLds>;
using LRFragW     = apply_data_layout_t<apply_transpose_t<MfmaFragW>, DataLayoutLds>;
using LRFragAlora = apply_data_layout_t<apply_transpose_t<MfmaFragAlora>, DataLayoutLds>;

// ---------------------------------------------------------------------------
// Device helper functions
// ---------------------------------------------------------------------------

ROCWMMA_DEVICE static inline void globalReadCoopX(GRBuffX& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void globalReadCoopW(GRBuffW& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    globalReadCoopAlora(GRBuffAlora& gr, InputT const* addr, uint32_t ld)
{
    load_matrix_sync(gr, addr, ld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopX(InputT* ldsAddr, GRBuffX const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(gr), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopW(InputT* ldsAddr, GRBuffW const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(gr)), ldsld);
}

ROCWMMA_DEVICE static inline void
    localWriteCoopAlora(InputT* ldsAddr, GRBuffAlora const& gr, uint32_t ldsld)
{
    store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(gr)), ldsld);
}

// Read BLOCKS_X X-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadX(MfmaFragX (&fragsX)[BLOCKS_X], InputT const* ldsAddr, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragX>;
    using FragShape = GetIOShape_t<LRFragX>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragX tmp;
        load_matrix_sync(tmp, ldsAddr, ldsld);
        fragsX[i] = apply_data_layout<DataLayoutX>(tmp);
        ldsAddr += blockStep;
    }
}

// Read BLOCKS_Y W-blocks from LDS
ROCWMMA_DEVICE static inline void
    localReadW(MfmaFragW (&fragsW)[BLOCKS_Y], InputT const* ldsAddr, uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<LRFragW>;
    using FragShape = GetIOShape_t<LRFragW>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragW tmp;
        load_matrix_sync(tmp, ldsAddr, ldsld);
        fragsW[i] = apply_data_layout<DataLayoutW>(apply_transpose(tmp));
        ldsAddr += blockStep;
    }
}

// Read the single A_lora block (BLOCKS_R == 1) from LDS.
ROCWMMA_DEVICE static inline void
    localReadAlora(MfmaFragAlora& fragAlora, InputT const* ldsAddr, uint32_t ldsld)
{
    LRFragAlora tmp;
    load_matrix_sync(tmp, ldsAddr, ldsld);
    fragAlora = apply_data_layout<DataLayoutAlora>(apply_transpose(tmp));
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

// Fill the BLOCKS_X x 1 LoRA-down accumulator fragments with a scalar
ROCWMMA_DEVICE static inline void clear_acc_fragments_S(MfmaFragAcc (&frags)[BLOCKS_X],
                                                        ComputeT val)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
        fill_fragment(frags[i], val);
}

// Generic warp-tile MMA accumulation: acc += fragsA * fragsB  (for X * W)
template <typename FragB>
ROCWMMA_DEVICE static inline void mfma_warp_tile(MfmaFragAcc (&accOut)[BLOCKS_X][BLOCKS_Y],
                                                 MfmaFragX const (&fragsX)[BLOCKS_X],
                                                 FragB const (&fragsB)[BLOCKS_Y],
                                                 MfmaFragAcc const (&accIn)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            mma_sync(accOut[i][j], fragsX[i], fragsB[j], accIn[i][j]);
}

// LoRA-down MMA accumulation: accS += fragsX * fragAlora  (S = X * A_lora)
// Only BLOCKS_R = 1 fragment in the rank dimension, so the inner loop is
// trivial and we keep it as a small unrolled helper.
ROCWMMA_DEVICE static inline void mfma_warp_tile_S(MfmaFragAcc (&accOut)[BLOCKS_X],
                                                   MfmaFragX const (&fragsX)[BLOCKS_X],
                                                   MfmaFragAlora const& fragAlora,
                                                   MfmaFragAcc const (&accIn)[BLOCKS_X])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
        mma_sync(accOut[i], fragsX[i], fragAlora, accIn[i]);
}

// Cast S accumulator (ComputeT) to InputT staging fragment and store to LDS.
// The element-index correspondence is preserved because both fragment types
// are accumulator-shaped on the same architecture; only the scalar element
// type changes.  Each warp writes its own [WARP_TILE_X x ROCWMMA_N]
// sub-tile of S.
ROCWMMA_DEVICE static inline void
    storeS_lds(InputT* ldsAddr, MfmaFragAcc const (&accS)[BLOCKS_X], uint32_t ldsld)
{
    using Mapper1d  = GetDataLayout_t<MfmaFragSStage>;
    using FragShape = GetIOShape_t<MfmaFragSStage>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        MfmaFragSStage stage;
#pragma unroll
        for(int e = 0; e < (int)accS[i].num_elements; e++)
            stage.x[e] = static_cast<InputT>(accS[i].x[e]);
        store_matrix_sync(ldsAddr, stage, ldsld);
        ldsAddr += blockStep;
    }
}

// Re-load S from LDS into BLOCKS_X matrix_a fragments for the inner GEMM.
ROCWMMA_DEVICE static inline void
    loadS_lds(MfmaFragS (&fragsS)[BLOCKS_X], InputT const* ldsAddr, uint32_t ldsld)
{
    using LRFragS   = apply_data_layout_t<MfmaFragS, DataLayoutLds>;
    using Mapper1d  = GetDataLayout_t<LRFragS>;
    using FragShape = GetIOShape_t<LRFragS>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldsld);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragS tmp;
        load_matrix_sync(tmp, ldsAddr, ldsld);
        fragsS[i] = apply_data_layout<DataLayoutX>(tmp);
        ldsAddr += blockStep;
    }
}

// Direct global load of B_lora into BLOCKS_Y matrix_b fragments.  B_lora is
// [R x N] with R = ROCWMMA_K, so each warp's slice spans WARP_TILE_Y cols
// and ROCWMMA_K rows -- a single mma's worth of K, and BLOCKS_Y mma's worth
// of N per warp.  No LDS staging needed because B_lora is read once.
ROCWMMA_DEVICE static inline void
    globalReadBlora(MfmaFragBlora (&fragsBlora)[BLOCKS_Y], InputT const* addr, uint32_t ld)
{
    using Mapper1d  = GetDataLayout_t<MfmaFragBlora>;
    using FragShape = GetIOShape_t<MfmaFragBlora>;
    auto blockStep  = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ld);
#pragma unroll
    for(int j = 0; j < BLOCKS_Y; j++)
    {
        load_matrix_sync(fragsBlora[j], addr, ld);
        addr += blockStep;
    }
}

// Inner GEMM: acc_lora = S * B_lora   (single K-step since R = ROCWMMA_K)
ROCWMMA_DEVICE static inline void mfma_warp_tile_lora(MfmaFragAcc (&accOut)[BLOCKS_X][BLOCKS_Y],
                                                      MfmaFragS const (&fragsS)[BLOCKS_X],
                                                      MfmaFragBlora const (&fragsB)[BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
            mma_sync(accOut[i][j], fragsS[i], fragsB[j], accOut[i][j]);
}

// Final FMA epilogue:  accY = accW + alpha * accLora   (all in ComputeT)
// Operates element-wise on accumulator fragments; element indices align
// because both fragments are accumulator-shaped with identical M/N/K.
ROCWMMA_DEVICE static inline void
    apply_lora_epilogue(MfmaFragAccCompute (&fragsY)[BLOCKS_X][BLOCKS_Y],
                        MfmaFragAcc const (&accW)[BLOCKS_X][BLOCKS_Y],
                        MfmaFragAcc const (&accLora)[BLOCKS_X][BLOCKS_Y],
                        ComputeT alpha)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
#pragma unroll
            for(int e = 0; e < (int)accW[i][j].num_elements; e++)
                fragsY[i][j].x[e] = accW[i][j].x[e] + alpha * accLora[i][j].x[e];
}

// Write Y to global memory via ComputeT -> OutputT fragment cast.
ROCWMMA_DEVICE static inline void globalWriteY(
    OutputT* gAddrY, MfmaFragAccCompute const (&fragsY)[BLOCKS_X][BLOCKS_Y], uint32_t ldy)
{
    using Mapper1d  = GetDataLayout_t<MfmaFragStoreOut>;
    using FragShape = GetIOShape_t<MfmaFragStoreOut>;
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldy);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldy);
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            MfmaFragStoreOut fragOut;
#pragma unroll
            for(int e = 0; e < (int)fragsY[i][j].num_elements; e++)
                fragOut.x[e] = static_cast<OutputT>(fragsY[i][j].x[e]);
            store_matrix_sync(gAddrY + offsetY, fragOut, ldy);
            offsetY += blockStepY;
        }
        gAddrY += blockStepX;
    }
}

// ---------------------------------------------------------------------------
// Main LoRA fused kernel
//
//   Y = X * W  +  alpha * (X * A_lora) * B_lora
//
//   X       [M x K] row_major
//   W       [K x N] row_major   (frozen base weights)
//   A_lora  [K x R] row_major   (LoRA down-projection, R = LORA_RANK = 16)
//   B_lora  [R x N] row_major   (LoRA up-projection)
//   Y       [M x N] row_major
// ---------------------------------------------------------------------------
constexpr uint32_t  kBlockThreads = TBLOCK_X * TBLOCK_Y;
ROCWMMA_KERNEL void __launch_bounds__(kBlockThreads) lora_adapter_fused(uint32_t      m,
                                                                        uint32_t      n,
                                                                        uint32_t      k,
                                                                        InputT const* x,
                                                                        InputT const* w,
                                                                        InputT const* a_lora,
                                                                        InputT const* b_lora,
                                                                        OutputT*      y,
                                                                        uint32_t      ldx,
                                                                        uint32_t      ldw,
                                                                        uint32_t      ldAlora,
                                                                        uint32_t      ldBlora,
                                                                        uint32_t      ldy,
                                                                        ComputeT      alpha)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // ------------------------------------------------------------------
        // Warp / tile coordinate setup
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
        using GRBuffXMap1d     = GetDataLayout_t<GRBuffX>;
        using GRBuffWMap1d     = GetDataLayout_t<GRBuffW>;
        using GRBuffAloraMap1d = GetDataLayout_t<GRBuffAlora>;

        auto globalReadOffsetX
            = GRBuffXMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), ldx);
        auto globalReadOffsetW
            = GRBuffWMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldw);
        // A_lora occupies the full R columns; column offset is always 0.
        auto globalReadOffsetAlora
            = GRBuffAloraMap1d::fromMatrixCoord(make_coord2d(0u, 0u), ldAlora);

        auto kStepOffsetX = GRBuffXMap1d::fromMatrixCoord(make_coord2d(0u, MACRO_TILE_K), ldx);
        auto kStepOffsetW = GRBuffWMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldw);
        auto kStepOffsetAlora
            = GRBuffAloraMap1d::fromMatrixCoord(make_coord2d(MACRO_TILE_K, 0u), ldAlora);

        // ------------------------------------------------------------------
        // Initial global pre-fetch
        // ------------------------------------------------------------------
        GRBuffX     grBuffX;
        GRBuffW     grBuffW;
        GRBuffAlora grBuffAlora;

        globalReadCoopX(grBuffX, x + globalReadOffsetX, ldx);
        globalReadCoopW(grBuffW, w + globalReadOffsetW, ldw);
        globalReadCoopAlora(grBuffAlora, a_lora + globalReadOffsetAlora, ldAlora);

        globalReadOffsetX += kStepOffsetX;
        globalReadOffsetW += kStepOffsetW;
        globalReadOffsetAlora += kStepOffsetAlora;

        // ------------------------------------------------------------------
        // LDS layout (col_major, 3 K-loop segments stacked vertically per
        // ping-pong buffer; S-stage region appended after both buffers).
        //
        //   Lo buffer   | Hi buffer   | S-stage
        //   ------------+-------------+----------
        //   [X|W^T|A^T] | [X|W^T|A^T] | [S col_major]
        // ------------------------------------------------------------------
        HIP_DYNAMIC_SHARED(InputT, ldsBase);

        using LWBuffXShape     = GetIOShape_t<LWBuffX>;
        using LWBuffWShape     = GetIOShape_t<LWBuffW>;
        using LWBuffAloraShape = GetIOShape_t<LWBuffAlora>;
        using LWBuffXMap1d     = GetDataLayout_t<LWBuffX>;
        using LWBuffWMap1d     = GetDataLayout_t<LWBuffW>;

        constexpr uint32_t ldsWidth = MACRO_TILE_K;
        constexpr uint32_t ldsHeight
            = LWBuffXShape::BlockHeight + LWBuffWShape::BlockHeight + LWBuffAloraShape::BlockHeight;
        constexpr uint32_t sizeLdsKloop = ldsHeight * ldsWidth;
        constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

        auto* ldsPtrLo = ldsBase;
        auto* ldsPtrHi = ldsPtrLo + sizeLdsKloop;
        auto* ldsPtrS  = ldsPtrHi + sizeLdsKloop; // appended after both ping-pong buffers

        // S-stage: col_major, height = MACRO_TILE_X, width = LORA_RANK
        constexpr uint32_t ldsldS
            = std::is_same_v<DataLayoutLds, row_major> ? LORA_RANK : MACRO_TILE_X;

        // Absolute write offsets within one K-loop ping-pong buffer.
        auto ldsWriteOffsetX = 0u;
        auto ldsWriteOffsetW
            = LWBuffXMap1d::fromMatrixCoord(make_coord2d(LWBuffXShape::BlockHeight, 0u), ldsld);
        auto ldsWriteOffsetAlora
            = ldsWriteOffsetW
              + LWBuffWMap1d::fromMatrixCoord(make_coord2d(LWBuffWShape::BlockHeight, 0u), ldsld);

        // Per-warp read offsets into each segment.  X uses warp_x's M-row,
        // W uses warp_y's N-col, A_lora has only one column-slice (R-wide)
        // shared across all warp_y values, so its read offset is fixed.
        auto ldsReadOffsetX
            = ldsWriteOffsetX
              + LWBuffXMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetW
            = ldsWriteOffsetW
              + LWBuffWMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetAlora = ldsWriteOffsetAlora; // single R-tile, same for every warp

        // S-stage write offset (per warp): warp_x's M-row, single R-block.
        // S is in its OWN col_major buffer (ldsPtrS) so offsets are
        // computed relative to ldsPtrS, not the ping-pong buffer.
        using LRFragSReadLayout = apply_data_layout_t<MfmaFragS, DataLayoutLds>;
        using LRFragSMap1d      = GetDataLayout_t<LRFragSReadLayout>;
        auto ldsSWarpOffset
            = LRFragSMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsldS);

        // ------------------------------------------------------------------
        // Write initial prefetch to Lo buffer
        // ------------------------------------------------------------------
        localWriteCoopX(ldsPtrLo + ldsWriteOffsetX, grBuffX, ldsld);
        localWriteCoopW(ldsPtrLo + ldsWriteOffsetW, grBuffW, ldsld);
        localWriteCoopAlora(ldsPtrLo + ldsWriteOffsetAlora, grBuffAlora, ldsld);

        // ------------------------------------------------------------------
        // Initialize accumulators for the base GEMM (X*W) and the LoRA-down
        // GEMM (X*A_lora).  acc_W lives [BLOCKS_X][BLOCKS_Y] per warp;
        // acc_S lives [BLOCKS_X] per warp because R = ROCWMMA_N (one block).
        // ------------------------------------------------------------------
        MfmaFragAcc fragsAccW[BLOCKS_X][BLOCKS_Y];
        MfmaFragAcc fragsAccS[BLOCKS_X];
        clear_acc_fragments(fragsAccW, ComputeT(0));
        if(get<1>(localWarpCoord) == 0u)
        {
            clear_acc_fragments_S(fragsAccS, ComputeT(0));
        }

        synchronize_workgroup();

        // ------------------------------------------------------------------
        // K-loop with double-buffer prefetch (identical scheme to
        // simple_gemm_swiglu).  Each iteration:
        //   1. read current X / W / A_lora tiles from Lo
        //   2. issue prefetch of NEXT tiles into registers
        //   3. accumulate acc_W += X * W   and   acc_S += X * A_lora
        //   4. write the prefetched registers into Hi
        //   5. swap Lo <-> Hi
        // ------------------------------------------------------------------
        for(uint32_t currentK = MACRO_TILE_K; currentK < k; currentK += MACRO_TILE_K)
        {
            MfmaFragX     fragsX[BLOCKS_X];
            MfmaFragW     fragsW[BLOCKS_Y];
            MfmaFragAlora fragAlora;

            localReadX(fragsX, ldsPtrLo + ldsReadOffsetX, ldsld);
            localReadW(fragsW, ldsPtrLo + ldsReadOffsetW, ldsld);
            localReadAlora(fragAlora, ldsPtrLo + ldsReadOffsetAlora, ldsld);

            globalReadCoopX(grBuffX, x + globalReadOffsetX, ldx);
            globalReadCoopW(grBuffW, w + globalReadOffsetW, ldw);
            globalReadCoopAlora(grBuffAlora, a_lora + globalReadOffsetAlora, ldAlora);

            globalReadOffsetX += kStepOffsetX;
            globalReadOffsetW += kStepOffsetW;
            globalReadOffsetAlora += kStepOffsetAlora;

            mfma_warp_tile(fragsAccW, fragsX, fragsW, fragsAccW);
            if(get<1>(localWarpCoord) == 0u)
            {
                mfma_warp_tile_S(fragsAccS, fragsX, fragAlora, fragsAccS);
            }

            localWriteCoopX(ldsPtrHi + ldsWriteOffsetX, grBuffX, ldsld);
            localWriteCoopW(ldsPtrHi + ldsWriteOffsetW, grBuffW, ldsld);
            localWriteCoopAlora(ldsPtrHi + ldsWriteOffsetAlora, grBuffAlora, ldsld);

            synchronize_workgroup();

            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;
        }

        // ------------------------------------------------------------------
        // Tail: accumulate the last K-step still in Lo
        // ------------------------------------------------------------------
        {
            MfmaFragX     fragsX[BLOCKS_X];
            MfmaFragW     fragsW[BLOCKS_Y];
            MfmaFragAlora fragAlora;

            localReadX(fragsX, ldsPtrLo + ldsReadOffsetX, ldsld);
            localReadW(fragsW, ldsPtrLo + ldsReadOffsetW, ldsld);
            localReadAlora(fragAlora, ldsPtrLo + ldsReadOffsetAlora, ldsld);

            mfma_warp_tile(fragsAccW, fragsX, fragsW, fragsAccW);
            if(get<1>(localWarpCoord) == 0u)
            {
                mfma_warp_tile_S(fragsAccS, fragsX, fragAlora, fragsAccS);
            }
        }

        // ------------------------------------------------------------------
        // Stage S to LDS so it can be re-read as a matrix_a fragment.
        //
        // Only warp_y == 0 computes S for each warp_x row slice and stores it
        // to LDS. Other warp_y rows later reload the staged S tile from LDS.
        // This single-writer policy avoids write-write races on the S-stage
        // LDS buffer.
        //
        // We MUST go through LDS because the lane-to-element mapping of
        // an accumulator fragment is different from a matrix_a fragment;
        // a register-only cast cannot rearrange data across lanes.
        // ------------------------------------------------------------------
        if(get<1>(localWarpCoord) == 0u)
        {
            storeS_lds(ldsPtrS + ldsSWarpOffset, fragsAccS, ldsldS);
        }

        synchronize_workgroup();

        // ------------------------------------------------------------------
        // Inner GEMM: acc_lora = S * B_lora      (single K-step, R = 16)
        //
        // Each warp loads its OWN slice:
        //   - fragsS[BLOCKS_X]      from LDS (warp_x picks rows)
        //   - fragsBlora[BLOCKS_Y]  from global (warp_y picks columns)
        //
        // B_lora is read just once per block, so we load it directly from
        // global memory without LDS staging.
        // ------------------------------------------------------------------
        MfmaFragS     fragsS[BLOCKS_X];
        MfmaFragBlora fragsBlora[BLOCKS_Y];

        loadS_lds(fragsS, ldsPtrS + ldsSWarpOffset, ldsldS);

        // Global B_lora pointer for this warp's N-slice.  R rows x WARP_TILE_Y
        // cols.
        using BloraMap1d = GetDataLayout_t<MfmaFragBlora>;
        auto globalReadOffsetBlora
            = BloraMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(warpTileCoord)), ldBlora);
        globalReadBlora(fragsBlora, b_lora + globalReadOffsetBlora, ldBlora);

        MfmaFragAcc fragsAccLora[BLOCKS_X][BLOCKS_Y];
        clear_acc_fragments(fragsAccLora, ComputeT(0));
        mfma_warp_tile_lora(fragsAccLora, fragsS, fragsBlora);

        // ------------------------------------------------------------------
        // FMA epilogue:  Y = acc_W + alpha * acc_lora    (in ComputeT)
        // ------------------------------------------------------------------
        MfmaFragAccCompute fragsY[BLOCKS_X][BLOCKS_Y];
        apply_lora_epilogue(fragsY, fragsAccW, fragsAccLora, alpha);

        // ------------------------------------------------------------------
        // Store Y to global memory
        // ------------------------------------------------------------------
        using MfmaFragStoreOutMap1d = GetDataLayout_t<MfmaFragStoreOut>;
        globalWriteY(y + MfmaFragStoreOutMap1d::fromMatrixCoord(warpTileCoord, ldy), fragsY, ldy);
    }
}

// ---------------------------------------------------------------------------
// CPU reference:  Y[i,j] = sum_k X[i,k]*W[k,j] + alpha * sum_r
// S[i,r]*Blora[r,j]
//                 with    S[i,r] = sum_k X[i,k]*Alora[k,r]
// All matrices are row_major.
// ---------------------------------------------------------------------------
static void lora_cpu_ref(uint32_t      m,
                         uint32_t      n,
                         uint32_t      k,
                         InputT const* x,
                         InputT const* w,
                         InputT const* a_lora,
                         InputT const* b_lora,
                         OutputT*      y,
                         uint32_t      ldx,
                         uint32_t      ldw,
                         uint32_t      ldAlora,
                         uint32_t      ldBlora,
                         uint32_t      ldy,
                         float         alpha,
                         uint32_t      r)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };

    // S = X * A_lora  -> [m x r]
    std::vector<float> S(static_cast<size_t>(m) * r, 0.0f);
    for(uint32_t i = 0; i < m; ++i)
    {
        for(uint32_t rr = 0; rr < r; ++rr)
        {
            float acc = 0.0f;
            for(uint32_t h = 0; h < k; ++h)
                acc += static_cast<float>(x[rowMjr(i, h, ldx)])
                       * static_cast<float>(a_lora[rowMjr(h, rr, ldAlora)]);
            S[i * r + rr] = acc;
        }
    }

    // Y = X * W + alpha * S * B_lora
    for(uint32_t i = 0; i < m; ++i)
    {
        for(uint32_t j = 0; j < n; ++j)
        {
            float w_acc = 0.0f;
            for(uint32_t h = 0; h < k; ++h)
                w_acc += static_cast<float>(x[rowMjr(i, h, ldx)])
                         * static_cast<float>(w[rowMjr(h, j, ldw)]);

            float lora_acc = 0.0f;
            for(uint32_t rr = 0; rr < r; ++rr)
                lora_acc += S[i * r + rr] * static_cast<float>(b_lora[rowMjr(rr, j, ldBlora)]);

            y[rowMjr(i, j, ldy)] = static_cast<OutputT>(w_acc + alpha * lora_acc);
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
ROCWMMA_HOST void run_lora_sample(uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  float    alpha          = 1.0f,
                                  bool     printInfo      = false,
                                  bool     skipValidation = false)
{
    if(printInfo)
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
    uint32_t ldx     = k; // X       [M x K]
    uint32_t ldw     = n; // W       [K x N]
    uint32_t ldAlora = LORA_RANK; // A_lora  [K x R]
    uint32_t ldBlora = n; // B_lora  [R x N]
    uint32_t ldy     = n; // Y       [M x N]

    std::cout << "Initializing host data (m=" << m << " n=" << n << " k=" << k << " r=" << LORA_RANK
              << " alpha=" << alpha << ")...\n";

    std::vector<InputT>  matX(static_cast<size_t>(m) * k);
    std::vector<InputT>  matW(static_cast<size_t>(k) * n);
    std::vector<InputT>  matAlora(static_cast<size_t>(k) * LORA_RANK);
    std::vector<InputT>  matBlora(static_cast<size_t>(LORA_RANK) * n);
    std::vector<OutputT> matY(static_cast<size_t>(m) * n,
                              std::numeric_limits<OutputT>::signaling_NaN());

    // fillRand fills mat[i*cols+j] (row_major), values are small integers 0..4.
    // Same 1/16 scaling rationale as simple_gemm_swiglu: prevents the
    // composite W + alpha*P sum from overflowing FP16.
    constexpr float kScale = 1.0f / 16.0f;
    fillRand(matX.data(), m, k);
    fillRand(matW.data(), k, n);
    fillRand(matAlora.data(), k, LORA_RANK);
    fillRand(matBlora.data(), LORA_RANK, n);
    for(auto& v : matX)
        v = static_cast<InputT>(static_cast<float>(v) * kScale);
    for(auto& v : matW)
        v = static_cast<InputT>(static_cast<float>(v) * kScale);
    for(auto& v : matAlora)
        v = static_cast<InputT>(static_cast<float>(v) * kScale);
    for(auto& v : matBlora)
        v = static_cast<InputT>(static_cast<float>(v) * kScale);

    std::cout << "Allocating device memory...\n";

    InputT*  d_x;
    InputT*  d_w;
    InputT*  d_alora;
    InputT*  d_blora;
    OutputT* d_y;

    CHECK_HIP_ERROR(hipMalloc(&d_x, m * k * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_w, k * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_alora, k * LORA_RANK * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_blora, LORA_RANK * n * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&d_y, m * n * sizeof(OutputT)));

    CHECK_HIP_ERROR(hipMemcpy(d_x, matX.data(), m * k * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_w, matW.data(), k * n * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_alora, matAlora.data(), k * LORA_RANK * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_blora, matBlora.data(), LORA_RANK * n * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_y, matY.data(), m * n * sizeof(OutputT), hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceil_div(m, get<0>(macroTileSize)),
                        rocwmma::ceil_div(n, get<1>(macroTileSize)));

    std::cout << "gridDim  (" << gridDim.x << " " << gridDim.y << ")"
              << "  blockDim (" << blockDim.x << " " << blockDim.y << ")\n";

    // LDS: 2 ping-pong K-loop buffers + 1 S-stage buffer.
    using LWBuffXShape     = GetIOShape_t<LWBuffX>;
    using LWBuffWShape     = GetIOShape_t<LWBuffW>;
    using LWBuffAloraShape = GetIOShape_t<LWBuffAlora>;
    constexpr uint32_t ldsSegHeight
        = LWBuffXShape::BlockHeight + LWBuffWShape::BlockHeight + LWBuffAloraShape::BlockHeight;
    constexpr uint32_t ldsKloopBytes  = 2u * sizeof(InputT) * ldsSegHeight * MACRO_TILE_K;
    constexpr uint32_t ldsSstageBytes = sizeof(InputT) * MACRO_TILE_X * LORA_RANK;
    int                ldsUsage       = static_cast<int>(ldsKloopBytes + ldsSstageBytes);
    std::cout << "LDS usage: " << ldsUsage << " bytes (" << ldsUsage / 1024 << " KiB)"
              << "  [K-loop=" << ldsKloopBytes << "  S-stage=" << ldsSstageBytes << "]\n";

    auto kernelLambda = [&]() {
        hipExtLaunchKernelGGL(lora_adapter_fused,
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
                              d_x,
                              d_w,
                              d_alora,
                              d_blora,
                              d_y,
                              ldx,
                              ldw,
                              ldAlora,
                              ldBlora,
                              ldy,
                              static_cast<ComputeT>(alpha));
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

    // Total FLOPs per run:
    //   X*W      : 2 * M * N * K
    //   X*A_lora : 2 * M * R * K
    //   S*B_lora : 2 * M * N * R
    //   FMA      :     M * N      (negligible, ignored)
    double mF           = static_cast<double>(m);
    double nF           = static_cast<double>(n);
    double kF           = static_cast<double>(k);
    double rF           = static_cast<double>(LORA_RANK);
    double gFlopsPerRun = (2.0 * mF * nF * kF + 2.0 * mF * rF * kF + 2.0 * mF * nF * rF) * 1e-9;
    double tFlopsPerSec
        = gFlopsPerRun * recordRuns / (static_cast<double>(elapsedMs) * 1e-3) * 1e-3;

    std::cout << std::left << std::setw(8) << "TBlkX" << std::setw(8) << "TBlkY" << std::setw(6)
              << "BlkM" << std::setw(6) << "BlkN" << std::setw(6) << "BlkK" << std::setw(6) << "R"
              << std::setw(8) << "MatM" << std::setw(8) << "MatN" << std::setw(8) << "MatK"
              << std::setw(8) << "ldx" << std::setw(8) << "ldw" << std::setw(10) << "ldAlora"
              << std::setw(10) << "ldBlora" << std::setw(8) << "ldy" << std::setw(20)
              << "elapsedMs(total)" << std::setw(20) << "GFlops(total)" << std::setw(12)
              << "TFlops/s" << std::setw(10) << "Warmups" << std::setw(10) << "Runs" << "\n";

    std::cout << std::left << std::setw(8) << hTBLOCK_X << std::setw(8) << hTBLOCK_Y << std::setw(6)
              << hROCWMMA_M << std::setw(6) << hROCWMMA_N << std::setw(6) << hROCWMMA_K
              << std::setw(6) << LORA_RANK << std::setw(8) << m << std::setw(8) << n << std::setw(8)
              << k << std::setw(8) << ldx << std::setw(8) << ldw << std::setw(10) << ldAlora
              << std::setw(10) << ldBlora << std::setw(8) << ldy << std::setw(20) << elapsedMs
              << std::setw(20) << (gFlopsPerRun * recordRuns) << std::setw(12) << tFlopsPerSec
              << std::setw(10) << warmups << std::setw(10) << recordRuns << "\n";

#if !NDEBUG
    if(skipValidation)
    {
        std::cout << "Skipping validation as requested.\n";
        CHECK_HIP_ERROR(hipFree(d_x));
        CHECK_HIP_ERROR(hipFree(d_w));
        CHECK_HIP_ERROR(hipFree(d_alora));
        CHECK_HIP_ERROR(hipFree(d_blora));
        CHECK_HIP_ERROR(hipFree(d_y));
        std::cout << "Finished!\n";
        return;
    }

    std::cout << "\nValidating against CPU reference...\n";

    CHECK_HIP_ERROR(hipMemcpy(matY.data(), d_y, m * n * sizeof(OutputT), hipMemcpyDeviceToHost));

    uint64_t refFlops = (uint64_t)(m) * (uint64_t)(n) * (uint64_t)(k)
                        + (uint64_t)(m) * (uint64_t)(LORA_RANK) * (uint64_t)(k)
                        + (uint64_t)(m) * (uint64_t)(n) * (uint64_t)(LORA_RANK);
    // threshold:512M flops
    if(refFlops > (512ULL * 1024ULL * 1024ULL))
    {
        std::cout << "[Note] Running CPU reference validation for large problem (" << m << "x" << n
                  << "x" << k << ").Please wait.  This may take a while...\n";
    }

    std::vector<OutputT> matYref(static_cast<size_t>(m) * n,
                                 std::numeric_limits<OutputT>::signaling_NaN());
    lora_cpu_ref(m,
                 n,
                 k,
                 matX.data(),
                 matW.data(),
                 matAlora.data(),
                 matBlora.data(),
                 matYref.data(),
                 ldx,
                 ldw,
                 ldAlora,
                 ldBlora,
                 ldy,
                 alpha,
                 LORA_RANK);

    auto res = compareEqual(matY.data(), matYref.data(), m * n);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n";
#endif

    CHECK_HIP_ERROR(hipFree(d_x));
    CHECK_HIP_ERROR(hipFree(d_w));
    CHECK_HIP_ERROR(hipFree(d_alora));
    CHECK_HIP_ERROR(hipFree(d_blora));
    CHECK_HIP_ERROR(hipFree(d_y));

    std::cout << "Finished!\n";
}

// ---------------------------------------------------------------------------
// Usage:
//   ./simple_lora_adapter_fusion          # Quick validation only (64 x 256 x
//   128)
//   ./simple_lora_adapter_fusion --all    # Also run full LLaMA-2 attention
//   sizes
//   ./simple_lora_adapter_fusion --skip-validation # Skip validation
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << "Community Sample: LoRA Adapter Fusion" << std::endl;
    std::cout << "This sample demonstrates: fused base GEMM + low-rank adapter delta"
              << " (Y = X*W + alpha*(X*A)*B) using rocWMMA" << std::endl;

    bool runAll         = false;
    bool skipValidation = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--all")
            runAll = true;
        if(arg == "--skip-validation")
            skipValidation = true;
    }

    // Quick validation case: small enough for any GPU; all multiples of tile
    // size. run_lora_sample(seq_len, hidden_out, hidden_in, alpha)
    run_lora_sample(64, 256, 128, /*alpha=*/0.5f, /*printInfo=*/true, skipValidation);

    if(runAll)
    {
        // LLaMA-2-inspired attention projection shape.
        // Uses the logical GEMM convention:
        //   Y[seq_len x hidden] = X[seq_len x hidden] * W[hidden x hidden]
        //                       + alpha * (X * A_lora) * B_lora
        // with sample-chosen seq_len = 64, rank = 16, and alpha = 0.5.
        // LLaMA-2-7B uses hidden_size = 4096.
        run_lora_sample(64, 4096, 4096, /*alpha=*/0.5f, /*printInfo=*/false, skipValidation);

        // LLaMA-2-inspired attention projection shape.
        // LLaMA-2-13B uses hidden_size = 5120.
        run_lora_sample(64, 5120, 5120, /*alpha=*/0.5f, /*printInfo=*/false, skipValidation);

        // LLaMA-2-inspired FFN gate/up projection shape.
        // Uses the logical GEMM convention:
        //   Y[seq_len x intermediate] = X[seq_len x hidden] * W[hidden x
        //   intermediate]
        //                             + alpha * (X * A_lora) * B_lora
        // LLaMA-2-7B uses hidden_size = 4096 and intermediate_size = 11008.
        run_lora_sample(64, 11008, 4096, /*alpha=*/0.5f, /*printInfo=*/false, skipValidation);
    }
    else
    {
        std::cout << "\nTip: pass --all to also run LLaMA-2 7b/13b attention/FFN "
                     "sizes.\n";
        std::cout << "Tip: pass --skip-validation to skip validation.\n";
    }

    std::cout << "Sample completed successfully!" << std::endl;
    return 0;
}
