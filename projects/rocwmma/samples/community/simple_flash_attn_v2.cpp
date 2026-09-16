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

/* COMMUNITY SAMPLE: Flash Attention v2 Forward (fused dual-GEMM + online softmax)
 *
 * ============================================================================
 * 1. WHAT IS FLASH ATTENTION?
 * ============================================================================
 *
 * Flash Attention (Dao et al., 2022 / 2023) is the de-facto attention kernel
 * used by every modern LLM inference and training stack.  It computes
 *
 *     O = softmax(Q * K^T * scale) * V
 *
 * WITHOUT ever materialising the full [N x N] attention score matrix in
 * global memory.  Instead, it uses an "online softmax" formulation that
 * processes K and V in tiles, keeping the running statistics
 * (m_i = row max,  l_i = row exp-sum) and the partial output O in registers
 * the whole time.
 *
 * Flash Attention v2 (Dao, 2023) made one structural change relative to v1:
 * it moved the OUTER LOOP to Q.  Each thread block claims one Q tile, then
 * iterates over the K/V sequence.  This is the variant implemented here.
 *
 * References:
 *   [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
 *       Attention with IO-Awareness", NeurIPS 2022.
 *       https://arxiv.org/abs/2205.14135
 *   [2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism
 *       and Work Partitioning", 2023.
 *       https://arxiv.org/abs/2307.08691
 *   [3] Dao-AILab/flash-attention v2.8.4 -- reference implementation
 *       (softmax(Q @ K^T * softmax_scale) @ V).
 *
 * ============================================================================
 * 2. ALGORITHM (one head, one Q-tile, executed by ONE thread block)
 * ============================================================================
 *
 *   // Q_i = [Br x D]  driven by blockIdx.x  (loaded once, kept in LDS)
 *   m_i = -inf  (per row)
 *   l_i = 0     (per row)
 *   O_i = 0     (per row, [Br x D] in registers)
 *
 *   for j in 0..N/Bc:                  // K/V sequence loop (double-buffered)
 *       K_j = K[j*Bc : (j+1)*Bc, :]    // [Bc x D]
 *       V_j = V[j*Bc : (j+1)*Bc, :]    // [Bc x D]
 *
 *       // ---- GEMM #1 ----
 *       S_ij = Q_i * K_j^T             // [Br x Bc]
 *       S_ij *= scale                  // 1/sqrt(D)
 *       if causal: mask upper-triangle of S_ij with -inf
 *
 *       // ---- Online softmax ----
 *       m_new   = max(m_i, rowmax(S_ij))
 *       alpha   = exp(m_i - m_new)
 *       P_ij    = exp(S_ij - m_new)
 *       l_new   = alpha * l_i + rowsum(P_ij)
 *
 *       // ---- Rescale old O, then accumulate ----
 *       O_i     = alpha * O_i + P_ij * V_j   // GEMM #2
 *       m_i, l_i = m_new, l_new
 *
 *   O_i = O_i / l_i                    // final normalisation
 *   write O_i back to global
 *
 * Two key points:
 *
 *   (a) S_ij and P_ij are NEVER stored in global memory.  They are staged
 *       through LDS only briefly (see section 3) so the per-row softmax is
 *       readable; O_i lives in MFMA accumulator registers across the whole
 *       K/V loop.
 *
 *   (b) When a new tile reveals a larger row-max (m_new > m_i), the
 *       previously accumulated O_i is "rescaled" by alpha = exp(m_i - m_new).
 *       This is the algebraic trick that makes the online softmax exact.
 *
 * ============================================================================
 * 3. PERFORMANCE STRUCTURE (what this sample borrows from simple_gemm_swiglu)
 * ============================================================================
 *
 * This sample follows the same architecture as simple_gemm_swiglu.cpp and
 * simple_lora_adapter_fusion.cpp:
 *
 *   - COOPERATIVE FRAGMENT LOADS.  Q, K and V tiles are moved from global
 *     memory into LDS with rocWMMA cooperative fragments
 *     (fragment_scheduler::coop_row_major_2d + apply_data_layout), so every
 *     thread in the block contributes vectorized loads instead of the scalar
 *     element loop a naive kernel would use.
 *
 *   - K/V DOUBLE BUFFERING (ping-pong).  While the current K/V tile is being
 *     consumed by the two GEMMs and the softmax, the NEXT tile's global read
 *     is already in flight, overlapping global-memory latency with compute.
 *     Two LDS K/V buffers (Lo / Hi) alternate every iteration.
 *
 *   - REGISTER-RESIDENT OUTPUT.  O_i never leaves MFMA accumulator registers
 *     across the K/V loop; the only LDS round-trip on O is the per-row alpha
 *     rescale (see KNOWN LIMITATIONS for why).
 *
 * The S -> softmax -> P bridge still stages through LDS.  This mirrors the
 * B2B-GEMM staging in simple_lora_adapter_fusion.cpp (storeS_lds / loadS_lds):
 * the lane-to-element mapping of an accumulator fragment differs from a
 * matrix_a fragment, so a register-only reinterpret cannot rearrange the data
 * across lanes -- LDS is the portable bridge.  Replacing it with cross-lane
 * shuffles is the next optimisation step (see KNOWN LIMITATIONS).
 *
 * ============================================================================
 * 4. WHAT YOU WILL LEARN FROM THIS SAMPLE
 * ============================================================================
 *
 *   - How to express attention as TWO back-to-back GEMMs feeding each
 *     other through register / LDS pipelines.
 *   - How online softmax keeps an exact result without materialising the
 *     attention matrix.
 *   - How cooperative global reads + LDS double buffering overlap memory
 *     latency with MFMA compute (identical scheme to simple_gemm_swiglu).
 *   - How a col_major matrix_b fragment reads K^T directly from a row_major
 *     K tile, for free (no explicit transpose).
 *   - How to apply per-row scalar rescaling to a partial output that spans
 *     multiple accumulator fragments.
 *   - How the wave-row-stripe partition (each wave owns Br/WAVES rows of Q)
 *     eliminates the need for cross-wave softmax communication.
 *   - The MfmaFragAcc -> MfmaFragStoreOut cast pattern for fp32 -> fp16
 *     output, identical to perf_hgemm.cpp / simple_gemm_swiglu.cpp.
 *
 * ============================================================================
 * 5. KERNEL DATA-FLOW OVERVIEW
 * ============================================================================
 *
 *   Global Q,K,V --coop load--> LDS (K/V double-buffered)             .
 *                                |                                    .
 *      Q stays;  K, V ping-pong each iteration                        .
 *                                |                                    .
 *                          +-----+-----+                              .
 *                          v           v                              .
 *               GEMM #1: Q * K^T   (K read as col_major = K^T)        .
 *                          |                                          .
 *                       frag_S (fp32 accumulator)                     .
 *                          |                                          .
 *                  store -> S_lds (fp32)                              .
 *                          |                                          .
 *                  per-row online softmax (LDS-mediated)              .
 *                          |                                          .
 *                  +-------+-------+                                  .
 *                  v               v                                  .
 *              P_lds (fp16)    update m_i, l_i,                       .
 *                  |           rescale frag_O by alpha                .
 *                  v                                                  .
 *               GEMM #2: O += P * V                                   .
 *                          |                                          .
 *                       frag_O (fp32)  -- persists across iterations  .
 *                          |                                          .
 *               (after all tiles)  O <- O / l_i                       .
 *                          |                                          .
 *                       cast -> fp16, store to global                 .
 *
 * ============================================================================
 * 6. PARALLEL DECOMPOSITION
 * ============================================================================
 *
 *   Grid:  (ceil(N / Br),  H,  B)             // one CTA per Q-tile, head, batch
 *   Block: (TBLOCK_X, 1, 1)                    // = WAVES * WARP_SIZE threads
 *
 *   Within a CTA:
 *     - Each wave owns ROWS_PER_WAVE = Br / WAVES contiguous rows of Q.
 *     - All waves cooperate to load each K/V tile into LDS.
 *     - All waves participate in GEMM #1 and GEMM #2 over their own row stripe.
 *     - Each wave does the softmax for its own rows -- NO CROSS-WAVE
 *       SYNCHRONISATION OF (m_i, l_i) IS NEEDED.  This is the key
 *       partitioning choice that keeps the softmax simple.
 *
 *      ___________________ Br = 64 ___________________
 *     |                                                |
 *     | wave 0  rows  [ 0, 16) -- owns its own m,l,O   |
 *     | wave 1  rows  [16, 32) -- owns its own m,l,O   |
 *     | wave 2  rows  [32, 48) -- owns its own m,l,O   |
 *     | wave 3  rows  [48, 64) -- owns its own m,l,O   |
 *     |________________________________________________|
 *
 * ============================================================================
 * 7. LDS LAYOUT  (single dynamic shared pool, row_major tiles)
 * ============================================================================
 *
 *   K and V are stored row_major so that GEMM #1 can read K as a col_major
 *   matrix_b fragment -- which IS K^T -- without any explicit transpose.
 *
 *   Segment    | Type   | Shape           | Bytes | Purpose
 *   -----------+--------+-----------------+-------+--------------------------
 *   Q_lds      | half   | [Br x D]        |  8 KB | Q tile, loaded ONCE
 *   KV_lo      | half   | [Bc x D]*2      |  8 KB | K|V tile (ping)
 *   KV_hi      | half   | [Bc x D]*2      |  8 KB | K|V tile (pong / prefetch)
 *   P_lds      | half   | [Br x Bc]       |  4 KB | softmax output for GEMM #2
 *   SO_lds     | float  | [Br x max(Bc,D)]| 16 KB | aliased: S (softmax) and
 *              |        |                 |       | O (per-row alpha rescale)
 *   -----------+--------+-----------------+-------+--------------------------
 *   Total (Br=64, Bc=32, D=64)              44 KB   fits one CU, no opt-in
 *
 *   WHY Bc = 32 (not 64)?  Double-buffering K AND V needs two live K/V
 *   buffers.  With Bc = 64 the pool would be 64 KB, requiring the
 *   >48 KB dynamic-shared opt-in (hipFuncAttributeMaxDynamicSharedMemorySize)
 *   and leaving zero headroom.  Bc = 32 keeps both K/V buffers double-buffered
 *   within the conventional 48 KB budget.  Br and D stay at 64.
 *
 *   The S_lds / O_lds aliasing (SO_lds) is safe because:
 *     - S_lds is written and read entirely within the softmax pass.
 *     - O_lds is written and read within the alpha-rescale pass, which runs
 *       AFTER the softmax pass of the same iteration is complete.
 *
 * ============================================================================
 * 8. REQUIREMENTS
 * ============================================================================
 *
 *   - Minimum ROCm version: ROCm 6.0+
 *   - GPU architectures: gfx9 / gfx11 / gfx12
 *       (tested on RDNA4 gfx1201 / RX9070; other archs are compile-time
 *        parameter paths and have not been validated)
 *   - Data types: float16 input (Q,K,V,O), float32 compute (S, softmax, O)
 *   - Tensor layout: contiguous [B, H, N, D] row-major
 *   - Head dimension D must equal Dhead (compile-time constant, 64)
 *   - Sequence length N must be a multiple of Br (=64) and Bc (=32)
 *
 * ============================================================================
 * 9. KNOWN LIMITATIONS
 * ============================================================================
 *
 *   - Head dimension fixed to 64 (covers GPT-2 / GPT-J / LLaMA D=64 heads).
 *     Larger D (=128) requires shrinking Bc or splitting D into chunks.
 *   - Softmax row-reductions go through LDS; a production kernel uses cross-
 *     lane shuffles (__shfl_xor) inside accumulator fragments.  Codifying the
 *     (element_idx -> row) mapping per architecture is the next optimisation.
 *   - O rescaling round-trips through LDS each iteration for the same reason.
 *   - No backward pass.
 *   - No dropout, no ALiBi, no sliding-window mask.
 *   - Causal mask is supported but applied scalar-wise during the LDS softmax
 *     pass (also a candidate for fragment-level optimisation).
 *   - No boundary handling: dimensions must be tile-aligned (see above).
 *   - Performance is not tuned for production use (educational sample).
 *   - LDS usage: ~44 KiB per block.
 *   - Input values are damped by 1/8 to keep softmax logits in a sane FP16
 *     range; real workloads may need different scaling strategies.
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
//
// Both architectures use WAVES = 4 waves stacked along the Q-row direction
// (1D thread block).  The halved WARP_SIZE on gfx11/12 halves TBLOCK_X, but
// the per-wave row stripe (ROWS_PER_WAVE) is identical because WAVES is the
// same -- so the macro tile (Br rows) is unchanged across architectures.
// ---------------------------------------------------------------------------
namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        WAVES     = 4u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64,
        TBLOCK_X  = WAVES * WARP_SIZE, // 256
        TBLOCK_Y  = 1u
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        WAVES     = 4u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32,
        TBLOCK_X  = WAVES * WARP_SIZE, // 128
        TBLOCK_Y  = 1u
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
using InputT   = float16_t; // Q, K, V, O
using OutputT  = float16_t;
using ComputeT = float32_t; // S, P (softmax-time), O accumulator

// K and V live in LDS row_major (same as their global [N x D] layout) so that
// GEMM #1 can read K as a col_major matrix_b fragment == K^T for free.
using DataLayoutLds = row_major;

// ---------------------------------------------------------------------------
// Tile dimensions (fixed for D = 64)
// ---------------------------------------------------------------------------
constexpr uint32_t Br    = 64; // Q rows per CTA
constexpr uint32_t Bc    = 32; // K/V tile rows per inner iteration (see LDS note)
constexpr uint32_t Dhead = 64; // head dimension (fixed)

constexpr uint32_t ROWS_PER_WAVE = Br / WAVES; // 16 -- one MFMA M tile
constexpr uint32_t M_FRAGS       = ROWS_PER_WAVE / ROCWMMA_M; // 1
constexpr uint32_t BC_FRAGS      = Bc / ROCWMMA_N; // 2 (N of GEMM#1, K of GEMM#2)
constexpr uint32_t D_FRAGS       = Dhead / ROCWMMA_K; // 4 (K of GEMM#1, N of GEMM#2)

// ---------------------------------------------------------------------------
// Fragment types
//   GEMM #1: S [Br x Bc] = Q [Br x D] * K^T [D x Bc]      -- one wave owns 16 rows
//   GEMM #2: O [Br x D]  += P [Br x Bc] * V  [Bc x D]
// Both GEMMs use 16 x 16 x 16 MFMA tiles.
//
// The K^T view is achieved by declaring the matrix_b fragment with col_major
// layout: K's row_major memory of shape [Bc x D] is bitwise identical to a
// col_major matrix of shape [D x Bc] -- which IS K^T.
// ---------------------------------------------------------------------------
using FragQ = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, row_major>;
using FragK = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, col_major>; // K^T
using FragP = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, row_major>;
using FragV = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, row_major>;

// Accumulators (compute precision).  MfmaFragAcc holds S and O in fp32;
// MfmaFragStoreOut is the OutputT-typed accumulator we cast to right before
// the final store_matrix_sync, mirroring perf_hgemm.cpp / simple_gemm_swiglu.
using MfmaFragAcc      = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;
using MfmaFragStoreOut = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, row_major>;

// ---------------------------------------------------------------------------
// Cooperative global-read fragments (whole block contributes vectorized loads).
//
// We treat each global->LDS tile move as a matrix_a-shaped cooperative copy of
// height = (tile rows) and width = ROCWMMA_K, looped over D_FRAGS K-strips.
// The matrix_a/matrix_b "role" here is irrelevant -- LDS is just bytes; the
// compute-time fragment types (FragQ/FragK/FragV) re-interpret those bytes.
// ---------------------------------------------------------------------------
using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;

using GRBuffQ  = fragment<matrix_a, Br, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutLds, CoopScheduler>;
using GRBuffKV = fragment<matrix_a, Bc, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutLds, CoopScheduler>;

// ---------------------------------------------------------------------------
// LDS layout helpers (element counts)
// ---------------------------------------------------------------------------
constexpr uint32_t kQLdsHalf  = Br * Dhead; // 4096 halfs  (8 KB)
constexpr uint32_t kKVLdsHalf = Bc * Dhead; // 2048 halfs per K or V tile
constexpr uint32_t kPLdsHalf  = Br * Bc; // 2048 halfs  (4 KB)
// SO_lds is fp32 and shared between S [Br x Bc] and O [Br x Dhead].
constexpr uint32_t kSOLdsFloat = Br * (Bc > Dhead ? Bc : Dhead); // 4096 floats (16 KB)

// One K/V double-buffer holds K then V back-to-back.
constexpr uint32_t kKVBufHalf = 2u * kKVLdsHalf; // K + V

// Total LDS in bytes:  Q + 2*(K|V) + P (half) + SO (float)
constexpr uint32_t kLdsBytes
    = (kQLdsHalf + 2u * kKVBufHalf + kPLdsHalf) * sizeof(InputT) + kSOLdsFloat * sizeof(ComputeT);
// = (4096 + 2*4096 + 2048)*2 + 4096*4 = 28672 + 16384 = 45056 bytes = 44 KB

// Row_major leading dimension of each [rows x Dhead] tile in LDS.
constexpr uint32_t kLdsLdTile = Dhead;

// ===========================================================================
// Device helper functions
// ===========================================================================

// ---------------------------------------------------------------------------
// Cooperative global -> register read of one [rows x Dhead] tile, in D_FRAGS
// strips of ROCWMMA_K columns.  `gmem` points at element (0,0) of the tile;
// `ld` is the row stride (= Dhead for a contiguous [N x D] head).
// ---------------------------------------------------------------------------
template <typename GRBuffT>
ROCWMMA_DEVICE static inline void
    globalReadTile(GRBuffT (&gr)[D_FRAGS], InputT const* gmem, uint32_t ld)
{
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        load_matrix_sync(gr[d], gmem + d * ROCWMMA_K, ld);
}

// ---------------------------------------------------------------------------
// Cooperative register -> LDS write of one [rows x Dhead] tile.  Each strip is
// stored at its column offset d*ROCWMMA_K within the row_major LDS tile.
// ---------------------------------------------------------------------------
template <typename GRBuffT>
ROCWMMA_DEVICE static inline void
    localWriteTile(InputT* lds, GRBuffT const (&gr)[D_FRAGS], uint32_t ldsld)
{
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        store_matrix_sync(lds + d * ROCWMMA_K, apply_data_layout<DataLayoutLds>(gr[d]), ldsld);
}

// ---------------------------------------------------------------------------
// Pre-load this wave's Q row-stripe into D_FRAGS matrix_a fragments.
// Q stays constant for the whole K/V loop, so this happens once.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void
    loadQFrags(FragQ (&fragQ)[D_FRAGS], InputT const* Q_lds, uint32_t wrow)
{
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        load_matrix_sync(fragQ[d], Q_lds + wrow * Dhead + d * ROCWMMA_K, Dhead);
}

// ---------------------------------------------------------------------------
// GEMM #1:  S = Q * K^T   for this wave's 16 rows.
//   frag_S[BC_FRAGS] accumulates over the D dimension (D_FRAGS K-steps).
//   K is read as col_major (== K^T) from the row_major K tile in LDS.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void
    gemm1_QKt(MfmaFragAcc (&fragS)[BC_FRAGS], FragQ const (&fragQ)[D_FRAGS], InputT const* K_lds)
{
#pragma unroll
    for(uint32_t n = 0; n < BC_FRAGS; ++n)
        fill_fragment(fragS[n], ComputeT(0));

#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
    {
#pragma unroll
        for(uint32_t n = 0; n < BC_FRAGS; ++n)
        {
            FragK         fragK;
            InputT const* k_src = K_lds + (n * ROCWMMA_N) * Dhead + d * ROCWMMA_K;
            load_matrix_sync(fragK, k_src, Dhead);
            mma_sync(fragS[n], fragQ[d], fragK, fragS[n]);
        }
    }
}

// ---------------------------------------------------------------------------
// Store this wave's frag_S row stripe to S_lds (fp32 row_major [Br x Bc]).
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void
    storeSFrags(ComputeT* S_lds, MfmaFragAcc const (&fragS)[BC_FRAGS], uint32_t wrow)
{
#pragma unroll
    for(uint32_t n = 0; n < BC_FRAGS; ++n)
        store_matrix_sync(S_lds + wrow * Bc + n * ROCWMMA_N, fragS[n], Bc, mem_row_major);
}

// ---------------------------------------------------------------------------
// GEMM #2:  O += P * V   for this wave's 16 rows.
//   frag_O[D_FRAGS] accumulates over the Bc dimension (BC_FRAGS K-steps).
//   P is read row_major from P_lds; V is read row_major from the V tile.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void
    gemm2_PV(MfmaFragAcc (&fragO)[D_FRAGS], InputT const* P_lds, InputT const* V_lds, uint32_t wrow)
{
#pragma unroll
    for(uint32_t k = 0; k < BC_FRAGS; ++k)
    {
        FragP         fragP;
        InputT const* p_src = P_lds + wrow * Bc + k * ROCWMMA_K;
        load_matrix_sync(fragP, p_src, Bc);

#pragma unroll
        for(uint32_t d = 0; d < D_FRAGS; ++d)
        {
            FragV         fragV;
            InputT const* v_src = V_lds + (k * ROCWMMA_K) * Dhead + d * ROCWMMA_N;
            load_matrix_sync(fragV, v_src, Dhead);
            mma_sync(fragO[d], fragP, fragV, fragO[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-row online softmax (LDS-mediated)
//
//   Inputs:
//     S_lds  : [Br x Bc]  fp32  -- raw S tile from GEMM #1, destructively
//                                  replaced in-place by P (still fp32)
//     P_lds  : [Br x Bc]  fp16  -- softmax output, ready as matrix_a for GEMM#2
//     m_i, l_i : per-row scalars in registers, owned by lane = (row % wave)
//
//   Output:
//     m_i, l_i updated in-place
//     alpha_out returned per row so the caller can rescale frag_O
//
//   Each lane R (0 <= R < ROWS_PER_WAVE) handles its own row.  Lanes
//   R >= ROWS_PER_WAVE are idle for this routine.
//
//   OPTIMISATION TODO: replace the LDS round-trip with cross-lane reductions
//                      inside accumulator fragments using __shfl_xor.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void onlineSoftmaxLdsMediated(ComputeT* S_lds, // [Br x Bc] fp32
                                                           InputT*   P_lds, // [Br x Bc] fp16
                                                           uint32_t  wave_row_off,
                                                           ComputeT& m_i,
                                                           ComputeT& l_i,
                                                           ComputeT& alpha_out,
                                                           ComputeT  scale,
                                                           bool      causal,
                                                           int32_t   causal_q_row_base,
                                                           int32_t   causal_kv_col_base)
{
    const uint32_t lane     = threadIdx.x % WARP_SIZE;
    const uint32_t row_in_w = lane; // 0..WARP_SIZE-1
    const bool     active   = (row_in_w < ROWS_PER_WAVE);

    if(active)
    {
        const uint32_t row_abs = wave_row_off + row_in_w; // [0, Br)
        ComputeT*      row_ptr = S_lds + row_abs * Bc;

        // ---- 1. Apply scale + causal mask, find row max ----
        ComputeT row_max = -std::numeric_limits<ComputeT>::infinity();
#pragma unroll
        for(uint32_t c = 0; c < Bc; ++c)
        {
            ComputeT v = row_ptr[c] * scale;
            if(causal)
            {
                const int32_t q_global = causal_q_row_base + (int32_t)row_abs;
                const int32_t k_global = causal_kv_col_base + (int32_t)c;
                if(k_global > q_global)
                    v = -std::numeric_limits<ComputeT>::infinity();
            }
            row_ptr[c] = v;
            if(v > row_max)
                row_max = v;
        }

        // ---- 2. Online statistics update ----
        ComputeT m_new = (m_i > row_max) ? m_i : row_max;

        // alpha = exp(m_i - m_new); first iter has m_i = -inf -> alpha = 0.
        // Special-case isinf(m_i) to dodge a -inf - -inf = NaN.
        ComputeT alpha;
        if(m_i == -std::numeric_limits<ComputeT>::infinity())
            alpha = ComputeT(0); // first iteration: no old O to rescale
        else
            alpha = __expf(m_i - m_new);
        alpha_out = alpha;

        // ---- 3. Compute P, row sum ----
        ComputeT row_sum = ComputeT(0);
#pragma unroll
        for(uint32_t c = 0; c < Bc; ++c)
        {
            ComputeT p = __expf(row_ptr[c] - m_new);
            row_ptr[c] = p; // overwrite S_lds with P (still fp32)
            row_sum += p;
        }

        // ---- 4. l_i update ----
        l_i = alpha * l_i + row_sum;
        m_i = m_new;

        // ---- 5. Cast P to fp16 and write to P_lds ----
        InputT* p_row_ptr = P_lds + row_abs * Bc;
#pragma unroll
        for(uint32_t c = 0; c < Bc; ++c)
            p_row_ptr[c] = static_cast<InputT>(row_ptr[c]);
    }
    else
    {
        alpha_out = ComputeT(1); // idle lanes don't rescale
    }
}

// ---------------------------------------------------------------------------
// Rescale frag_O by per-row alpha (LDS round-trip)
//
//   alpha is a per-ROW scalar, but frag_O is laid out in accumulator-fragment
//   register tiles whose (lane, element) -> row mapping is arch-dependent.
//   The simplest 100%-correct trick:
//     1. Store frag_O to LDS (fp32 [Br x Dhead], row_major).
//     2. Each row-owning lane multiplies its row's elements by alpha.
//     3. Reload frag_O from LDS.
//
//   OPTIMISATION TODO: replace with in-register per-element multiplication
//                      once the (element_idx -> row) mapping is codified.
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void rescaleOLdsMediated(MfmaFragAcc (&frag_O)[D_FRAGS],
                                                      ComputeT* O_lds,
                                                      uint32_t  wave_row_off,
                                                      ComputeT  alpha,
                                                      bool      first_iter)
{
    if(first_iter)
        return; // alpha would be 0 (= -inf rescale); skip work

    const uint32_t lane     = threadIdx.x % WARP_SIZE;
    const uint32_t row_in_w = lane;
    const bool     active   = (row_in_w < ROWS_PER_WAVE);

    // ---- 1. Store frag_O to LDS, this wave's row stripe ----
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        store_matrix_sync(
            O_lds + wave_row_off * Dhead + d * ROCWMMA_N, frag_O[d], Dhead, mem_row_major);
    synchronize_workgroup();

    // ---- 2. Per-row scalar multiply ----
    if(active)
    {
        const uint32_t row_abs = wave_row_off + row_in_w;
        ComputeT*      row_ptr = O_lds + row_abs * Dhead;
#pragma unroll
        for(uint32_t d = 0; d < Dhead; ++d)
            row_ptr[d] *= alpha;
    }
    synchronize_workgroup();

    // ---- 3. Reload into frag_O ----
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        load_matrix_sync(
            frag_O[d], O_lds + wave_row_off * Dhead + d * ROCWMMA_N, Dhead, mem_row_major);
}

// ---------------------------------------------------------------------------
// Final normalisation O <- O / l_i  (LDS-mediated for the same reason)
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void finalNormalizeLdsMediated(MfmaFragAcc (&frag_O)[D_FRAGS],
                                                            ComputeT* O_lds,
                                                            uint32_t  wave_row_off,
                                                            ComputeT  l_i)
{
    const uint32_t lane     = threadIdx.x % WARP_SIZE;
    const uint32_t row_in_w = lane;
    const bool     active   = (row_in_w < ROWS_PER_WAVE);

#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        store_matrix_sync(
            O_lds + wave_row_off * Dhead + d * ROCWMMA_N, frag_O[d], Dhead, mem_row_major);
    synchronize_workgroup();

    if(active)
    {
        const uint32_t row_abs = wave_row_off + row_in_w;
        ComputeT*      row_ptr = O_lds + row_abs * Dhead;
        const ComputeT inv_l   = (l_i > ComputeT(0)) ? (ComputeT(1) / l_i) : ComputeT(0);
#pragma unroll
        for(uint32_t d = 0; d < Dhead; ++d)
            row_ptr[d] *= inv_l;
    }
    synchronize_workgroup();

#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
        load_matrix_sync(
            frag_O[d], O_lds + wave_row_off * Dhead + d * ROCWMMA_N, Dhead, mem_row_major);
}

// ---------------------------------------------------------------------------
// Write this wave's O row-stripe to global memory (ComputeT -> OutputT cast
// via an intermediate accumulator fragment, matching simple_gemm_swiglu).
// ---------------------------------------------------------------------------
ROCWMMA_DEVICE static inline void globalWriteO(OutputT* O_row_base,
                                               MfmaFragAcc const (&frag_O)[D_FRAGS])
{
#pragma unroll
    for(uint32_t d = 0; d < D_FRAGS; ++d)
    {
        MfmaFragStoreOut fragOut;
#pragma unroll
        for(uint32_t e = 0; e < (uint32_t)frag_O[d].num_elements; ++e)
            fragOut.x[e] = static_cast<OutputT>(frag_O[d].x[e]);
        // MfmaFragStoreOut carries row_major in its type -> 3-arg store form
        // (matches simple_gemm_swiglu's globalWriteD).
        store_matrix_sync(O_row_base + d * ROCWMMA_N, fragOut, Dhead);
    }
}

// ---------------------------------------------------------------------------
// Main kernel
//
// Tensor layout: contiguous [B, H, N, D] row-major
//   ptr[b, h, n, d] = base + ((b*H + h)*N + n)*D + d
//
// Grid : (ceil(N/Br), H, B)
// Block: (TBLOCK_X)
// ---------------------------------------------------------------------------
ROCWMMA_KERNEL void __launch_bounds__(TBLOCK_X) flash_attn_v2_fwd(InputT const* __restrict__ Q,
                                                                  InputT const* __restrict__ K,
                                                                  InputT const* __restrict__ V,
                                                                  OutputT* __restrict__ O,
                                                                  uint32_t B,
                                                                  uint32_t H,
                                                                  uint32_t N,
                                                                  ComputeT scale,
                                                                  bool     causal)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        // The batch count B is encoded in the grid (one CTA per Q-tile/head/
        // batch via blockIdx.z); it is not needed inside the kernel body.
        // Mark it used to avoid -Wunused-parameter on the device path.
        (void)B;

        // ------------------------------------------------------------------
        // Block / wave / lane setup
        // ------------------------------------------------------------------
        const uint32_t q_blk = blockIdx.x;
        const uint32_t head  = blockIdx.y;
        const uint32_t batch = blockIdx.z;
        const uint32_t wave  = threadIdx.x / WARP_SIZE; // 0..WAVES-1
        const uint32_t wrow  = wave * ROWS_PER_WAVE; // wave's first row in [0, Br)

        const uint32_t bh_seq_d = N * Dhead; // per-head elements
        const uint32_t bh_off   = (batch * H + head) * bh_seq_d; // (b, h) base offset
        InputT const*  Q_bh     = Q + bh_off;
        InputT const*  K_bh     = K + bh_off;
        InputT const*  V_bh     = V + bh_off;
        OutputT*       O_bh     = O + bh_off;

        // ------------------------------------------------------------------
        // LDS partition  (single dynamic shared pool)
        //   [ Q_lds | KV_lo (K|V) | KV_hi (K|V) | P_lds | SO_lds(fp32) ]
        // ------------------------------------------------------------------
        HIP_DYNAMIC_SHARED(void*, smem);
        InputT*   Q_lds  = reinterpret_cast<InputT*>(smem);
        InputT*   KV_lo  = Q_lds + kQLdsHalf;
        InputT*   KV_hi  = KV_lo + kKVBufHalf;
        InputT*   P_lds  = KV_hi + kKVBufHalf;
        ComputeT* SO_lds = reinterpret_cast<ComputeT*>(P_lds + kPLdsHalf);

        // Within a K/V double-buffer: K at offset 0, V at offset kKVLdsHalf.
        auto* kvLo = KV_lo;
        auto* kvHi = KV_hi;

        // ------------------------------------------------------------------
        // 1. Load Q tile to LDS  (cooperative, all threads), once.
        //    Q_i corresponds to global Q rows [q_blk*Br, q_blk*Br + Br).
        // ------------------------------------------------------------------
        const uint32_t q_row_base_global = q_blk * Br;
        {
            GRBuffQ grQ[D_FRAGS];
            globalReadTile(grQ, Q_bh + q_row_base_global * Dhead, Dhead);
            localWriteTile(Q_lds, grQ, kLdsLdTile);
        }
        synchronize_workgroup();

        // ------------------------------------------------------------------
        // 2. Pre-load Q fragments (this wave's row stripe, all D fragments).
        // ------------------------------------------------------------------
        FragQ frag_Q[D_FRAGS];
        loadQFrags(frag_Q, Q_lds, wrow);

        // ------------------------------------------------------------------
        // 3. Initialise running output and per-row stats.
        //    frag_O lives in registers across the entire K/V loop.
        //    m_i, l_i are per-row scalars, owned by lane R for row R.
        // ------------------------------------------------------------------
        MfmaFragAcc frag_O[D_FRAGS];
#pragma unroll
        for(uint32_t d = 0; d < D_FRAGS; ++d)
            fill_fragment(frag_O[d], ComputeT(0));

        ComputeT m_i = -std::numeric_limits<ComputeT>::infinity();
        ComputeT l_i = ComputeT(0);

        // ------------------------------------------------------------------
        // 4. Causal optimisation: K-tiles strictly to the right of Q's tile
        //    diagonal contribute nothing -- skip them.  Max global k for this
        //    Q tile is q_row_base_global + Br - 1.
        // ------------------------------------------------------------------
        const uint32_t n_kv_tiles = N / Bc;
        const uint32_t last_tile
            = causal ? rocwmma::ceil_div(q_row_base_global + Br, Bc) : n_kv_tiles;

        // ------------------------------------------------------------------
        // 5. Prologue: prefetch tile 0 (K and V) into the Lo buffer.
        // ------------------------------------------------------------------
        {
            GRBuffKV grK[D_FRAGS];
            GRBuffKV grV[D_FRAGS];
            globalReadTile(grK, K_bh + 0u * Bc * Dhead, Dhead);
            globalReadTile(grV, V_bh + 0u * Bc * Dhead, Dhead);
            localWriteTile(kvLo, grK, kLdsLdTile); // K at offset 0
            localWriteTile(kvLo + kKVLdsHalf, grV, kLdsLdTile); // V after K
        }
        synchronize_workgroup();

        // ------------------------------------------------------------------
        // 6. Main K/V loop with double-buffer prefetch.
        //
        //    Iteration j (current tile in kvLo):
        //      a. prefetch NEXT tile (j+1) from global into registers
        //      b. GEMM #1 -> softmax -> rescale O -> GEMM #2   (consume kvLo)
        //      c. write prefetched NEXT tile into kvHi
        //      d. sync, swap kvLo <-> kvHi
        //    The last tile has no successor, so it is handled as a tail with
        //    no prefetch (mirrors simple_gemm_swiglu's K-loop + tail split).
        // ------------------------------------------------------------------
        for(uint32_t j = 0; j + 1 < last_tile; ++j)
        {
            const uint32_t kv_col_base_global = j * Bc;

            // -- 6a. Prefetch next tile (j+1) into registers ---------------
            GRBuffKV grK[D_FRAGS];
            GRBuffKV grV[D_FRAGS];
            globalReadTile(grK, K_bh + (j + 1) * Bc * Dhead, Dhead);
            globalReadTile(grV, V_bh + (j + 1) * Bc * Dhead, Dhead);

            // -- 6b. Compute current tile (consume kvLo) ------------------
            InputT const* K_cur = kvLo; // [Bc x D] row_major
            InputT const* V_cur = kvLo + kKVLdsHalf; // [Bc x D] row_major

            MfmaFragAcc frag_S[BC_FRAGS];
            gemm1_QKt(frag_S, frag_Q, K_cur);

            ComputeT* S_lds = SO_lds; // alias
            storeSFrags(S_lds, frag_S, wrow);
            synchronize_workgroup();

            ComputeT alpha;
            onlineSoftmaxLdsMediated(S_lds,
                                     P_lds,
                                     wrow,
                                     m_i,
                                     l_i,
                                     alpha,
                                     scale,
                                     causal,
                                     (int32_t)q_row_base_global,
                                     (int32_t)kv_col_base_global);
            synchronize_workgroup();

            // SO_lds is now reused as O scratch (S already consumed).
            ComputeT* O_lds_alias = SO_lds;
            rescaleOLdsMediated(frag_O, O_lds_alias, wrow, alpha, /*first_iter=*/(j == 0));
            // rescale internally calls synchronize_workgroup twice

            gemm2_PV(frag_O, P_lds, V_cur, wrow);

            // -- 6c. Write prefetched next tile into kvHi -----------------
            localWriteTile(kvHi, grK, kLdsLdTile);
            localWriteTile(kvHi + kKVLdsHalf, grV, kLdsLdTile);

            // -- 6d. Barrier before swap; the barrier also guards SO/P
            //        scratch reuse by the next iteration. ----------------
            synchronize_workgroup();

            auto* tmp = kvLo;
            kvLo      = kvHi;
            kvHi      = tmp;
        }

        // ------------------------------------------------------------------
        // 7. Tail: the last K/V tile is resident in kvLo.
        // ------------------------------------------------------------------
        {
            const uint32_t j                  = last_tile - 1;
            const uint32_t kv_col_base_global = j * Bc;

            InputT const* K_cur = kvLo;
            InputT const* V_cur = kvLo + kKVLdsHalf;

            MfmaFragAcc frag_S[BC_FRAGS];
            gemm1_QKt(frag_S, frag_Q, K_cur);

            ComputeT* S_lds = SO_lds;
            storeSFrags(S_lds, frag_S, wrow);
            synchronize_workgroup();

            ComputeT alpha;
            onlineSoftmaxLdsMediated(S_lds,
                                     P_lds,
                                     wrow,
                                     m_i,
                                     l_i,
                                     alpha,
                                     scale,
                                     causal,
                                     (int32_t)q_row_base_global,
                                     (int32_t)kv_col_base_global);
            synchronize_workgroup();

            ComputeT* O_lds_alias = SO_lds;
            rescaleOLdsMediated(frag_O, O_lds_alias, wrow, alpha, /*first_iter=*/(j == 0));

            gemm2_PV(frag_O, P_lds, V_cur, wrow);
            synchronize_workgroup(); // ensure P/V reads done before SO reuse below
        }

        // ------------------------------------------------------------------
        // 8. Final normalisation O <- O / l_i
        // ------------------------------------------------------------------
        finalNormalizeLdsMediated(frag_O, SO_lds, wrow, l_i);

        // ------------------------------------------------------------------
        // 9. Cast fp32 -> fp16 and store O to global (each wave's 16-row stripe).
        // ------------------------------------------------------------------
        globalWriteO(O_bh + (q_row_base_global + wrow) * Dhead, frag_O);
    }
}

// ---------------------------------------------------------------------------
// CPU reference: plain attention
//   O[b,h,i,d] = sum_j softmax(scale * sum_k Q[b,h,i,k] * K[b,h,j,k])[j] * V[b,h,j,d]
// ---------------------------------------------------------------------------
static void flash_attn_cpu_ref(InputT const* Q,
                               InputT const* K,
                               InputT const* V,
                               OutputT*      O,
                               uint32_t      B,
                               uint32_t      H,
                               uint32_t      N,
                               uint32_t      D,
                               float         scale,
                               bool          causal)
{
    auto idx = [&](uint32_t b, uint32_t h, uint32_t n, uint32_t d) {
        return ((b * H + h) * N + n) * D + d;
    };

    std::vector<float> S(N);

    for(uint32_t b = 0; b < B; ++b)
    {
        for(uint32_t h = 0; h < H; ++h)
        {
            for(uint32_t i = 0; i < N; ++i)
            {
                // 1. Compute scaled scores S[j] = scale * <Q_i, K_j>, with mask
                float row_max = -std::numeric_limits<float>::infinity();
                for(uint32_t j = 0; j < N; ++j)
                {
                    if(causal && j > i)
                    {
                        S[j] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float acc = 0.0f;
                    for(uint32_t k = 0; k < D; ++k)
                        acc += static_cast<float>(Q[idx(b, h, i, k)])
                               * static_cast<float>(K[idx(b, h, j, k)]);
                    acc *= scale;
                    S[j] = acc;
                    if(acc > row_max)
                        row_max = acc;
                }

                // 2. Softmax (offline, for reference correctness)
                float sum = 0.0f;
                for(uint32_t j = 0; j < N; ++j)
                {
                    if(S[j] == -std::numeric_limits<float>::infinity())
                        S[j] = 0.0f;
                    else
                        S[j] = std::expf(S[j] - row_max);
                    sum += S[j];
                }
                float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                for(uint32_t j = 0; j < N; ++j)
                    S[j] *= inv_sum;

                // 3. O_i = sum_j P[j] * V_j
                for(uint32_t d = 0; d < D; ++d)
                {
                    float acc = 0.0f;
                    for(uint32_t j = 0; j < N; ++j)
                        acc += S[j] * static_cast<float>(V[idx(b, h, j, d)]);
                    O[idx(b, h, i, d)] = static_cast<OutputT>(acc);
                }
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
// Returns true if the sample actually ran to completion; false if it returned
// early (unsupported architecture / wave size or invalid problem size) so the
// caller can report an accurate final status.
ROCWMMA_HOST bool run_flash_attn_sample(uint32_t B,
                                        uint32_t H,
                                        uint32_t N,
                                        bool     causal,
                                        bool     printInfo      = false,
                                        bool     skipValidation = false)
{
    if(printInfo)
        printDeviceInfo();

    // Architecture / wave-size sanity checks (mirror simple_gemm_swiglu)
    auto warpSize = getWarpSize();
    if((isGfx11() || isGfx12()) && warpSize != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size for gfx11/12!\n";
        return false;
    }
    if(isGfx9() && warpSize != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size for gfx9!\n";
        return false;
    }

    // Reject degenerate sizes before the alignment check: a zero B/H/N passes
    // the modulo test but would yield an empty/invalid grid and out-of-bounds
    // global reads in the kernel prologue (which unconditionally prefetches
    // KV tile 0).
    if(B == 0 || H == 0 || N == 0)
    {
        std::cout << "B, H, and N must all be non-zero.\n";
        return false;
    }

    if(N % Br != 0 || N % Bc != 0)
    {
        std::cout << "N must be a multiple of " << Br << " (Br) and " << Bc << " (Bc).\n";
        return false;
    }

    uint32_t hTBLOCK_X = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;

    constexpr uint32_t D     = Dhead; // fixed
    const float        scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::cout << "Initializing host data (B=" << B << " H=" << H << " N=" << N << " D=" << D
              << " causal=" << (causal ? "true" : "false") << ")...\n";

    const size_t         total = (size_t)B * H * N * D;
    std::vector<InputT>  hQ(total), hK(total), hV(total);
    std::vector<OutputT> hO(total, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(hQ.data(), 1, total);
    fillRand(hK.data(), 1, total);
    fillRand(hV.data(), 1, total);

    // Scale inputs to keep softmax logits in a sane range.
    // fillRand produces small ints 0..4; with D=64, raw <Q,K> can reach 64*16 = 1024,
    // and scale=1/8 brings that down to 128 -- still huge for exp().
    // We further damp inputs by 1/8 each so post-scale logits are O(2) -- safe.
    constexpr float kInDamp = 1.0f / 8.0f;
    for(auto& x : hQ)
        x = static_cast<InputT>(static_cast<float>(x) * kInDamp);
    for(auto& x : hK)
        x = static_cast<InputT>(static_cast<float>(x) * kInDamp);
    for(auto& x : hV)
        x = static_cast<InputT>(static_cast<float>(x) * kInDamp);

    std::cout << "Allocating device memory...\n";
    InputT * dQ, *dK, *dV;
    OutputT* dO;
    CHECK_HIP_ERROR(hipMalloc(&dQ, total * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dK, total * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dV, total * sizeof(InputT)));
    CHECK_HIP_ERROR(hipMalloc(&dO, total * sizeof(OutputT)));

    CHECK_HIP_ERROR(hipMemcpy(dQ, hQ.data(), total * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dK, hK.data(), total * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dV, hV.data(), total * sizeof(InputT), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dO, hO.data(), total * sizeof(OutputT), hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, 1, 1);
    auto gridDim  = dim3(rocwmma::ceil_div(N, (uint32_t)Br), H, B);

    std::cout << "gridDim  (" << gridDim.x << " " << gridDim.y << " " << gridDim.z
              << ")  blockDim (" << blockDim.x << ")\n"
              << "LDS usage: " << kLdsBytes << " bytes (" << (kLdsBytes / 1024) << " KiB)\n";

    auto kernelLambda = [&]() {
        hipExtLaunchKernelGGL(flash_attn_v2_fwd,
                              gridDim,
                              blockDim,
                              kLdsBytes,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              dQ,
                              dK,
                              dV,
                              dO,
                              B,
                              H,
                              N,
                              scale,
                              causal);
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

    // FLOP estimate per forward (non-causal): 4 * B * H * N * N * D
    //   = 2*B*H*N*N*D for QK^T  +  2*B*H*N*N*D for P*V
    //   (softmax/normalise are O(B*H*N*N) -- ignored)
    double flops_per_run = 4.0 * (double)B * (double)H * (double)N * (double)N * (double)D;
    if(causal)
        flops_per_run *= 0.5; // upper triangle skipped on average
    double tflops = flops_per_run * recordRuns / (elapsedMs * 1e-3) * 1e-12;

    std::cout << std::left << std::setw(6) << "B" << std::setw(6) << "H" << std::setw(8) << "N"
              << std::setw(6) << "D" << std::setw(10) << "causal" << std::setw(14) << "elapsedMs"
              << std::setw(14) << "TFlops/s" << std::setw(8) << "Warm" << std::setw(8) << "Runs\n";
    std::cout << std::left << std::setw(6) << B << std::setw(6) << H << std::setw(8) << N
              << std::setw(6) << D << std::setw(10) << (causal ? "yes" : "no") << std::setw(14)
              << elapsedMs << std::setw(14) << tflops << std::setw(8) << warmups << std::setw(8)
              << recordRuns << "\n";

#if !NDEBUG
    if(skipValidation)
    {
        std::cout << "Skipping validation as requested.\n";
        CHECK_HIP_ERROR(hipFree(dQ));
        CHECK_HIP_ERROR(hipFree(dK));
        CHECK_HIP_ERROR(hipFree(dV));
        CHECK_HIP_ERROR(hipFree(dO));
        std::cout << "Finished!\n";
        return true;
    }

    std::cout << "\nValidating against CPU reference...\n";
    CHECK_HIP_ERROR(hipMemcpy(hO.data(), dO, total * sizeof(OutputT), hipMemcpyDeviceToHost));

    std::vector<OutputT> hOref(total, std::numeric_limits<OutputT>::signaling_NaN());
    flash_attn_cpu_ref(hQ.data(), hK.data(), hV.data(), hOref.data(), B, H, N, D, scale, causal);

    auto res = compareEqual(hO.data(), hOref.data(), total);
    std::cout << (std::get<0>(res) ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max relative error: " << std::get<1>(res) << "\n";
#else
    (void)skipValidation;
#endif

    CHECK_HIP_ERROR(hipFree(dQ));
    CHECK_HIP_ERROR(hipFree(dK));
    CHECK_HIP_ERROR(hipFree(dV));
    CHECK_HIP_ERROR(hipFree(dO));

    std::cout << "Finished!\n";
    return true;
}

// ---------------------------------------------------------------------------
// Usage:
//   ./simple_flash_attn_v2          # quick validation (B=1, H=2, N=128, non-causal + causal)
//   ./simple_flash_attn_v2 --all    # also a few realistic LLM-ish shapes
//   ./simple_flash_attn_v2 --skip-validation   # skip CPU validation
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << "Community Sample: Flash Attention v2 Forward\n";
    std::cout << "This sample demonstrates: fused attention with online softmax, "
              << "dual-GEMM register pipeline, cooperative loads + K/V double "
              << "buffering -- the core of every modern LLM attention kernel.\n";

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

    bool ok = true;

    // Quick validation: small enough to run on any GPU.
    ok &= run_flash_attn_sample(
        /*B=*/1, /*H=*/2, /*N=*/128, /*causal=*/false, /*printInfo=*/true, skipValidation);
    ok &= run_flash_attn_sample(
        /*B=*/1, /*H=*/2, /*N=*/128, /*causal=*/true, /*printInfo=*/false, skipValidation);

    if(runAll)
    {
        // GPT-2 small style (D=64 head): N=1024, H=12.
        ok &= run_flash_attn_sample(
            /*B=*/1, /*H=*/12, /*N=*/1024, /*causal=*/true, /*printInfo=*/false, skipValidation);
        // Longer context.
        ok &= run_flash_attn_sample(
            /*B=*/1, /*H=*/12, /*N=*/2048, /*causal=*/true, /*printInfo=*/false, skipValidation);
    }
    else
    {
        std::cout << "Tip: pass --all to also run larger shapes.\n";
        std::cout << "Tip: pass --skip-validation to skip CPU validation.\n";
    }

    if(ok)
        std::cout << "Sample completed successfully!" << std::endl;
    else
        std::cout << "Sample finished, but one or more runs were skipped "
                     "(unsupported architecture / wave size or invalid size); "
                     "see messages above."
                  << std::endl;

    return ok ? 0 : 1;
}
