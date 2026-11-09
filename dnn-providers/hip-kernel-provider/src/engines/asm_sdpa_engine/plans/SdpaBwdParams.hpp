// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

namespace asm_sdpa_engine
{

/// Accumulator precision for dQ in the backward DQDKDV kernel (AITER's
/// v3_atomic_fp32).
///
/// - A32: dQ is accumulated in FP32 into a dq_acc workspace, then a DQ_CONVERT
///   kernel writes dQ (ODO -> DQDKDV -> DQ_CONVERT).
/// - A16: dQ is accumulated in 16 bits. On gfx942 the DQDKDV kernel writes dQ
///   directly (ODO -> DQDKDV); on gfx950 it accumulates into a Q-dtype dq_acc
///   workspace and a DQ_SHUFFLE kernel writes dQ (ODO -> DQDKDV -> DQ_SHUFFLE).
enum class AccumulatorType : uint8_t
{
    A32,
    A16
};

/// CSV `mask` ordinal of the gfx950 dqdkdv kernels for a sliding window with
/// bottom-right alignment (AITER mha_bwd.cu asm_mask_type()). Ordinals 0-3 are
/// plan_utils::MaskType; this one exists only in the gfx950 backward registry.
inline constexpr int32_t BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW = 4;

/**
 * @brief Parameters for SDPA backward kernel execution.
 *
 * Holds tensor UIDs, dimensions, strides, and attention scale
 * extracted from the operation graph.  All strides are in elements
 * (converted to bytes at kernel-launch time).
 */
struct SdpaBwdParams
{
    // --- Tensor UIDs ---
    // Inputs
    int64_t qUid;
    int64_t kUid;
    int64_t vUid;
    int64_t oUid;
    int64_t doUid;
    int64_t statsUid; // LSE from forward pass

    // Outputs
    int64_t dqUid;
    int64_t dkUid;
    int64_t dvUid;

    // --- Tensor dimensions ---
    unsigned int batchSize; // B
    unsigned int numHeadsQ; // H_q
    unsigned int numHeadsKv; // H_kv
    unsigned int seqLenQ; // S_q
    unsigned int seqLenKv; // S_kv (= S_k)
    unsigned int headDimQk; // D_qk, unpadded (the kernels take the graph's head dims)
    unsigned int headDimV; // D_v, unpadded

    // --- Tensor strides (in elements) ---
    // Q: [B, H_q, S_q, D_qk]
    unsigned int qStrideSeq;
    unsigned int qStrideHead;
    unsigned int qStrideBatch;

    // K: [B, H_kv, S_kv, D_qk]
    unsigned int kStrideSeq;
    unsigned int kStrideHead;
    unsigned int kStrideBatch;

    // V: [B, H_kv, S_kv, D_v]
    unsigned int vStrideSeq;
    unsigned int vStrideHead;
    unsigned int vStrideBatch;

    // O: [B, H_q, S_q, D_v]
    unsigned int oStrideSeq;
    unsigned int oStrideHead;
    unsigned int oStrideBatch;

    // dO: [B, H_q, S_q, D_v]
    unsigned int doStrideSeq;
    unsigned int doStrideHead;
    unsigned int doStrideBatch;

    // dQ: [B, H_q, S_q, D_qk]
    unsigned int dqStrideSeq;
    unsigned int dqStrideHead;
    unsigned int dqStrideBatch;

    // dK: [B, H_kv, S_kv, D_qk]
    unsigned int dkStrideSeq;
    unsigned int dkStrideHead;
    unsigned int dkStrideBatch;

    // dV: [B, H_kv, S_kv, D_v]
    unsigned int dvStrideSeq;
    unsigned int dvStrideHead;
    unsigned int dvStrideBatch;

    // Stats (LSE) and D: packed [B, H_q, S_q] in FP32. buildPlan derives these
    // from Q's dims; they are not the stats tensor's strides.
    unsigned int statsStrideHead;
    unsigned int statsStrideBatch;

    // Attention scale — resolved at execute via resolveScalarOperand().
    // Supports compile-time constant, runtime-with-default, and pure runtime
    // user-supplied (RFC 0016 pass-by-value) states.
    hipdnn_plugin_sdk::ScalarOperand attnScale;

    // Per-stage tile sizes; populated by SdpaBwdPlanBuilder from the resolved
    // CSV configs ('ts' column) and consumed by SdpaBwdPlan grid math.
    struct KernelTiles
    {
        unsigned int ts; // K/V or convert tile size (column 'ts' in the AITER CSV)

        // Ceil-divide an extent by `ts` to get the corresponding grid-x dimension.
        // Returns 0 if `ts` is unset (KernelTiles default-initialised) so callers
        // can fail loudly at launch time rather than divide-by-zero.
        // When `isCausal` is true the grid is halved (rounded up) because causal
        // kernels tile only the triangular region of the attention matrix.
        constexpr unsigned int gridDim(unsigned int extent, bool isCausal = false) const noexcept
        {
            unsigned int gd = ts == 0U ? 0U : (extent + ts - 1U) / ts;
            if(isCausal)
            {
                gd = (gd + 1U) / 2U;
            }
            return gd;
        }
    };
    KernelTiles odoTiles{}; // from cfg_fmha_bwd_odo
    KernelTiles dqdkdvTiles{}; // from cfg_fmha_bwd_dqdkdv
    KernelTiles postTiles{}; // from cfg_fmha_bwd_dq_convert or cfg_fmha_bwd_dq_shuffle

    AccumulatorType accumulatorType = AccumulatorType::A32;

    // Device architecture the kernels were resolved for (e.g. "gfx942").
    std::string archString;

    // dqdkdv CSV `mask` ordinal of the resolved kernel:
    //   0 = NO_MASK, 1 = TOP_LEFT_CAUSAL, 2 = BOTTOM_RIGHT_CAUSAL,
    //   3 = SLIDING_WINDOW (top-left on gfx950, either alignment on gfx942),
    //   4 = BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW (gfx950 only)
    int32_t maskOrdinal = 0;

    // FlashAttention window sizes for the sliding-window masks (3 and 4),
    // -1 for unbounded. Left at -1 for the other masks.
    int32_t windowLeft = -1;
    int32_t windowRight = -1;

    // Diagonal alignment of the sliding window. True = top-left (query i sits
    // on key i), false = bottom-right (query i sits on key i + S_kv - S_q).
    // computeMaskCoordinates() uses it for the gfx942 window kernels.
    bool topLeftAlignment = true;
};

} // namespace asm_sdpa_engine
