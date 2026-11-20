// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "../asm/SdpaBwdKernelArgs.hpp"
#include "SdpaBwdParams.hpp"
#include "SdpaKernelUtils.hpp"
#include "SdpaPlanUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace asm_sdpa_engine
{

// =============================================================================
// MhaBwdArgs — convenience struct bridging SdpaBwdParams → kernel args
// =============================================================================
// Subset of AITER's mha_bwd_args (csrc/include/mha_bwd.h) covering the
// batch-mode, no-bias, no-dropout path.  Lets the build* helpers below mirror
// AITER's mha_bwd.cu field assignments line-for-line, making future AITER
// updates a textual diff.
//
// Omitted vs AITER's full mha_bwd_args: dispatch flags (use_asm_v3 etc.),
// bias/dbias, dropout, group-mode seq pointers, sink attention, allocators,
// and the mask fields (dqdkdvMaskCoordinates() reads them from SdpaBwdParams).
//
// AITER provenance: csrc/include/mha_bwd.h (AITER v0.1.29, commit a0c4b69)
//
// Naming convention: field names match AITER where possible.  Strides are in
// *elements* here; they are converted to bytes in the build helpers.

// NOLINTBEGIN(readability-identifier-naming)
struct MhaBwdArgs
{
    // Tensor pointers (set in execute from device buffers + workspace)
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr;
    const void* lse_ptr; // stats/LSE from forward pass
    const void* do_ptr;
    void* d_ptr; // workspace: D reduction buffer [B, H_q, S_q] FP32
    void* dq_ptr; // output dQ
    void* dk_ptr; // output dK
    void* dv_ptr; // output dV
    void* dq_acc_ptr; // workspace dq_acc; nullptr without a post stage

    // Accumulator type (AITER v3_atomic_fp32) and whether a post stage reads
    // dq_acc (AITER need_post_processing)
    AccumulatorType accType;
    bool need_post_processing;

    // Dimensions
    unsigned int seqlen_q;
    unsigned int seqlen_k;
    unsigned int batch;
    unsigned int nhead_q;
    unsigned int nhead_k;
    unsigned int hdim_q;
    unsigned int hdim_v;
    float scale;

    // Strides (all in elements, NOT bytes)
    // Q
    unsigned int stride_q;
    unsigned int nhead_stride_q;
    unsigned int batch_stride_q;
    // K
    unsigned int stride_k;
    unsigned int nhead_stride_k;
    unsigned int batch_stride_k;
    // V
    unsigned int stride_v;
    unsigned int nhead_stride_v;
    unsigned int batch_stride_v;
    // O
    unsigned int stride_o;
    unsigned int nhead_stride_o;
    unsigned int batch_stride_o;
    // dO
    unsigned int stride_do;
    unsigned int nhead_stride_do;
    unsigned int batch_stride_do;
    // dQ
    unsigned int stride_dq;
    unsigned int nhead_stride_dq;
    unsigned int batch_stride_dq;
    // dK
    unsigned int stride_dk;
    unsigned int nhead_stride_dk;
    unsigned int batch_stride_dk;
    // dV
    unsigned int stride_dv;
    unsigned int nhead_stride_dv;
    unsigned int batch_stride_dv;

    // LSE/D buffer strides (elements, FP32 [B, H_q, S_q])
    //
    // "lsed" = LSE-and-D.  In AITER, both the LSE stats tensor and the D
    // reduction workspace are allocated as contiguous torch::empty() with
    // identical shape [B, H_q, S_q], so they share the same strides.  AITER
    // uses a single stride set (nhead_stride_lsed, batch_stride_lsed) for
    // both the ODO kernel's D output (Hs_d, BAs_d) and the DQDKDV kernel's
    // LSE input (Hs_lsed).
    //
    // In hip-kernel-provider the D buffer is carved from a contiguous
    // workspace allocation, and SdpaBwdPlanBuilder::buildPlan sets both strides
    // to the packed [B, H_q, S_q] values. isApplicable declines any stats tensor
    // that is not packed on every axis of size > 1, so these strides describe
    // the LSE tensor too, preserving AITER's shared-stride convention.
    unsigned int nhead_stride_lsed;
    unsigned int batch_stride_lsed;

    // dq_acc layout (strides in elements), sdpaBwdDqAccLayout()
    size_t dq_acc_element_size;
    unsigned int stride_dq_acc;
    int64_t nhead_stride_dq_acc;
    int64_t batch_stride_dq_acc;
};
// NOLINTEND(readability-identifier-naming)

// =============================================================================
// Build helpers — mirror AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() (commit a0c4b69)
// =============================================================================
//
// Pure (they perform no device calls), so they can be unit-tested without
// loading a kernel module.

inline constexpr unsigned int K_BWD_BF16_SIZE = 2;
inline constexpr unsigned int K_BWD_FP32_SIZE = 4;

/// The window coordinates (mask_y, mask_x) the DQDKDV kernel takes for the
/// mask in `params`, or (0, 0) for a mask the kernel bakes in. Mirrors AITER
/// v0.1.29 mha_bwd.cu fmha_v3_bwd(): the gfx950 window kernels (masks 3 and 4)
/// take the FlashAttention window sizes as they are, negative for unbounded,
/// and align the window to each batch themselves; the gfx942 window kernel
/// (mask 3) takes the coordinates computeMaskCoordinates() derives from the
/// window sizes, the sequence lengths and the alignment.
inline std::pair<int32_t, int32_t> dqdkdvMaskCoordinates(const SdpaBwdParams& params)
{
    const bool isWindow
        = params.maskOrdinal == static_cast<int32_t>(plan_utils::MaskType::SLIDING_WINDOW)
          || params.maskOrdinal == BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW;
    if(!isWindow)
    {
        return {0, 0};
    }
    if(params.archString == "gfx950")
    {
        return {params.windowLeft, params.windowRight}; // {mask_y, mask_x}
    }
    return plan_utils::computeMaskCoordinates(params.windowLeft,
                                              params.windowRight,
                                              static_cast<int32_t>(params.seqLenQ),
                                              static_cast<int32_t>(params.seqLenKv),
                                              params.topLeftAlignment);
}

// AITER reference: mha_bwd.cu fmha_v3_bwd(), odo_args
inline fmha_bwd_odo_args buildOdoArgs(const MhaBwdArgs& a)
{
    fmha_bwd_odo_args odo{};
    odo.ptr_o = a.o_ptr;
    odo.ptr_do = a.do_ptr;
    odo.ptr_d = a.d_ptr;
    odo.Hs_o = a.nhead_stride_o * K_BWD_BF16_SIZE;
    odo.BAs_o = a.batch_stride_o * K_BWD_BF16_SIZE;
    odo.Seqs_o = a.stride_o * K_BWD_BF16_SIZE;
    odo.Hs_do = a.nhead_stride_do * K_BWD_BF16_SIZE;
    odo.BAs_do = a.batch_stride_do * K_BWD_BF16_SIZE;
    odo.Seqs_do = a.stride_do * K_BWD_BF16_SIZE;
    odo.Hs_d = a.nhead_stride_lsed * K_BWD_FP32_SIZE;
    odo.BAs_d = a.batch_stride_lsed * K_BWD_FP32_SIZE;
    odo.Seqs_d = 1 * K_BWD_FP32_SIZE; // contiguous along seq dim
    odo.seqlen_q = a.seqlen_q;
    // D_qk, unpadded, as AITER passes it (mha_bwd.cu:493), although the odo row
    // is selected by padded D_v: gfx950 D_qk = 192, D_v = 128 runs the hd128 odo
    // kernel with head_dim = 192.
    odo.head_dim = a.hdim_q;
    // The engine serves batch mode only, so the sequence-offset pointers are null.
    odo.ptr_qseq = nullptr;
    odo.ptr_qseq_padded = nullptr;
    return odo;
}

// AITER reference: mha_bwd.cu fmha_v3_bwd(), dqdkdv_args
//
// `tsKv` is the K/V tile size for the resolved kernel (CSV column 'ts').  In
// AITER it comes from the kernel-traits template parameter at the call site;
// here it is plumbed in from the dispatch tuple via SdpaBwdParams::dqdkdvTiles.
// Kept as an explicit parameter (not a field on MhaBwdArgs) to mirror AITER's
// mha_bwd_args layout exactly.
inline fmha_bwd_dqdkdv_args
    buildDqdkdvArgs(const MhaBwdArgs& a, unsigned int tsKv, const SdpaBwdParams& p)
{
    fmha_bwd_dqdkdv_args dqdkdv{};

    // AITER mha_bwd.cu:599: dq_acc when a post stage writes dQ from it,
    // otherwise dQ itself.
    dqdkdv.ptr_dq = a.need_post_processing ? a.dq_acc_ptr : a.dq_ptr;
    dqdkdv.ptr_dk = a.dk_ptr;
    dqdkdv.ptr_dv = a.dv_ptr;

    // Inputs
    dqdkdv.ptr_q = a.q_ptr;
    dqdkdv.ptr_k = a.k_ptr;
    dqdkdv.ptr_v = a.v_ptr;
    dqdkdv.ptr_do = a.do_ptr;
    dqdkdv.ptr_lse = a.lse_ptr;
    dqdkdv.ptr_d = a.d_ptr;

    // Scalars
    dqdkdv.scalar = a.scale;
    dqdkdv.log2e = 1.44269504089f; // log2(e)
    dqdkdv.ratio = a.nhead_q / a.nhead_k; // GQA

    // Dimensions
    dqdkdv.seqlen_q = a.seqlen_q;
    dqdkdv.seqlen_k = a.seqlen_k;
    dqdkdv.head_dim_q = a.hdim_q;
    dqdkdv.head_dim_v = a.hdim_v;
    dqdkdv.nhead_q = a.nhead_q;

    // Tile size: tsKv * stride_k * sizeof(BF16)
    dqdkdv.Ts = tsKv * a.stride_k * K_BWD_BF16_SIZE;

    // Q strides (bytes)
    dqdkdv.Hs_q = a.nhead_stride_q * K_BWD_BF16_SIZE;
    dqdkdv.BAs_q = a.batch_stride_q * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_q = a.stride_q * K_BWD_BF16_SIZE;

    // K strides (bytes)
    dqdkdv.Hs_k = a.nhead_stride_k * K_BWD_BF16_SIZE;
    dqdkdv.BAs_k = a.batch_stride_k * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_k = a.stride_k * K_BWD_BF16_SIZE;

    // V strides (bytes)
    dqdkdv.Hs_v = a.nhead_stride_v * K_BWD_BF16_SIZE;
    dqdkdv.BAs_v = a.batch_stride_v * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_v = a.stride_v * K_BWD_BF16_SIZE;

    // dO strides (bytes)
    dqdkdv.Hs_do = a.nhead_stride_do * K_BWD_BF16_SIZE;
    dqdkdv.BAs_do = a.batch_stride_do * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_do = a.stride_do * K_BWD_BF16_SIZE;

    // dK strides (bytes)
    dqdkdv.Hs_dk = a.nhead_stride_dk * K_BWD_BF16_SIZE;
    dqdkdv.BAs_dk = a.batch_stride_dk * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_dk = a.stride_dk * K_BWD_BF16_SIZE;

    // dV strides (bytes)
    dqdkdv.Hs_dv = a.nhead_stride_dv * K_BWD_BF16_SIZE;
    dqdkdv.BAs_dv = a.batch_stride_dv * K_BWD_BF16_SIZE;
    dqdkdv.Seqs_dv = a.stride_dv * K_BWD_BF16_SIZE;

    // LSE stride (FP32)
    dqdkdv.Hs_lsed = a.nhead_stride_lsed * K_BWD_FP32_SIZE;

    // The engine serves batch mode only, so the sequence-offset pointers are null.
    dqdkdv.ptr_qseq = nullptr;
    dqdkdv.ptr_kseq = nullptr;
    dqdkdv.ptr_qseq_padded = nullptr;
    dqdkdv.ptr_kseq_padded = nullptr;

    // AITER mha_bwd.cu:658: S_q for A32, S_q rounded up to 16 for A16 (the rows
    // of its 16-bit dq_acc).
    dqdkdv.max_seqlen_dq
        = (a.accType == AccumulatorType::A32) ? a.seqlen_q : (a.seqlen_q + 15U) / 16U * 16U;

    // Window attention mask coordinates. The masks other than the sliding
    // windows are baked into the kernel binary, which ignores mask_y/mask_x.
    std::tie(dqdkdv.mask_y, dqdkdv.mask_x) = dqdkdvMaskCoordinates(p);

    return dqdkdv;
}

// AITER reference: mha_bwd.cu fmha_v3_bwd(), post_args (:730-750). The same
// arguments drive DQ_CONVERT (FP32 dq_acc) and DQ_SHUFFLE (16-bit dq_acc).
inline fmha_bwd_post_kernel_args buildPostArgs(const MhaBwdArgs& a)
{
    fmha_bwd_post_kernel_args post{};

    post.ptr_dq_acc = a.dq_acc_ptr;
    post.ptr_dq = a.dq_ptr;
    post.Hs_dq_acc = static_cast<uint64_t>(a.nhead_stride_dq_acc) * a.dq_acc_element_size;
    post.BAs_dq_acc = static_cast<uint64_t>(a.batch_stride_dq_acc) * a.dq_acc_element_size;
    post.Seqs_dq_acc = static_cast<uint32_t>(a.stride_dq_acc * a.dq_acc_element_size);
    post.Hs_dq = static_cast<uint64_t>(a.nhead_stride_dq) * K_BWD_BF16_SIZE;
    post.BAs_dq = static_cast<uint64_t>(a.batch_stride_dq) * K_BWD_BF16_SIZE;
    post.Seqs_dq = a.stride_dq * K_BWD_BF16_SIZE;
    post.seqlen_q = a.seqlen_q;
    post.head_dim = a.hdim_q;
    // The engine serves batch mode only, so the sequence-offset pointers are null.
    post.ptr_qseq = nullptr;
    post.ptr_qseq_padded = nullptr;

    return post;
}

// Build MhaBwdArgs from SdpaBwdParams + runtime pointers + resolved scale
inline MhaBwdArgs buildMhaBwdArgs(const SdpaBwdParams& p,
                                  float resolvedAttnScale,
                                  const void* qPtr,
                                  const void* kPtr,
                                  const void* vPtr,
                                  const void* oPtr,
                                  const void* doPtr,
                                  const void* lsePtr,
                                  void* dqPtr,
                                  void* dkPtr,
                                  void* dvPtr,
                                  void* dBufPtr,
                                  void* dqAccPtr,
                                  bool needPostProcessing)
{
    MhaBwdArgs a{};

    // Tensor pointers
    a.q_ptr = qPtr;
    a.k_ptr = kPtr;
    a.v_ptr = vPtr;
    a.o_ptr = oPtr;
    a.lse_ptr = lsePtr;
    a.do_ptr = doPtr;
    a.d_ptr = dBufPtr;
    a.dq_ptr = dqPtr;
    a.dk_ptr = dkPtr;
    a.dv_ptr = dvPtr;
    a.dq_acc_ptr = dqAccPtr;
    a.accType = p.accumulatorType;
    a.need_post_processing = needPostProcessing;

    // Dimensions
    a.seqlen_q = p.seqLenQ;
    a.seqlen_k = p.seqLenKv;
    a.batch = p.batchSize;
    a.nhead_q = p.numHeadsQ;
    a.nhead_k = p.numHeadsKv;
    a.hdim_q = p.headDimQk;
    a.hdim_v = p.headDimV;
    a.scale = resolvedAttnScale;

    // Q strides (elements)
    a.stride_q = p.qStrideSeq;
    a.nhead_stride_q = p.qStrideHead;
    a.batch_stride_q = p.qStrideBatch;

    // K strides
    a.stride_k = p.kStrideSeq;
    a.nhead_stride_k = p.kStrideHead;
    a.batch_stride_k = p.kStrideBatch;

    // V strides
    a.stride_v = p.vStrideSeq;
    a.nhead_stride_v = p.vStrideHead;
    a.batch_stride_v = p.vStrideBatch;

    // O strides
    a.stride_o = p.oStrideSeq;
    a.nhead_stride_o = p.oStrideHead;
    a.batch_stride_o = p.oStrideBatch;

    // dO strides
    a.stride_do = p.doStrideSeq;
    a.nhead_stride_do = p.doStrideHead;
    a.batch_stride_do = p.doStrideBatch;

    // dQ strides
    a.stride_dq = p.dqStrideSeq;
    a.nhead_stride_dq = p.dqStrideHead;
    a.batch_stride_dq = p.dqStrideBatch;

    // dK strides
    a.stride_dk = p.dkStrideSeq;
    a.nhead_stride_dk = p.dkStrideHead;
    a.batch_stride_dk = p.dkStrideBatch;

    // dV strides
    a.stride_dv = p.dvStrideSeq;
    a.nhead_stride_dv = p.dvStrideHead;
    a.batch_stride_dv = p.dvStrideBatch;

    // LSE/D strides: packed [B, H_q, S_q], derived from Q's dims by buildPlan
    a.nhead_stride_lsed = p.statsStrideHead;
    a.batch_stride_lsed = p.statsStrideBatch;

    // dq_acc layout, AITER mha_bwd.cu:577-596. The DQDKDV kernels address
    // dq_acc with 32-bit offsets within a batch, and the pssk = 0 kernels across
    // batches too, although the post kernel strides are 64-bit. The builder
    // declines graphs whose offsets would wrap: sdpaBwdDqAccFitsU32 (isApplicable
    // for A32, buildPlan for A16) and the dqdkdv lookup's pssk = 0 guard.
    const auto dqAcc
        = sdpaBwdDqAccLayout(p.batchSize, p.numHeadsQ, p.seqLenQ, p.headDimQk, p.accumulatorType);
    a.dq_acc_element_size = dqAcc.elementBytes;
    a.stride_dq_acc = static_cast<unsigned int>(dqAcc.stride);
    a.nhead_stride_dq_acc = static_cast<int64_t>(dqAcc.nheadStride);
    a.batch_stride_dq_acc = static_cast<int64_t>(dqAcc.batchStride);

    return a;
}

} // namespace asm_sdpa_engine
