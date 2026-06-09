// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlan.hpp"
#include "asm/SdpaBwdKernelArgs.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace
{

// =============================================================================
// MhaBwdArgs — convenience struct mirroring AITER's mha_bwd_args
// =============================================================================
// This intermediate struct holds all high-level parameters (tensor pointers,
// element strides, dimensions, scale) so the build* helpers below can mirror
// AITER's mha_bwd.cu line-for-line, making future AITER
// updates a textual diff.
//
// AITER provenance: csrc/include/mha_bwd.h::mha_bwd_args (commit 9522048)
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
    void* dq_ptr; // output dQ (BF16)
    void* dk_ptr; // output dK (BF16)
    void* dv_ptr; // output dV (BF16)
    void* dq_acc_ptr; // workspace: FP32 dQ accumulator (a32 only; nullptr for a16)

    // Accumulator type — determines 2-kernel (a16) vs 3-kernel (a32) path
    asm_sdpa_engine::AccumulatorType accType;

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
    // workspace allocation, and the LSE stats tensor is created with
    // contiguous strides for the same shape — so the stride values are
    // always identical, preserving AITER's shared-stride convention.
    unsigned int nhead_stride_lsed;
    unsigned int batch_stride_lsed;

    // dq_acc strides (elements, FP32 contiguous [B, H_q, S_q, D_qk])
    unsigned int stride_dq_acc;
    int64_t nhead_stride_dq_acc;
    int64_t batch_stride_dq_acc;
};
// NOLINTEND(readability-identifier-naming)

// =============================================================================
// Build helpers — mirror AITER mha_bwd.cu (commit 9522048)
// =============================================================================

constexpr unsigned int K_BF16_SIZE = 2;
constexpr unsigned int K_FP32_SIZE = 4;

constexpr unsigned int K_BWD_BLOCK_DIM = 256;

// AITER reference: mha_bwd.cu::run_fmha_bwd_odo() (commit 9522048)
asm_sdpa_engine::fmha_bwd_odo_args buildOdoArgs(const MhaBwdArgs& a)
{
    asm_sdpa_engine::fmha_bwd_odo_args odo{};
    odo.ptr_o = a.o_ptr;
    odo.ptr_do = a.do_ptr;
    odo.ptr_d = a.d_ptr;
    odo.Hs_o = a.nhead_stride_o * K_BF16_SIZE;
    odo.BAs_o = a.batch_stride_o * K_BF16_SIZE;
    odo.Seqs_o = a.stride_o * K_BF16_SIZE;
    odo.Hs_do = a.nhead_stride_do * K_BF16_SIZE;
    odo.BAs_do = a.batch_stride_do * K_BF16_SIZE;
    odo.Seqs_do = a.stride_do * K_BF16_SIZE;
    odo.Hs_d = a.nhead_stride_lsed * K_FP32_SIZE;
    odo.BAs_d = a.batch_stride_lsed * K_FP32_SIZE;
    odo.Seqs_d = 1 * K_FP32_SIZE; // contiguous along seq dim
    odo.seqlen_q = a.seqlen_q;
    odo.head_dim = a.hdim_q;
    odo.ptr_qseq = nullptr; // batch mode (POC)
    odo.ptr_qseq_padded = nullptr;
    return odo;
}

// AITER reference: mha_bwd.cu::run_fmha_bwd_dqdkdv() (commit 9522048)
//
// `tsKv` is the K/V tile size for the resolved kernel (CSV column 'ts').  In
// AITER it comes from the kernel-traits template parameter at the call site;
// here it is plumbed in from the dispatch tuple via SdpaBwdParams::dqdkdvTiles.
// Kept as an explicit parameter (not a field on MhaBwdArgs) to mirror AITER's
// mha_bwd_args layout exactly.
asm_sdpa_engine::fmha_bwd_dqdkdv_args buildDqdkdvArgs(const MhaBwdArgs& a, unsigned int tsKv)
{
    asm_sdpa_engine::fmha_bwd_dqdkdv_args dqdkdv{};

    // A32: write dQ to FP32 dq_acc workspace (DQ_CONVERT casts it to BF16 afterward).
    // A16: write dQ directly to the output BF16 buffer.
    dqdkdv.ptr_dq = (a.accType == asm_sdpa_engine::AccumulatorType::A32) ? a.dq_acc_ptr : a.dq_ptr;
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
    dqdkdv.Ts = tsKv * a.stride_k * K_BF16_SIZE;

    // Q strides (bytes)
    dqdkdv.Hs_q = a.nhead_stride_q * K_BF16_SIZE;
    dqdkdv.BAs_q = a.batch_stride_q * K_BF16_SIZE;
    dqdkdv.Seqs_q = a.stride_q * K_BF16_SIZE;

    // K strides (bytes)
    dqdkdv.Hs_k = a.nhead_stride_k * K_BF16_SIZE;
    dqdkdv.BAs_k = a.batch_stride_k * K_BF16_SIZE;
    dqdkdv.Seqs_k = a.stride_k * K_BF16_SIZE;

    // V strides (bytes)
    dqdkdv.Hs_v = a.nhead_stride_v * K_BF16_SIZE;
    dqdkdv.BAs_v = a.batch_stride_v * K_BF16_SIZE;
    dqdkdv.Seqs_v = a.stride_v * K_BF16_SIZE;

    // dO strides (bytes)
    dqdkdv.Hs_do = a.nhead_stride_do * K_BF16_SIZE;
    dqdkdv.BAs_do = a.batch_stride_do * K_BF16_SIZE;
    dqdkdv.Seqs_do = a.stride_do * K_BF16_SIZE;

    // dK strides (bytes)
    dqdkdv.Hs_dk = a.nhead_stride_dk * K_BF16_SIZE;
    dqdkdv.BAs_dk = a.batch_stride_dk * K_BF16_SIZE;
    dqdkdv.Seqs_dk = a.stride_dk * K_BF16_SIZE;

    // dV strides (bytes)
    dqdkdv.Hs_dv = a.nhead_stride_dv * K_BF16_SIZE;
    dqdkdv.BAs_dv = a.batch_stride_dv * K_BF16_SIZE;
    dqdkdv.Seqs_dv = a.stride_dv * K_BF16_SIZE;

    // LSE stride (FP32)
    dqdkdv.Hs_lsed = a.nhead_stride_lsed * K_FP32_SIZE;

    // Group mode pointers — nullptr for batch mode (POC)
    dqdkdv.ptr_qseq = nullptr;
    dqdkdv.ptr_kseq = nullptr;
    dqdkdv.ptr_qseq_padded = nullptr;
    dqdkdv.ptr_kseq_padded = nullptr;

    // a32: max_seqlen_dq = seqlen_q (AITER: v3_atomic_fp32 path)
    // a16: max_seqlen_dq = 0 (AITER convention: a16 path, no dq_convert)
    dqdkdv.max_seqlen_dq = (a.accType == asm_sdpa_engine::AccumulatorType::A32) ? a.seqlen_q : 0;

    // No window mask for POC
    dqdkdv.mask_x = -1;
    dqdkdv.mask_y = -1;

    return dqdkdv;
}

// AITER reference: mha_bwd.cu::run_fmha_bwd_convert_dq() (commit 9522048)
// Only called for a32 accumulator kernels.
asm_sdpa_engine::fmha_bwd_post_kernel_args buildPostArgs(const MhaBwdArgs& a)
{
    asm_sdpa_engine::fmha_bwd_post_kernel_args post{};

    // dq_acc is FP32 (4 bytes per element)
    post.ptr_dq_acc = a.dq_acc_ptr;
    post.ptr_dq = a.dq_ptr;
    post.Hs_dq_acc = static_cast<uint32_t>(a.nhead_stride_dq_acc) * K_FP32_SIZE;
    post.BAs_dq_acc = static_cast<uint32_t>(a.batch_stride_dq_acc) * K_FP32_SIZE;
    post.Seqs_dq_acc = a.stride_dq_acc * K_FP32_SIZE;
    post.Hs_dq = a.nhead_stride_dq * K_BF16_SIZE;
    post.BAs_dq = a.batch_stride_dq * K_BF16_SIZE;
    post.Seqs_dq = a.stride_dq * K_BF16_SIZE;
    post.seqlen_q = a.seqlen_q;
    post.head_dim = a.hdim_q;
    post.ptr_qseq = nullptr; // batch mode (POC)
    post.ptr_qseq_padded = nullptr;

    return post;
}

// Build MhaBwdArgs from SdpaBwdParams + runtime pointers
MhaBwdArgs buildMhaBwdArgs(const asm_sdpa_engine::SdpaBwdParams& p,
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
                           void* dqAccPtr)
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

    // Dimensions
    a.seqlen_q = p.seqLenQ;
    a.seqlen_k = p.seqLenKv;
    a.batch = p.batchSize;
    a.nhead_q = p.numHeadsQ;
    a.nhead_k = p.numHeadsKv;
    a.hdim_q = p.headDimQk;
    a.hdim_v = p.headDimV;
    a.scale = p.attnScale;

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

    // LSE/D strides (from stats tensor strides)
    a.nhead_stride_lsed = p.statsStrideHead;
    a.batch_stride_lsed = p.statsStrideBatch;

    // dq_acc strides — contiguous [B, H_q, S_q, D_qk] in FP32
    a.stride_dq_acc = p.headDimQk; // D_qk
    a.nhead_stride_dq_acc = static_cast<int64_t>(p.seqLenQ) * p.headDimQk; // S_q * D_qk
    a.batch_stride_dq_acc
        = static_cast<int64_t>(p.numHeadsQ) * p.seqLenQ * p.headDimQk; // H_q * S_q * D_qk

    // The kernel args structs use uint32_t for byte strides.  Verify that
    // stride_in_elements * K_FP32_SIZE fits before we silently truncate
    // in buildPostArgs().  Overflow would cause the DQ_CONVERT kernel to
    // read/write the wrong memory addresses.
    // Only relevant for a32 (a16 has no dq_acc buffer or DQ_CONVERT kernel).
    // TODO: Move this validation to frontend graph validation or operator creation
    // so oversized tensors are rejected before plan building.
    if(a.accType == asm_sdpa_engine::AccumulatorType::A32)
    {
        constexpr auto K_U32_MAX = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        if(a.nhead_stride_dq_acc * K_FP32_SIZE > K_U32_MAX
           || a.batch_stride_dq_acc * K_FP32_SIZE > K_U32_MAX)
        {
            HIPDNN_PLUGIN_LOG_ERROR("dq_acc byte strides overflow uint32_t "
                                    "(nhead_stride="
                                    << a.nhead_stride_dq_acc * K_FP32_SIZE
                                    << ", batch_stride=" << a.batch_stride_dq_acc * K_FP32_SIZE
                                    << ", max=" << K_U32_MAX << ")");
        }
    }

    return a;
}

} // anonymous namespace

namespace asm_sdpa_engine
{

// =============================================================================
// Constructors
// =============================================================================

SdpaBwdPlan::SdpaBwdPlan(HipModuleGuard odoKernel,
                         HipModuleGuard dqdkdvKernel,
                         std::optional<HipModuleGuard> postKernel,
                         SdpaBwdParams params)
    : _odoKernel(std::move(odoKernel))
    , _dqdkdvKernel(std::move(dqdkdvKernel))
    , _postKernel(std::move(postKernel))
    , _params(params)
{
}

SdpaBwdPlan::SdpaBwdPlan(HipModuleGuard odoKernel,
                         HipModuleGuard dqdkdvKernel,
                         SdpaBwdParams params)
    : _odoKernel(std::move(odoKernel))
    , _dqdkdvKernel(std::move(dqdkdvKernel))
    , _postKernel(std::nullopt)
    , _params(params)
{
}

// =============================================================================
// getWorkspaceSize
// =============================================================================

size_t SdpaBwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    return sdpaBwdWorkspaceSize(_params.batchSize,
                                _params.numHeadsQ,
                                _params.seqLenQ,
                                _params.headDimQk,
                                _params.accumulatorType);
}

// =============================================================================
// execute — 3-kernel orchestration
// =============================================================================

void SdpaBwdPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    // 1. Validate workspace.  getMaxWorkspaceSize() always reports a non-zero
    // size for backward SDPA, so a null workspace pointer here is a contract
    // violation by the caller.
    if(workspace == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlan::execute: workspace is null but backward SDPA requires a non-zero "
            "workspace (see getMaxWorkspaceSize())");
    }

    // 2. Build UID->ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // 3. Resolve tensor pointers
    void* qPtr = uidToPtrMap.at(_params.qUid);
    void* kPtr = uidToPtrMap.at(_params.kUid);
    void* vPtr = uidToPtrMap.at(_params.vUid);
    void* oPtr = uidToPtrMap.at(_params.oUid);
    void* doPtr = uidToPtrMap.at(_params.doUid);
    void* lsePtr = uidToPtrMap.at(_params.statsUid);
    void* dqPtr = uidToPtrMap.at(_params.dqUid);
    void* dkPtr = uidToPtrMap.at(_params.dkUid);
    void* dvPtr = uidToPtrMap.at(_params.dvUid);

    // 4. Carve workspace into sub-buffers.
    // A32: dq_acc follows D buffer (DQDKDV accumulates FP32 dQ there, then DQ_CONVERT casts).
    // A16: DQDKDV writes dQ directly to the output buffer; dq_acc is not allocated.
    auto* dBufPtr = workspace;
    // A32: dq_acc buffer follows D buffer in workspace.
    // A16: no dq_acc buffer (nullptr) — DQDKDV writes dQ directly to user output.
    void* dqAccPtr = nullptr;
    if(_params.accumulatorType == AccumulatorType::A32)
    {
        dqAccPtr = static_cast<char*>(workspace)
                   + sdpaBwdDBufferSize(_params.batchSize, _params.numHeadsQ, _params.seqLenQ);
    }

    // 5. Build convenience args struct (mirrors AITER mha_bwd_args).
    // Byte-stride uint32 overflow was already rejected by isApplicable.
    const MhaBwdArgs mhaArgs = buildMhaBwdArgs(
        _params, qPtr, kPtr, vPtr, oPtr, doPtr, lsePtr, dqPtr, dkPtr, dvPtr, dBufPtr, dqAccPtr);

    // 6. Launch kernels on the same stream.
    // a32: 3 kernels — ODO → DQDKDV → DQ_CONVERT (sequential dependencies)
    // a16: 2 kernels — ODO → DQDKDV (dQ written directly in BF16)
    // Launching on the same stream guarantees ordering without explicit barriers.
    auto stream = handle.getStream();

    // 6a. Build args and launch kernel 1: ODO
    auto odoArgs = buildOdoArgs(mhaArgs);

    const unsigned int gdxOdo = _params.odoTiles.gridDim(mhaArgs.seqlen_q);

    if(!launchKernel("SDPA backward ODO",
                     _odoKernel.function(),
                     &odoArgs,
                     sizeof(odoArgs),
                     gdxOdo,
                     mhaArgs.nhead_q,
                     mhaArgs.batch,
                     K_BWD_BLOCK_DIM,
                     stream))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for SDPA backward ODO");
    }
    plan_utils::throwOnLaunchPostError("SDPA backward ODO");

    // 6b. Build args and launch kernel 2: DQDKDV
    auto dqdkdvArgs = buildDqdkdvArgs(mhaArgs, _params.dqdkdvTiles.ts);

    const unsigned int gdxDqdkdv = _params.dqdkdvTiles.gridDim(mhaArgs.seqlen_k);

    // A32: zero dq_acc before DQDKDV. The atomic-accumulator kernel adds per-K-tile
    // dQ contributions atomically and does not pre-zero; stale residue from a
    // prior workspace lease would silently corrupt dQ. AITER allocates dq_accum
    // via torch::zeros (aiter/csrc/py_itfs_cu/asm_mha_bwd.cu:137 at commit 9522048).
    // A16 writes dQ directly — no accumulator buffer needed, skip the memset.
    if(_params.accumulatorType == AccumulatorType::A32)
    {
        const size_t dqAccBytes = sdpaBwdDqAccBufferSize(
            _params.batchSize, _params.numHeadsQ, _params.seqLenQ, _params.headDimQk);
        const hipError_t memsetErr = hipMemsetAsync(dqAccPtr, 0, dqAccBytes, stream);
        if(memsetErr != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("SdpaBwdPlan::execute: failed to zero dq_acc workspace before SDPA "
                            "backward DQDKDV, error: ")
                    + hipGetErrorString(memsetErr));
        }
    }

    if(!launchKernel("SDPA backward DQDKDV",
                     _dqdkdvKernel.function(),
                     &dqdkdvArgs,
                     sizeof(dqdkdvArgs),
                     gdxDqdkdv,
                     mhaArgs.nhead_q,
                     mhaArgs.batch,
                     K_BWD_BLOCK_DIM,
                     stream))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for SDPA backward DQDKDV");
    }
    plan_utils::throwOnLaunchPostError("SDPA backward DQDKDV");

    // 6c. DQ_CONVERT (FP32 → BF16) — A32 path only.
    // A16 wrote dQ directly to the output BF16 buffer in step 6b; no cast needed.
    if(_params.accumulatorType == AccumulatorType::A32)
    {
        auto postArgs = buildPostArgs(mhaArgs);

        const unsigned int gdxPost = _params.dqConvertTiles.gridDim(mhaArgs.seqlen_q);

        if(!launchKernel("SDPA backward DQ_CONVERT",
                         _postKernel->function(),
                         &postArgs,
                         sizeof(postArgs),
                         gdxPost,
                         mhaArgs.nhead_q,
                         mhaArgs.batch,
                         K_BWD_BLOCK_DIM,
                         stream))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for SDPA backward DQ_CONVERT");
        }
        plan_utils::throwOnLaunchPostError("SDPA backward DQ_CONVERT");
    }
}

} // namespace asm_sdpa_engine
