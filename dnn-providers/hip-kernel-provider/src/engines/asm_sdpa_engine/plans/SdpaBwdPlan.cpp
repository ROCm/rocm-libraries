// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlan.hpp"
#include "asm/SdpaBwdKernelArgs.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace
{

constexpr size_t K_WORKSPACE_ALIGNMENT_BYTES = 64;

constexpr size_t alignUp(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

// =============================================================================
// MhaBwdArgs — convenience struct mirroring AITER's mha_bwd_args
// =============================================================================
// This intermediate struct holds all high-level parameters (tensor pointers,
// element strides, dimensions, scale) so the populate* helpers below can mirror
// AITER's mha_bwd.cu (lines 430-597) line-for-line, making future AITER
// updates a textual diff.
//
// AITER provenance: csrc/include/mha_bwd.h (struct mha_bwd_args, lines 15-149)
//
// Naming convention: field names match AITER where possible.  Strides are in
// *elements* here; they are converted to bytes in the populate helpers.

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
    void* dq_acc_ptr; // workspace: FP32 dQ accumulator [B, H_q, S_q, D_qk]

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
    unsigned int nhead_stride_lsed;
    unsigned int batch_stride_lsed;

    // dq_acc strides (elements, FP32 contiguous [B, H_q, S_q, D_qk])
    unsigned int stride_dq_acc;
    int64_t nhead_stride_dq_acc;
    int64_t batch_stride_dq_acc;
};
// NOLINTEND(readability-identifier-naming)

// =============================================================================
// Populate helpers — mirror AITER mha_bwd.cu lines 430-597
// =============================================================================

constexpr unsigned int K_BF16_SIZE = 2;
constexpr unsigned int K_FP32_SIZE = 4;

// AITER reference: mha_bwd.cu lines 430-448
asm_sdpa_engine::fmha_bwd_odo_args populateOdoArgs(const MhaBwdArgs& a)
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

// AITER reference: mha_bwd.cu lines 460-561
asm_sdpa_engine::fmha_bwd_dqdkdv_args populateDqdkdvArgs(const MhaBwdArgs& a)
{
    asm_sdpa_engine::fmha_bwd_dqdkdv_args dqdkdv{};

    // Outputs — a32 accumulator: always write dQ to dq_acc workspace
    dqdkdv.ptr_dq = a.dq_acc_ptr;
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

    // Tile size: ts_kv * stride_k * sizeof(BF16)
    constexpr unsigned int K_TS_KV = 192;
    dqdkdv.Ts = K_TS_KV * a.stride_k * K_BF16_SIZE;

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

    // a32 accumulator: max_seqlen_dq = seqlen_q (AITER: v3_atomic_fp32 path)
    dqdkdv.max_seqlen_dq = a.seqlen_q;

    // No window mask for POC
    dqdkdv.mask_x = -1;
    dqdkdv.mask_y = -1;

    return dqdkdv;
}

// AITER reference: mha_bwd.cu lines 571-597
asm_sdpa_engine::fmha_bwd_post_kernel_args populatePostArgs(const MhaBwdArgs& a)
{
    asm_sdpa_engine::fmha_bwd_post_kernel_args post{};

    // a32 accumulator: dq_acc is FP32 (4 bytes per element)
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

    return a;
}

// Helper to launch a single kernel via HIP_LAUNCH_PARAM
hipError_t launchKernel(hipFunction_t func,
                        void* args,
                        size_t argSize,
                        unsigned int gridX,
                        unsigned int gridY,
                        unsigned int gridZ)
{
    constexpr unsigned int K_BLOCK_DIM = 256;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    return hipModuleLaunchKernel(func,
                                 gridX,
                                 gridY,
                                 gridZ,
                                 K_BLOCK_DIM,
                                 1,
                                 1,
                                 0, // shared memory (kernel uses LDS internally)
                                 nullptr, // stream (use default)
                                 nullptr, // kernel args (not used with config)
                                 config);
}

void unloadModuleSafe(hipModule_t module, const char* name)
{
    if(module != nullptr)
    {
        hipError_t err = hipModuleUnload(module);
        if(err != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR("Failed to unload "
                                    << name << " module, error: " << hipGetErrorString(err));
        }
    }
}

} // anonymous namespace

namespace asm_sdpa_engine
{

// =============================================================================
// Constructor / Destructor / Move
// =============================================================================

SdpaBwdPlan::SdpaBwdPlan(hipModule_t odoModule,
                         hipFunction_t odoFunc,
                         hipModule_t dqdkdvModule,
                         hipFunction_t dqdkdvFunc,
                         hipModule_t postModule,
                         hipFunction_t postFunc,
                         SdpaBwdParams params)
    : _odoModule(odoModule)
    , _dqdkdvModule(dqdkdvModule)
    , _postModule(postModule)
    , _odoFunc(odoFunc)
    , _dqdkdvFunc(dqdkdvFunc)
    , _postFunc(postFunc)
    , _params(params)
{
}

SdpaBwdPlan::~SdpaBwdPlan()
{
    unloadModuleSafe(_odoModule, "ODO");
    unloadModuleSafe(_dqdkdvModule, "DQDKDV");
    unloadModuleSafe(_postModule, "DQ_CONVERT");
}

SdpaBwdPlan::SdpaBwdPlan(SdpaBwdPlan&& other) noexcept
    : _odoModule(other._odoModule)
    , _dqdkdvModule(other._dqdkdvModule)
    , _postModule(other._postModule)
    , _odoFunc(other._odoFunc)
    , _dqdkdvFunc(other._dqdkdvFunc)
    , _postFunc(other._postFunc)
    , _params(other._params)
{
    other._odoModule = nullptr;
    other._dqdkdvModule = nullptr;
    other._postModule = nullptr;
    other._odoFunc = nullptr;
    other._dqdkdvFunc = nullptr;
    other._postFunc = nullptr;
}

SdpaBwdPlan& SdpaBwdPlan::operator=(SdpaBwdPlan&& other) noexcept
{
    if(this != &other)
    {
        unloadModuleSafe(_odoModule, "ODO");
        unloadModuleSafe(_dqdkdvModule, "DQDKDV");
        unloadModuleSafe(_postModule, "DQ_CONVERT");

        _odoModule = other._odoModule;
        _dqdkdvModule = other._dqdkdvModule;
        _postModule = other._postModule;
        _odoFunc = other._odoFunc;
        _dqdkdvFunc = other._dqdkdvFunc;
        _postFunc = other._postFunc;
        _params = other._params;

        other._odoModule = nullptr;
        other._dqdkdvModule = nullptr;
        other._postModule = nullptr;
        other._odoFunc = nullptr;
        other._dqdkdvFunc = nullptr;
        other._postFunc = nullptr;
    }
    return *this;
}

// =============================================================================
// getWorkspaceSize
// =============================================================================

size_t SdpaBwdPlan::getWorkspaceSize(const HipKernelHandle& /*handle*/) const
{
    // D buffer: [B, H_q, S_q] in FP32
    size_t dSize = static_cast<size_t>(_params.batchSize) * _params.numHeadsQ * _params.seqLenQ
                   * sizeof(float);
    dSize = alignUp(dSize, K_WORKSPACE_ALIGNMENT_BYTES);

    // dq_acc buffer: [B, H_q, S_q, D_qk] in FP32 (a32 accumulator)
    size_t dqAccSize = static_cast<size_t>(_params.batchSize) * _params.numHeadsQ * _params.seqLenQ
                       * _params.headDimQk * sizeof(float);
    dqAccSize = alignUp(dqAccSize, K_WORKSPACE_ALIGNMENT_BYTES);

    return dSize + dqAccSize;
}

// =============================================================================
// execute — 3-kernel orchestration
// =============================================================================

void SdpaBwdPlan::execute(const HipKernelHandle& /*handle*/,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    // 1. Build UID->ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // 2. Resolve tensor pointers
    void* qPtr = uidToPtrMap.at(_params.qUid);
    void* kPtr = uidToPtrMap.at(_params.kUid);
    void* vPtr = uidToPtrMap.at(_params.vUid);
    void* oPtr = uidToPtrMap.at(_params.oUid);
    void* doPtr = uidToPtrMap.at(_params.doUid);
    void* lsePtr = uidToPtrMap.at(_params.statsUid);
    void* dqPtr = uidToPtrMap.at(_params.dqUid);
    void* dkPtr = uidToPtrMap.at(_params.dkUid);
    void* dvPtr = uidToPtrMap.at(_params.dvUid);

    // 3. Carve workspace into sub-buffers
    size_t dBufSize = static_cast<size_t>(_params.batchSize) * _params.numHeadsQ * _params.seqLenQ
                      * sizeof(float);
    dBufSize = alignUp(dBufSize, K_WORKSPACE_ALIGNMENT_BYTES);

    auto* dBufPtr = workspace;
    auto* dqAccPtr = static_cast<char*>(workspace) + dBufSize;

    // 4. Build convenience args struct (mirrors AITER mha_bwd_args)
    MhaBwdArgs mhaArgs = buildMhaBwdArgs(
        _params, qPtr, kPtr, vPtr, oPtr, doPtr, lsePtr, dqPtr, dkPtr, dvPtr, dBufPtr, dqAccPtr);

    // 5. Populate and launch kernel 1: ODO
    auto odoArgs = populateOdoArgs(mhaArgs);

    constexpr unsigned int K_TS_ODO = 128; // from CSV: fmha_bwd_odo.csv
    unsigned int gdxOdo = (mhaArgs.seqlen_q + K_TS_ODO - 1) / K_TS_ODO;

    hipError_t err
        = launchKernel(_odoFunc, &odoArgs, sizeof(odoArgs), gdxOdo, mhaArgs.nhead_q, mhaArgs.batch);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to launch ODO kernel, error: " << hipGetErrorString(err));
        return;
    }

    // 6. Populate and launch kernel 2: DQDKDV
    auto dqdkdvArgs = populateDqdkdvArgs(mhaArgs);

    constexpr unsigned int K_TS_KV = 192; // from CSV: fmha_bwd_dqdkdv.csv
    unsigned int gdxDqdkdv = (mhaArgs.seqlen_k + K_TS_KV - 1) / K_TS_KV;

    err = launchKernel(
        _dqdkdvFunc, &dqdkdvArgs, sizeof(dqdkdvArgs), gdxDqdkdv, mhaArgs.nhead_q, mhaArgs.batch);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Failed to launch DQDKDV kernel, error: " << hipGetErrorString(err));
        return;
    }

    // 7. Populate and launch kernel 3: DQ_CONVERT (FP32 → BF16)
    auto postArgs = populatePostArgs(mhaArgs);

    constexpr unsigned int K_TS_DQ = 64; // from CSV: fmha_bwd_dq_convert.csv (hd128, rtne)
    unsigned int gdxPost = (mhaArgs.seqlen_q + K_TS_DQ - 1) / K_TS_DQ;

    err = launchKernel(
        _postFunc, &postArgs, sizeof(postArgs), gdxPost, mhaArgs.nhead_q, mhaArgs.batch);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Failed to launch DQ_CONVERT kernel, error: " << hipGetErrorString(err));
        return;
    }

    HIPDNN_PLUGIN_LOG_INFO("SDPA backward kernels launched: ODO grid=["
                           << gdxOdo << "," << mhaArgs.nhead_q << "," << mhaArgs.batch
                           << "] DQDKDV grid=[" << gdxDqdkdv << "," << mhaArgs.nhead_q << ","
                           << mhaArgs.batch << "] POST grid=[" << gdxPost << "," << mhaArgs.nhead_q
                           << "," << mhaArgs.batch << "]");
}

} // namespace asm_sdpa_engine
