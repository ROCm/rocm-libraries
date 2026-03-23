// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlan.hpp"
#include "asm/AsmSdpaFwdKernelArgs.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace sdpa_kernel_provider
{

SdpaKernelPlan::SdpaKernelPlan(hipModule_t kernelModule,
                               hipFunction_t function,
                               int64_t qUid,
                               int64_t kUid,
                               int64_t vUid,
                               int64_t oUid,
                               unsigned int batchSize,
                               unsigned int numHeadsQ,
                               unsigned int numHeadsKv,
                               unsigned int seqLenQ,
                               unsigned int seqLenKv,
                               unsigned int headDimQk,
                               unsigned int headDimV,
                               unsigned int qStrideSeq,
                               unsigned int qStrideRow,
                               unsigned int qStrideHead,
                               unsigned int qStrideBatch,
                               unsigned int kStrideSeq,
                               unsigned int kStrideHead,
                               unsigned int kStrideBatch,
                               unsigned int vStrideSeq,
                               unsigned int vStrideHead,
                               unsigned int vStrideBatch,
                               unsigned int oStrideSeq,
                               unsigned int oStrideHead,
                               unsigned int oStrideBatch,
                               float attnScale)
    : _module(kernelModule)
    , _function(function)
    , _qUid(qUid)
    , _kUid(kUid)
    , _vUid(vUid)
    , _oUid(oUid)
    , _batchSize(batchSize)
    , _numHeadsQ(numHeadsQ)
    , _numHeadsKv(numHeadsKv)
    , _seqLenQ(seqLenQ)
    , _seqLenKv(seqLenKv)
    , _headDimQk(headDimQk)
    , _headDimV(headDimV)
    , _qStrideSeq(qStrideSeq)
    , _qStrideRow(qStrideRow)
    , _qStrideHead(qStrideHead)
    , _qStrideBatch(qStrideBatch)
    , _kStrideSeq(kStrideSeq)
    , _kStrideHead(kStrideHead)
    , _kStrideBatch(kStrideBatch)
    , _vStrideSeq(vStrideSeq)
    , _vStrideHead(vStrideHead)
    , _vStrideBatch(vStrideBatch)
    , _oStrideSeq(oStrideSeq)
    , _oStrideHead(oStrideHead)
    , _oStrideBatch(oStrideBatch)
    , _attnScale(attnScale)
{
}

SdpaKernelPlan::~SdpaKernelPlan()
{
    if(_module != nullptr)
    {
        hipError_t err = hipModuleUnload(_module);
        if(err != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to unload kernel module, error: " << hipGetErrorString(err));
        }
    }
}

SdpaKernelPlan::SdpaKernelPlan(SdpaKernelPlan&& other) noexcept
    : _module(other._module)
    , _function(other._function)
    , _qUid(other._qUid)
    , _kUid(other._kUid)
    , _vUid(other._vUid)
    , _oUid(other._oUid)
    , _batchSize(other._batchSize)
    , _numHeadsQ(other._numHeadsQ)
    , _numHeadsKv(other._numHeadsKv)
    , _seqLenQ(other._seqLenQ)
    , _seqLenKv(other._seqLenKv)
    , _headDimQk(other._headDimQk)
    , _headDimV(other._headDimV)
    , _qStrideSeq(other._qStrideSeq)
    , _qStrideRow(other._qStrideRow)
    , _qStrideHead(other._qStrideHead)
    , _qStrideBatch(other._qStrideBatch)
    , _kStrideSeq(other._kStrideSeq)
    , _kStrideHead(other._kStrideHead)
    , _kStrideBatch(other._kStrideBatch)
    , _vStrideSeq(other._vStrideSeq)
    , _vStrideHead(other._vStrideHead)
    , _vStrideBatch(other._vStrideBatch)
    , _oStrideSeq(other._oStrideSeq)
    , _oStrideHead(other._oStrideHead)
    , _oStrideBatch(other._oStrideBatch)
    , _attnScale(other._attnScale)
{
    // Transfer ownership - set source to nullptr to prevent double-free
    other._module = nullptr;
    other._function = nullptr;
}

SdpaKernelPlan& SdpaKernelPlan::operator=(SdpaKernelPlan&& other) noexcept
{
    if(this != &other)
    {
        // Clean up existing resource
        if(_module != nullptr)
        {
            hipError_t err = hipModuleUnload(_module);
            if(err != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "Failed to unload kernel module during move assignment, error: "
                    << hipGetErrorString(err));
            }
        }

        // Transfer ownership
        _module = other._module;
        _function = other._function;
        _qUid = other._qUid;
        _kUid = other._kUid;
        _vUid = other._vUid;
        _oUid = other._oUid;
        _batchSize = other._batchSize;
        _numHeadsQ = other._numHeadsQ;
        _numHeadsKv = other._numHeadsKv;
        _seqLenQ = other._seqLenQ;
        _seqLenKv = other._seqLenKv;
        _headDimQk = other._headDimQk;
        _headDimV = other._headDimV;
        _qStrideSeq = other._qStrideSeq;
        _qStrideRow = other._qStrideRow;
        _qStrideHead = other._qStrideHead;
        _qStrideBatch = other._qStrideBatch;
        _kStrideSeq = other._kStrideSeq;
        _kStrideHead = other._kStrideHead;
        _kStrideBatch = other._kStrideBatch;
        _vStrideSeq = other._vStrideSeq;
        _vStrideHead = other._vStrideHead;
        _vStrideBatch = other._vStrideBatch;
        _oStrideSeq = other._oStrideSeq;
        _oStrideHead = other._oStrideHead;
        _oStrideBatch = other._oStrideBatch;
        _attnScale = other._attnScale;

        // Set source to nullptr to prevent double-free
        other._module = nullptr;
        other._function = nullptr;
    }
    return *this;
}

size_t SdpaKernelPlan::getWorkspaceSize(const SdpaKernelHandle& /*handle*/) const
{
    // Forward-only kernel requires no workspace (uses 64KB LDS internally)
    return 0;
}

void SdpaKernelPlan::execute(const SdpaKernelHandle& /*handle*/,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers,
                             void* /*workspace*/) const
{
    // Build UID→ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // Get tensor pointers
    void* qPtr = uidToPtrMap.at(_qUid);
    void* kPtr = uidToPtrMap.at(_kUid);
    void* vPtr = uidToPtrMap.at(_vUid);
    void* oPtr = uidToPtrMap.at(_oUid);

    // Populate kernel args struct
    fmha_fwd_v3_args args{};

    // Output/input pointers
    args.ptr_o = oPtr;
    args.ptr_q = qPtr;
    args.ptr_k = kPtr;
    args.ptr_v = vPtr;
    args.ptr_lse = nullptr; // POC: no LSE output (withStats = false)

    // Attention scale
    args.scalar = _attnScale;

    // Q dimensions and strides (convert to bytes: stride * sizeof(bfloat16))
    constexpr unsigned int K_BF16_SIZE = 2;
    args.s_seq_len = _seqLenQ;
    args.s_Seqs = _qStrideSeq * K_BF16_SIZE;
    args.s_Ts = _qStrideRow * K_BF16_SIZE;
    args.s_Hs = _qStrideHead * K_BF16_SIZE;
    args.s_Bs = _qStrideBatch * K_BF16_SIZE;

    // GQA ratio
    args.s_gqa = _numHeadsQ / _numHeadsKv;

    // K strides (in bytes)
    args.s_k_Seqs = _kStrideSeq * K_BF16_SIZE;
    args.s_k_Hs = _kStrideHead * K_BF16_SIZE;
    args.s_k_Bs = _kStrideBatch * K_BF16_SIZE;

    // Options
    args.s_opt = 0; // Default: no special options (RTNE rounding)
    args.s_lse = 0; // POC: don't compute LSE

    // KV dimensions
    args.s_kv_seq_len = _seqLenKv;
    args.s_qk_head_dim = _headDimQk;
    args.s_v_head_dim = _headDimV;
    args.s_q_head_num = _numHeadsQ;

    // V strides (in bytes)
    args.s_v_Seqs = _vStrideSeq * K_BF16_SIZE;
    args.s_v_Hs = _vStrideHead * K_BF16_SIZE;
    args.s_v_Bs = _vStrideBatch * K_BF16_SIZE;

    // O strides (in bytes)
    args.s_o_Seqs = _oStrideSeq * K_BF16_SIZE;
    args.s_o_Hs = _oStrideHead * K_BF16_SIZE;
    args.s_o_Bs = _oStrideBatch * K_BF16_SIZE;

    // Variable-length sequence pointers (nullptr for batch mode)
    args.ptr_qseq = nullptr;
    args.ptr_kseq = nullptr;

    // LSE stride (not used since ptr_lse = nullptr)
    args.s_lse_Hs = 0;

    // Padding pointers (nullptr for batch mode)
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;

    // FP8 descale pointers (nullptr for BF16)
    args.ptr_q_descale = nullptr;
    args.ptr_k_descale = nullptr;
    args.ptr_v_descale = nullptr;

    // FP8 descale strides (unused)
    args.s_descale_q_Bs = 0;
    args.s_descale_q_Hs = 0;
    args.s_descale_k_Bs = 0;
    args.s_descale_k_Hs = 0;
    args.s_descale_v_Bs = 0;
    args.s_descale_v_Hs = 0;

    // Compute grid dimensions
    // From AITER: gdx = (S_q + ts_qo - 1) / ts_qo, where ts_qo = 256
    constexpr unsigned int K_TS_QO = 256;
    unsigned int gridDimX = (_seqLenQ + K_TS_QO - 1) / K_TS_QO;
    unsigned int gridDimY = _numHeadsQ;
    unsigned int gridDimZ = _batchSize;

    // Block dimensions (fixed for this kernel)
    constexpr unsigned int K_BLOCK_DIM_X = 512;
    constexpr unsigned int K_BLOCK_DIM_Y = 1;
    constexpr unsigned int K_BLOCK_DIM_Z = 1;

    // Launch kernel using HIP_LAUNCH_PARAM mechanism
    // This is required for passing large argument structures(656 bytes) to ASM kernels
    size_t argSize = sizeof(args);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    hipError_t err = hipModuleLaunchKernel(_function,
                                           gridDimX,
                                           gridDimY,
                                           gridDimZ, // grid dimensions
                                           K_BLOCK_DIM_X,
                                           K_BLOCK_DIM_Y,
                                           K_BLOCK_DIM_Z, // block dimensions
                                           0, // shared memory bytes (kernel uses LDS internally)
                                           nullptr, // stream (use default)
                                           nullptr, // kernel arguments (not used with config)
                                           config); // extra options (HIP_LAUNCH_PARAM config)

    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to launch kernel, error: " << hipGetErrorString(err));
        return;
    }

    HIPDNN_PLUGIN_LOG_INFO("SDPA kernel launched: grid=["
                           << gridDimX << "," << gridDimY << "," << gridDimZ << "] block=["
                           << K_BLOCK_DIM_X << "," << K_BLOCK_DIM_Y << "," << K_BLOCK_DIM_Z << "]");
}

}
