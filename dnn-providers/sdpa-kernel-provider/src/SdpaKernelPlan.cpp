// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlan.hpp"
#include "asm/AsmSdpaFwdKernelArgs.hpp"
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hip/hip_runtime.h>
#include <unordered_map>

namespace sdpa_kernel_provider
{

SdpaKernelPlan::SdpaKernelPlan(
    hipModule_t module,
    hipFunction_t function,
    int64_t qUid, int64_t kUid, int64_t vUid, int64_t oUid,
    size_t batchSize, size_t numHeadsQ, size_t numHeadsKv,
    size_t seqLenQ, size_t seqLenKv, size_t headDimQk, size_t headDimV,
    size_t qStrideSeq, size_t qStrideRow, size_t qStrideHead, size_t qStrideBatch,
    size_t kStrideSeq, size_t kStrideHead, size_t kStrideBatch,
    size_t vStrideSeq, size_t vStrideHead, size_t vStrideBatch,
    size_t oStrideSeq, size_t oStrideHead, size_t oStrideBatch,
    float attnScale)
    : _module(module)
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
            HIPDNN_PLUGIN_LOG_ERROR("Failed to unload kernel module, error: "
                                    << hipGetErrorString(err));
        }
    }
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
    // 1. Build UID→ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // 2. Get tensor pointers
    void* qPtr = uidToPtrMap.at(_qUid);
    void* kPtr = uidToPtrMap.at(_kUid);
    void* vPtr = uidToPtrMap.at(_vUid);
    void* oPtr = uidToPtrMap.at(_oUid);

    // 3. Populate kernel args struct
    fmha_fwd_v3_args args{};

    // Output/input pointers
    args.ptr_o = oPtr;
    args.ptr_q = qPtr;
    args.ptr_k = kPtr;
    args.ptr_v = vPtr;
    args.ptr_lse = nullptr;  // POC: no LSE output (withStats = false)

    // Attention scale
    args.scalar = _attnScale;

    // Q dimensions and strides (convert to bytes: stride * sizeof(bfloat16))
    constexpr size_t bf16_size = 2;
    args.s_seq_len = static_cast<unsigned int>(_seqLenQ);
    args.s_Seqs = static_cast<unsigned int>(_qStrideSeq * bf16_size);
    args.s_Ts = static_cast<unsigned int>(_qStrideRow * bf16_size);
    args.s_Hs = static_cast<unsigned int>(_qStrideHead * bf16_size);
    args.s_Bs = static_cast<unsigned int>(_qStrideBatch * bf16_size);

    // GQA ratio
    args.s_gqa = static_cast<unsigned int>(_numHeadsQ / _numHeadsKv);

    // K strides (in bytes)
    args.s_k_Seqs = static_cast<unsigned int>(_kStrideSeq * bf16_size);
    args.s_k_Hs = static_cast<unsigned int>(_kStrideHead * bf16_size);
    args.s_k_Bs = static_cast<unsigned int>(_kStrideBatch * bf16_size);

    // Options
    args.s_opt = 0;  // Default: no special options (RTNE rounding)
    args.s_lse = 0;  // POC: don't compute LSE

    // KV dimensions
    args.s_kv_seq_len = static_cast<unsigned int>(_seqLenKv);
    args.s_qk_head_dim = static_cast<unsigned int>(_headDimQk);
    args.s_v_head_dim = static_cast<unsigned int>(_headDimV);
    args.s_q_head_num = static_cast<unsigned int>(_numHeadsQ);

    // V strides (in bytes)
    args.s_v_Seqs = static_cast<unsigned int>(_vStrideSeq * bf16_size);
    args.s_v_Hs = static_cast<unsigned int>(_vStrideHead * bf16_size);
    args.s_v_Bs = static_cast<unsigned int>(_vStrideBatch * bf16_size);

    // O strides (in bytes)
    args.s_o_Seqs = static_cast<unsigned int>(_oStrideSeq * bf16_size);
    args.s_o_Hs = static_cast<unsigned int>(_oStrideHead * bf16_size);
    args.s_o_Bs = static_cast<unsigned int>(_oStrideBatch * bf16_size);

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

    // 4. Compute grid dimensions
    // From AITER: gdx = (S_q + ts_qo - 1) / ts_qo, where ts_qo = 256
    constexpr size_t ts_qo = 256;
    unsigned int gdx = static_cast<unsigned int>((_seqLenQ + ts_qo - 1) / ts_qo);
    unsigned int gdy = static_cast<unsigned int>(_numHeadsQ);
    unsigned int gdz = static_cast<unsigned int>(_batchSize);

    // Block dimensions (fixed for this kernel)
    constexpr unsigned int bdx = 512;
    constexpr unsigned int bdy = 1;
    constexpr unsigned int bdz = 1;

    // 5. Launch kernel
    void* kernelArgs[] = {&args};

    hipError_t err = hipModuleLaunchKernel(
        _function,
        gdx, gdy, gdz,  // grid dimensions
        bdx, bdy, bdz,  // block dimensions
        0,              // shared memory bytes (kernel uses LDS internally)
        nullptr,        // stream (use default)
        kernelArgs,     // kernel arguments
        nullptr);       // extra options

    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to launch kernel, error: " << hipGetErrorString(err));
        return;
    }

    HIPDNN_PLUGIN_LOG_INFO("SDPA kernel launched: grid=[" << gdx << "," << gdy << "," << gdz
                           << "] block=[" << bdx << "," << bdy << "," << bdz << "]");
}

}
