// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlan.hpp"
#include "asm/SdpaFwdKernelArgs.hpp"
#include "plans/SdpaFwdLaunchParams.hpp"
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace asm_sdpa_engine
{

SdpaFwdPlan::SdpaFwdPlan(CachedModule kernel, SdpaFwdParams params)
    : _kernel(std::move(kernel))
    , _params(std::move(params))
{
}

size_t SdpaFwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    // Forward-only kernel requires no workspace (uses 64KB LDS internally)
    return 0;
}

void SdpaFwdPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* /*workspace*/) const
{
    // Build UID->ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // Get tensor pointers
    void* qPtr = uidToPtrMap.at(_params.qUid);
    void* kPtr = uidToPtrMap.at(_params.kUid);
    void* vPtr = uidToPtrMap.at(_params.vUid);
    void* oPtr = uidToPtrMap.at(_params.oUid);

    // Populate kernel args struct
    fmha_fwd_v3_args args{};

    // Output/input pointers
    args.ptr_o = oPtr;
    args.ptr_q = qPtr;
    args.ptr_k = kPtr;
    args.ptr_v = vPtr;
    if(_params.lseUid >= 0)
    {
        args.ptr_lse = uidToPtrMap.at(_params.lseUid);
    }
    else
    {
        args.ptr_lse = nullptr;
    }

    // Attention scale
    args.scalar = _params.attnScale;

    // Q dimensions and strides (convert to bytes: stride * sizeof(bfloat16))
    // TODO: When adding the fp8 kernels, modify this to check for the datatype
    constexpr unsigned int K_BF16_SIZE = 2;
    args.s_seq_len = _params.seqLenQ;
    args.s_Seqs = _params.qStrideSeq * K_BF16_SIZE;
    args.s_Ts = _params.tileSizeQo * _params.qStrideRow * K_BF16_SIZE;
    args.s_Hs = _params.qStrideHead * K_BF16_SIZE;
    args.s_Bs = _params.qStrideBatch * K_BF16_SIZE;

    // GQA ratio
    args.s_gqa = _params.numHeadsQ / _params.numHeadsKv;

    // K strides (in bytes)
    args.s_k_Seqs = _params.kStrideSeq * K_BF16_SIZE;
    args.s_k_Hs = _params.kStrideHead * K_BF16_SIZE;
    args.s_k_Bs = _params.kStrideBatch * K_BF16_SIZE;

    // Options and grid dimensions
    const auto launchParams = computeFwdLaunchParams(_params);
    args.s_opt = launchParams.tuneOpt;
    args.s_lse = (_params.lseUid >= 0) ? 1 : 0;

    // KV dimensions
    args.s_kv_seq_len = _params.seqLenKv;
    args.s_qk_head_dim = _params.headDimQk;
    args.s_v_head_dim = _params.headDimV;
    args.s_q_head_num = _params.numHeadsQ;

    // V strides (in bytes)
    args.s_v_Seqs = _params.vStrideSeq * K_BF16_SIZE;
    args.s_v_Hs = _params.vStrideHead * K_BF16_SIZE;
    args.s_v_Bs = _params.vStrideBatch * K_BF16_SIZE;

    // O strides (in bytes)
    args.s_o_Seqs = _params.oStrideSeq * K_BF16_SIZE;
    args.s_o_Hs = _params.oStrideHead * K_BF16_SIZE;
    args.s_o_Bs = _params.oStrideBatch * K_BF16_SIZE;

    // Variable-length sequence pointers (nullptr for batch mode)
    args.ptr_qseq = nullptr;
    args.ptr_kseq = nullptr;

    // LSE stride (head dimension, in bytes)
    constexpr unsigned int K_FP32_SIZE = 4;
    args.s_lse_Hs = (_params.lseUid >= 0) ? _params.lseStrideHead * K_FP32_SIZE : 0;

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

    launchKernel("fwd",
                 _kernel->function(),
                 &args,
                 sizeof(args),
                 launchParams.gridDimX,
                 launchParams.gridDimY,
                 launchParams.gridDimZ,
                 launchParams.blockDimX,
                 handle.getStream());
}

} // namespace asm_sdpa_engine
