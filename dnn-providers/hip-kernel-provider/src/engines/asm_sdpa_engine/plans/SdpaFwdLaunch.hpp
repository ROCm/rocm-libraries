// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

#include "asm/SdpaFwdKernelArgs.hpp"
#include "plans/SdpaFwdLaunchParams.hpp"
#include "plans/SdpaFwdParams.hpp"
#include "plans/SdpaKernelUtils.hpp"

/**
 * @file SdpaFwdLaunch.hpp
 * @brief Packs the forward kernarg struct and launches it.
 *
 * Separate from SdpaFwdPlan because the struct's layout is the assembly's ABI, and more
 * than one plan type now reaches the same code objects: the hand-written engine and the
 * descriptor-backed one. Two copies of a field-by-field pack against one ABI is one
 * copy too many.
 */
namespace asm_sdpa_engine
{

/**
 * @brief Launches a forward code object against @p params and the caller's buffers.
 *
 * @param function  The resolved kernel, whose kernarg layout must be fmha_fwd_v3_args.
 * @param params    Everything read off the graph at plan build.
 *
 * @throws HipdnnPluginException if the launch fails.
 */
inline void launchForward(hipFunction_t function,
                          const SdpaFwdParams& params,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          hipStream_t stream)
{
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    fmha_fwd_v3_args args{};

    args.ptr_o = uidToPtrMap.at(params.oUid);
    args.ptr_q = uidToPtrMap.at(params.qUid);
    args.ptr_k = uidToPtrMap.at(params.kUid);
    args.ptr_v = uidToPtrMap.at(params.vUid);
    args.ptr_lse = params.lseUid >= 0 ? uidToPtrMap.at(params.lseUid) : nullptr;

    // Resolved at execute so a runtime pass-by-value scale is read from the buffers the
    // caller supplied for this launch, not the ones matching saw.
    args.scalar
        = static_cast<float>(hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
            params.attnScale, deviceBuffers, numDeviceBuffers)));

    // Strides reach the kernel in bytes. Every shipped forward kernel is bf16.
    // TODO: When adding the fp8 kernels, derive this from the kernel's dtype.
    constexpr unsigned int K_BF16_SIZE = 2;
    args.s_seq_len = params.seqLenQ;
    args.s_Seqs = params.qStrideSeq * K_BF16_SIZE;
    args.s_Ts = params.tileSizeQo * params.qStrideRow * K_BF16_SIZE;
    args.s_Hs = params.qStrideHead * K_BF16_SIZE;
    args.s_Bs = params.qStrideBatch * K_BF16_SIZE;

    args.s_gqa = params.numHeadsQ / params.numHeadsKv;

    args.s_k_Seqs = params.kStrideSeq * K_BF16_SIZE;
    args.s_k_Hs = params.kStrideHead * K_BF16_SIZE;
    args.s_k_Bs = params.kStrideBatch * K_BF16_SIZE;

    const auto launchParams = computeFwdLaunchParams(params);
    args.s_opt = launchParams.tuneOpt;
    args.s_lse = params.lseUid >= 0 ? 1 : 0;

    args.s_kv_seq_len = params.seqLenKv;
    args.s_qk_head_dim = params.headDimQk;
    args.s_v_head_dim = params.headDimV;
    args.s_q_head_num = params.numHeadsQ;

    args.s_v_Seqs = params.vStrideSeq * K_BF16_SIZE;
    args.s_v_Hs = params.vStrideHead * K_BF16_SIZE;
    args.s_v_Bs = params.vStrideBatch * K_BF16_SIZE;

    args.s_o_Seqs = params.oStrideSeq * K_BF16_SIZE;
    args.s_o_Hs = params.oStrideHead * K_BF16_SIZE;
    args.s_o_Bs = params.oStrideBatch * K_BF16_SIZE;

    // Variable-length sequence and padding pointers are batch-mode nulls.
    args.ptr_qseq = nullptr;
    args.ptr_kseq = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;

    constexpr unsigned int K_FP32_SIZE = 4;
    args.s_lse_Hs = params.lseUid >= 0 ? params.lseStrideHead * K_FP32_SIZE : 0;

    // FP8 descale pointers and strides, unused by the bf16 kernels.
    args.ptr_q_descale = nullptr;
    args.ptr_k_descale = nullptr;
    args.ptr_v_descale = nullptr;
    args.s_descale_q_Bs = 0;
    args.s_descale_q_Hs = 0;
    args.s_descale_k_Bs = 0;
    args.s_descale_k_Hs = 0;
    args.s_descale_v_Bs = 0;
    args.s_descale_v_Hs = 0;

    if(!launchKernel("fwd",
                     function,
                     &args,
                     sizeof(args),
                     launchParams.gridDimX,
                     launchParams.gridDimY,
                     launchParams.gridDimZ,
                     launchParams.blockDimX,
                     stream))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "hipModuleLaunchKernel failed for SDPA forward");
    }
}

} // namespace asm_sdpa_engine
