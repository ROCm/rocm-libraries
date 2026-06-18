// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlan.hpp"
#include "asm/SdpaFwdKernelArgs.hpp"
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>
#include <utility>

namespace asm_sdpa_engine
{

SdpaFwdPlan::SdpaFwdPlan(HipModuleGuard kernel, SdpaFwdParams params)
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
    args.ptr_lse = nullptr; // POC: no LSE output (withStats = false)

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

    // Options
    uint32_t tuneOpt = 5;
    // if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    if(!_params.noMask && ((_params.numHeadsQ % 8 != 0) || (_params.seqLenQ > 16384)))
    {
        tuneOpt -= 2;
    }
    if(_params.headDimQk == 192 && _params.headDimV == 128 && _params.archString == "gfx942")
    {
        tuneOpt = 0;
    }

    args.s_opt = tuneOpt;
    args.s_lse = 0; // POC: don't compute LSE

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
    unsigned int gridDimX = (_params.seqLenQ + _params.tileSizeQo - 1) / _params.tileSizeQo;
    unsigned int gridDimY = _params.numHeadsQ;
    const unsigned int gridDimZ = _params.batchSize;

    if(_params.headDimQk == 192 && _params.headDimV == 128 && _params.archString == "gfx942")
    {
        std::swap(gridDimX, gridDimY);
    }

    const unsigned int blockDimX = _params.headDimQk == 192 && _params.headDimV == 128 ? 256 : 512;

    launchKernel("fwd",
                 _kernel.function(),
                 &args,
                 sizeof(args),
                 gridDimX,
                 gridDimY,
                 gridDimZ,
                 blockDimX,
                 handle.getStream());
}

} // namespace asm_sdpa_engine
