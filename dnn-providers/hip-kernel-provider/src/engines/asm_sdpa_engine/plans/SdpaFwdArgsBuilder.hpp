// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "../asm/SdpaFwdKernelArgs.hpp"
#include "SdpaFwdLaunchParams.hpp"
#include "SdpaFwdParams.hpp"

#include <cstdint>
#include <unordered_map>

namespace asm_sdpa_engine
{

/**
 * @brief Build the packed forward kernel-argument struct from plan params and a
 *        UID->device-pointer map.
 *
 * Pure (performs no device calls), so it can be unit-tested without loading a
 * kernel module. Q/K/V strides use params.inBytesPerElement (1 for fp8, 2 for
 * bf16); the output stays 2-byte BF16. FP8 descale pointers are populated when
 * the corresponding descale UIDs are set; per-tensor descales carry zero strides.
 */
inline fmha_fwd_v3_args buildFwdKernelArgs(const SdpaFwdParams& params,
                                           const std::unordered_map<int64_t, void*>& uidToPtrMap)
{
    fmha_fwd_v3_args args{};

    // Output/input pointers
    args.ptr_o = uidToPtrMap.at(params.oUid);
    args.ptr_q = uidToPtrMap.at(params.qUid);
    args.ptr_k = uidToPtrMap.at(params.kUid);
    args.ptr_v = uidToPtrMap.at(params.vUid);
    args.ptr_lse = (params.lseUid >= 0) ? uidToPtrMap.at(params.lseUid) : nullptr;

    // Attention scale (args.scalar) is a runtime pass-by-value operand resolved by
    // the caller at execute via resolveScalarOperand(), so it is not set here — this
    // keeps buildFwdKernelArgs pure (no device_buffers dependency, unit-testable).

    // Input bytes-per-element: 1 for fp8, 2 for bf16. Output is always 2-byte BF16.
    const unsigned int inBpe = params.inBytesPerElement;
    constexpr unsigned int K_BF16_SIZE = 2;

    // Q dimensions and strides (element strides converted to byte strides)
    args.s_seq_len = params.seqLenQ;
    args.s_Seqs = params.qStrideSeq * inBpe;
    args.s_Ts = params.tileSizeQo * params.qStrideRow * inBpe;
    args.s_Hs = params.qStrideHead * inBpe;
    args.s_Bs = params.qStrideBatch * inBpe;

    // GQA ratio
    args.s_gqa = params.numHeadsQ / params.numHeadsKv;

    // K strides (in bytes)
    args.s_k_Seqs = params.kStrideSeq * inBpe;
    args.s_k_Hs = params.kStrideHead * inBpe;
    args.s_k_Bs = params.kStrideBatch * inBpe;

    // Options and grid dimensions
    const auto launchParams = computeFwdLaunchParams(params);
    args.s_opt = launchParams.tuneOpt;
    args.s_lse = (params.lseUid >= 0) ? 1 : 0;

    // KV dimensions
    args.s_kv_seq_len = params.seqLenKv;
    args.s_qk_head_dim = params.headDimQk;
    args.s_v_head_dim = params.headDimV;
    args.s_q_head_num = params.numHeadsQ;

    // V strides (in bytes)
    args.s_v_Seqs = params.vStrideSeq * inBpe;
    args.s_v_Hs = params.vStrideHead * inBpe;
    args.s_v_Bs = params.vStrideBatch * inBpe;

    // O strides (in bytes) — output stays BF16
    args.s_o_Seqs = params.oStrideSeq * K_BF16_SIZE;
    args.s_o_Hs = params.oStrideHead * K_BF16_SIZE;
    args.s_o_Bs = params.oStrideBatch * K_BF16_SIZE;

    // Variable-length sequence pointers (nullptr for batch mode)
    args.ptr_qseq = nullptr;
    args.ptr_kseq = nullptr;

    // LSE stride (head dimension, in bytes)
    constexpr unsigned int K_FP32_SIZE = 4;
    args.s_lse_Hs = (params.lseUid >= 0) ? params.lseStrideHead * K_FP32_SIZE : 0;

    // Padding pointers (nullptr for batch mode)
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;

    // FP8 descale pointers (nullptr for BF16). Mirrors AITER's init_fmha_fwd_v3_args
    // fp8 branch: the kernel dequantizes Q/K/V through these per-tensor scalars.
    args.ptr_q_descale = (params.qDescaleUid >= 0) ? uidToPtrMap.at(params.qDescaleUid) : nullptr;
    args.ptr_k_descale = (params.kDescaleUid >= 0) ? uidToPtrMap.at(params.kDescaleUid) : nullptr;
    args.ptr_v_descale = (params.vDescaleUid >= 0) ? uidToPtrMap.at(params.vDescaleUid) : nullptr;

    // FP8 descale strides. Per-tensor descales are scalars (no batch/head variation),
    // so all strides are zero. Per-(batch, KV-head) descales are a future extension.
    args.s_descale_q_Bs = 0;
    args.s_descale_q_Hs = 0;
    args.s_descale_k_Bs = 0;
    args.s_descale_k_Hs = 0;
    args.s_descale_v_Bs = 0;
    args.s_descale_v_Hs = 0;

    return args;
}

} // namespace asm_sdpa_engine
