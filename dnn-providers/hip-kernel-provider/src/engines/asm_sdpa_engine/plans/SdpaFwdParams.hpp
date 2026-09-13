// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaPlanUtils.hpp"

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

namespace asm_sdpa_engine
{

/**
 * @brief Parameters for SDPA forward kernel execution.
 *
 * Holds tensor UIDs, dimensions, strides, and attention scale
 * extracted from the operation graph.
 */
struct SdpaFwdParams
{
    // Tensor UIDs
    int64_t qUid;
    int64_t kUid;
    int64_t vUid;
    int64_t oUid;
    int64_t lseUid = -1; // LSE output, -1 = disabled

    // FP8 descale tensor UIDs (-1 = absent, i.e. non-fp8 path). When present these
    // dequantize the fp8 Q/K/V inputs; the kernel reads them via ptr_*_descale.
    int64_t qDescaleUid = -1;
    int64_t kDescaleUid = -1;
    int64_t vDescaleUid = -1;

    // Bytes per element for the Q/K/V inputs: 1 for fp8, 2 for bf16. The output is
    // always 2-byte BF16 regardless. Element strides are multiplied by this to get
    // the byte strides the kernel expects.
    unsigned int inBytesPerElement = 2;

    // Tensor dimensions
    unsigned int batchSize; // B
    unsigned int numHeadsQ; // H_q
    unsigned int numHeadsKv; // H_kv
    unsigned int seqLenQ; // S_q
    unsigned int seqLenKv; // S_kv
    unsigned int headDimQk; // D_qk (128 for POC)
    unsigned int headDimV; // D_v

    // Q tensor strides (in elements)
    unsigned int qStrideSeq;
    unsigned int qStrideRow;
    unsigned int qStrideHead;
    unsigned int qStrideBatch;

    // K tensor strides (in elements)
    unsigned int kStrideSeq;
    unsigned int kStrideHead;
    unsigned int kStrideBatch;

    // V tensor strides (in elements)
    unsigned int vStrideSeq;
    unsigned int vStrideHead;
    unsigned int vStrideBatch;

    // O tensor strides (in elements)
    unsigned int oStrideSeq;
    unsigned int oStrideHead;
    unsigned int oStrideBatch;

    // LSE tensor stride (in elements).  The forward kernel args struct
    // (fmha_fwd_v3_args) only exposes s_lse_Hs — no batch stride field.
    unsigned int lseStrideHead = 0;

    // Tile size
    unsigned int tileSizeQo;

    // Architecture
    std::string archString;

    // Mask type
    plan_utils::MaskType maskType;

    // Attention scale — resolved at execute via resolveScalarOperand().
    // Supports compile-time constant, runtime-with-default, and pure runtime
    // user-supplied (RFC 0016 pass-by-value) states.
    hipdnn_plugin_sdk::ScalarOperand attnScale;
};

} // namespace asm_sdpa_engine
