// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <string>

#include "ck_dsl_runtime/dispatcher.hpp"

namespace ck_dsl_plugin {
namespace CkDslAttnParamParser {

// Stage-1 output for an SDPA (forward) graph node. Mirrors the CK FMHA
// provider's SdpaAttributes parsing; the backend differs (ck_dsl kernels).
struct ParsedAttnParams {
    int64_t q_uid = 0, k_uid = 0, v_uid = 0, o_uid = 0;
    int64_t bias_uid = 0, lse_uid = 0;
    long batch = 0, nhead_q = 0, nhead_k = 0;
    long seqlen_q = 0, seqlen_k = 0, hdim_q = 0, hdim_v = 0;
    std::string dtype;    // "fp16", "bf16"
    bool is_bhsd = true;  // [B,H,S,D] vs [B,S,H,D]
    int mask_type = 0;    // 0=none,1=top_left,2=bottom_right,3=window
    int bias_type = 0;    // 0=none,1=elementwise,2=alibi
    bool has_lse = false;
    float scale = 0.0f;  // 0 -> 1/sqrt(hdim_q)
};

// hipDNN SDPA tensors use logical dims [B,H,S,D]. Return true when those
// logical strides describe physical BHSD-contiguous storage (H-major over S),
// which the current ck_dsl attention kernel cannot consume directly.
bool isPhysicalBhsdLayout(int64_t strideH, int64_t strideS);

bool isSdpaGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);
ParsedAttnParams parseSdpaGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);
ck_dsl::Problem buildProblem(const ParsedAttnParams& p, const std::string& arch);

}  // namespace CkDslAttnParamParser
}  // namespace ck_dsl_plugin
