// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <string>

#include "ck_dsl_runtime/dispatcher.hpp"

namespace ck_dsl_plugin {
namespace CkDslParamParser {

// Physical memory layout of the B operand of a Matmul whose logical dims are
// [K, N] (C[M,N] = A[M,K] x B[K,N]).
//
//   RowMajor_KN : B contiguous in N  -> strides {N, 1} (the "NN"/row-major B).
//   RCR_NK      : B contiguous in K  -> strides {1, K} (B physically stored
//                 [N,K]); this is the shipped ck_dsl GEMM ABI
//                 (C[m,n]=sum_k A[m,k]*B[n,k]).
//   Unknown     : strides absent, ambiguous, or neither of the above.
enum class BLayout { Unknown, RowMajor_KN, RCR_NK };

// Stage-1 output for a matmul/GEMM graph node.
struct ParsedGemmParams {
    int64_t a_uid = 0, b_uid = 0, c_uid = 0;
    long M = 0, N = 0, K = 0;
    std::string dtype;  // "fp16", "bf16"
    BLayout b_layout = BLayout::Unknown;
};

bool isGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

// Detect the B operand's physical layout from its logical dims [K,N] and the
// declared strides. Returns BLayout::Unknown when strides are missing or do not
// match a supported layout. `dims`/`strides` are the last-two-dim views.
BLayout detectBLayout(long K, long N, int64_t stride_outer, int64_t stride_inner);

// Map a layout to the dispatcher Problem.layout string.
const char* bLayoutName(BLayout l);

// True when the shipped ck_dsl GEMM can execute the given B layout directly.
// Today only the RCR ABI (B stored [N,K]) is supported; row-major [K,N] needs a
// pre-transpose or an NN kernel (neither shipped yet) and is rejected cleanly.
bool isSupportedBLayout(BLayout l);

ParsedGemmParams parseGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

ck_dsl::Problem buildProblem(const ParsedGemmParams& p, const std::string& arch);

}  // namespace CkDslParamParser
}  // namespace ck_dsl_plugin
