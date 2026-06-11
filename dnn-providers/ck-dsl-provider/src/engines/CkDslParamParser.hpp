// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <string>

#include "ck_dsl_runtime/dispatcher.hpp"

namespace ck_dsl_plugin {
namespace CkDslParamParser {

// Stage-1 output for a matmul/GEMM graph node.
struct ParsedGemmParams {
    int64_t a_uid = 0, b_uid = 0, c_uid = 0;
    long M = 0, N = 0, K = 0;
    std::string dtype;  // "fp16", "bf16"
};

bool isGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

ParsedGemmParams parseGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

ck_dsl::Problem buildProblem(const ParsedGemmParams& p, const std::string& arch);

}  // namespace CkDslParamParser
}  // namespace ck_dsl_plugin
