// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <string>

#include "ck_dsl_runtime/dispatcher.hpp"

namespace ck_dsl_plugin {
namespace CkDslConvParamParser {

// 2D forward convolution (implicit-GEMM): NHWC input, KRSC weights, NHWK output.
struct ParsedConvParams {
    int64_t x_uid = 0, w_uid = 0, y_uid = 0;
    int N = 0, Hi = 0, Wi = 0, C = 0, K = 0, G = 1, R = 0, S = 0;
    int sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
    std::string dtype;
    int Ho() const {
        return (Hi + 2 * pH - dH * (R - 1) - 1) / sH + 1;
    }
    int Wo() const {
        return (Wi + 2 * pW - dW * (S - 1) - 1) / sW + 1;
    }
};

bool isConvGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);
ParsedConvParams parseConvGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);
ck_dsl::Problem buildProblem(const ParsedConvParams& p, const std::string& arch);

}  // namespace CkDslConvParamParser
}  // namespace ck_dsl_plugin
