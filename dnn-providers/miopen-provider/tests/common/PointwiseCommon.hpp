// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace pointwise_common
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_test_sdk::utilities::createValidPointwiseGraph;

inline flatbuffers::FlatBufferBuilder validBinaryGraph(PointwiseMode mode = PointwiseMode::ADD)
{
    return createValidPointwiseGraph(mode,
                                     {1, 3, 4, 4}, // inputDims
                                     {1, 3, 4, 4}, // outputDims
                                     std::vector<int64_t>{48, 16, 4, 1}, // inputStrides
                                     std::vector<int64_t>{48, 16, 4, 1}, // outputStrides
                                     std::vector<int64_t>{1, 3, 1, 1}, // secondInputDims
                                     std::vector<int64_t>{3, 1, 1, 1}); // secondInputStrides
}

struct ModeCase
{
    PointwiseMode mode;
    const char* name;
};

inline const std::vector<ModeCase>& getBinaryModeCases()
{
    static const std::vector<ModeCase> s_cases = {{PointwiseMode::ADD, "Add"},
                                                  {PointwiseMode::SUB, "Sub"},
                                                  {PointwiseMode::MUL, "Mul"},
                                                  {PointwiseMode::MAX_OP, "MaxOp"},
                                                  {PointwiseMode::MIN_OP, "MinOp"}};
    return s_cases;
}

} // namespace pointwise_common
