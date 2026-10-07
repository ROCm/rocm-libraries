// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// The half cross-correlation graph with unit stride and dilation, no padding, and the x
// dims `xDims`.
inline flatbuffers::FlatBufferBuilder
    buildConvFwdPackedGraphWithX(const std::vector<int64_t>& xDims)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    return buildConvFwdGraph(data_objects::DataType::HALF,
                             data_objects::ConvMode::CROSS_CORRELATION,
                             {1, 1},
                             {1, 1},
                             {0, 0},
                             {0, 0},
                             xDims);
}

inline flatbuffers::FlatBufferBuilder buildConvFwdPackedGraph()
{
    return buildConvFwdPackedGraphWithX({1, 1, 3, 3});
}

// The x input has h != width, so a launch that swaps the two gives another output.
inline flatbuffers::FlatBufferBuilder buildConvFwdPackedRoundTripGraph()
{
    return buildConvFwdPackedGraphWithX({1, 1, 3, 5});
}

// Fills each element `index` of the half tensor `tensor` with (index % period) + 1.
inline void fillConvFwdPackedCycle(hipdnn_data_sdk::utilities::ITensor& tensor, size_t period)
{
    using hipdnn_data_sdk::types::half;

    std::vector<half> values(tensor.elementSpace());
    for(size_t index = 0; index < values.size(); ++index)
    {
        values[index] = half(static_cast<float>((index % period) + 1));
    }
    tensor.fillWithData(values.data(), values.size() * sizeof(half));
}

// Input values differ by position and by row, so a launch that reads the wrong operand or
// the wrong extent gives another result.
inline void fillConvFwdPackedInputs(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    fillConvFwdPackedCycle(tensors.getTensor(CONV_X_UID), 7);
    fillConvFwdPackedCycle(tensors.getTensor(CONV_W_UID), 3);
}

inline void expectConvFwdPackedOutput(const std::vector<uint8_t>& output)
{
    using hipdnn_data_sdk::types::half;

    // x is 3x5 with rows {1 2 3 4 5}, {6 7 1 2 3} and {4 5 6 7 1}. w is 2x2 with rows
    // {1 2} and {3 1}. y is the 2x4 cross-correlation of x and w. Each value is exact in half.
    std::vector<half> expected;
    for(const float value : {30.F, 30.F, 16.F, 23.F, 37.F, 30.F, 30.F, 30.F})
    {
        expected.emplace_back(value);
    }
    EXPECT_EQ(valuesOf<half>(output), expected);
}

// The packed ConvFwd set, with the kernel of block size 64.
inline constexpr PackedPlanCase CONV_FWD_PACKED_PLAN_CASE{
    "ConvFwd",
    CONV_FWD.engineName,
    CONV_FWD_DESCRIPTOR,
    &CONV_FWD,
    CONV_Y_UID,
    &hip_kernel_provider::testing::unitKpackRoot,
    &buildConvFwdPackedGraph,
    &buildConvFwdPackedRoundTripGraph,
    &fillConvFwdPackedInputs,
    &expectConvFwdPackedOutput,
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
