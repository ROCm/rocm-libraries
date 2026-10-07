// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// The engine of the packed Pointwise set.
inline constexpr std::string_view POINTWISE_PACKED_ENGINE = "hipkernel:pointwise_packed";

inline flatbuffers::FlatBufferBuilder buildPointwisePackedGraph()
{
    return buildPointwiseGraph();
}

inline void fillPointwisePackedInputs(hipdnn_test_sdk::utilities::GraphTensorBundle& tensors)
{
    tensors.getTensor(INPUT_A_UID).fillTensorWithValue(1.5F);
    tensors.getTensor(INPUT_B_UID).fillTensorWithValue(2.25F);
}

inline void expectPointwisePackedOutput(const std::vector<uint8_t>& output)
{
    EXPECT_EQ(valuesOf<float>(output), std::vector<float>{3.75F});
}

// The packed Pointwise set, with the kernel of block size 256.
inline constexpr PackedPlanCase POINTWISE_PACKED_PLAN_CASE{
    "Pointwise",
    POINTWISE_PACKED_ENGINE,
    POINTWISE_B256_DESCRIPTOR,
    &POINTWISE_ADD,
    OUTPUT_UID,
    &hip_kernel_provider::testing::archiveFixtureRoot,
    &buildPointwisePackedGraph,
    &buildPointwisePackedGraph,
    &fillPointwisePackedInputs,
    &expectPointwisePackedOutput,
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
