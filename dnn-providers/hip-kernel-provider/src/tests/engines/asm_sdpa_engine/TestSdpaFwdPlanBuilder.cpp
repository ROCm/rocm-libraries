// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"

namespace asm_sdpa_engine
{
namespace
{

class TestSdpaFwdPlanBuilder : public ::testing::Test
{
protected:
    SdpaFwdPlanBuilder _planBuilder;
    HipKernelHandle _handle;
};

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsTrueForSdpaGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions (withStats = false by default)
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the workspace size from the plan builder
    HipKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Forward-only kernel uses LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor (stats_tensor_uid), not workspace
    // The default test graph has withStats = false, so workspace should be 0
    EXPECT_EQ(workspaceSize, 0u);
}

} // namespace
} // namespace asm_sdpa_engine
