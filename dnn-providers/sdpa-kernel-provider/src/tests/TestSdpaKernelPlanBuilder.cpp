// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelHandle.hpp"
#include "SdpaKernelPlanBuilder.hpp"

namespace sdpa_kernel_provider
{
namespace
{

class TestSdpaKernelPlanBuilder : public ::testing::Test
{
protected:
    SdpaKernelPlanBuilder _planBuilder;
    SdpaKernelHandle _handle;
};

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelPlanBuilder, IsApplicableReturnsTrueForSdpaGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

} // namespace
} // namespace sdpa_kernel_provider
