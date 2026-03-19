// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "SdpaKernelEngine.hpp"
#include "SdpaKernelHandle.hpp"
#include "SdpaKernelPlanBuilder.hpp"

namespace sdpa_kernel_provider
{
namespace
{

class TestSdpaKernelEngine : public ::testing::Test
{
protected:
    SdpaKernelEngine _engine;
    SdpaKernelHandle _handle;

    void SetUp() override
    {
        _engine.addPlanBuilder(std::make_unique<SdpaKernelPlanBuilder>());
    }
};

TEST_F(TestSdpaKernelEngine, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_engine.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaKernelEngine, IsApplicableReturnsTrueForSdpaGraph)
{
    // Create a SDPA forward inference graph
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFpropGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(_engine.isApplicable(_handle, graphWrapper));
}

} // namespace
} // namespace sdpa_kernel_provider
