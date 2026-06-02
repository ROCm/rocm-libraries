// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "engines/plans/MiopenConvBwdPlan.hpp"

using namespace miopen_plugin;

class TestGpuConvBwdPlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

TEST(TestConvBwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    const ConvBwdParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.dx());
    EXPECT_NO_THROW(params.w());
    EXPECT_NO_THROW(params.dy());
    EXPECT_NO_THROW(params.conv());
}

TEST(TestConvBwdParams, ThrowsOnAssymetricPadding)
{
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0}; // Asymmetic padding
    const std::vector<int64_t> convPostPadding = {1, 1};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvBwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvBwdParams, ThrowsOnInvalidPostPaddingVectorSize)
{
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvBwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvBwdParams, ThrowsOnInvalidPaddingVectorsSize)
{
    // Create a convolution graph with invalid conv dims
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0, 0}; // Invalid pre padding vector size
    const std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvBwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvBwdParams, ThrowsOnInvalidStrideVectorSize)
{
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1}; // Invalid strides vector size
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvBwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvBwdParams, ThrowsOnInvalidDilationVectorSize)
{
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1}; // Invalid dilation vector size
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvBwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvBwdPlan, CreatesPlanWithValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvBwdParams params(*attrs, graph.getTensorMap());

    // Create plan
    const HipdnnMiopenSettings executionSettings;
    ConvBwdPlan(*_handle, std::move(params), executionSettings);
}

TEST_F(TestGpuConvBwdPlan, PlanUsesDefaultWorkspaceSizeWhenNoLimitSet)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    ConvBwdParams params(*attrs, graph.getTensorMap());

    const size_t defaultSize = 4096;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(defaultSize);

    const ConvBwdPlan plan(*_handle, std::move(params), settings);
    EXPECT_EQ(plan.getWorkspaceSize(*_handle), defaultSize);
}

TEST_F(TestGpuConvBwdPlan, PlanUsesKnobLimitOverDefault)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    ConvBwdParams params(*attrs, graph.getTensorMap());

    const size_t defaultSize = 4096;
    const size_t knobLimit = 2048;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(defaultSize);
    settings.setWorkspaceSizeLimit(knobLimit);

    const ConvBwdPlan plan(*_handle, std::move(params), settings);
    EXPECT_EQ(plan.getWorkspaceSize(*_handle), knobLimit);
}

TEST_F(TestGpuConvBwdPlan, ThrowsOnInvalidDims)
{
    // Create a convolution graph with invalid conv dims
    const std::vector<int64_t> dxDims = {1, 1, 1, 1};
    const std::vector<int64_t> dxStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 4, 4}; // dy too big
    const std::vector<int64_t> dyStrides = {1, 1, 4, 16};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvBwdParams params(*attrs, graph.getTensorMap());

    // Create plan and expect exception
    const HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(ConvBwdPlan(*_handle, std::move(params), executionSettings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
