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
#include "engines/plans/MiopenConvFwdPlan.hpp"

using namespace miopen_plugin;

class TestGpuConvFwdPlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

TEST(TestConvFwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    const ConvFwdParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.w());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.conv());
}

TEST(TestConvFwdParams, ThrowsOnAssymetricPadding)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0}; // Asymmetic padding
    const std::vector<int64_t> convPostPadding = {1, 1};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidPostPaddingVectorSize)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidPaddingVectorsSize)
{
    // Create a convolution graph with invalid conv dims
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0, 0}; // Invalid pre padding vector size
    const std::vector<int64_t> convPostPadding = {0, 0, 0}; // Invalid post padding vector size
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidStrideVectorSize)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1}; // Invalid strides vector size
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvFwdParams, ThrowsOnInvalidDilationVectorSize)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> wStrides = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1}; // Invalid dilation vector size
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params and expect exception
    EXPECT_THROW(ConvFwdParams(*attrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvFwdPlan, CreatesPlanWithValidGraph)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvFwdParams params(*attrs, graph.getTensorMap());

    // Create plan
    const HipdnnMiopenSettings executionSettings;
    ConvFwdPlan(*_handle, std::move(params), executionSettings);
}

TEST_F(TestGpuConvFwdPlan, ThrowsOnInvalidDims)
{
    // Create a convolution graph with invalid conv dims
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> xStrides = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1}; // Invalid w tensor dims
    const std::vector<int64_t> wStrides = {1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> yStrides = {1, 1, 1, 1};
    const std::vector<int64_t> convPrePadding = {0, 0};
    const std::vector<int64_t> convPostPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    ConvFwdParams params(*attrs, graph.getTensorMap());

    // Create plan and expect exception
    const HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(ConvFwdPlan(*_handle, std::move(params), executionSettings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvFwdPlan, PlanUsesDefaultWorkspaceSizeWhenNoLimitSet)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    ConvFwdParams params(*attrs, graph.getTensorMap());

    const size_t defaultSize = 4096;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(defaultSize);

    const ConvFwdPlan plan(*_handle, std::move(params), settings);
    EXPECT_EQ(plan.getWorkspaceSize(*_handle), defaultSize);
}

TEST_F(TestGpuConvFwdPlan, PlanUsesKnobLimitOverDefault)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    ConvFwdParams params(*attrs, graph.getTensorMap());

    const size_t defaultSize = 4096;
    const size_t knobLimit = 2048;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(defaultSize);
    settings.setWorkspaceSizeLimit(knobLimit);

    const ConvFwdPlan plan(*_handle, std::move(params), settings);
    EXPECT_EQ(plan.getWorkspaceSize(*_handle), knobLimit);
}

TEST(TestConvFwdParams, AcceptsDeterministicEnabledFlag)
{
    // Create a valid convolution graph
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the convolution node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params with deterministic enabled
    EXPECT_NO_THROW(ConvFwdParams(*attrs, graph.getTensorMap(), true));

    // Construct params with deterministic disabled (default)
    EXPECT_NO_THROW(ConvFwdParams(*attrs, graph.getTensorMap(), false));
}
