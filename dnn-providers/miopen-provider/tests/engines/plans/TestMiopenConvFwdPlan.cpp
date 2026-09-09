// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include "MiopenApi.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

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

TEST(TestConvFwdParams, PadsOneDimensionalGraphToTwoDimensions)
{
    // NCL tensors with one spatial dimension. MIOpen has no 1D convolution, so
    // the provider pads the tensors to 4D and the convolution to 2 spatial dims.
    const std::vector<int64_t> xDims = {1, 4, 8};
    const std::vector<int64_t> xStrides = {32, 8, 1};
    const std::vector<int64_t> wDims = {4, 4, 3};
    const std::vector<int64_t> wStrides = {12, 3, 1};
    const std::vector<int64_t> yDims = {1, 4, 6};
    const std::vector<int64_t> yStrides = {24, 6, 1};
    const std::vector<int64_t> convPrePadding = {0};
    const std::vector<int64_t> convPostPadding = {0};
    const std::vector<int64_t> convStrides = {1};
    const std::vector<int64_t> convDilation = {1};

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

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    const ConvFwdParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(params.spatialDimCount(), 1);

    for(const auto* tensor : {&params.x(), &params.w(), &params.y()})
    {
        int dimCount = 0;
        EXPECT_EQ(miopenGetTensorDescriptorSize(tensor->tensorDescriptor(), &dimCount),
                  miopenStatusSuccess);
        EXPECT_EQ(dimCount, 4);
    }

    int convSpatialDimCount = 0;
    EXPECT_EQ(miopenGetConvolutionSpatialDim(params.conv().convDescriptor(), &convSpatialDimCount),
              miopenStatusSuccess);
    EXPECT_EQ(convSpatialDimCount, 2);
}

TEST(TestConvFwdParams, PadsOneDimensionalChannelsLastGraphWithChannelStride)
{
    // NLC tensors: the padded trailing dimension must take the channel count as
    // its stride, not 1, or MIOpen reads the tensor as channels-first.
    constexpr int64_t CHANNEL_COUNT = 4;
    const std::vector<int64_t> xDims = {1, CHANNEL_COUNT, 8};
    const std::vector<int64_t> xStrides = {32, 1, CHANNEL_COUNT};
    const std::vector<int64_t> wDims = {4, CHANNEL_COUNT, 3};
    const std::vector<int64_t> wStrides = {12, 1, CHANNEL_COUNT};
    const std::vector<int64_t> yDims = {1, CHANNEL_COUNT, 6};
    const std::vector<int64_t> yStrides = {24, 1, CHANNEL_COUNT};
    const std::vector<int64_t> convPrePadding = {0};
    const std::vector<int64_t> convPostPadding = {0};
    const std::vector<int64_t> convStrides = {1};
    const std::vector<int64_t> convDilation = {1};

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

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(attrs, nullptr);

    const ConvFwdParams params(*attrs, graph.getTensorMap());

    for(const auto* tensor : {&params.x(), &params.w(), &params.y()})
    {
        std::vector<int> dims(4);
        std::vector<int> strides(4);
        miopenDataType_t dataType{};
        EXPECT_EQ(miopenGetTensorDescriptor(
                      tensor->tensorDescriptor(), &dataType, dims.data(), strides.data()),
                  miopenStatusSuccess);
        EXPECT_EQ(dims[3], 1);
        EXPECT_EQ(strides[3], CHANNEL_COUNT);
    }
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
