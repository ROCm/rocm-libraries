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
#include "engines/plans/MiopenConvFwdBiasActivPlan.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "tests/common/TestWorkarounds.hpp"

using namespace miopen_plugin;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

class TestGpuConvFwdBiasActivPlan : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    // Per ROCm/rocm-libraries#6979 these tests must skip when the device has
    // no applicable CBA solver. Each test passes its own constructed graph so
    // the skip check uses the same shape the test would actually exercise.
    std::unique_ptr<HipdnnMiopenHandle> _handle;
    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
};

TEST(TestConvFwdBiasActivParams, InitializeFromValidConvFwdActivGraph)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());
}

TEST(TestConvFwdBiasActivParams, InitializeFromValidConvFwdBiasActivGraph)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());
}

TEST(TestConvFwdBiasActivParams, ThrowOnWrongActivationMode)
{
    const std::vector<int64_t> xDims = {1, 1, 1, 1};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 1, 1};
    const std::vector<int64_t> convPadding = {0, 0};
    const std::vector<int64_t> convStrides = {1, 1};
    const std::vector<int64_t> convDilation = {1, 1};

    auto builder = createValidConvFwdBiasActivGraph(xDims,
                                                    generateStrides(xDims),
                                                    wDims,
                                                    generateStrides(wDims),
                                                    yDims,
                                                    generateStrides(yDims),
                                                    convPadding,
                                                    convPadding,
                                                    convStrides,
                                                    convDilation,
                                                    PointwiseMode::ADD); // Invalid activation mode
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    EXPECT_THROW(ConvFwdBiasActivParams(*convAttr, biasAttr, *activAttr, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraph)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraphCompileOnly)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings, true, false);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdActivGraphWorkspaceOnly)
{
    auto builder = createValidConvFwdActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* activAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, nullptr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings, false, true);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraph)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraphCompileOnly)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings, true, false);
}

TEST_F(TestGpuConvFwdBiasActivPlan, CreatePlanWithValidConvFwdBiasActivGraphWorkspaceOnly)
{
    auto builder = createValidConvFwdBiasActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    SKIP_IF_NO_APPLICABLE_CBA_ENGINE(*_handle, _planBuilder, graph);

    const auto& node0 = graph.getNode(0);
    const auto* convAttr = node0.attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(convAttr, nullptr);

    const auto& node1 = graph.getNode(1);
    const auto* biasAttr = node1.attributes_as_PointwiseAttributes();
    ASSERT_NE(biasAttr, nullptr);

    const auto& node2 = graph.getNode(2);
    const auto* activAttr = node2.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttr, nullptr);

    ConvFwdBiasActivParams params(*convAttr, biasAttr, *activAttr, graph.getTensorMap());

    HipdnnMiopenSettings executionSettings;
    ConvFwdBiasActivPlan(*_handle, std::move(params), executionSettings, false, true);
}
