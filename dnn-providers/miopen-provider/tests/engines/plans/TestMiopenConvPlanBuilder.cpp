// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "MiopenApi.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/ConvolutionCommon.hpp"
#include "engines/plans/MiopenConvBwdPlan.hpp"
#include "engines/plans/MiopenConvFwdPlan.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"
#include "engines/plans/MiopenConvWrwPlan.hpp"

using namespace miopen_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

class TestMiopenConvPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenConvPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

class TestGpuMiopenConvPlanBuilder : public TestMiopenConvPlanBuilder
{
protected:
    void SetUp() override
    {
        TestMiopenConvPlanBuilder::SetUp();
        // GTEST_SKIP() in the base only unwinds the base SetUp() frame; without this
        // guard, _handle would still be constructed and miopenCreate() would throw on
        // no-device, turning the intended skip into a fixture failure.
        if(IsSkipped())
        {
            return;
        }
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    void executePlan(const hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>& plan,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        const size_t workspaceSize = plan.getWorkspaceSize(*_handle);
        const hipdnn_data_sdk::utilities::Workspace workspace(workspaceSize);

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensors;

        const auto& tensorMap = graph.getTensorMap();
        for(const auto& [uid, tensorAttrPtr] : tensorMap)
        {
            if(!tensorAttrPtr->virtual_())
            {
                auto dims = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                    tensorAttrPtr->dims());
                auto strides
                    = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                        tensorAttrPtr->strides());

                auto tensor = hipdnn_test_sdk::detail::createTensor(
                    tensorAttrPtr->data_type(), dims, strides);

                deviceBuffers.push_back({tensorAttrPtr->uid(), tensor->rawDeviceData()});
                tensors.push_back(std::move(tensor));
            }
        }

        EXPECT_NO_THROW(plan.execute(*_handle,
                                     deviceBuffers.data(),
                                     static_cast<uint32_t>(deviceBuffers.size()),
                                     workspace.get()));
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

class TestGpuMiopenConvPlanBuilderShapes
    : public TestGpuMiopenConvPlanBuilder,
      public ::testing::WithParamInterface<test_conv_common::ConvTestCase>
{
protected:
    void buildAndExecute(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         std::optional<size_t> workspaceLimit)
    {
        const MockEngineConfig mockEngineConfig;
        HipdnnMiopenSettings settings;
        if(workspaceLimit.has_value())
        {
            settings.setWorkspaceSizeLimit(*workspaceLimit);
        }
        HipdnnMiopenContext ctx;
        ctx.setExecutionSettings(settings);
        _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);
        executePlan(ctx.plan(), graph);
    }
};

static std::vector<test_conv_common::ConvTestCase> getWorkspaceRangeShapes()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();
    return {
        // 1D (NCL) shapes. MIOpen has no 1D convolution, so the provider pads
        // these to 2D internally.
        {{1, 16, 16}, {1, 16, 1}, {0}, {0}, {1}, {1}, seed},
        {{1, 16, 16}, {1, 16, 3}, {1}, {1}, {1}, {1}, seed},
        {{2, 32, 16}, {4, 8, 3}, {1}, {1}, {2}, {2}, seed},
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}, seed},
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, seed},
        {{8, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        {{1, 16, 16, 8}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
    };
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {4, 4, 4, 4},
        {64, 16, 4, 1},
        {4, 4, 1, 1},
        {4, 1, 1, 1},
        {4, 4, 4, 4},
        {64, 16, 4, 1},
        {0, 0},
        {0, 0},
        {1, 1},
        {1, 1},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }
}

// True when MIOpen itself reports a forward solution for the graph. The decline tests below only
// mean something when it does: otherwise MIOpen rejects the graph on its own and the provider's
// check is never exercised.
static bool miopenReportsForwardSolution(const HipdnnMiopenHandle& handle, const IGraph& graph)
{
    try
    {
        const auto& attr
            = graph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();
        const ConvFwdParams params(attr, graph.getTensorMap());

        size_t solutionCount = 0;
        const auto status = miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &solutionCount);
        return status == miopenStatusSuccess && solutionCount != 0;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

// The graph builders give all three tensors one data type; this sets them individually, in the
// order the tensors appear in the graph.
static void setConvTensorTypes(const flatbuffers::FlatBufferBuilder& builder,
                               hipdnn_flatbuffers_sdk::data_objects::DataType firstType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType secondType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType thirdType)
{
    auto* mutableTensors
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer())
              ->mutable_tensors();
    EXPECT_TRUE(mutableTensors->GetMutableObject(0)->mutate_data_type(firstType));
    EXPECT_TRUE(mutableTensors->GetMutableObject(1)->mutate_data_type(secondType));
    EXPECT_TRUE(mutableTensors->GetMutableObject(2)->mutate_data_type(thirdType));
}

// Controls: a uniform packed forward convolution (one data type, x, w and y all channels-first or
// all channels-last) stays applicable.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForUniformPackedForwardTensors)
{
    const std::vector<int64_t> xDims = {2, 4, 8, 8};
    const std::vector<int64_t> wDims = {4, 4, 3, 3};
    const std::vector<int64_t> yDims = {2, 4, 6, 6};

    for(const bool channelsLast : {false, true})
    {
        const auto strides = [channelsLast](const std::vector<int64_t>& dims) {
            return channelsLast
                       ? hipdnn_data_sdk::utilities::generateStrides(
                             dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size()))
                       : hipdnn_data_sdk::utilities::generateStrides(dims);
        };

        for(const auto dataType : {hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16})
        {
            SCOPED_TRACE(std::string(channelsLast ? "channels-last " : "channels-first ")
                         + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));

            auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                               strides(xDims),
                                                                               wDims,
                                                                               strides(wDims),
                                                                               yDims,
                                                                               strides(yDims),
                                                                               {0, 0},
                                                                               {0, 0},
                                                                               {1, 1},
                                                                               {1, 1},
                                                                               dataType);
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
                builder.GetBufferPointer(), builder.GetSize());

            EXPECT_TRUE(_planBuilder.isApplicable(*_handle, graph));
        }
    }
}

// ALMIOPEN-3017: x channels-first, w channels-last, y channels-first (the F15 1D bf16 shape).
// MIOpen reports a solution and Find may pick a solver that computes this wrongly.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedLayoutForward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {8, 16, 16},
        {256, 16, 1},
        {32, 16, 1},
        {16, 1, 16},
        {8, 32, 16},
        {512, 16, 1},
        {0},
        {0},
        {1},
        {1},
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(!miopenReportsForwardSolution(*_handle, graph))
    {
        GTEST_SKIP() << "MIOpen already declines this graph by itself";
    }

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph));
}

// ALMIOPEN-3018: x with padded rows (strides keep the channels-first order, so the layout is
// uniform) next to packed w and y.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForNonPackedForward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph({2, 4, 8, 8},
                                                                       {512, 128, 16, 1},
                                                                       {4, 4, 3, 3},
                                                                       {36, 9, 3, 1},
                                                                       {2, 4, 6, 6},
                                                                       {144, 36, 6, 1},
                                                                       {0, 0},
                                                                       {0, 0},
                                                                       {1, 1},
                                                                       {1, 1});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(!miopenReportsForwardSolution(*_handle, graph))
    {
        GTEST_SKIP() << "MIOpen already declines this graph by itself";
    }

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph));
}

// ALMIOPEN-3015: x and w HALF with y FLOAT, layouts uniform and packed. MIOpen reports a solution
// and then fails at execution or returns garbage.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedDataTypeForward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {2, 4, 8, 8},
        {256, 64, 8, 1},
        {4, 4, 3, 3},
        {36, 9, 3, 1},
        {2, 4, 6, 6},
        {144, 36, 6, 1},
        {0, 0},
        {0, 0},
        {1, 1},
        {1, 1},
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    setConvTensorTypes(builder,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(!miopenReportsForwardSolution(*_handle, graph))
    {
        GTEST_SKIP() << "MIOpen already declines this graph by itself";
    }

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph));
}

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights
// ---------------------------------------------------------------------------------------------

using ConvDataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

enum class BackwardKind
{
    DATA,
    WEIGHTS
};

struct BackwardTensor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

// The tensors of a 2D backward convolution (unit stride, no padding, no dilation) in backward-data
// roles. Backward weights poses the same problem: x has the shape of dx and dw the shape of w.
struct BackwardGraphSpec
{
    BackwardTensor dy;
    BackwardTensor w;
    BackwardTensor dx;
    ConvDataType dataType = ConvDataType::FLOAT;
};

static std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLastLayout)
{
    return channelsLastLayout ? hipdnn_data_sdk::utilities::generateStrides(
                                    dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size()))
                              : hipdnn_data_sdk::utilities::generateStrides(dims);
}

static BackwardGraphSpec uniformPackedSpec(bool channelsLastLayout,
                                           ConvDataType dataType = ConvDataType::FLOAT)
{
    const std::vector<int64_t> dyDims = {2, 4, 6, 6};
    const std::vector<int64_t> wDims = {4, 4, 3, 3};
    const std::vector<int64_t> dxDims = {2, 4, 8, 8};

    return {{dyDims, packedStrides(dyDims, channelsLastLayout)},
            {wDims, packedStrides(wDims, channelsLastLayout)},
            {dxDims, packedStrides(dxDims, channelsLastLayout)},
            dataType};
}

// dy and dx channels-first, w channels-last (the F15 2D bf16 shape).
static BackwardGraphSpec mixedLayoutSpec()
{
    return {{{2, 32, 14, 14}, {6272, 196, 14, 1}},
            {{32, 16, 3, 3}, {144, 1, 48, 16}},
            {{2, 16, 16, 16}, {4096, 256, 16, 1}},
            ConvDataType::BFLOAT16};
}

// All channels-last, dx with padded rows.
static BackwardGraphSpec channelsLastNonPackedSpec()
{
    return {{{2, 4, 6, 6}, {144, 1, 24, 4}},
            {{4, 4, 3, 3}, {36, 1, 12, 4}},
            {{2, 4, 8, 8}, {320, 1, 40, 4}},
            ConvDataType::FLOAT};
}

// All channels-first, dx with padded rows.
static BackwardGraphSpec channelsFirstNonPackedSpec()
{
    return {{{2, 4, 6, 6}, {144, 36, 6, 1}},
            {{4, 4, 3, 3}, {36, 9, 3, 1}},
            {{2, 4, 8, 8}, {512, 128, 16, 1}},
            ConvDataType::FLOAT};
}

// A single-channel 46341 x 46341 image as dy and dx (2,147,488,281 elements) with a 1x1 filter.
static BackwardGraphSpec aboveSizeLimitSpec()
{
    const std::vector<int64_t> imageDims = {1, 1, 46341, 46341};
    const std::vector<int64_t> filterDims = {1, 1, 1, 1};

    return {{imageDims, packedStrides(imageDims, /*channelsLastLayout=*/false)},
            {filterDims, packedStrides(filterDims, /*channelsLastLayout=*/false)},
            {imageDims, packedStrides(imageDims, /*channelsLastLayout=*/false)},
            ConvDataType::HALF};
}

static flatbuffers::FlatBufferBuilder createBackwardGraph(BackwardKind kind,
                                                          const BackwardGraphSpec& spec)
{
    if(kind == BackwardKind::DATA)
    {
        return hipdnn_test_sdk::utilities::createValidConvBwdGraph(spec.dx.dims,
                                                                   spec.dx.strides,
                                                                   spec.w.dims,
                                                                   spec.w.strides,
                                                                   spec.dy.dims,
                                                                   spec.dy.strides,
                                                                   {0, 0},
                                                                   {0, 0},
                                                                   {1, 1},
                                                                   {1, 1},
                                                                   spec.dataType);
    }
    return hipdnn_test_sdk::utilities::createValidConvWrwGraph(spec.dx.dims,
                                                               spec.dx.strides,
                                                               spec.w.dims,
                                                               spec.w.strides,
                                                               spec.dy.dims,
                                                               spec.dy.strides,
                                                               {0, 0},
                                                               {0, 0},
                                                               {1, 1},
                                                               {1, 1},
                                                               spec.dataType);
}

// True when MIOpen itself reports a backward solution for the graph. The decline tests only mean
// something when it does: otherwise MIOpen rejects the graph on its own and the provider's check
// is never exercised.
static bool miopenReportsBackwardSolution(BackwardKind kind,
                                          const HipdnnMiopenHandle& handle,
                                          const IGraph& graph)
{
    try
    {
        size_t solutionCount = 0;
        if(kind == BackwardKind::DATA)
        {
            const auto& attr
                = graph.getNodeWrapper(0)
                      .attributesAs<
                          hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>();
            const ConvBwdParams params(attr, graph.getTensorMap());

            const auto status
                = miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                                params.dy().tensorDescriptor(),
                                                                params.w().tensorDescriptor(),
                                                                params.conv().convDescriptor(),
                                                                params.dx().tensorDescriptor(),
                                                                &solutionCount);
            return status == miopenStatusSuccess && solutionCount != 0;
        }

        const auto& attr
            = graph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>();
        const ConvWrwParams params(attr, graph.getTensorMap());

        const auto status
            = miopenConvolutionBackwardWeightsGetSolutionCount(handle.miopenHandle,
                                                               params.dy().tensorDescriptor(),
                                                               params.x().tensorDescriptor(),
                                                               params.conv().convDescriptor(),
                                                               params.dw().tensorDescriptor(),
                                                               &solutionCount);
        return status == miopenStatusSuccess && solutionCount != 0;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

// The provider must decline a graph MIOpen would otherwise run. Skipped when MIOpen declines it
// by itself.
static void expectBackwardDeclined(const MiopenConvPlanBuilder& planBuilder,
                                   const HipdnnMiopenHandle& handle,
                                   BackwardKind kind,
                                   const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(!miopenReportsBackwardSolution(kind, handle, graph))
    {
        GTEST_SKIP() << "MIOpen already declines this graph by itself";
    }

    EXPECT_FALSE(planBuilder.isApplicable(handle, graph));
}

// The provider must keep a graph MIOpen offers a solution for. Skipped when MIOpen has none, so a
// MIOpen limitation is not reported as a provider failure.
static void expectBackwardApplicable(const MiopenConvPlanBuilder& planBuilder,
                                     const HipdnnMiopenHandle& handle,
                                     BackwardKind kind,
                                     const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(!miopenReportsBackwardSolution(kind, handle, graph))
    {
        GTEST_SKIP() << "MIOpen reports no solution for this graph";
    }

    EXPECT_TRUE(planBuilder.isApplicable(handle, graph));
}

// Controls: a uniform packed backward convolution (one data type, all tensors channels-first or
// all channels-last) stays applicable, for backward data and backward weights.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForUniformPackedBackwardTensors)
{
    for(const auto kind : {BackwardKind::DATA, BackwardKind::WEIGHTS})
    {
        for(const bool channelsLastLayout : {false, true})
        {
            for(const auto dataType :
                {ConvDataType::FLOAT, ConvDataType::HALF, ConvDataType::BFLOAT16})
            {
                SCOPED_TRACE(
                    std::string(kind == BackwardKind::DATA ? "backward data " : "backward weights ")
                    + (channelsLastLayout ? "channels-last " : "channels-first ")
                    + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));

                auto builder
                    = createBackwardGraph(kind, uniformPackedSpec(channelsLastLayout, dataType));
                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
                    builder.GetBufferPointer(), builder.GetSize());

                EXPECT_TRUE(_planBuilder.isApplicable(*_handle, graph));
            }
        }
    }
}

// ALMIOPEN-3299: dy and dx channels-first, w channels-last. MIOpen reports a solution and returns
// wrong output.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedLayoutBackwardData)
{
    expectBackwardDeclined(_planBuilder,
                           *_handle,
                           BackwardKind::DATA,
                           createBackwardGraph(BackwardKind::DATA, mixedLayoutSpec()));
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedLayoutBackwardWeights)
{
    expectBackwardDeclined(_planBuilder,
                           *_handle,
                           BackwardKind::WEIGHTS,
                           createBackwardGraph(BackwardKind::WEIGHTS, mixedLayoutSpec()));
}

// ALMIOPEN-3301: channels-last tensors with padded rows (the layout stays uniform). MIOpen
// returns wrong output, up to 1e38, infinities and NaN.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForNonPackedChannelsLastBackwardData)
{
    expectBackwardDeclined(_planBuilder,
                           *_handle,
                           BackwardKind::DATA,
                           createBackwardGraph(BackwardKind::DATA, channelsLastNonPackedSpec()));
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       IsApplicableReturnsFalseForNonPackedChannelsLastBackwardWeights)
{
    expectBackwardDeclined(_planBuilder,
                           *_handle,
                           BackwardKind::WEIGHTS,
                           createBackwardGraph(BackwardKind::WEIGHTS, channelsLastNonPackedSpec()));
}

// Unlike forward, channels-first tensors with padded rows stay applicable: MIOpen computes them
// correctly.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForNonPackedChannelsFirstBackwardData)
{
    expectBackwardApplicable(_planBuilder,
                             *_handle,
                             BackwardKind::DATA,
                             createBackwardGraph(BackwardKind::DATA, channelsFirstNonPackedSpec()));
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       IsApplicableReturnsTrueForNonPackedChannelsFirstBackwardWeights)
{
    expectBackwardApplicable(
        _planBuilder,
        *_handle,
        BackwardKind::WEIGHTS,
        createBackwardGraph(BackwardKind::WEIGHTS, channelsFirstNonPackedSpec()));
}

// ALMIOPEN-3303: the first two tensors HALF and the last FLOAT, layouts uniform and packed. MIOpen
// reports a solution and then fails at execution or returns garbage. The tensors are in the order
// of the graph: dy, w, dx for backward data and x, dy, dw for backward weights.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedDataTypeBackwardData)
{
    auto builder = createBackwardGraph(
        BackwardKind::DATA, uniformPackedSpec(/*channelsLastLayout=*/false, ConvDataType::HALF));
    setConvTensorTypes(builder, ConvDataType::HALF, ConvDataType::HALF, ConvDataType::FLOAT);

    expectBackwardDeclined(_planBuilder, *_handle, BackwardKind::DATA, builder);
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForMixedDataTypeBackwardWeights)
{
    auto builder = createBackwardGraph(
        BackwardKind::WEIGHTS, uniformPackedSpec(/*channelsLastLayout=*/false, ConvDataType::HALF));
    setConvTensorTypes(builder, ConvDataType::HALF, ConvDataType::HALF, ConvDataType::FLOAT);

    expectBackwardDeclined(_planBuilder, *_handle, BackwardKind::WEIGHTS, builder);
}

// ALMIOPEN-3306: dy and dx of 2,147,488,281 elements. MIOpen offers a solver that faults on it
// during Find. Descriptor creation allocates nothing, so the graph is cheap to build.
TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsFalseForBackwardDataAboveSizeLimit)
{
    expectBackwardDeclined(_planBuilder,
                           *_handle,
                           BackwardKind::DATA,
                           createBackwardGraph(BackwardKind::DATA, aboveSizeLimitSpec()));
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    const HipdnnMiopenSettings settings;
    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(*_dummyHandle, mockGraph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(*_dummyHandle, graph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeReturnsValueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, mockGraph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedComputeType)
{
    const flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    auto mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenConvPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        const MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        const MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        const MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeFwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    const size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    const size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    const size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_P(TestGpuMiopenConvPlanBuilderShapes, WorkspaceRangeIsConsistentAndExecutableFwd)
{
    const auto& tc = GetParam();
    auto xStrides = hipdnn_data_sdk::utilities::generateStrides(tc.xDims);
    auto wStrides = hipdnn_data_sdk::utilities::generateStrides(tc.wDims);
    auto yStrides = hipdnn_data_sdk::utilities::generateStrides(tc.yDims);

    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(tc.xDims,
                                                                       xStrides,
                                                                       tc.wDims,
                                                                       wStrides,
                                                                       tc.yDims,
                                                                       yStrides,
                                                                       tc.convPrePadding,
                                                                       tc.convPostPadding,
                                                                       tc.convStride,
                                                                       tc.convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    EXPECT_LE(range.min, range.max);

    buildAndExecute(graph, std::nullopt);
    buildAndExecute(graph, range.min);
    buildAndExecute(graph, range.max);
}

TEST_P(TestGpuMiopenConvPlanBuilderShapes, WorkspaceRangeIsConsistentAndExecutableBwd)
{
    const auto& tc = GetParam();
    auto dxStrides = hipdnn_data_sdk::utilities::generateStrides(tc.xDims);
    auto wStrides = hipdnn_data_sdk::utilities::generateStrides(tc.wDims);
    auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(tc.yDims);

    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(tc.xDims,
                                                                       dxStrides,
                                                                       tc.wDims,
                                                                       wStrides,
                                                                       tc.yDims,
                                                                       dyStrides,
                                                                       tc.convPrePadding,
                                                                       tc.convPostPadding,
                                                                       tc.convStride,
                                                                       tc.convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    EXPECT_LE(range.min, range.max);

    buildAndExecute(graph, std::nullopt);
    buildAndExecute(graph, range.min);
    buildAndExecute(graph, range.max);
}

TEST_P(TestGpuMiopenConvPlanBuilderShapes, WorkspaceRangeIsConsistentAndExecutableWrw)
{
    const auto& tc = GetParam();
    auto xStrides = hipdnn_data_sdk::utilities::generateStrides(tc.xDims);
    auto dwStrides = hipdnn_data_sdk::utilities::generateStrides(tc.wDims);
    auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(tc.yDims);

    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph(tc.xDims,
                                                                       xStrides,
                                                                       tc.wDims,
                                                                       dwStrides,
                                                                       tc.yDims,
                                                                       dyStrides,
                                                                       tc.convPrePadding,
                                                                       tc.convPostPadding,
                                                                       tc.convStride,
                                                                       tc.convDilation);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    EXPECT_LE(range.min, range.max);

    buildAndExecute(graph, std::nullopt);
    buildAndExecute(graph, range.min);
    buildAndExecute(graph, range.max);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvPlanBuilderShapes,
                         ::testing::ValuesIn(getWorkspaceRangeShapes()));

TEST_F(TestGpuMiopenConvPlanBuilder, GetCustomKnobsReturnsWorkspaceSizeLimitKnob)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto customKnobs = _planBuilder.getCustomKnobs(*_handle, graph);
    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    auto workspaceKnobIt
        = std::find_if(customKnobs.begin(),
                       customKnobs.end(),
                       [](const hipdnn_flatbuffers_sdk::data_objects::KnobT& knob) {
                           return knob.knob_id == hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME;
                       });

    ASSERT_NE(workspaceKnobIt, customKnobs.end());

    const auto& workspaceKnob = *workspaceKnobIt;
    EXPECT_EQ(workspaceKnob.description, "Workspace size limit in bytes");

    // Check default value (should be max)
    ASSERT_TRUE(workspaceKnob.default_value.AsIntValue() != nullptr);
    EXPECT_EQ(workspaceKnob.default_value.AsIntValue()->value, static_cast<int64_t>(range.max));

    // Check constraint
    ASSERT_TRUE(workspaceKnob.constraint.AsIntConstraint() != nullptr);
    const auto& constraint = *workspaceKnob.constraint.AsIntConstraint();
    EXPECT_EQ(constraint.min_value, static_cast<int64_t>(range.min));
    EXPECT_EQ(constraint.max_value, static_cast<int64_t>(range.max));
    EXPECT_EQ(constraint.step, 1);
}

TEST_F(TestMiopenConvPlanBuilder, GetCustomKnobsReturnsEmptyWhenNotApplicable)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(0));

    auto customKnobs = _planBuilder.getCustomKnobs(*_dummyHandle, mockGraph);

    EXPECT_TRUE(customKnobs.empty());
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnInvalidWorkspaceSizeLimitKnobType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto stringValueOffset = configBuilder.CreateString("invalid_value");
    auto knobValue
        = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(configBuilder, stringValueOffset);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsThrowsOnNegativeWorkspaceSizeLimit)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(configBuilder, -1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnWorkspaceSizeLimitBelowRange)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    // Skip if min is 0 (already tested in BuildPlanThrowsOnNegativeWorkspaceSizeLimit)
    if(range.min == 0)
    {
        GTEST_SKIP() << "Skipping below-range test when min is 0";
    }

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.min) - 1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnWorkspaceSizeLimitAboveRange)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    if(range.max >= std::numeric_limits<int64_t>::max())
    {
        GTEST_SKIP() << "Skipping above-range test when max >= INT64_MAX";
    }

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.max) + 1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsWorkspaceSizeLimit)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    const auto testWorkspaceSize = range.min + ((range.max - range.min) / 2);

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(testWorkspaceSize));
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(
        *_handle, graph, configWrapper, executionSettings));

    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    ctx.setExecutionSettings(executionSettings);

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
    auto workspaceLimit = executionSettings.workspaceSizeLimit();
    ASSERT_TRUE(workspaceLimit.has_value());
    if(workspaceLimit.has_value())
    {
        EXPECT_EQ(*workspaceLimit, testWorkspaceSize);
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsDefaultsWorkspaceSizeLimitWhenNoKnob)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(
        *_handle, graph, configWrapper, executionSettings));

    const MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    ctx.setExecutionSettings(executionSettings);

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
    EXPECT_FALSE(executionSettings.workspaceSizeLimit().has_value());
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSize)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    const HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetDefaultWhenLimitIsSet)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.max));
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    EXPECT_TRUE(settings.defaultWorkspaceSize().has_value());
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefault)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}

// fp32 N64 C256 56x56, 3x3 forward. On gfx90a the default query reports a workspace far below the
// range max for this problem. These tests only query MIOpen: no device memory and no Find.
static flatbuffers::FlatBufferBuilder createLargeConvFwdGraph()
{
    return hipdnn_test_sdk::utilities::createValidConvFwdGraph({64, 256, 56, 56},
                                                               {802816, 3136, 56, 1},
                                                               {256, 256, 3, 3},
                                                               {2304, 9, 3, 1},
                                                               {64, 256, 56, 56},
                                                               {802816, 3136, 56, 1},
                                                               {1, 1},
                                                               {1, 1},
                                                               {1, 1},
                                                               {1, 1});
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeIsRangeMaxWhenBenchmarking)
{
    auto builder = createLargeConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(_planBuilder.getMaxWorkspaceSize(*_handle, graph, HipdnnMiopenSettings{})
       == _planBuilder.getWorkspaceSizeRange(*_handle, graph).max)
    {
        GTEST_SKIP() << "The default workspace already equals the range max here, so there is "
                        "nothing for benchmarking to change";
    }

    HipdnnMiopenSettings settings;
    settings.setBenchmarkingEnabled(true);

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings),
              _planBuilder.getWorkspaceSizeRange(*_handle, graph).max);
}

// Without benchmarking the builder keeps reporting what MIOpen reports for the problem.
TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeWithoutBenchmarkingIsTheMiopenDefault)
{
    auto builder = createLargeConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& attributes
        = graph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();
    const ConvFwdParams params(attributes, graph.getTensorMap(), /*deterministicEnabled=*/false);
    size_t expected = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(_handle->miopenHandle,
                                                       params.w().tensorDescriptor(),
                                                       params.x().tensorDescriptor(),
                                                       params.conv().convDescriptor(),
                                                       params.y().tensorDescriptor(),
                                                       &expected),
              miopenStatusSuccess);

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_handle, graph, HipdnnMiopenSettings{}), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeHonorsLimitWhenBenchmarking)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenSettings settings;
    settings.setBenchmarkingEnabled(true);
    settings.setWorkspaceSizeLimit(1);

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings), 1u);
}

TEST_F(TestGpuMiopenConvPlanBuilder, DefaultWorkspaceSizeIsRangeMaxWhenBenchmarking)
{
    auto builder = createLargeConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    if(_planBuilder.getMaxWorkspaceSize(*_handle, graph, HipdnnMiopenSettings{})
       == _planBuilder.getWorkspaceSizeRange(*_handle, graph).max)
    {
        GTEST_SKIP() << "The default workspace already equals the range max here, so there is "
                        "nothing for benchmarking to change";
    }

    flatbuffers::FlatBufferBuilder configBuilder;
    configBuilder.Finish(
        hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0));
    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings settings;
    settings.setBenchmarkingEnabled(true);
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(*settings.defaultWorkspaceSize(),
              _planBuilder.getWorkspaceSizeRange(*_handle, graph).max);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsZeroWhenNothingSet)
{
    const HipdnnMiopenSettings settings;
    EXPECT_EQ(settings.selectedWorkspaceSize(), 0u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsDefaultWhenOnlyDefaultSet)
{
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(1024);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 1024u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsLimitWhenOnlyLimitSet)
{
    HipdnnMiopenSettings settings;
    settings.setWorkspaceSizeLimit(512);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 512u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsLimitWhenBothSet)
{
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(1024);
    settings.setWorkspaceSizeLimit(512);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 512u);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSizeBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    const HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSizeWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
        buffer.data(), buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    const HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefaultBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefaultWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}
