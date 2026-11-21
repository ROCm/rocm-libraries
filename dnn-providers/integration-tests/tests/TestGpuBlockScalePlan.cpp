// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <tuple>

#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuBlockScalePlan.hpp"

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_test_sdk::utilities::createValidBlockScaleDequantizeGraph;
using hipdnn_test_sdk::utilities::createValidBlockScaleQuantizeGraph;

namespace
{
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::Graph;
using hipdnn_flatbuffers_sdk::data_objects::GraphT;
using hipdnn_flatbuffers_sdk::data_objects::UnPackGraph;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::CpuFpReferenceBlockScaleDequantize;
using hipdnn_test_sdk::utilities::CpuFpReferenceBlockScaleQuantize;
using hipdnn_test_sdk::utilities::nativeTypeToDataType;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

std::unique_ptr<GraphT> smallBlockScaleGraph(bool quantize)
{
    auto builder
        = quantize ? createValidBlockScaleQuantizeGraph() : createValidBlockScaleDequantizeGraph();
    auto graph = UnPackGraph(builder.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        tensor->dims = {2, 3, 5};
        tensor->strides = {15, 5, 1};
    }
    if(quantize)
    {
        auto* attributes = graph->nodes[0]->attributes.AsBlockScaleQuantizeAttributes();
        attributes->block_size = 2;
        graph->tensors[2]->dims = {2, 2, 5};
        graph->tensors[2]->strides = {10, 5, 1};
    }
    else
    {
        auto* attributes = graph->nodes[0]->attributes.AsBlockScaleDequantizeAttributes();
        attributes->block_size = {2, 2};
        graph->tensors[1]->dims = {2, 2, 3};
        graph->tensors[1]->strides = {6, 3, 1};
    }
    return graph;
}

flatbuffers::FlatBufferBuilder packGraph(const GraphT& graph)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

bool builderIsApplicable(const GraphWrapper& graph, bool quantize)
{
    if(quantize)
    {
        const GpuBlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT>
            builder;
        return builder.isApplicable(graph.getNode(0), graph.getTensorMap());
    }
    const GpuBlockScaleDequantizePlanBuilder<DataType::FLOAT,
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             DataType::FLOAT>
        builder;
    return builder.isApplicable(graph.getNode(0), graph.getTensorMap());
}

template <typename T>
void expectMatchingTensors(Tensor<T>& actual, Tensor<T>& expected)
{
    actual.memory().hostData();
    for(int64_t batch = 0; batch < actual.dims()[0]; ++batch)
    {
        for(int64_t row = 0; row < actual.dims()[1]; ++row)
        {
            for(int64_t column = 0; column < actual.dims()[2]; ++column)
            {
                EXPECT_FLOAT_EQ(static_cast<float>(actual.getHostValue(batch, row, column)),
                                static_cast<float>(expected.getHostValue(batch, row, column)));
            }
        }
    }
}

template <typename RealType = float, typename QuantizedType = float, typename ScaleType = float>
void executeQuantizeGraph(std::optional<int64_t> axis)
{
    auto graph = smallBlockScaleGraph(true);
    auto* attributes = graph->nodes[0]->attributes.AsBlockScaleQuantizeAttributes();
    attributes->axis = axis;
    const auto targetAxis = static_cast<size_t>(axis.value_or(2));
    auto& scaleAttr = *graph->tensors[2];
    scaleAttr.dims = graph->tensors[0]->dims;
    scaleAttr.dims[targetAxis] = (scaleAttr.dims[targetAxis] + 1) / 2;
    scaleAttr.strides = hipdnn_data_sdk::utilities::generateStrides(scaleAttr.dims);
    graph->tensors[0]->data_type = nativeTypeToDataType<RealType>();
    graph->tensors[1]->data_type = nativeTypeToDataType<QuantizedType>();
    graph->tensors[1]->strides = {1, 10, 2};
    scaleAttr.data_type = nativeTypeToDataType<ScaleType>();
    auto builder = packGraph(*graph);
    GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(gpuExecutor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
    Tensor<RealType> x({2, 3, 5});
    Tensor<QuantizedType> gpuY({2, 3, 5}, {1, 10, 2});
    Tensor<QuantizedType> cpuY({2, 3, 5}, {1, 10, 2});
    Tensor<ScaleType> gpuScale(scaleAttr.dims, scaleAttr.strides);
    Tensor<ScaleType> cpuScale(scaleAttr.dims, scaleAttr.strides);
    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t row = 0; row < 3; ++row)
        {
            for(int64_t column = 0; column < 5; ++column)
            {
                x.setHostValue(
                    static_cast<RealType>(static_cast<float>(batch * 15 + row * 5 + column - 12)),
                    batch,
                    row,
                    column);
            }
        }
    }
    gpuExecutor.execute(
        builder.GetBufferPointer(),
        builder.GetSize(),
        {{1, x.rawDeviceData()}, {2, gpuY.rawDeviceData()}, {3, gpuScale.rawDeviceData()}});
    gpuY.markDeviceModified();
    gpuScale.markDeviceModified();
    CpuFpReferenceBlockScaleQuantize::quantize(x, cpuY, cpuScale, 2, axis);
    expectMatchingTensors(gpuY, cpuY);
    expectMatchingTensors(gpuScale, cpuScale);
}

template <typename XType = float, typename ScaleType = float, typename YType = float>
void executeDequantizeGraph(bool isNegativeScale)
{
    auto graph = smallBlockScaleGraph(false);
    graph->nodes[0]->attributes.AsBlockScaleDequantizeAttributes()->is_negative_scale
        = isNegativeScale;
    graph->tensors[0]->data_type = nativeTypeToDataType<XType>();
    graph->tensors[1]->data_type = nativeTypeToDataType<ScaleType>();
    graph->tensors[2]->data_type = nativeTypeToDataType<YType>();
    graph->tensors[2]->strides = {1, 10, 2};
    auto builder = packGraph(*graph);
    GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(gpuExecutor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
    Tensor<XType> x({2, 3, 5});
    Tensor<ScaleType> scale({2, 2, 3});
    Tensor<YType> gpuY({2, 3, 5}, {1, 10, 2});
    Tensor<YType> cpuY({2, 3, 5}, {1, 10, 2});
    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t row = 0; row < 3; ++row)
        {
            for(int64_t column = 0; column < 5; ++column)
            {
                x.setHostValue(
                    static_cast<XType>(static_cast<float>(column - row)), batch, row, column);
            }
        }
        for(int64_t row = 0; row < 2; ++row)
        {
            for(int64_t column = 0; column < 3; ++column)
            {
                scale.setHostValue(static_cast<ScaleType>(static_cast<float>(row + column + 1)),
                                   batch,
                                   row,
                                   column);
            }
        }
    }
    gpuExecutor.execute(
        builder.GetBufferPointer(),
        builder.GetSize(),
        {{1, x.rawDeviceData()}, {2, scale.rawDeviceData()}, {3, gpuY.rawDeviceData()}});
    gpuY.markDeviceModified();
    CpuFpReferenceBlockScaleDequantize::dequantize(x, scale, cpuY, {2, 2}, isNegativeScale);
    expectMatchingTensors(gpuY, cpuY);
}

enum class InvalidMetadata
{
    ZERO_BLOCK_SIZE,
    NEGATIVE_BLOCK_SIZE,
    IO_DIMS_MISMATCH,
    SCALE_RANK_MISMATCH,
    SCALE_DIMS_MISMATCH,
    NON_FLOAT_COMPUTE,
    RANK_ABOVE_LIMIT
};
} // namespace

class TestGpuBlockScaleInvalidMetadata
    : public ::testing::TestWithParam<std::tuple<bool, InvalidMetadata>>
{
};

TEST_P(TestGpuBlockScaleInvalidMetadata, IsNotApplicable)
{
    const auto [quantize, invalid] = GetParam();
    auto graph = smallBlockScaleGraph(quantize);
    auto builder = packGraph(*graph);
    ASSERT_TRUE(
        builderIsApplicable(GraphWrapper(builder.GetBufferPointer(), builder.GetSize()), quantize));
    const size_t scaleIndex = quantize ? 2 : 1;
    const size_t yIndex = quantize ? 1 : 2;
    switch(invalid)
    {
    case InvalidMetadata::ZERO_BLOCK_SIZE:
    case InvalidMetadata::NEGATIVE_BLOCK_SIZE:
    {
        const auto blockSize = invalid == InvalidMetadata::ZERO_BLOCK_SIZE ? 0 : -1;
        if(quantize)
        {
            graph->nodes[0]->attributes.AsBlockScaleQuantizeAttributes()->block_size = blockSize;
        }
        else
        {
            graph->nodes[0]->attributes.AsBlockScaleDequantizeAttributes()->block_size
                = {blockSize, 2};
        }
        break;
    }
    case InvalidMetadata::IO_DIMS_MISMATCH:
        --graph->tensors[yIndex]->dims.back();
        break;
    case InvalidMetadata::SCALE_RANK_MISMATCH:
        graph->tensors[scaleIndex]->dims.pop_back();
        graph->tensors[scaleIndex]->strides.pop_back();
        break;
    case InvalidMetadata::SCALE_DIMS_MISMATCH:
        ++graph->tensors[scaleIndex]->dims.back();
        break;
    case InvalidMetadata::NON_FLOAT_COMPUTE:
        graph->nodes[0]->compute_data_type = DataType::DOUBLE;
        break;
    case InvalidMetadata::RANK_ABOVE_LIMIT:
        for(auto& tensor : graph->tensors)
        {
            tensor->dims.insert(tensor->dims.begin(), 3, 1);
            tensor->strides.insert(tensor->strides.begin(), 3, 30);
        }
        break;
    default:
        FAIL() << "Unknown invalid-metadata case";
    }
    builder = packGraph(*graph);
    const GraphWrapper wrapper(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(builderIsApplicable(wrapper, quantize));
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

INSTANTIATE_TEST_SUITE_P(InvalidArguments,
                         TestGpuBlockScaleInvalidMetadata,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(InvalidMetadata::ZERO_BLOCK_SIZE,
                                                              InvalidMetadata::NEGATIVE_BLOCK_SIZE,
                                                              InvalidMetadata::IO_DIMS_MISMATCH,
                                                              InvalidMetadata::SCALE_RANK_MISMATCH,
                                                              InvalidMetadata::SCALE_DIMS_MISMATCH,
                                                              InvalidMetadata::NON_FLOAT_COMPUTE,
                                                              InvalidMetadata::RANK_ABOVE_LIMIT)));

TEST(TestGpuBlockScalePlan, InvalidQuantizeAxisIsNotApplicable)
{
    for(const int64_t axis : {-1, 3})
    {
        auto graph = smallBlockScaleGraph(true);
        graph->nodes[0]->attributes.AsBlockScaleQuantizeAttributes()->axis = axis;
        auto builder = packGraph(*graph);
        EXPECT_FALSE(
            builderIsApplicable(GraphWrapper(builder.GetBufferPointer(), builder.GetSize()), true));
    }
}

TEST(TestGpuBlockScalePlan, MissingOrOversizedDequantizeBlockSizeIsNotApplicable)
{
    for(const auto& blockSize : std::vector<std::vector<int32_t>>{{}, {1, 1, 2, 2}})
    {
        auto graph = smallBlockScaleGraph(false);
        graph->nodes[0]->attributes.AsBlockScaleDequantizeAttributes()->block_size = blockSize;
        auto builder = packGraph(*graph);
        EXPECT_FALSE(builderIsApplicable(
            GraphWrapper(builder.GetBufferPointer(), builder.GetSize()), false));
    }
}

TEST(TestGpuBlockScalePlan, RuntimePassByValueTensorIsNotApplicable)
{
    for(const auto quantize : {true, false})
    {
        for(size_t tensorIndex = 0; tensorIndex < 3; ++tensorIndex)
        {
            SCOPED_TRACE(::testing::Message()
                         << "quantize=" << quantize << " tensorIndex=" << tensorIndex);
            auto graph = smallBlockScaleGraph(quantize);
            auto builder = packGraph(*graph);
            ASSERT_TRUE(builderIsApplicable(
                GraphWrapper(builder.GetBufferPointer(), builder.GetSize()), quantize));
            graph->tensors[tensorIndex]->is_runtime_pass_by_value = true;
            builder = packGraph(*graph);
            EXPECT_FALSE(builderIsApplicable(
                GraphWrapper(builder.GetBufferPointer(), builder.GetSize()), quantize));
            GpuReferenceGraphExecutor executor;
            EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
        }
    }
}

TEST(TestGpuBlockScalePlan, ExecuteQuantizeNonLastAxisMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    executeQuantizeGraph(1);
}

TEST(TestGpuBlockScalePlan, ExecuteQuantizeWithoutAxisMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    executeQuantizeGraph(std::nullopt);
}

TEST(TestGpuBlockScalePlan, ExecuteDequantizeNegativeScaleMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    executeDequantizeGraph(true);
}

TEST(TestGpuBlockScalePlanFp16, ExecuteCompactMxGraphsMatchCpu)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_data_sdk::types;
    executeQuantizeGraph<half, fp4_e2m1, fp8_e8m0>(1);
    executeDequantizeGraph<fp4_e2m1, fp8_e8m0, half>(false);
}

TEST(TestGpuBlockScalePlanFp16, ExecuteGraphsMatchCpu)
{
    SKIP_IF_NO_DEVICES();
    using hipdnn_data_sdk::types::half;
    executeQuantizeGraph<half, half>(1);
    executeDequantizeGraph<half, float, half>(false);
}

TEST(TestGpuBlockScalePlanBfp16, ExecuteGraphsMatchCpu)
{
    SKIP_IF_NO_DEVICES();
    using hipdnn_data_sdk::types::bfloat16;
    executeQuantizeGraph<bfloat16, bfloat16>(1);
    executeDequantizeGraph<bfloat16, float, bfloat16>(false);
}

TEST(TestGpuBlockScalePlan, QuantizeFloatGraphIsApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph();
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, DequantizeFloatGraphIsApplicable)
{
    auto graph = createValidBlockScaleDequantizeGraph();
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, CompactMxGraphIsApplicable)
{
    auto graph = createValidBlockScaleDequantizeGraph({65536, 2048, 64, 1},
                                                      {2, 32, 32, 64},
                                                      DataType::FP4_E2M1,
                                                      DataType::FLOAT,
                                                      false,
                                                      DataType::FP8_E8M0,
                                                      DataType::FLOAT);
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, InvalidScaleDimensionsAreNotApplicable)
{
    auto graph
        = createValidBlockScaleQuantizeGraph({2, 64, 32, 32}, {65536, 1024, 32, 1}, {2, 3, 32, 32});
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, RankAboveLimitIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({1, 2, 3, 4, 5, 6},
                                                    {720, 360, 120, 30, 6, 1},
                                                    {1, 1, 3, 4, 5, 6},
                                                    {360, 360, 120, 30, 6, 1});
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, UnsupportedCompactTypeCombinationIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                    {65536, 1024, 32, 1},
                                                    {2, 2, 32, 32},
                                                    {2048, 1024, 32, 1},
                                                    32,
                                                    DataType::FLOAT,
                                                    DataType::FP4_E2M1,
                                                    DataType::FLOAT);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, NonFloatComputeIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                    {65536, 1024, 32, 1},
                                                    {2, 2, 32, 32},
                                                    {2048, 1024, 32, 1},
                                                    32,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::DOUBLE);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, RaggedTensorIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph();
    auto ragged
        = hipdnn_integration_tests::test_utils::markFirstTensorRagged(graph.GetBufferPointer());
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(ragged.data(), ragged.size()));
}
