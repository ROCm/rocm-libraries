// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <type_traits>

#include <gtest/gtest.h>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleFwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include "../TestPlanCommon.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

flatbuffers::FlatBufferBuilder createCustomResampleFwdGraph(
    const std::vector<int64_t>& xDims,
    const std::vector<int64_t>& xStrides,
    const std::vector<int64_t>& yDims,
    const std::vector<int64_t>& yStrides,
    const std::vector<int64_t>& prePadding,
    const std::vector<int64_t>& postPadding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& window,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> indexDataType = std::nullopt,
    std::optional<bool> generateIndex = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));
    if(indexDataType.has_value())
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 3, "index", *indexDataType, &yStrides, &yDims));
    }

    ::flatbuffers::Optional<bool> flatbufferGenerateIndex = ::flatbuffers::nullopt;
    if(generateIndex.has_value())
    {
        const bool generateIndexValue = generateIndex.value_or(false);
        flatbufferGenerateIndex = ::flatbuffers::Optional<bool>(generateIndexValue);
    }

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        indexDataType.has_value() ? ::flatbuffers::Optional<int64_t>(3) : ::flatbuffers::nullopt,
        &prePadding,
        &postPadding,
        &stride,
        &window,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD,
        flatbufferGenerateIndex);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes,
        resampleAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdPlan>
    createPlanFromGraph(std::optional<bool> generateIndex = std::nullopt)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(generateIndex);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), ResampleFwdPlan{std::move(params)}};
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdPlan>
    createPlanFromCustomGraph(flatbuffers::FlatBufferBuilder&& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), ResampleFwdPlan{std::move(params)}};
}

} // namespace

TEST(TestResampleFwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    EXPECT_NO_THROW(
        const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type()));
}

TEST(TestResampleFwdParams, CapturesTensorPointersAndAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_EQ(params.index(), nullptr);
    EXPECT_EQ(params.prePadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.postPadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.stride(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.window(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.mode(), hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL);
    EXPECT_EQ(params.paddingMode(), hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);
}

TEST(TestResampleFwdParams, CapturesOptionalIndexTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    ASSERT_NE(params.index(), nullptr);
    EXPECT_EQ(params.index()->uid(), attr.index_tensor_uid().value());
    EXPECT_TRUE(params.generateIndex());
}

TEST(TestResampleFwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    const ResampleFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestResampleFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleFwdParams>);
}

TEST(TestResampleFwdPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleFwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestResampleFwdPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();
    const ResampleFwdPlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestResampleFwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleFwdPlan>);
}

TEST(TestResampleFwdPlan, CompileCallsCompilerWithCorrectKernelName)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("ResampleFwd"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

    EXPECT_CALL(mockCompiler, compile("ResampleFwd.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestResampleFwdPlan, CompileSetsExpectedDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(::testing::_))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& option) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), option)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_SPATIAL_DIMS=2"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_MODE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_PADDING_MODE=2"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT=4"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_WINDOW_H=2"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_WINDOW_W=2"));
}

TEST(TestResampleFwdPlan, CompileSetsIndexDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(::testing::_))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph(true);
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& option) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), option)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_INDEX_TYPE=int32_t"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_HAS_INDEX=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_GENERATE_INDEX=1"));
}

TEST(TestResampleFwdPlan, CompileSetsChannelLastStrideDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(::testing::_))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            return program;
        });

    auto [fbb, plan] = createPlanFromCustomGraph(createCustomResampleFwdGraph(
        {1, 3, 4, 4}, {48, 1, 12, 3}, {1, 3, 2, 2}, {12, 1, 6, 3}, {0, 0}, {0, 0}, {2, 2}, {2, 2}));
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& option) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), option)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_X_STRIDE_N=48"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_X_STRIDE_C=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_X_STRIDE_H=12"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_X_STRIDE_W=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_N=12"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_C=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_H=6"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_W=3"));
}

TEST(TestResampleFwdPlan, CompileRejectsUnsupportedTensorDimensions)
{
    auto [fbb, plan] = createPlanFromCustomGraph(createCustomResampleFwdGraph(
        {1, 1, 4}, {4, 4, 1}, {1, 1, 2}, {2, 2, 1}, {0}, {0}, {2}, {2}));
    auto deviceProps = createTestDeviceProps();

    const MockKernelCompiler mockCompiler;
    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleFwdPlan, CompileRejectsUnsupportedWorkgroups)
{
    auto [fbb, plan] = createPlanFromCustomGraph(
        createCustomResampleFwdGraph({1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {0, 0},
                                     {0, 0},
                                     {1, 1},
                                     {1, 1}));
    auto deviceProps = createTestDeviceProps();

    const MockKernelCompiler mockCompiler;
    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleFwdPlan, CompileRejectsUnsupportedIndexDataType)
{
    auto [fbb, plan] = createPlanFromCustomGraph(
        createCustomResampleFwdGraph({1, 1, 4, 4},
                                     {16, 16, 4, 1},
                                     {1, 1, 2, 2},
                                     {4, 4, 2, 1},
                                     {0, 0},
                                     {0, 0},
                                     {2, 2},
                                     {2, 2},
                                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                                     true));
    auto deviceProps = createTestDeviceProps();

    const MockKernelCompiler mockCompiler;
    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}
