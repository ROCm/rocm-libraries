// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdio>
#include <type_traits>

#include <gtest/gtest.h>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleFwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

flatbuffers::FlatBufferBuilder createValidResampleFwdGraphWithIndex()
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> xDims = {1, 1, 4, 4};
    const std::vector<int64_t> xStrides = {16, 16, 4, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    const std::vector<int64_t> yStrides = {4, 4, 2, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "index",
        hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        &yStrides,
        &yDims));

    const std::vector<int64_t> prePadding = {0, 0};
    const std::vector<int64_t> postPadding = {0, 0};
    const std::vector<int64_t> stride = {2, 2};
    const std::vector<int64_t> window = {2, 2};

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        ::flatbuffers::Optional<int64_t>(3),
        &prePadding,
        &postPadding,
        &stride,
        &window,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD,
        ::flatbuffers::Optional<bool>(true));

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

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdPlan> createPlanFromGraph()
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), ResampleFwdPlan{std::move(params)}};
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdPlan> createPlanFromGraphWithIndex()
{
    auto builder = createValidResampleFwdGraphWithIndex();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), ResampleFwdPlan{std::move(params)}};
}

hipDeviceProp_t createTestDeviceProps(const char* archName = "gfx942")
{
    hipDeviceProp_t deviceProps = {};
    deviceProps.multiProcessorCount = 60;
    deviceProps.warpSize = 64;
    std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", archName);
    return deviceProps;
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
    auto builder = createValidResampleFwdGraphWithIndex();
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

    auto [fbb, plan] = createPlanFromGraphWithIndex();
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
