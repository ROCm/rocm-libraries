// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../TestPlanCommon.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormFwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormFwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(LayernormFwdParams(*attrs, graph.getTensorMap()));
}

TEST(TestLayernormFwdParams, HasCorrectTensorPointersForSingleNode)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormFwdParams params(*attrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_EQ(params.mean(), nullptr);
    EXPECT_NE(params.epsilon(), nullptr);
    EXPECT_EQ(params.invVariance(), nullptr);
}

TEST(TestLayernormFwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormFwdParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attrs->x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attrs->y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrs->scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attrs->bias_tensor_uid());
    EXPECT_EQ(params.epsilon()->uid(), attrs->epsilon_tensor_uid());
}

TEST(TestLayernormFwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormFwdParams params(attr, graph.getTensorMap());
    const LayernormFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestLayernormFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormFwdParams>);
}

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, LayernormFwdPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormFwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), LayernormFwdPlan{std::move(params)}};
}

} // namespace

TEST(TestLayernormFwdPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormFwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestLayernormFwdPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();

    const LayernormFwdPlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestLayernormFwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormFwdPlan>);
}

TEST(TestLayernormFwdPlan, CompileCallsCompilerWithCorrectKernelName)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("LayernormFwd"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

    EXPECT_CALL(mockCompiler, compile("LayernormFwd.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestLayernormFwdPlan, CompileIncludesOffloadArchOption)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel(::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

    EXPECT_CALL(mockCompiler,
                compile(::testing::_, ::testing::Contains(std::string("--offload-arch=gfx942"))))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps("gfx942");

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestLayernormFwdPlanFp32, CompileSetsCorrectDefines)
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

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdPlanFp16, CompileSetsCorrectDefines)
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

    auto [fbb, plan] = createPlanFromGraph({150528, 50176, 224, 1},
                                           {1, 3, 224, 224},
                                           hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdPlanBfp16, CompileSetsCorrectDefines)
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

    auto [fbb, plan]
        = createPlanFromGraph({150528, 50176, 224, 1},
                              {1, 3, 224, 224},
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdPlan, CompileWithUnsupportedDimensionThrows)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormFwdParams params(attr, graph.getTensorMap());
    LayernormFwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}
