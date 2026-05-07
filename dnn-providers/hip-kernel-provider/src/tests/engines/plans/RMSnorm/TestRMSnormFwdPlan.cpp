// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <cstdio>
#include <utility>

#include <gtest/gtest.h>

#include "engines/plans/RMSnorm/RMSnormFwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

// ============================================================================
// RMSnormFwdParams - construction from valid graph data
// ============================================================================
TEST(TestRMSnormFwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    EXPECT_NO_THROW(RMSnormFwdParams params(attr, graph.getTensorMap()));
}

TEST(TestRMSnormFwdParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_EQ(params.bias(), nullptr); // Optional and not set in `createValidRMSNormGraph`
    EXPECT_EQ(params.invRMS(), nullptr); // Optional and not set in `createValidRMSNormGraph`
}

TEST(TestRMSnormFwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
}

TEST(TestRMSnormFwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());
    RMSnormFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestRMSnormFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormFwdParams>);
}
// ============================================================================
// RMSnormFwdPlan - helpers
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, RMSnormFwdPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidRMSNormGraph(strides, dims, inputDataType);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), RMSnormFwdPlan{std::move(params)}};
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

// ============================================================================
// RMSnormFwdPlan - basic behavior
// ============================================================================

TEST(TestRMSnormFwdPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    HipKernelHandle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    HipKernelHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormFwdPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();

    RMSnormFwdPlan moved(std::move(plan));
    HipKernelHandle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormFwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormFwdPlan>);
}

// ============================================================================
// RMSnormFwdPlan - compile
// ============================================================================

TEST(TestRMSnormFwdPlan, CompileCallsCompilerWithCorrectKernelName)
{
    MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("RMSnormFwd"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

    EXPECT_CALL(mockCompiler, compile("RMSNormFwd.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestRMSnormFwdPlan, CompileIncludesOffloadArchOption)
{
    MockKernelCompiler mockCompiler;

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

TEST(TestRMSnormFwdPlan, CompileFp32SetsCorrectDefines)
{
    MockKernelCompiler mockCompiler;

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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
}

TEST(TestRMSnormFwdPlan, CompileFp16SetsCorrectDefines)
{
    MockKernelCompiler mockCompiler;

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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
}

TEST(TestRMSnormFwdPlan, CompileBfp16SetsCorrectDefines)
{
    MockKernelCompiler mockCompiler;

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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=ushort"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=ushort"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
}

TEST(TestRMSnormFwdPlan, CompileWithUnsupportedDimensionThrows)
{
    MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());
    RMSnormFwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdPlan, CompileWithUnsupportedWorkgroupsThrows)
{
    MockKernelCompiler mockCompiler;

    // Number of workgroups exeeds UINT32_MAX
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());
    RMSnormFwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}
