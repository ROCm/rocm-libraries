// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../TestPlanCommon.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormBwdPlan.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormBwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(LayernormBwdParams(*attrs, graph.getTensorMap()));
}

TEST(TestLayernormBwdParams, HasCorrectTensorPointersForSingleNode)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormBwdParams params(*attrs, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_NE(params.epsilonValue(nullptr, 0), std::nullopt);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0).value(), 1e-5, 1e-10);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
}

TEST(TestLayernormBwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormBwdParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attrs->dy_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attrs->x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrs->scale_tensor_uid());
    EXPECT_EQ(params.mean()->uid(), attrs->mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attrs->inv_variance_tensor_uid());
    EXPECT_NE(params.epsilonValue(nullptr, 0), std::nullopt);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0).value(), 1e-5, 1e-10);
    EXPECT_EQ(params.dx()->uid(), attrs->dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attrs->dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attrs->dbias_tensor_uid());
}

TEST(TestLayernormBwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormBwdParams params(attr, graph.getTensorMap());
    const LayernormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.dx(), nullptr);
}

TEST(TestLayernormBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormBwdParams>);
}

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, LayernormBwdPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                        bool optionalTensors = true)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        strides, dims, optionalTensors, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormBwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), LayernormBwdPlan{std::move(params)}};
}

} // namespace

TEST(TestLayernormBwdPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormBwdPlan, GetWorkspaceSizeReturnsZero)
{
    SKIP_IF_NO_DEVICES(); // getWorkspaceSize requires a device

    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestLayernormBwdPlan, GetWorkspaceSizeParallelReturnsNonzero)
{
    SKIP_IF_NO_DEVICES(); // getWorkspaceSize requires a device

    const Handle handle;
    // The answer is dependent on the number of multiprocessors
    auto multiProcessorCount = static_cast<int64_t>(
        hip_kernel_provider_common::getDeviceProperties(handle.getStream()).multiProcessorCount);
    auto [fbb, plan]
        = createPlanFromGraph({3072 * multiProcessorCount, 1024 * multiProcessorCount, 32, 1},
                              {4, 3, 32 * multiProcessorCount, 32});

    // The answer is `sizeof(float) * 2 * innerSize * parallelSize` = `4 * 2 * (32 * 32 * 3 * multiProcessorCount) * 2` = `49152 * multiProcessorCount`
    EXPECT_EQ(plan.getWorkspaceSize(handle), 49152u * multiProcessorCount);
}

TEST(TestLayernormBwdPlan, GetWorkspaceSizeWithoutOptionalTensorsReturnsNonzero)
{
    SKIP_IF_NO_DEVICES(); // getWorkspaceSize requires a device

    auto [fbb, plan] = createPlanFromGraph({150528, 50176, 224, 1},
                                           {2, 3, 224, 224},
                                           hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                           false);
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 16u);
}

TEST(TestLayernormBwdPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();

    const LayernormBwdPlan moved(std::move(plan));
    const Handle handle;

    SKIP_IF_NO_DEVICES(); // getWorkspaceSize requires a device

    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestLayernormBwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormBwdPlan>);
}

TEST(TestLayernormBwdPlan, CompileCallsCompilerWithCorrectKernelName)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockKernelScaleBias = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockKernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("LayernormBwd"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));
    EXPECT_CALL(*mockProgram, getKernel("LayernormBwdScaleBias"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernelScaleBias))));

    EXPECT_CALL(mockCompiler, compile("LayernormBwd.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestLayernormBwdPlan, CompileIncludesOffloadArchOption)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockKernelScaleBias = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockKernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("LayernormBwd"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));
    EXPECT_CALL(*mockProgram, getKernel("LayernormBwdScaleBias"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernelScaleBias))));

    EXPECT_CALL(mockCompiler,
                compile(::testing::_, ::testing::Contains(std::string("--offload-arch=gfx942"))))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps("gfx942");

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestLayernormBwdPlanFp32, CompileSetsCorrectDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBias = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBias"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBias))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdPlanFp32, CompileSetsCorrectDefinesParallel)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBiasParallel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto kernelScaleBiasReduceSum = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasParallel"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBiasParallel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasReduceSum"))
                .WillOnce(
                    ::testing::Return(::testing::ByMove(std::move(kernelScaleBiasReduceSum))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph({768, 256, 16, 1}, {1024, 3, 16, 16});
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdPlanFp16, CompileSetsCorrectDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBias = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBias"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBias))));
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdPlanFp16, CompileSetsCorrectDefinesParallel)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBiasParallel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto kernelScaleBiasReduceSum = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasParallel"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBiasParallel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasReduceSum"))
                .WillOnce(
                    ::testing::Return(::testing::ByMove(std::move(kernelScaleBiasReduceSum))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph(
        {768, 256, 16, 1}, {1024, 3, 16, 16}, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdPlanBfp16, CompileSetsCorrectDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBias = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBias"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBias))));
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=0"));
}

TEST(TestLayernormBwdPlanBfp16, CompileSetsCorrectDefinesParallel)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto kernelScaleBiasParallel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasParallel,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto kernelScaleBiasReduceSum = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*kernelScaleBiasReduceSum,
                        setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("LayernormBwd"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasParallel"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernelScaleBiasParallel))));
            EXPECT_CALL(*program, getKernel("LayernormBwdScaleBiasReduceSum"))
                .WillOnce(
                    ::testing::Return(::testing::ByMove(std::move(kernelScaleBiasReduceSum))));
            return program;
        });

    auto [fbb, plan]
        = createPlanFromGraph({768, 256, 16, 1},
                              {1024, 3, 16, 16},
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYERNORM_PARALLEL_SIZE=32"));
}

TEST(TestLayernormBwdPlan, CompileWithUnsupportedDimensionThrows)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        {12, 4, 1}, {1, 3, 4}, true, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormBwdParams params(attr, graph.getTensorMap());
    LayernormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}
