// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <utility>

#include <gtest/gtest.h>

#include "engines/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace hip_kernel_provider::batchnorm::test
{

// ============================================================================
// BatchnormFwdTrainingParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdTrainingParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    EXPECT_NO_THROW(BatchnormFwdTrainingParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdTrainingParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.mean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attr.inv_variance_tensor_uid());
}

TEST(TestBatchnormFwdTrainingParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());
    BatchnormFwdTrainingParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, true);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, false);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}

TEST(TestBatchnormFwdTrainingParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdTrainingParams>);
}

// ============================================================================
// BatchnormFwdTrainingPlan - helpers
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, BatchnormFwdTrainingPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        bool withMeanVariance = true)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        strides, dims, withMeanVariance);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormFwdTrainingParams params(attr, graph.getTensorMap());
    return {std::move(builder), BatchnormFwdTrainingPlan{std::move(params)}};
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
// BatchnormFwdTrainingPlan - basic behavior
// ============================================================================

TEST(TestBatchnormFwdTrainingPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    HipKernelHandle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingPlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    HipKernelHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormFwdTrainingPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();

    BatchnormFwdTrainingPlan moved(std::move(plan));
    HipKernelHandle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormFwdTrainingPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdTrainingPlan>);
}

// ============================================================================
// BatchnormFwdTrainingPlan - compile
// ============================================================================

TEST(TestBatchnormFwdTrainingPlan, CompileCallsCompilerWithCorrectSingleKernel)
{
    MockKernelCompiler mockCompiler;
    std::vector<std::string> capturedOptions;

    EXPECT_CALL(mockCompiler, compile("BatchNormFwdTrainSpatial.cpp", ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel("BatchNormFwdTrainSpatial"))
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
}

TEST(TestBatchnormFwdTrainingPlan, CompileCallsCompilerWithCorrectMultipleKernels)
{
    MockKernelCompiler mockCompiler;
    auto deviceProps = createTestDeviceProps();

    auto createKernel = []() {
        auto k = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*k, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*k, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        return k;
    };

    auto mockProgram = std::make_unique<MockCompiledProgram>();

    EXPECT_CALL(*mockProgram, getKernel("BatchNormFwdTrainSpatialMeanVariance"))
        .WillOnce(::testing::Return(::testing::ByMove(createKernel())));

    EXPECT_CALL(*mockProgram, getKernel("BatchNormFwdTrainSpatialFinalMeanVariance"))
        .WillOnce(::testing::Return(::testing::ByMove(createKernel())));

    EXPECT_CALL(*mockProgram, getKernel("BatchNormFwdTrainSpatialNorm"))
        .WillOnce(::testing::Return(::testing::ByMove(createKernel())));

    EXPECT_CALL(mockCompiler, compile("BatchNormFwdTrainSpatial.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph({147456, 4608, 576, 1}, {32, 8, 24, 24}, true);

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestBatchnormFwdTrainingPlan, CompileIncludesOffloadArchOption)
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

TEST(TestBatchnormFwdTrainingPlan, CompileDefaultSetsCorrectDefines)
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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_SAVE_MEAN_VARIANCE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RUNNING_RESULT=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_C=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_HW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NHW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NCHW=150528"));
}

// NOTE: Missing tests for FP16 and BFP16 mix options since createValidBatchnormFwdTrainingGraph
// currently creates graphs only with FP32 input data type. We can add tests for those options
// once we have graph builders that can create graphs with FP16/BFP16 input tensors.

TEST(TestBatchnormFwdTrainingPlan, CompileNchwLayoutSetsCorrectDefine)
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

    // NCHW strides: N=C*H*W, C=H*W, H=W, W=1
    auto [fbb, plan] = createPlanFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYOUT_NHWC=0"));
}

TEST(TestBatchnormFwdTrainingPlan, CompileNhwcLayoutSetsCorrectDefine)
{
    MockKernelCompiler mockCompiler;
    std::vector<std::string> capturedOptions;

    auto createKernel = []() {
        auto k = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*k, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*k, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        return k;
    };

    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(::testing::_))
                .Times(3)
                .WillRepeatedly(
                    ::testing::Invoke([&](const std::string&) { return createKernel(); }));
            return program;
        });

    // NHWC strides: N=H*W*C, H=W*C, W=C, C=1
    auto [fbb, plan] = createPlanFromGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYOUT_NHWC=1"));
}

TEST(TestBatchnormFwdTrainingPlan, HasNoSaveStatsSetsCorrectDefines)
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

    // NCHW strides: N=C*H*W, C=H*W, H=W, W=1
    auto [fbb, plan] = createPlanFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_SAVE_MEAN_VARIANCE=0"));
}

} // namespace hip_kernel_provider::batchnorm::test
