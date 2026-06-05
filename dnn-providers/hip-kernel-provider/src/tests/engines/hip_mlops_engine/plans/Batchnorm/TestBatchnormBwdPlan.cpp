// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormBwdPlan.hpp"

#include "../TestPlanCommon.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace hip_kernel_provider::batchnorm
{

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_NE(params.savedMean(), nullptr);
    EXPECT_NE(params.savedInvVariance(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.bias(), nullptr);
}

TEST(TestBatchnormBwdParams, HandlesMissingOptionalSavedStats)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_FALSE(params.hasSavedStats());
    EXPECT_EQ(params.savedMean(), nullptr);
    EXPECT_EQ(params.savedInvVariance(), nullptr);
}

TEST(TestBatchnormBwdParams, InitializesFusedActivationWithAllTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    ASSERT_EQ(graph.nodeCount(), 3u);

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_EQ(params.dy()->uid(), pointwiseAttrs->in_0_tensor_uid());
}

TEST(TestBatchnormBwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormBwdParams params(*attrs, graph.getTensorMap());
    const BatchnormBwdPlan plan(std::move(params));
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormBwdPlan, FusedModeHasActivationAndBias)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
}

TEST(TestBatchnormBwdPlan, CompileDefaultSetsCorrectDefines)
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

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();
    BatchnormBwdParams params(*bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());
    BatchnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_RUNNING_RESULT=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_C=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_HW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NHW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHW=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NCHW=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N_ELEMENTS=HIP_PLUGIN_BN_N"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MAXN=65"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXN=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXHW=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VECTORIZE=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=3"));
}

} // namespace hip_kernel_provider::batchnorm
