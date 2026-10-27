// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <utility>

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdInferencePlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include "../TestPlanCommon.hpp"
#include "BatchnormAlignmentTestUtils.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace hip_kernel_provider::batchnorm::test
{

// ============================================================================
// BatchnormFwdInferenceParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdInferenceParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    EXPECT_NO_THROW(const BatchnormFwdInferenceParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const BatchnormFwdInferenceParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);

    // No activation in this graph, so these should be nullopt / nullptr
    EXPECT_EQ(params.optActivation(), std::nullopt);
    EXPECT_EQ(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdInferenceParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const BatchnormFwdInferenceParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.estMean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attr.inv_variance_tensor_uid());
}

TEST(TestBatchnormFwdInferenceParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());
    const BatchnormFwdInferenceParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestBatchnormFwdInferenceParams, ConstructGraphWithActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    EXPECT_NO_THROW(
        const BatchnormFwdInferenceParams params(attr, activAttrs, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceParams, HasCorrectTensorPointersForGraphWithActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    const BatchnormFwdInferenceParams params(attr, activAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_NE(params.optActivation(), std::nullopt);
    EXPECT_NE(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdInferenceParams, TensorPointersMatchExpectedUidsForGraphWithActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    const BatchnormFwdInferenceParams params(attr, activAttrs, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.estMean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), attr.inv_variance_tensor_uid());
    EXPECT_EQ(params.activationOut()->uid(), activAttrs.out_0_tensor_uid());
}

TEST(TestBatchnormFwdInferenceParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdInferenceParams>);
}

// ============================================================================
// BatchnormFwdInferencePlan - helpers
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, BatchnormFwdInferencePlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());
    return {std::move(builder), BatchnormFwdInferencePlan{std::move(params)}};
}

} // namespace

// ============================================================================
// BatchnormFwdInferencePlan - basic behavior
// ============================================================================

TEST(TestBatchnormFwdInferencePlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdInferencePlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormFwdInferencePlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();

    const BatchnormFwdInferencePlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormFwdInferencePlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdInferencePlan>);
}

// ============================================================================
// BatchnormFwdInferencePlan - compile
// ============================================================================

TEST(TestBatchnormFwdInferencePlan, CompileCallsCompilerWithCorrectKernelName)
{
    const MockKernelCompiler mockCompiler;

    auto mockKernel = std::make_unique<MockRunnableKernel>();
    EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

    auto mockProgram = std::make_unique<MockCompiledProgram>();
    EXPECT_CALL(*mockProgram, getKernel("BatchNormFwdInferSpatialEstInvVar"))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

    EXPECT_CALL(mockCompiler, compile("BatchNormFwdInferSpatial.cpp", ::testing::_))
        .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);
}

TEST(TestBatchnormFwdInferencePlan, CompileIncludesOffloadArchOption)
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

TEST(TestBatchnormFwdInferencePlanFp32, CompileDefaultSetsCorrectDefines)
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_RUNNING_RESULT=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_C=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_HW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NCHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=256"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N_ELEMENTS=HIP_PLUGIN_BN_N"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXN=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=256"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXHW=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=255"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=0"));
}

TEST(TestBatchnormFwdInferencePlanFp32, CompileSetsCorrectDefines)
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
}

TEST(TestBatchnormFwdInferencePlanFp16, CompileSetsCorrectDefines)
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=1"));
}

TEST(TestBatchnormFwdInferencePlanBfp16, CompileSetsCorrectDefines)
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

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=1"));
}

TEST(TestBatchnormFwdInferencePlan, CompileNchwLayoutSetsCorrectDefine)
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

TEST(TestBatchnormFwdInferencePlan, CompileNhwcLayoutSetsCorrectDefine)
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

TEST(TestBatchnormFwdInferencePlan, CompileWithUnsupportedDimensionThrows)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());
    BatchnormFwdInferencePlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// BatchnormFwdInferencePlan - declared tensor alignment
// ============================================================================

namespace
{

// Compiles the plan of an inference graph whose named tensors declare the given alignments, and
// returns the vector size it compiled with.
std::string compiledVectorSize(const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dims,
                               const AlignmentByName& alignments)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(strides, dims);
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), alignments);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(patched.data(),
                                                                           patched.size());
    const auto& attr = *graph.getNode(0).attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferencePlan plan{BatchnormFwdInferenceParams(attr, graph.getTensorMap())};
    const auto options = compileAndCaptureOptions(
        [&](const IKernelCompiler& compiler) { plan.compile(compiler, createTestDeviceProps()); });

    const std::string prefix = "-DHIP_PLUGIN_BN_VEC_SIZE=";
    for(const auto& option : options)
    {
        if(option.rfind(prefix, 0) == 0)
        {
            return option.substr(prefix.size());
        }
    }
    return {};
}

} // namespace

// Bug caught: the vector size chosen from the shape alone, so x is loaded four elements at a time
// from a pointer whose declared alignment covers only two or one.
TEST(TestBatchnormFwdInferencePlan, CompileNarrowsTheVectorSizeToTheDeclaredInputAlignment)
{
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
    const std::vector<int64_t> dims = {1, 3, 224, 224};

    EXPECT_EQ(compiledVectorSize(strides, dims, {}), "4");
    EXPECT_EQ(compiledVectorSize(strides, dims, {{"x", 8}}), "2");
    EXPECT_EQ(compiledVectorSize(strides, dims, {{"x", 4}}), "1");
}

// Bug caught: the vector size narrowed at exactly the alignment it needs (off by one on the
// boundary), or widened by a declaration that is not a multiple of the 16 bytes it needs.
TEST(TestBatchnormFwdInferencePlan, CompileKeepsTheVectorSizeAtExactlyTheAlignmentItNeeds)
{
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
    const std::vector<int64_t> dims = {1, 3, 224, 224};

    EXPECT_EQ(compiledVectorSize(strides, dims, {{"x", 16}}), "4");
    EXPECT_EQ(compiledVectorSize(strides, dims, {{"x", 32}}), "4");
    EXPECT_EQ(compiledVectorSize(strides, dims, {{"x", 24}}), "2");
}

// Bug caught: only x checked, so y is stored four elements at a time to a pointer declaring less.
TEST(TestBatchnormFwdInferencePlan, CompileNarrowsTheVectorSizeToTheDeclaredOutputAlignment)
{
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
    const std::vector<int64_t> dims = {1, 3, 224, 224};

    EXPECT_EQ(compiledVectorSize(strides, dims, {{"y", 8}}), "2");
    EXPECT_EQ(compiledVectorSize(strides, dims, {{"y", 16}}), "4");
}

// Bug caught: the per-channel tensors capping the vector size in NCHW, where the kernel loads one
// element of them, or not capping it in NHWC, where it loads a vector.
TEST(TestBatchnormFwdInferencePlan, CompileBoundsTheVectorSizeByPerChannelTensorsOnlyInNhwc)
{
    const std::vector<int64_t> nchwStrides = {150528, 50176, 224, 1};
    const std::vector<int64_t> nchwDims = {1, 3, 224, 224};
    EXPECT_EQ(compiledVectorSize(nchwStrides,
                                 nchwDims,
                                 {{"scale", 4}, {"bias", 4}, {"est_mean", 4}, {"est_variance", 4}}),
              "4");

    const std::vector<int64_t> nhwcStrides = {16384, 1, 1024, 64};
    const std::vector<int64_t> nhwcDims = {1, 64, 16, 16};
    EXPECT_EQ(compiledVectorSize(nhwcStrides, nhwcDims, {}), "4");
    EXPECT_EQ(compiledVectorSize(nhwcStrides, nhwcDims, {{"est_mean", 8}}), "2");
    EXPECT_EQ(compiledVectorSize(nhwcStrides, nhwcDims, {{"est_variance", 4}}), "1");
    EXPECT_EQ(compiledVectorSize(nhwcStrides, nhwcDims, {{"scale", 16}}), "4");
}

// Bug caught: a tensor declaring less than its element size compiled anyway, instead of the plan
// refusing it.
TEST(TestBatchnormFwdInferencePlan, CompileThrowsBelowTheElementAlignment)
{
    const MockKernelCompiler mockCompiler;

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"x", 2}});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(patched.data(),
                                                                           patched.size());
    const auto& attr = *graph.getNode(0).attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferencePlan plan{BatchnormFwdInferenceParams(attr, graph.getTensorMap())};

    EXPECT_THROW(plan.compile(mockCompiler, createTestDeviceProps()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// Bug caught: the fused output tensor ignored, so the virtual batchnorm output's alignment bounds
// the vector size instead of the activation output the kernel writes.
TEST(TestBatchnormFwdInferenceParams, MaxVectorSizeReadsTheFusedActivationOutput)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"Dy", 8}, {"y", 1}});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(patched.data(),
                                                                           patched.size());
    const auto& attr = *graph.getNode(0).attributes_as_BatchnormInferenceAttributes();
    const auto& activAttrs = *graph.getNode(1).attributes_as_PointwiseAttributes();

    const BatchnormFwdInferenceParams params(attr, activAttrs, graph.getTensorMap());

    EXPECT_EQ(params.maxVectorSize(false), 2U);
    EXPECT_EQ(params.maxVectorSize(true), 2U);
}

} // namespace hip_kernel_provider::batchnorm::test
