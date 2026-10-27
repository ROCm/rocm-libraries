// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormBwdPlan.hpp"

#include "../TestPlanCommon.hpp"
#include "BatchnormAlignmentTestUtils.hpp"
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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXN=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXHW=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
}

TEST(TestBatchnormBwdPlan, CompileWithUnsupportedDimensionThrows)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 1}, {1, 3, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormBackwardAttributes();

    BatchnormBwdParams params(attr, graph.getTensorMap());
    BatchnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// BatchnormBwdPlan - declared tensor alignment
// ============================================================================

namespace
{

// Compiles the plan of a backward graph whose named tensors declare the given alignments and
// returns the compile options.
std::vector<std::string> bwdCompileOptionsWithAlignments(const std::vector<int64_t>& strides,
                                                         const std::vector<int64_t>& dims,
                                                         const test::AlignmentByName& alignments)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(strides, dims);
    const auto patched = test::withNamedAlignments(builder.GetBufferPointer(), alignments);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(patched.data(),
                                                                           patched.size());
    const auto& attr = *graph.getNode(0).attributes_as_BatchnormBackwardAttributes();

    BatchnormBwdPlan plan{BatchnormBwdParams(attr, graph.getTensorMap())};
    return test::compileAndCaptureOptions(
        [&](const IKernelCompiler& compiler) { plan.compile(compiler, createTestDeviceProps()); });
}

} // namespace

// Bug caught: the single kernel's four-element x and dy reads or two-element dx writes left running
// on tensors declaring less, instead of the spatial multiple kernel at a narrower vector size, or
// that fallback taken at exactly the alignment the wide accesses need.
TEST(TestBatchnormBwdPlan, CompileFallsBackToSpatialMultipleWhenWideAccessesAreUncovered)
{
    // hw = 50176 selects the single kernel with wide reads and writes.
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
    const std::vector<int64_t> dims = {1, 3, 224, 224};

    const auto exact
        = bwdCompileOptionsWithAlignments(strides, dims, {{"x", 16}, {"dy", 16}, {"dx", 8}});
    EXPECT_TRUE(test::hasCompileOption(exact, "-DHIP_PLUGIN_BN_VARIANT=1"));

    for(const test::AlignmentByName& alignments : {test::AlignmentByName{{"x", 8}},
                                                   test::AlignmentByName{{"dy", 8}},
                                                   test::AlignmentByName{{"dx", 4}}})
    {
        const auto narrowed = bwdCompileOptionsWithAlignments(strides, dims, alignments);
        EXPECT_TRUE(test::hasCompileOption(narrowed, "-DHIP_PLUGIN_BN_VARIANT=2"));
        EXPECT_TRUE(test::hasCompileOption(narrowed, "-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    }
}

// Bug caught: the NHWC vector size kept at 4 although the per-channel tensors declare less than the
// float4 loads need, or dropped at exactly the 16 bytes those loads need.
TEST(TestBatchnormBwdPlan, CompileFoldsTheDeclaredAlignmentIntoTheNhwcVectorSize)
{
    const std::vector<int64_t> strides = {16384, 1, 1024, 64};
    const std::vector<int64_t> dims = {128, 64, 16, 16};

    EXPECT_TRUE(test::hasCompileOption(bwdCompileOptionsWithAlignments(strides, dims, {}),
                                       "-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(
        test::hasCompileOption(bwdCompileOptionsWithAlignments(strides, dims, {{"dscale", 16}}),
                               "-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(
        test::hasCompileOption(bwdCompileOptionsWithAlignments(strides, dims, {{"dscale", 8}}),
                               "-DHIP_PLUGIN_BN_VEC_SIZE=1"));
}

// Bug caught: a problem that has no kernel configuration within the declared alignment compiled
// the wide-access kernel anyway.
TEST(TestBatchnormBwdPlan, CompileThrowsWhenNoConfigurationFitsTheDeclaredAlignment)
{
    const MockKernelCompiler mockCompiler;

    // hw = 17 * 241 = 4097 selects the single kernel, and a single sample leaves the spatial
    // multiple kernel no room to stash.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({8194, 4097, 241, 1},
                                                                            {1, 2, 17, 241});
    const auto patched = test::withNamedAlignments(builder.GetBufferPointer(), {{"dx", 4}});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(patched.data(),
                                                                           patched.size());
    const auto& attr = *graph.getNode(0).attributes_as_BatchnormBackwardAttributes();

    BatchnormBwdPlan plan{BatchnormBwdParams(attr, graph.getTensorMap())};

    EXPECT_THROW(plan.compile(mockCompiler, createTestDeviceProps()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace hip_kernel_provider::batchnorm
