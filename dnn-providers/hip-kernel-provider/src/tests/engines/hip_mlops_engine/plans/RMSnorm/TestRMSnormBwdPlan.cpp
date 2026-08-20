// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <utility>

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

// ============================================================================
// RMSnormBwdParams - construction from valid graph data
// ============================================================================

TEST(TestRMSnormBwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    EXPECT_NO_THROW(const RMSnormBwdParams params(attr, graph.getTensorMap()));
}

TEST(TestRMSnormBwdParams, ConstructsFromTripleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    EXPECT_NO_THROW(const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap()));
}

TEST(TestRMSnormBwdParams, HasCorrectTensorPointersWithOptionalAttributesForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value()); // Not set by single node graph
    EXPECT_EQ(params.y(), nullptr); // Not set by single node graph
}

TEST(TestRMSnormBwdParams, HasCorrectTensorPointersWithOptionalAttributesForTripleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, TensorPointersMatchExpectedUidsForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attr.dy_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.invRMS()->uid(), attr.inv_rms_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attr.dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attr.dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attr.dbias_tensor_uid().value());
}

TEST(TestRMSnormBwdParams, TensorPointersMatchExpectedUidsForTripleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attrActivation.in_0_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attrBwd.x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrBwd.scale_tensor_uid());
    EXPECT_EQ(params.invRMS()->uid(), attrBwd.inv_rms_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attrBwd.dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attrBwd.dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attrBwd.dbias_tensor_uid().value());
    EXPECT_EQ(params.y()->uid(), attrActivation.in_1_tensor_uid().value());
}

TEST(TestRMSnormBwdParams, OptionalTensorsAreNullWhenNotProvidedForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_EQ(params.dbias(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, OptionalTensorsAreNullWhenNotProvidedForTripleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_EQ(params.dbias(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, IsMoveConstructibleForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attr, graph.getTensorMap());
    const RMSnormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
}

TEST(TestRMSnormBwdParams, IsMoveConstructibleForTripleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    const RMSnormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestRMSnormBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormBwdParams>);
}

// ============================================================================
// RMSnormBwdPlan - basic behavior
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, RMSnormBwdPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        bool hasOptionalAttributes = true,
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        strides, dims, hasOptionalAttributes, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), RMSnormBwdPlan{std::move(params)}};
}

std::pair<flatbuffers::FlatBufferBuilder, RMSnormBwdPlan>
    createFusedActivationPlanFromGraph(const std::vector<int64_t>& strides
                                       = {150528, 50176, 224, 1},
                                       const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                       bool hasOptionalAttributes = true,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                                       = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        strides, dims, hasOptionalAttributes, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    return {std::move(builder), RMSnormBwdPlan{std::move(params)}};
}

hipDeviceProp_t createTestDeviceProps(const char* archName = "gfx942")
{
    hipDeviceProp_t deviceProps = {};
    deviceProps.multiProcessorCount = 60;
    deviceProps.warpSize = 64;
    std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", archName);
    return deviceProps;
}

std::pair<std::unique_ptr<MockKernelCompiler>, std::unique_ptr<std::vector<std::string>>>
    setupMockCompileChain(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                          const std::vector<int64_t>& dims = {1, 3, 224, 224},
                          hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                          = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto mockCompiler = std::make_unique<MockKernelCompiler>();
    auto capturedOptions = std::make_unique<std::vector<std::string>>();
    EXPECT_CALL(*mockCompiler, compile("RMSNormBwd.cpp", ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            *capturedOptions = options;

            // First mock kernel for BwdData
            auto mockKernel1 = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*mockKernel1, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*mockKernel1, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);

            // Second mock kernel for BwdWeightBias
            auto mockKernel2 = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*mockKernel2, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*mockKernel2, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);

            auto mockProgram = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*mockProgram, getKernel("RMSnormBwdData"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel1))));
            EXPECT_CALL(*mockProgram, getKernel("RMSnormBwdScaleBias"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel2))));

            return mockProgram;
        });

    auto [fbb, plan] = createPlanFromGraph(strides, dims, true, inputDataType);
    auto deviceProps = createTestDeviceProps();

    plan.compile(*mockCompiler, deviceProps);

    return {std::move(mockCompiler), std::move(capturedOptions)};
}

std::pair<std::unique_ptr<MockKernelCompiler>, std::unique_ptr<std::vector<std::string>>>
    setupFusedActivationMockCompileChain(
        const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
        const std::vector<int64_t>& dims = {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto mockCompiler = std::make_unique<MockKernelCompiler>();
    auto capturedOptions = std::make_unique<std::vector<std::string>>();
    EXPECT_CALL(*mockCompiler, compile("RMSNormBwd.cpp", ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            *capturedOptions = options;

            // First mock kernel for BwdData
            auto mockKernel1 = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*mockKernel1, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*mockKernel1, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);

            // Second mock kernel for BwdWeightBias
            auto mockKernel2 = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*mockKernel2, setBlockSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);
            EXPECT_CALL(*mockKernel2, setGridSize(::testing::_, ::testing::_, ::testing::_))
                .Times(1);

            auto mockProgram = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*mockProgram, getKernel("RMSnormBwdData"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel1))));
            EXPECT_CALL(*mockProgram, getKernel("RMSnormBwdScaleBias"))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel2))));

            return mockProgram;
        });

    auto [fbb, plan] = createFusedActivationPlanFromGraph(strides, dims, true, inputDataType);
    auto deviceProps = createTestDeviceProps();

    plan.compile(*mockCompiler, deviceProps);

    return {std::move(mockCompiler), std::move(capturedOptions)};
}

} // namespace

TEST(TestRMSnormBwdPlan, GetWorkspaceSizeReturnsZeroForSingleNodeGraph)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormBwdPlan, GetWorkspaceSizeReturnsZeroForTripleNodeGraph)
{
    auto [fbb, plan] = createFusedActivationPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormBwdPlan, IsMoveConstructibleForSingleNodeGraph)
{
    auto [fbb, plan] = createPlanFromGraph();

    const RMSnormBwdPlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormBwdPlan, IsMoveConstructibleForTripleNodeGraph)
{
    auto [fbb, plan] = createFusedActivationPlanFromGraph();

    const RMSnormBwdPlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestRMSnormBwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormBwdPlan>);
}

// ============================================================================
// RMSnormBwdPlan - compile
// ============================================================================

TEST(TestRMSnormBwdPlan, CompileCallsCompilerWithCorrectKernelNameForSingleNodeGraph)
{
    setupMockCompileChain();
}

TEST(TestRMSnormBwdPlan, CompileCallsCompilerWithCorrectKernelNameForTripleNodeGraph)
{
    setupFusedActivationMockCompileChain();
}

TEST(TestRMSnormBwdPlanFp32, CompileSetsCorrectDefinesForSingleNodeGraph)
{
    auto [mockCompiler, capturedOptions] = setupMockCompileChain();

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdPlanFp32, CompileSetsCorrectDefinesForTripleNodeGraph)
{
    auto [mockCompiler, capturedOptions] = setupFusedActivationMockCompileChain();

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdPlanFp16, CompileSetsCorrectDefinesForSingleNodeGraph)
{
    auto [mockCompiler, capturedOptions]
        = setupMockCompileChain({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdPlanFp16, CompileSetsCorrectDefinesForTripleNodeGraph)
{
    auto [mockCompiler, capturedOptions] = setupFusedActivationMockCompileChain(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdPlanBfp16, CompileSetsCorrectDefinesForSingleNodeGraph)
{
    auto [mockCompiler, capturedOptions]
        = setupMockCompileChain({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdPlanBfp16, CompileSetsCorrectDefinesForTripleNodeGraph)
{
    auto [mockCompiler, capturedOptions] = setupFusedActivationMockCompileChain(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdPlan, CompileWithChannelLastInputCorrectlySetsDefinesForSingleNodeGraph)
{
    auto [mockCompiler, capturedOptions] = setupMockCompileChain({150528, 1, 672, 3});

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdPlan, CompileWithChannelLastInputCorrectlySetsDefinesForTripleNodeGraph)
{
    auto [mockCompiler, capturedOptions]
        = setupFusedActivationMockCompileChain({150528, 1, 672, 3});

    auto hasOption = [&, &opts = capturedOptions](const std::string& opt) {
        return std::find(opts->begin(), opts->end(), opt) != opts->end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdPlan, CompileWithUnsupportedWorkgroupsThrowsForSingleNodeGraph)
{
    const MockKernelCompiler mockCompiler;

    // Number of workgroups exeeds UINT32_MAX
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph({4, 4, 2, 1},
                                                                          {UINT32_MAX, 1, 2, 2});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attr, graph.getTensorMap());
    RMSnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdPlan, CompileWithUnsupportedWorkgroupsThrowsForTripleNodeGraph)
{
    const MockKernelCompiler mockCompiler;

    // Number of workgroups exeeds UINT32_MAX
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    RMSnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdPlan, CompileWithUnsupportedDimensionThrowsForSingleNodeGraph)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder
        = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph({12, 4, 1}, {1, 3, 4}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attr, graph.getTensorMap());
    RMSnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdPlan, CompileWithUnsupportedDimensionThrowsForTripleNodeGraph)
{
    const MockKernelCompiler mockCompiler;

    // 3D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {12, 4, 1}, {1, 3, 4}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(2);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    RMSnormBwdPlan plan(std::move(params));

    auto deviceProps = createTestDeviceProps();

    EXPECT_THROW(plan.compile(mockCompiler, deviceProps), hipdnn_plugin_sdk::HipdnnPluginException);
}
