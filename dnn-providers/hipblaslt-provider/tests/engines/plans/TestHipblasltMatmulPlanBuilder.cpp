// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "TestWorkarounds.hpp"
#include "engines/plans/HipblasltMatmulPlanBuilder.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

namespace
{

using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// Re-serializes a test-helper matmul graph with the virtual layout a frontend emits:
// every fused intermediate (the matmul output when bias or activation follows, the
// bias output when activation follows) is virtual and every other tensor is not.
// Each tensor named in `flipped` then has its virtual flag inverted. The test graph
// helper names the tensors A, B, bias, C_matmul, C_bias and C.
flatbuffers::FlatBufferBuilder frontendVirtualLayout(const flatbuffers::FlatBufferBuilder& source,
                                                     const std::set<std::string>& flipped = {})
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(source.GetBufferPointer());
    const size_t nodeCount = graph->nodes.size();
    for(auto& tensor : graph->tensors)
    {
        const bool intermediate = (tensor->name == "C_matmul" && nodeCount > 1)
                                  || (tensor->name == "C_bias" && nodeCount > 2);
        tensor->virtual_ = intermediate != (flipped.count(tensor->name) != 0);
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(builder, graph.get()));
    return builder;
}

// Re-serializes the graph with the named tensor's strides replaced.
flatbuffers::FlatBufferBuilder withTensorStrides(const flatbuffers::FlatBufferBuilder& source,
                                                 const std::string& tensorName,
                                                 const std::vector<int64_t>& strides)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(source.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->name == tensorName)
        {
            tensor->strides = strides;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(builder, graph.get()));
    return builder;
}

flatbuffers::FlatBufferBuilder createFusedGraph(bool doBias,
                                                PointwiseMode activMode,
                                                const std::set<std::string>& flipped = {})
{
    return frontendVirtualLayout(
        createValidMatmulBiasActivGraph(
            {4, 8}, {8, 1}, {8, 5}, {5, 1}, {4, 5}, {5, 1}, doBias, activMode),
        flipped);
}

// Re-serializes the graph with the named tensor's data type replaced.
flatbuffers::FlatBufferBuilder
    withTensorDataType(const flatbuffers::FlatBufferBuilder& source,
                       const std::string& tensorName,
                       hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(source.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->name == tensorName)
        {
            tensor->data_type = type;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(builder, graph.get()));
    return builder;
}

} // namespace

// Host-only fixture: the default handle is never used because the virtual and
// bias checks run before plan construction. Without a GPU handle every graph is
// eventually rejected, so the decline cause is asserted from the plugin log.
class TestHipblasltMatmulPlanBuilderDecline : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Other suites install their own plugin logging callback; route plugin logs to
        // the shared recorder so tests can assert decline causes.
        hipdnn_plugin_sdk::logging::initializeCallbackLogging(
            "hipblaslt_plugin_tests",
            hipdnn_test_sdk::utilities::initializeTestLogRecordingShared());
    }

    HipblasltMatmulPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle{};
};

// One fusion shape and the tensors whose virtual flag is flipped, one per case,
// with the role the decline message names.
struct VirtualCase
{
    const char* shape;
    bool doBias;
    PointwiseMode activMode;
    const char* tensorName;
    const char* role;
};

class TestHipblasltMatmulPlanBuilderDeclineRoles : public TestHipblasltMatmulPlanBuilderDecline,
                                                   public ::testing::WithParamInterface<VirtualCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderDeclineRoles, IsApplicableDeclinesFlippedVirtualFlag)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const auto& param = GetParam();
    auto builder = createFusedGraph(param.doBias, param.activMode, {param.tensorName});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                          std::string("Matmul: ") + param.role + " tensor must be"))
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(
    Roles,
    TestHipblasltMatmulPlanBuilderDeclineRoles,
    ::testing::Values(
        VirtualCase{"matmul", false, PointwiseMode::UNSET, "A", "A"},
        VirtualCase{"matmul", false, PointwiseMode::UNSET, "B", "B"},
        VirtualCase{"matmul", false, PointwiseMode::UNSET, "C_matmul", "matmul output"},
        VirtualCase{"matmul_bias", true, PointwiseMode::UNSET, "A", "A"},
        VirtualCase{"matmul_bias", true, PointwiseMode::UNSET, "B", "B"},
        VirtualCase{"matmul_bias", true, PointwiseMode::UNSET, "bias", "bias input"},
        VirtualCase{"matmul_bias", true, PointwiseMode::UNSET, "C_matmul", "matmul output"},
        VirtualCase{"matmul_bias", true, PointwiseMode::UNSET, "C_bias", "bias output"},
        VirtualCase{"matmul_act", false, PointwiseMode::RELU_FWD, "A", "A"},
        VirtualCase{"matmul_act", false, PointwiseMode::RELU_FWD, "B", "B"},
        VirtualCase{"matmul_act", false, PointwiseMode::RELU_FWD, "C_matmul", "matmul output"},
        VirtualCase{"matmul_act", false, PointwiseMode::RELU_FWD, "C", "activation output"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "A", "A"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "B", "B"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "bias", "bias input"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "C_matmul", "matmul output"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "C_bias", "bias output"},
        VirtualCase{"matmul_bias_act", true, PointwiseMode::RELU_FWD, "C", "activation output"}),
    [](const ::testing::TestParamInfo<VirtualCase>& info) {
        return std::string(info.param.shape) + "_" + info.param.tensorName;
    });

// The virtual layout a frontend emits for each fusion shape is not declined for
// virtuality.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDoesNotDeclineFrontendVirtualLayout)
{
    const std::array<std::pair<bool, PointwiseMode>, 4> shapes{{{false, PointwiseMode::UNSET},
                                                                {true, PointwiseMode::UNSET},
                                                                {false, PointwiseMode::RELU_FWD},
                                                                {true, PointwiseMode::RELU_FWD}}};
    for(const auto& [doBias, activMode] : shapes)
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        auto builder = createFusedGraph(doBias, activMode);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        (void)_planBuilder.isApplicable(_handle, graph);
        EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "tensor must be"))
            << recorder.getRecordedLogsAsString();
    }
}

// The bias of a [4, 5] output has dims [1, 5]; hipBLASLt reads it as a packed
// vector of 5 elements. Strides of size-1 dims do not affect the layout.
struct BiasStridesCase
{
    std::vector<int64_t> strides;
    bool packed;
};

class TestHipblasltMatmulPlanBuilderBiasStridesCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<BiasStridesCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderBiasStridesCases, IsApplicableDeclinesNonPackedBias)
{
    for(const auto activMode : {PointwiseMode::UNSET, PointwiseMode::RELU_FWD})
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        auto builder = withTensorStrides(
            createValidMatmulBiasActivGraph(
                {4, 8}, {8, 1}, {8, 5}, {5, 1}, {4, 5}, {5, 1}, true, activMode),
            "bias",
            GetParam().strides);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        (void)_planBuilder.isApplicable(_handle, graph);
        EXPECT_EQ(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Matmul: bias strides must be packed"),
                  !GetParam().packed)
            << recorder.getRecordedLogsAsString();
    }
}

INSTANTIATE_TEST_SUITE_P(Strides,
                         TestHipblasltMatmulPlanBuilderBiasStridesCases,
                         ::testing::Values(BiasStridesCase{{5, 1}, true},
                                           BiasStridesCase{{0, 1}, true},
                                           BiasStridesCase{{99, 1}, true},
                                           BiasStridesCase{{5, 2}, false},
                                           BiasStridesCase{{5, 0}, false},
                                           BiasStridesCase{{1, 5}, false}));

// The bias input is validated on its own data type, independent of the matmul
// output type (which stays fp32 here).
struct BiasTypeCase
{
    hipdnn_flatbuffers_sdk::data_objects::DataType type;
    bool served;
};

class TestHipblasltMatmulPlanBuilderBiasTypeCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<BiasTypeCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderBiasTypeCases, IsApplicableDeclinesUnservedBiasType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorDataType(
        createValidMatmulBiasActivGraph({4, 8},
                                        {8, 1},
                                        {8, 5},
                                        {5, 1},
                                        {4, 5},
                                        {5, 1},
                                        true,
                                        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        DataType::FLOAT),
        "bias",
        GetParam().type);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_EQ(
        recorder.hasLogContaining(HIPDNN_SEV_INFO, "Bias node input and output data types must be"),
        !GetParam().served)
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(
    Types,
    TestHipblasltMatmulPlanBuilderBiasTypeCases,
    ::testing::Values(BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, true},
                      BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::HALF, true},
                      BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16, true},
                      BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE, false},
                      BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::INT8, false},
                      BiasTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                                   false}));

class TestHipblasltMatmulPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        SKIP_IF_WORKAROUND_ISSUE_9962(_handle);
        ASSERT_EQ(hipblasLtCreate(&_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle.hipblasltHandle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    HipblasltMatmulPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle;
};

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicable)
{
    // Too many nodes (> 3)
    {
        MockGraph const mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, mockGraph));
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported broadcastable batch dimensions
    {
        std::vector<int64_t> const aDims = {2, 4, 8};
        std::vector<int64_t> const aStrides = {32, 8, 1};
        std::vector<int64_t> const bDims = {1, 8, 5};
        std::vector<int64_t> const bStrides = {40, 5, 1};
        std::vector<int64_t> const cDims = {2, 4, 5};
        std::vector<int64_t> const cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Different tensor ranks
    {
        std::vector<int64_t> const aDims = {2, 4, 8};
        std::vector<int64_t> const aStrides = {32, 8, 1};
        std::vector<int64_t> const bDims = {8, 5};
        std::vector<int64_t> const bStrides = {5, 1};
        std::vector<int64_t> const cDims = {2, 4, 5};
        std::vector<int64_t> const cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Incorrect C batch dimension
    {
        std::vector<int64_t> const aDims = {2, 4, 8};
        std::vector<int64_t> const aStrides = {32, 8, 1};
        std::vector<int64_t> const bDims = {3, 8, 5};
        std::vector<int64_t> const bStrides = {40, 5, 1};
        std::vector<int64_t> const cDims = {6, 4, 5};
        std::vector<int64_t> const cStrides = {20, 5, 1};

        auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported data type
    {
        std::vector<int64_t> const aDims = {2, 4, 8};
        std::vector<int64_t> const aStrides = {32, 8, 1};
        std::vector<int64_t> const bDims = {2, 8, 5};
        std::vector<int64_t> const bStrides = {40, 5, 1};
        std::vector<int64_t> const cDims = {2, 4, 5};
        std::vector<int64_t> const cStrides = {20, 5, 1};

        auto builder
            = createValidMatmulGraph(aDims,
                                     aStrides,
                                     bDims,
                                     bStrides,
                                     cDims,
                                     cStrides,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);

        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported compute data type
    {
        flatbuffers::FlatBufferBuilder const builder = createValidMatmulGraph();

        auto mutableGraph
            = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
        mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
            hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias only (no activation)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (GELU)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + activation only (no bias, standard ReLU)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.0f));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (Swish)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD,
            std::nullopt,
            std::nullopt,
            1.0f));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (clamp)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.f,
            6.f));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + activation (ReLU with no clip or slope set)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (ReLU6: upper clip only)
    {
        auto builder = frontendVirtualLayout(createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            std::nullopt,
            6.f));
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported leaky ReLU (no hipBLASLt epilogue), even with an explicit zero lower clip
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.f,
            std::nullopt,
            std::nullopt,
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            0.01f);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }

    // Unsupported compute data type on activation node
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);

        auto mutableGraph
            = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
        mutableGraph->mutable_nodes()->GetMutableObject(2)->mutate_compute_data_type(
            hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    }
}

TEST_F(TestHipblasltMatmulPlanBuilder, GetWorkspaceSize)
{
    // Too many nodes (> 3)
    {
        MockGraph const mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, mockGraph), HipdnnPluginException);
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, graph), HipdnnPluginException);
    }

    // Supported Graph with matmul only
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
    }

    // Supported Graph with matmul + bias only (no activation)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
    }

    // Supported Graph with matmul + bias + activation
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
    }

    // Supported Graph with matmul + activation only (no bias)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.0f);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
    }
}

TEST_F(TestHipblasltMatmulPlanBuilder, BuildPlan)
{
    // Too many nodes (> 3)
    {
        MockGraph const mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_handle, mockGraph, ctx), HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }

    // Unsupported Graph with batchnorm
    {
        auto builder = createValidBatchnormInferenceGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_handle, graph, ctx), HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }

    // Supported Graph with matmul
    {
        auto builder = createValidMatmulGraph();
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    // Supported Graph with matmul + bias only (no activation)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    // Supported Graph with matmul + bias + activation
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    // Supported Graph with matmul + activation only (no bias)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.0f);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}
