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

// Re-serializes a test-helper matmul graph with the virtual flag of each tensor named in
// `flipped` inverted. The helper emits the virtual layout a frontend produces and names
// the tensors A, B, bias, C_matmul, C_bias and C.
flatbuffers::FlatBufferBuilder flipVirtualFlags(const flatbuffers::FlatBufferBuilder& source,
                                                const std::set<std::string>& flipped)
{
    return hipdnn_test_sdk::utilities::mutateGraph(source, [&flipped](auto& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(flipped.count(tensor->name) != 0)
            {
                tensor->virtual_ = !tensor->virtual_;
            }
        }
    });
}

// Re-serializes the graph with the named tensor's strides replaced.
flatbuffers::FlatBufferBuilder withTensorStrides(const flatbuffers::FlatBufferBuilder& source,
                                                 const std::string& tensorName,
                                                 const std::vector<int64_t>& strides)
{
    return hipdnn_test_sdk::utilities::mutateGraph(source, [&](auto& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == tensorName)
            {
                tensor->strides = strides;
            }
        }
    });
}

// Bias node whose operands are in_0 = bias, in_1 = matmul output when `biasFirst` is set.
flatbuffers::FlatBufferBuilder
    createMatmulBiasGraph(bool doBias, PointwiseMode activMode, bool biasFirst = false)
{
    return createValidMatmulBiasActivGraph({4, 8},
                                           {8, 1},
                                           {8, 5},
                                           {5, 1},
                                           {4, 5},
                                           {5, 1},
                                           doBias,
                                           activMode,
                                           std::nullopt,
                                           std::nullopt,
                                           std::nullopt,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                           std::nullopt,
                                           biasFirst);
}

flatbuffers::FlatBufferBuilder createFusedGraph(bool doBias,
                                                PointwiseMode activMode,
                                                const std::set<std::string>& flipped = {},
                                                bool biasFirst = false)
{
    return flipVirtualFlags(createMatmulBiasGraph(doBias, activMode, biasFirst), flipped);
}

// Re-serializes the graph with the named tensor's data type replaced.
flatbuffers::FlatBufferBuilder
    withTensorDataType(const flatbuffers::FlatBufferBuilder& source,
                       const std::string& tensorName,
                       hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    return hipdnn_test_sdk::utilities::mutateGraph(source, [&](auto& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == tensorName)
            {
                tensor->data_type = type;
            }
        }
    });
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

// The bias is whichever bias-node operand is not the matmul output, in either position.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDeclinesVirtualBiasInFirstOperand)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = createFusedGraph(true, PointwiseMode::RELU_FWD, {"bias"}, true);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(
        recorder.hasLogContaining(HIPDNN_SEV_INFO, "Matmul: bias input tensor must be non-virtual"))
        << recorder.getRecordedLogsAsString();
}

TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDoesNotDeclineFrontendLayoutBiasFirst)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = createFusedGraph(true, PointwiseMode::RELU_FWD, {}, true);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "tensor must be"))
        << recorder.getRecordedLogsAsString();
}

// A frontend-layout graph that reaches the activation compute-type check and fails only it.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDeclinesActivationComputeTypeMismatch)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = hipdnn_test_sdk::utilities::mutateGraph(
        createFusedGraph(true, PointwiseMode::GELU_APPROX_TANH_FWD), [](auto& graph) {
            graph.nodes.at(2)->compute_data_type
                = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
        });
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_INFO, "Activation node compute data type must be equal to matmul"))
        << recorder.getRecordedLogsAsString();
}

TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDeclinesLeakyRelu)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder
        = createValidMatmulBiasActivGraph({4, 8},
                                          {8, 1},
                                          {8, 5},
                                          {5, 1},
                                          {4, 5},
                                          {5, 1},
                                          false,
                                          PointwiseMode::RELU_FWD,
                                          0.f,
                                          std::nullopt,
                                          std::nullopt,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                          0.01f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Leaky relu"))
        << recorder.getRecordedLogsAsString();
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
        for(const bool biasFirst : {false, true})
        {
            const auto recorder
                = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
            auto builder = withTensorStrides(
                createMatmulBiasGraph(true, activMode, biasFirst), "bias", GetParam().strides);
            GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

            (void)_planBuilder.isApplicable(_handle, graph);
            EXPECT_EQ(
                recorder.hasLogContaining(HIPDNN_SEV_INFO, "Matmul: bias strides must be packed"),
                !GetParam().packed)
                << "biasFirst=" << biasFirst << "\n"
                << recorder.getRecordedLogsAsString();
        }
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
// output type (which stays fp32 here). A HALF or BF16 bias on an FP32 graph passes the
// type check; hipBLASLt declines it when it has no kernel.
struct BiasTypeCase
{
    hipdnn_flatbuffers_sdk::data_objects::DataType type;
    bool passesTypeCheck;
};

class TestHipblasltMatmulPlanBuilderBiasTypeCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<BiasTypeCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderBiasTypeCases, IsApplicableDeclinesUnservedBiasType)
{
    for(const bool biasFirst : {false, true})
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        auto builder = withTensorDataType(
            createMatmulBiasGraph(true, PointwiseMode::UNSET, biasFirst), "bias", GetParam().type);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        (void)_planBuilder.isApplicable(_handle, graph);
        EXPECT_EQ(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                            "Bias node input and output data types must be"),
                  !GetParam().passesTypeCheck)
            << "biasFirst=" << biasFirst << "\n"
            << recorder.getRecordedLogsAsString();
    }
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

// Frontend-layout fused graph with batched operands. Tensors are named as in the helper:
// A, B, C_matmul (virtual when fused) and C (the final output of the activation).
static flatbuffers::FlatBufferBuilder createBatchedGraph(const std::vector<int64_t>& aDims,
                                                         const std::vector<int64_t>& aStrides,
                                                         const std::vector<int64_t>& bDims,
                                                         const std::vector<int64_t>& bStrides,
                                                         const std::vector<int64_t>& cDims,
                                                         const std::vector<int64_t>& cStrides,
                                                         PointwiseMode activMode)
{
    return createValidMatmulBiasActivGraph(
        aDims, aStrides, bDims, bStrides, cDims, cStrides, false, activMode);
}

// Re-serializes the graph with the named tensor's dims replaced.
static flatbuffers::FlatBufferBuilder withTensorDims(const flatbuffers::FlatBufferBuilder& source,
                                                     const std::string& tensorName,
                                                     const std::vector<int64_t>& dims)
{
    return hipdnn_test_sdk::utilities::mutateGraph(source, [&](auto& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == tensorName)
            {
                tensor->dims = dims;
            }
        }
    });
}

// hipBLASLt takes one flattened batch count and one batch stride, so the batch axes must
// nest contiguously. The matmul output is the final tensor of a one-node graph, so
// "C_matmul" names the output there. With an activation the matmul output is a virtual
// intermediate and "C" is the final output the layout takes its strides from.
struct BatchStrideCase
{
    const char* name;
    const char* tensorName;
    std::vector<int64_t> strides;
    bool uniform;
    PointwiseMode activMode = PointwiseMode::UNSET;
};

class TestHipblasltMatmulPlanBuilderBatchStrideCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<BatchStrideCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderBatchStrideCases, IsApplicableDeclinesNonUniformBatchStrides)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    // Dense [2, 3, M, K] x [2, 3, K, N] with a two-axis batch.
    auto builder = withTensorStrides(createBatchedGraph({2, 3, 4, 8},
                                                        {96, 32, 8, 1},
                                                        {2, 3, 8, 5},
                                                        {120, 40, 5, 1},
                                                        {2, 3, 4, 5},
                                                        {60, 20, 5, 1},
                                                        GetParam().activMode),
                                     GetParam().tensorName,
                                     GetParam().strides);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_EQ(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"), !GetParam().uniform)
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(
    Strides,
    TestHipblasltMatmulPlanBuilderBatchStrideCases,
    ::testing::Values(
        BatchStrideCase{"DenseA", "A", {96, 32, 8, 1}, true},
        BatchStrideCase{"GapInOuterAxisA", "A", {160, 32, 8, 1}, false},
        BatchStrideCase{"GapInOuterAxisB", "B", {200, 40, 5, 1}, false},
        BatchStrideCase{"GapInOuterAxisOutput", "C_matmul", {100, 20, 5, 1}, false},
        BatchStrideCase{"PaddedInnerAxisA", "A", {192, 64, 8, 1}, true},
        BatchStrideCase{
            "FusedGapInOuterAxisFinal", "C", {100, 20, 5, 1}, false, PointwiseMode::RELU_FWD},
        BatchStrideCase{"FusedNestedFinal", "C", {60, 20, 5, 1}, true, PointwiseMode::RELU_FWD}),
    [](const auto& info) { return std::string(info.param.name); });

// A singleton trailing batch axis can carry an arbitrary stride; it never advances, so the
// flattened batch stride comes from the next axis with extent above one.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableAcceptsSingletonBatchAxisStrideOnA)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorStrides(createBatchedGraph({4, 1, 4, 8},
                                                        {32, 32, 8, 1},
                                                        {4, 1, 8, 5},
                                                        {40, 40, 5, 1},
                                                        {4, 1, 4, 5},
                                                        {20, 20, 5, 1},
                                                        PointwiseMode::UNSET),
                                     "A",
                                     {32, 7, 8, 1});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"))
        << recorder.getRecordedLogsAsString();
}

// A matrix layout takes its leading dimension from the stride of the matrix's non-unit-stride
// axis; a zero or negative one is declined before any plan is built. A is [4, 8] and B is
// [8, 5], so strides {s, 1} are row-major and {1, s} column-major with ld = s.
struct LeadingDimensionCase
{
    const char* name;
    const char* tensorName;
    std::vector<int64_t> strides;
};

class TestHipblasltMatmulPlanBuilderLeadingDimensionCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<LeadingDimensionCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderLeadingDimensionCases,
       IsApplicableDeclinesNonPositiveLeadingDimension)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorStrides(
        createBatchedGraph({4, 8}, {8, 1}, {8, 5}, {5, 1}, {4, 5}, {5, 1}, PointwiseMode::UNSET),
        GetParam().tensorName,
        GetParam().strides);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Leading dimension must be positive"))
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(
    Strides,
    TestHipblasltMatmulPlanBuilderLeadingDimensionCases,
    ::testing::Values(LeadingDimensionCase{"ARowMajorNegative", "A", {-1, 1}},
                      LeadingDimensionCase{"AColumnMajorNegative", "A", {1, -1}},
                      LeadingDimensionCase{"ARowMajorZero", "A", {0, 1}},
                      LeadingDimensionCase{"BRowMajorNegative", "B", {-1, 1}},
                      LeadingDimensionCase{"BColumnMajorNegative", "B", {1, -1}},
                      LeadingDimensionCase{"BColumnMajorZero", "B", {1, 0}}),
    [](const auto& info) { return std::string(info.param.name); });

// The stride of a size-1 dimension is never a leading dimension, so a non-positive one is
// not declined for it: A [1, 8] reads its leading dimension from the column count.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableIgnoresNonPositiveStrideOfSizeOneDim)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder
        = createBatchedGraph({1, 8}, {-1, 1}, {8, 5}, {5, 1}, {1, 5}, {5, 1}, PointwiseMode::UNSET);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "Leading dimension must be positive"))
        << recorder.getRecordedLogsAsString();
}

// A fused final output with a singleton batch axis keeps the matmul output's flattened batch
// stride, so it is not declined for a batch stride or batch dimensions mismatch.
TEST_F(TestHipblasltMatmulPlanBuilderDecline,
       IsApplicableDoesNotDeclineFusedFinalWithSingletonBatchAxis)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    // The virtual matmul output is packed; the final output keeps a stride of 5 on the
    // singleton axis.
    auto builder = withTensorStrides(createBatchedGraph({2, 1, 4, 8},
                                                        {32, 32, 8, 1},
                                                        {2, 1, 8, 5},
                                                        {40, 40, 5, 1},
                                                        {2, 1, 4, 5},
                                                        {20, 5, 5, 1},
                                                        PointwiseMode::RELU_FWD),
                                     "C_matmul",
                                     {20, 20, 5, 1});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "final output batch"))
        << recorder.getRecordedLogsAsString();
}

// A batch axis of extent one never advances, so its stride is irrelevant.
TEST_F(TestHipblasltMatmulPlanBuilderDecline,
       IsApplicableAcceptsArbitraryStrideOnSingletonOuterBatchAxis)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorStrides(createBatchedGraph({1, 3, 4, 8},
                                                        {96, 32, 8, 1},
                                                        {1, 3, 8, 5},
                                                        {120, 40, 5, 1},
                                                        {1, 3, 4, 5},
                                                        {60, 20, 5, 1},
                                                        PointwiseMode::UNSET),
                                     "A",
                                     {999, 32, 8, 1});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"))
        << recorder.getRecordedLogsAsString();
}

// An input broadcast along the whole batch uses no batch stride, so its strides are not checked.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableAcceptsUnusedBroadcastBatchStrides)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = createBatchedGraph({2, 3, 4, 8},
                                      {96, 32, 8, 1},
                                      {1, 1, 8, 5},
                                      {777, 99, 5, 1},
                                      {2, 3, 4, 5},
                                      {60, 20, 5, 1},
                                      PointwiseMode::UNSET);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"))
        << recorder.getRecordedLogsAsString();
}

// The layout takes the batch offset from the final output, so a padded final output is
// served and is not declined for its batch stride.
TEST_F(TestHipblasltMatmulPlanBuilderDecline,
       IsApplicableDoesNotDeclineFusedFinalWithPaddedBatchStride)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorStrides(createBatchedGraph({2, 4, 8},
                                                        {32, 8, 1},
                                                        {2, 8, 5},
                                                        {40, 5, 1},
                                                        {2, 4, 5},
                                                        {20, 5, 1},
                                                        PointwiseMode::RELU_FWD),
                                     "C",
                                     {27, 5, 1});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "final output batch"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "are not uniform"))
        << recorder.getRecordedLogsAsString();
}

TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDeclinesFinalBatchDimsMismatch)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorDims(createBatchedGraph({2, 4, 8},
                                                     {32, 8, 1},
                                                     {2, 8, 5},
                                                     {40, 5, 1},
                                                     {2, 4, 5},
                                                     {20, 5, 1},
                                                     PointwiseMode::RELU_FWD),
                                  "C",
                                  {3, 4, 5});
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_INFO, "final output batch dimensions must equal the matmul output"))
        << recorder.getRecordedLogsAsString();
}

// hipBLASLt documents only equal-typed A and B for the plain matmul tuples.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableDeclinesMixedABTypes)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder
        = withTensorDataType(createFusedGraph(false, PointwiseMode::UNSET), "B", DataType::HALF);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "A and B data types must be equal"))
        << recorder.getRecordedLogsAsString();
}

TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableAcceptsEqualHalfABTypes)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = withTensorDataType(
        withTensorDataType(createFusedGraph(false, PointwiseMode::UNSET), "A", DataType::HALF),
        "B",
        DataType::HALF);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "A and B data types must be equal"))
        << recorder.getRecordedLogsAsString();
}

// The cuDNN v8 mixed-precision fused shape: half operands, bias and final output, with
// float virtual intermediates (matmul and bias outputs). No type check may decline it.
TEST_F(TestHipblasltMatmulPlanBuilderDecline, IsApplicableAcceptsCudnnV8MixedPrecisionFusedGraph)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = createFusedGraph(true, PointwiseMode::RELU_FWD);
    for(const char* name : {"A", "B", "bias", "C"})
    {
        builder = withTensorDataType(builder, name, DataType::HALF);
    }
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    (void)_planBuilder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "data type"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "tensor must be"))
        << recorder.getRecordedLogsAsString();
}

// The activation output is the tensor the plan writes; its type must be one hipBLASLt serves.
struct ActivOutputTypeCase
{
    hipdnn_flatbuffers_sdk::data_objects::DataType type;
    bool served;
};

class TestHipblasltMatmulPlanBuilderActivOutputTypeCases
    : public TestHipblasltMatmulPlanBuilderDecline,
      public ::testing::WithParamInterface<ActivOutputTypeCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderActivOutputTypeCases,
       IsApplicableDeclinesUnservedActivationOutputType)
{
    for(const bool doBias : {false, true})
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        auto builder = withTensorDataType(
            createFusedGraph(doBias, PointwiseMode::RELU_FWD), "C", GetParam().type);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        (void)_planBuilder.isApplicable(_handle, graph);
        EXPECT_EQ(
            recorder.hasLogContaining(HIPDNN_SEV_INFO, "Activation node output data type must be"),
            !GetParam().served)
            << "doBias=" << doBias << "\n"
            << recorder.getRecordedLogsAsString();
    }
}

INSTANTIATE_TEST_SUITE_P(
    Types,
    TestHipblasltMatmulPlanBuilderActivOutputTypeCases,
    ::testing::Values(
        ActivOutputTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, true},
        ActivOutputTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::HALF, true},
        ActivOutputTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16, true},
        ActivOutputTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::INT8, false},
        ActivOutputTypeCase{hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE, false}));

// The plan binds A, B, the bias input and the final output to device buffers, so a
// pass-by-value tensor (compile-time or runtime scalar) on any of them has no buffer.
struct ScalarCase
{
    const char* name;
    bool doBias;
    PointwiseMode activMode;
    const char* tensorName;
    const char* role;
    bool runtime = false;
};

class TestHipblasltMatmulPlanBuilderScalarCases : public TestHipblasltMatmulPlanBuilderDecline,
                                                  public ::testing::WithParamInterface<ScalarCase>
{
};

TEST_P(TestHipblasltMatmulPlanBuilderScalarCases, IsApplicableDeclinesPassByValue)
{
    const auto& param = GetParam();
    const auto base = createFusedGraph(param.doBias, param.activMode);
    {
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        GraphWrapper const graph(base.GetBufferPointer(), base.GetSize());
        (void)_planBuilder.isApplicable(_handle, graph);
        EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "must not be pass-by-value"))
            << recorder.getRecordedLogsAsString();
    }

    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto builder = hipdnn_test_sdk::utilities::mutateGraph(base, [&](auto& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == param.tensorName)
            {
                if(param.runtime)
                {
                    tensor->is_runtime_pass_by_value = true;
                }
                else
                {
                    tensor->value.Set(hipdnn_flatbuffers_sdk::data_objects::Float32Value(2.0f));
                }
            }
        }
    });
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                          std::string("Matmul: ") + param.role
                                              + " tensor must not be pass-by-value"))
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(
    Roles,
    TestHipblasltMatmulPlanBuilderScalarCases,
    ::testing::Values(
        ScalarCase{"A", true, PointwiseMode::RELU_FWD, "A", "A"},
        ScalarCase{"B", true, PointwiseMode::RELU_FWD, "B", "B"},
        ScalarCase{"BiasInput", true, PointwiseMode::RELU_FWD, "bias", "bias input"},
        ScalarCase{"BiasInputRuntime", true, PointwiseMode::RELU_FWD, "bias", "bias input", true},
        ScalarCase{"BiasActivOutput", true, PointwiseMode::RELU_FWD, "C", "activation output"},
        ScalarCase{"BiasOnlyOutput", true, PointwiseMode::UNSET, "C_bias", "bias output"},
        ScalarCase{"ActivOnlyOutput", false, PointwiseMode::RELU_FWD, "C", "activation output"},
        ScalarCase{"MatmulOutput", false, PointwiseMode::UNSET, "C_matmul", "matmul output"}),
    [](const auto& info) { return std::string(info.param.name); });

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

// Size-1 batch axes are ignored, so the plan builder serves these graphs. Values are not
// compared here.
TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableAcceptsSingletonBatchAxisGraphs)
{
    {
        auto builder = withTensorStrides(createBatchedGraph({2, 1, 4, 8},
                                                            {32, 32, 8, 1},
                                                            {2, 1, 8, 5},
                                                            {40, 40, 5, 1},
                                                            {2, 1, 4, 5},
                                                            {20, 5, 5, 1},
                                                            PointwiseMode::RELU_FWD),
                                         "C_matmul",
                                         {20, 20, 5, 1});
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }
    {
        auto builder = withTensorStrides(createBatchedGraph({4, 1, 4, 8},
                                                            {32, 32, 8, 1},
                                                            {4, 1, 8, 5},
                                                            {40, 40, 5, 1},
                                                            {4, 1, 4, 5},
                                                            {20, 20, 5, 1},
                                                            PointwiseMode::UNSET),
                                         "A",
                                         {32, 7, 8, 1});
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }
}

// The boundary graphs the host-only tests build are applicable on a device. Those tests
// only check that one decline message is absent; here the verdict itself is asserted.
TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableAcceptsBatchBoundaryGraphs)
{
    const auto expectApplicable
        = [this](const flatbuffers::FlatBufferBuilder& builder, const char* what) {
              SCOPED_TRACE(what);
              GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
              EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
          };

    expectApplicable(withTensorStrides(createBatchedGraph({1, 3, 4, 8},
                                                          {96, 32, 8, 1},
                                                          {1, 3, 8, 5},
                                                          {120, 40, 5, 1},
                                                          {1, 3, 4, 5},
                                                          {60, 20, 5, 1},
                                                          PointwiseMode::UNSET),
                                       "A",
                                       {999, 32, 8, 1}),
                     "arbitrary stride on a singleton outer batch axis");
    expectApplicable(createBatchedGraph({2, 3, 4, 8},
                                        {96, 32, 8, 1},
                                        {1, 1, 8, 5},
                                        {777, 99, 5, 1},
                                        {2, 3, 4, 5},
                                        {60, 20, 5, 1},
                                        PointwiseMode::UNSET),
                     "unused broadcast batch strides");
    expectApplicable(createBatchedGraph({2, 3, 4, 8},
                                        {96, 32, 8, 1},
                                        {2, 3, 8, 5},
                                        {120, 40, 5, 1},
                                        {2, 3, 4, 5},
                                        {60, 20, 5, 1},
                                        PointwiseMode::UNSET),
                     "dense two-axis batch");
    expectApplicable(withTensorStrides(createBatchedGraph({2, 3, 4, 8},
                                                          {96, 32, 8, 1},
                                                          {2, 3, 8, 5},
                                                          {120, 40, 5, 1},
                                                          {2, 3, 4, 5},
                                                          {60, 20, 5, 1},
                                                          PointwiseMode::UNSET),
                                       "A",
                                       {192, 64, 8, 1}),
                     "padded nested batch strides on A");
    expectApplicable(createBatchedGraph({2, 3, 4, 8},
                                        {96, 32, 8, 1},
                                        {2, 3, 8, 5},
                                        {120, 40, 5, 1},
                                        {2, 3, 4, 5},
                                        {60, 20, 5, 1},
                                        PointwiseMode::RELU_FWD),
                     "fused final output nested like the matmul output");
    expectApplicable(withTensorStrides(createBatchedGraph({2, 4, 8},
                                                          {32, 8, 1},
                                                          {2, 8, 5},
                                                          {40, 5, 1},
                                                          {2, 4, 5},
                                                          {20, 5, 1},
                                                          PointwiseMode::RELU_FWD),
                                       "C",
                                       {27, 5, 1}),
                     "fused final output with a padded batch stride");
}

// Bias in either operand position and every packed bias stride the host tests accept.
TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableAcceptsBiasBoundaryGraphs)
{
    for(const auto activMode : {PointwiseMode::UNSET, PointwiseMode::RELU_FWD})
    {
        for(const bool biasFirst : {false, true})
        {
            for(const auto& strides : std::vector<std::vector<int64_t>>{{5, 1}, {0, 1}, {99, 1}})
            {
                SCOPED_TRACE("activation=" + std::to_string(static_cast<int>(activMode))
                             + " biasFirst=" + std::to_string(biasFirst) + " biasStrides="
                             + std::to_string(strides[0]) + "," + std::to_string(strides[1]));
                auto builder = withTensorStrides(
                    createMatmulBiasGraph(true, activMode, biasFirst), "bias", strides);
                GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
                EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
            }
        }
    }
}

// The data-type boundary graphs the host-only tests accept: equal half A and B, and the
// cuDNN v8 mixed-precision fused shape with float virtual intermediates.
TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableAcceptsHalfOperandGraphs)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    {
        auto builder = withTensorDataType(
            withTensorDataType(createFusedGraph(false, PointwiseMode::UNSET), "A", DataType::HALF),
            "B",
            DataType::HALF);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }
    {
        auto builder = createFusedGraph(true, PointwiseMode::RELU_FWD);
        for(const char* name : {"A", "B", "bias", "C"})
        {
            builder = withTensorDataType(builder, name, DataType::HALF);
        }
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }
}

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

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (GELU)
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

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + activation only (no bias, standard ReLU)
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

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (Swish)
    {
        auto builder = createValidMatmulBiasActivGraph(
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
            1.0f);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (clamp)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            0.f,
            6.f);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + activation (ReLU with no clip or slope set)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            false,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
        GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
    }

    // Supported graph with matmul + bias + activation (ReLU6: upper clip only)
    {
        auto builder = createValidMatmulBiasActivGraph(
            {4, 8},
            {8, 1},
            {8, 5},
            {5, 1},
            {4, 5},
            {5, 1},
            true,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            std::nullopt,
            6.f);
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

namespace
{
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// Tensor order in createValidMatmulBiasActivGraph: A, B, C_matmul, [bias, C_bias], C.
void setTensorTypes(flatbuffers::FlatBufferBuilder& builder,
                    const std::vector<std::pair<flatbuffers::uoffset_t, DataType>>& types)
{
    auto* mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    for(const auto& [index, type] : types)
    {
        ASSERT_TRUE(
            mutableGraph->mutable_tensors()->GetMutableObject(index)->mutate_data_type(type));
    }
}

bool isApplicableWithTypes(const HipblasltMatmulPlanBuilder& planBuilder,
                           const HipdnnEnginePluginHandle& handle,
                           bool doBias,
                           PointwiseMode activMode,
                           const std::vector<std::pair<flatbuffers::uoffset_t, DataType>>& types)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8}, {8, 1}, {8, 5}, {5, 1}, {4, 5}, {5, 1}, doBias, activMode);
    setTensorTypes(builder, types);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    return planBuilder.isApplicable(handle, graph);
}
} // namespace

// Bias only: A, B, C_matmul, bias, C_bias(final).
TEST_F(TestHipblasltMatmulPlanBuilder, DeclinesNarrowMatmulOutputBeforeBias)
{
    // Mutant: dropping the matmul-output check makes the narrower HALF intermediate applicable.
    EXPECT_TRUE(isApplicableWithTypes(_planBuilder, _handle, true, PointwiseMode::UNSET, {}));
    EXPECT_FALSE(isApplicableWithTypes(
        _planBuilder, _handle, true, PointwiseMode::UNSET, {{2, DataType::HALF}}));
}

// Bias + activation: A, B, C_matmul, bias, C_bias, C(final).
TEST_F(TestHipblasltMatmulPlanBuilder, DeclinesNarrowBiasOutputBeforeActivation)
{
    // Mutant: dropping the bias-output check makes the HALF bias output applicable.
    EXPECT_TRUE(isApplicableWithTypes(_planBuilder, _handle, true, PointwiseMode::RELU_FWD, {}));
    EXPECT_FALSE(isApplicableWithTypes(
        _planBuilder, _handle, true, PointwiseMode::RELU_FWD, {{4, DataType::HALF}}));
}

TEST_F(TestHipblasltMatmulPlanBuilder, AcceptsFloatIntermediatesUnderNarrowOutput)
{
    // Mutant: a rule requiring intermediates to equal the final output declines FLOAT ones.
    for(const auto narrow : {DataType::HALF, DataType::BFLOAT16})
    {
        EXPECT_TRUE(isApplicableWithTypes(_planBuilder,
                                          _handle,
                                          true,
                                          PointwiseMode::RELU_FWD,
                                          {{0, narrow}, {1, narrow}, {5, narrow}}));
    }
}

// hipBLASLt computes the epilogue on the FP32 accumulator, so an intermediate that equals a
// narrow final output type is still declined.
TEST_F(TestHipblasltMatmulPlanBuilder, DeclinesNarrowIntermediatesEqualToNarrowOutput)
{
    // Mutant: an "or equal to the final output type" exemption accepts these graphs.
    for(const auto narrow : {DataType::HALF, DataType::BFLOAT16})
    {
        // Bias + activation, every tensor narrow: the matmul output (2) is the first decline.
        EXPECT_FALSE(isApplicableWithTypes(
            _planBuilder,
            _handle,
            true,
            PointwiseMode::RELU_FWD,
            {{0, narrow}, {1, narrow}, {2, narrow}, {3, narrow}, {4, narrow}, {5, narrow}}));
        // Bias + activation, FLOAT matmul output: the narrow bias output (4) alone declines.
        EXPECT_FALSE(isApplicableWithTypes(
            _planBuilder,
            _handle,
            true,
            PointwiseMode::RELU_FWD,
            {{0, narrow}, {1, narrow}, {3, narrow}, {4, narrow}, {5, narrow}}));
        // Bias only, narrow matmul output (2) equal to the narrow final output (4).
        EXPECT_FALSE(isApplicableWithTypes(
            _planBuilder,
            _handle,
            true,
            PointwiseMode::UNSET,
            {{0, narrow}, {1, narrow}, {2, narrow}, {3, narrow}, {4, narrow}}));
        // Positive control: the same bias-only graph with a FLOAT matmul output is accepted.
        EXPECT_TRUE(isApplicableWithTypes(_planBuilder,
                                          _handle,
                                          true,
                                          PointwiseMode::UNSET,
                                          {{0, narrow}, {1, narrow}, {3, narrow}, {4, narrow}}));
        // Activation only (A, B, C_matmul, C): a narrow matmul output equal to the narrow final
        // output is declined with no bias node in the graph.
        EXPECT_FALSE(isApplicableWithTypes(_planBuilder,
                                           _handle,
                                           false,
                                           PointwiseMode::RELU_FWD,
                                           {{0, narrow}, {1, narrow}, {2, narrow}, {3, narrow}}));
        // Positive control: the same activation-only graph with a FLOAT matmul output.
        EXPECT_TRUE(isApplicableWithTypes(_planBuilder,
                                          _handle,
                                          false,
                                          PointwiseMode::RELU_FWD,
                                          {{0, narrow}, {1, narrow}, {3, narrow}}));
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
