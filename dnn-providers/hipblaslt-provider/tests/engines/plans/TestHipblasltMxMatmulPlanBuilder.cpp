// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "TestWorkarounds.hpp"
#include "engines/plans/HipblasltMxMatmulPlanBuilder.hpp"
#include "engines/plans/MxArchSupport.hpp"
#include "engines/plans/MxTestUtils.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipblaslt_plugin_test::createMxGraphWithLayout;
using hipblaslt_plugin_test::MxOperandLayout;

namespace
{

enum class MxTensor
{
    XA, // A input — should be non-virtual
    SCALE_A, // A scale — should be non-virtual
    YA, // A dequant output — should be virtual
    XB, // B input — should be non-virtual
    SCALE_B, // B scale — should be non-virtual
    YB, // B dequant output — should be virtual
    C, // matmul output — should be non-virtual
    NONE, // no tensor corrupted
};

// Builds an otherwise-valid canonical MX graph with the given tensor's is_virtual
// flag flipped from its required value, to exercise the builder's virtuality checks.
// The dequantize nodes carry the given is_negative_scale flags.
flatbuffers::FlatBufferBuilder createMxGraphWithWrongVirtual(MxTensor target,
                                                             bool negativeScaleA = false,
                                                             bool negativeScaleB = false)
{
    // Returns the is_virtual flag for a tensor, flipped from its valid value when it
    // is the corruption target.
    const auto flag
        = [target](MxTensor self, bool valid) { return (self == target) ? !valid : valid; };

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensors;

    const std::vector<int64_t> xADims = {32, 128};
    const std::vector<int64_t> xAStrides = {1, 32}; // col-major (opA=T)
    const std::vector<int64_t> scaleADims = {32, 4};
    const std::vector<int64_t> scaleAStrides = {4, 1};
    const std::vector<int64_t> yADims = {32, 128};
    const std::vector<int64_t> yAStrides = {128, 1};
    const std::vector<int64_t> xBDims = {128, 32};
    const std::vector<int64_t> xBStrides = {32, 1}; // row-major (opB=N)
    const std::vector<int64_t> scaleBDims = {4, 32};
    const std::vector<int64_t> scaleBStrides = {32, 1};
    const std::vector<int64_t> yBDims = {128, 32};
    const std::vector<int64_t> yBStrides = {32, 1};
    const std::vector<int64_t> cDims = {32, 32};
    const std::vector<int64_t> cStrides = {32, 1};

    const int64_t xAUid = 1;
    const int64_t scaleAUid = 2;
    const int64_t yAUid = 3;
    const int64_t xBUid = 4;
    const int64_t scaleBUid = 5;
    const int64_t yBUid = 6;
    const int64_t cUid = 7;

    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xAUid, "x_a", DT::FP8_E4M3, &xAStrides, &xADims, flag(MxTensor::XA, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleAUid,
        "scale_a",
        DT::FP8_E8M0,
        &scaleAStrides,
        &scaleADims,
        flag(MxTensor::SCALE_A, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yAUid, "y_a", DT::FLOAT, &yAStrides, &yADims, flag(MxTensor::YA, true)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xBUid, "x_b", DT::FP8_E4M3, &xBStrides, &xBDims, flag(MxTensor::XB, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleBUid,
        "scale_b",
        DT::FP8_E8M0,
        &scaleBStrides,
        &scaleBDims,
        flag(MxTensor::SCALE_B, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yBUid, "y_b", DT::FLOAT, &yBStrides, &yBDims, flag(MxTensor::YB, true)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, cUid, "c", DT::HALF, &cStrides, &cDims, flag(MxTensor::C, false)));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    const std::vector<int32_t> blockSizeVec = {32};

    auto deqAttrA
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xAUid, scaleAUid, yAUid, &blockSizeVec, negativeScaleA);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_a",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrA.Union()));

    auto deqAttrB
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xBUid, scaleBUid, yBUid, &blockSizeVec, negativeScaleB);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_b",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrB.Union()));

    auto matmulAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(builder, yAUid, yBUid, cUid);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "matmul",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
        matmulAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder, "mx_virtual_test", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensors, &nodes);
    builder.Finish(graphOffset);
    return builder;
}

} // namespace

// ===========================================================================
// Fixtures
// ===========================================================================

// CPU fixture: graph-level negative cases, no device required. The handle has no
// hipBLASLt handle, so isApplicable() also returns false for a valid graph that
// reaches the heuristic: these tests show that a graph is declined, not why. The
// packing rules are tested in the GPU fixture, against a packed control.
class TestHipblasltMxMatmulPlanBuilder : public ::testing::Test
{
protected:
    // Other suites install their own plugin logging callback; route plugin logs to the
    // shared recorder so tests can assert decline causes.
    void SetUp() override
    {
        hipdnn_plugin_sdk::logging::initializeCallbackLogging(
            "hipblaslt_plugin_tests",
            hipdnn_test_sdk::utilities::initializeTestLogRecordingShared());
    }

    HipblasltMxMatmulPlanBuilder _builder;
    HipdnnEnginePluginHandle _handle{};
};

// GPU fixture: positive cases that construct a real plan via the hipBLASLt
// handle. Gated on a device + a live hipblasLt handle. Split into a base so the
// typed suites below reuse the same setup.
class MxGpuPlanBuilderTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        const auto archName = hipdnn_test_sdk::utilities::currentDeviceArch();
        if(!hipblaslt_plugin_test::isMxSupportedArch(archName))
        {
            GTEST_SKIP() << "MX block-scaled GEMM is not supported on " << archName;
        }
        ASSERT_EQ(hipblasLtCreate(&_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle.hipblasltHandle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    HipblasltMxMatmulPlanBuilder _builder;
    HipdnnEnginePluginHandle _handle;
};

class TestGpuHipblasltMxMatmulPlanBuilder : public MxGpuPlanBuilderTestBase
{
};

// Canonical valid MX graph, varying only the element types under test.
static flatbuffers::FlatBufferBuilder
    createMxGraphWithTypes(DT xTypeA, DT cType, DT xTypeB = DT::UNSET)
{
    return createValidMxMatmulGraph({32, 128},
                                    {1, 32},
                                    {128, 32},
                                    {32, 1},
                                    {32, 32},
                                    {32, 1},
                                    {32, 4},
                                    {4, 32},
                                    xTypeA,
                                    cType,
                                    DT::FP8_E8M0,
                                    DT::FP8_E8M0,
                                    DT::FLOAT,
                                    32,
                                    false,
                                    false,
                                    xTypeB);
}

namespace
{

bool onArch(const char* arch)
{
    return archMatches(
        hipdnn_test_sdk::utilities::currentDeviceArch(), arch, ArchMatchMode::PREFIX);
}

bool onGfx1250()
{
    return hipblaslt_plugin_test::isGfx1250Target(hipdnn_test_sdk::utilities::currentDeviceArch());
}

} // namespace

// ===========================================================================
// isApplicable — positive cases (GPU-gated via SKIP_IF_NO_DEVICES in SetUp)
// ===========================================================================

// Input element type x output element type. The dtype pair lives in the type
// parameter rather than the test name, so the case name stays free of datatype
// keywords; -- see MxPlanConfig in TestBlockScaleDequantizePlan.cpp for the same
// pattern over DataType enum values.
template <DT XType, DT OutType>
struct MxDtypeConfig
{
    static constexpr DT X_TYPE = XType;
    static constexpr DT OUT_TYPE = OutType;
};

using MxDtypeConfigs = ::testing::Types<MxDtypeConfig<DT::FP8_E4M3, DT::HALF>,
                                        MxDtypeConfig<DT::FP8_E5M2, DT::BFLOAT16>,
                                        MxDtypeConfig<DT::FP8_E4M3, DT::FLOAT>,
                                        MxDtypeConfig<DT::FP4_E2M1, DT::HALF>,
                                        MxDtypeConfig<DT::FP4_E2M1, DT::BFLOAT16>,
                                        MxDtypeConfig<DT::FP4_E2M1, DT::FLOAT>,
                                        MxDtypeConfig<DT::FP6_E2M3, DT::HALF>,
                                        MxDtypeConfig<DT::FP6_E2M3, DT::BFLOAT16>,
                                        MxDtypeConfig<DT::FP6_E2M3, DT::FLOAT>,
                                        MxDtypeConfig<DT::FP6_E3M2, DT::HALF>,
                                        MxDtypeConfig<DT::FP6_E3M2, DT::BFLOAT16>,
                                        MxDtypeConfig<DT::FP6_E3M2, DT::FLOAT>>;

template <typename ConfigType>
class TestGpuHipblasltMxMatmulPlanBuilderDtypes : public MxGpuPlanBuilderTestBase
{
};

TYPED_TEST_SUITE(TestGpuHipblasltMxMatmulPlanBuilderDtypes, MxDtypeConfigs, );

TYPED_TEST(TestGpuHipblasltMxMatmulPlanBuilderDtypes, IsApplicable)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createMxGraphWithTypes(TypeParam::X_TYPE, TypeParam::OUT_TYPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(this->_builder.isApplicable(this->_handle, graph))
        << "x=" << EnumNameDataType(TypeParam::X_TYPE)
        << " out=" << EnumNameDataType(TypeParam::OUT_TYPE);
}

// Mixed operand pairs: A and B carry different MX element types. Output is HALF
// throughout -- the output axis is covered by the suite above.
template <DT AType, DT BType>
struct MxMixedConfig
{
    static constexpr DT A_TYPE = AType;
    static constexpr DT B_TYPE = BType;
};

using MxMixedConfigs = ::testing::Types<MxMixedConfig<DT::FP8_E4M3, DT::FP4_E2M1>,
                                        MxMixedConfig<DT::FP4_E2M1, DT::FP8_E4M3>,
                                        MxMixedConfig<DT::FP6_E2M3, DT::FP6_E3M2>,
                                        MxMixedConfig<DT::FP6_E3M2, DT::FP6_E2M3>,
                                        MxMixedConfig<DT::FP6_E2M3, DT::FP8_E4M3>,
                                        MxMixedConfig<DT::FP6_E2M3, DT::FP4_E2M1>>;

template <typename ConfigType>
class TestGpuHipblasltMxMatmulPlanBuilderMixedDtypes : public MxGpuPlanBuilderTestBase
{
};

TYPED_TEST_SUITE(TestGpuHipblasltMxMatmulPlanBuilderMixedDtypes, MxMixedConfigs, );

TYPED_TEST(TestGpuHipblasltMxMatmulPlanBuilderMixedDtypes, IsApplicable)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createMxGraphWithTypes(TypeParam::A_TYPE, DT::HALF, TypeParam::B_TYPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(this->_builder.isApplicable(this->_handle, graph))
        << "a=" << EnumNameDataType(TypeParam::A_TYPE)
        << " b=" << EnumNameDataType(TypeParam::B_TYPE);
}

// Dequant nodes emitted in B-then-A order must still be recognized: A/B are
// resolved by matching dequant Y outputs to matmul inputs, not by node position.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableHandlesSwappedDequantOrder)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       32,
                                       false,
                                       true /*swapDequantOrder*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// ===========================================================================
// isApplicable — negative cases (all CPU-safe, no GPU call needed)
// ===========================================================================

// Plain matmul (1 node) must NOT match the MX builder
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsPlainMatmul)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Wrong node count (not 3)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongNodeCount)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_FALSE(_builder.isApplicable(_handle, mockGraph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongNodeCount4)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    EXPECT_FALSE(_builder.isApplicable(_handle, mockGraph));
}

// Non-MX input types
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonMxInputHalf)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::HALF,
                                       DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonMxInputFloat)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FLOAT,
                                       DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// FP8 output type not allowed
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8Output)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::FP8_E4M3);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8E8M0Output)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::FP8_E8M0);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// m not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsMNotDiv16)
{
    auto fb = createValidMxMatmulGraph(
        {33, 128}, {1, 33}, {128, 32}, {32, 1}, {33, 32}, {32, 1}, {33, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// n not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNNotDiv16)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 33}, {33, 1}, {32, 33}, {33, 1}, {32, 4}, {4, 33});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// K not divisible by 128 (96 is block-aligned at 32 but not 128-aligned)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsKNotDiv128)
{
    auto fb = createValidMxMatmulGraph(
        {32, 96}, {1, 32}, {96, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 3}, {3, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// block_size != 32
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongBlockSize)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       16 /*blockSize*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Operands and scales must be packed row- or column-major matrices: rocRoller
// ignores leading dimensions, so a padded one would be read as packed (#14290).
// Each case changes the strides of one tensor of a K-contiguous graph that is
// accepted when packed (on every MX arch), so the decline comes from those strides.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonPackedTensors)
{
    const auto graphWithStrides = [](const std::vector<int64_t>& aStrides,
                                     const std::vector<int64_t>& bStrides,
                                     const std::vector<int64_t>& scaleAStrides,
                                     const std::vector<int64_t>& scaleBStrides) {
        return createValidMxMatmulGraph({32, 128},
                                        aStrides,
                                        {128, 32},
                                        bStrides,
                                        {32, 32},
                                        {32, 1},
                                        {32, 4},
                                        {4, 32},
                                        DT::FP8_E4M3,
                                        DT::BFLOAT16,
                                        DT::FP8_E8M0,
                                        DT::FP8_E8M0,
                                        DT::FLOAT,
                                        32,
                                        false,
                                        false,
                                        DT::UNSET,
                                        scaleAStrides,
                                        scaleBStrides);
    };
    const std::vector<int64_t> aPacked = {128, 1};
    const std::vector<int64_t> bPacked = {1, 128};
    const std::vector<int64_t> scaleAPacked = {4, 1};
    const std::vector<int64_t> scaleBPacked = {1, 4};

    auto packed = graphWithStrides(aPacked, bPacked, scaleAPacked, scaleBPacked);
    GraphWrapper const packedGraph(packed.GetBufferPointer(), packed.GetSize());
    ASSERT_TRUE(_builder.isApplicable(_handle, packedGraph)) << "packed control";

    struct Case
    {
        const char* name;
        flatbuffers::FlatBufferBuilder graph;
    };
    std::vector<Case> cases;
    // A row-major [32, 128] with a row stride of 136.
    cases.push_back({"padded A", graphWithStrides({136, 1}, bPacked, scaleAPacked, scaleBPacked)});
    // B column-major [128, 32] with a column stride of 136.
    cases.push_back({"padded B", graphWithStrides(aPacked, {1, 136}, scaleAPacked, scaleBPacked)});
    // Neither axis of A has unit stride.
    cases.push_back(
        {"non-matrix A", graphWithStrides({256, 2}, bPacked, scaleAPacked, scaleBPacked)});
    // scale_a row-major [32, 4] with a row stride of 8.
    cases.push_back({"padded scale_a", graphWithStrides(aPacked, bPacked, {8, 1}, scaleBPacked)});
    // scale_b column-major [4, 32] with a column stride of 8.
    cases.push_back({"padded scale_b", graphWithStrides(aPacked, bPacked, scaleAPacked, {1, 8})});

    for(auto& c : cases)
    {
        GraphWrapper const graph(c.graph.GetBufferPointer(), c.graph.GetSize());
        EXPECT_FALSE(_builder.isApplicable(_handle, graph)) << c.name;
    }
}

// The same rule for the NT layout (M-contiguous A, N-contiguous B, row-major
// scales), which the provider accepted before it checked packing: a padded operand
// or scale was then read as packed. gfx950 only; gfx1250 declines NT for layout.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonPackedTensorsNt)
{
    if(!onArch("gfx950"))
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    const auto graphWithStrides = [](const std::vector<int64_t>& aStrides,
                                     const std::vector<int64_t>& bStrides,
                                     const std::vector<int64_t>& scaleAStrides,
                                     const std::vector<int64_t>& scaleBStrides) {
        return createValidMxMatmulGraph({32, 128},
                                        aStrides,
                                        {128, 32},
                                        bStrides,
                                        {32, 32},
                                        {32, 1},
                                        {32, 4},
                                        {4, 32},
                                        DT::FP8_E4M3,
                                        DT::HALF,
                                        DT::FP8_E8M0,
                                        DT::FP8_E8M0,
                                        DT::FLOAT,
                                        32,
                                        false,
                                        false,
                                        DT::UNSET,
                                        scaleAStrides,
                                        scaleBStrides);
    };
    const std::vector<int64_t> aPacked = {1, 32};
    const std::vector<int64_t> bPacked = {32, 1};
    const std::vector<int64_t> scaleAPacked = {4, 1};
    const std::vector<int64_t> scaleBPacked = {32, 1};

    auto packed = graphWithStrides(aPacked, bPacked, scaleAPacked, scaleBPacked);
    GraphWrapper const packedGraph(packed.GetBufferPointer(), packed.GetSize());
    ASSERT_TRUE(_builder.isApplicable(_handle, packedGraph)) << "packed control";

    struct Case
    {
        const char* name;
        flatbuffers::FlatBufferBuilder graph;
    };
    std::vector<Case> cases;
    // A column-major [32, 128] with a column stride of 40.
    cases.push_back({"padded A", graphWithStrides({1, 40}, bPacked, scaleAPacked, scaleBPacked)});
    // B row-major [128, 32] with a row stride of 40.
    cases.push_back({"padded B", graphWithStrides(aPacked, {40, 1}, scaleAPacked, scaleBPacked)});
    // scale_a row-major [32, 4] with a row stride of 8.
    cases.push_back({"padded scale_a", graphWithStrides(aPacked, bPacked, {8, 1}, scaleBPacked)});
    // scale_b row-major [4, 32] with a row stride of 40.
    cases.push_back({"padded scale_b", graphWithStrides(aPacked, bPacked, scaleAPacked, {40, 1})});

    for(auto& c : cases)
    {
        GraphWrapper const graph(c.graph.GetBufferPointer(), c.graph.GetSize());
        EXPECT_FALSE(_builder.isApplicable(_handle, graph)) << c.name;
    }
}

// The graph-only checks need no device. Every packed operand order passes them, with
// packed scales in either order: the plan prepares a scale stored in the other order.
TEST(TestMxGraphConstraints, AcceptsPackedOperandsWithScalesInEitherOrder)
{
    for(const auto layout : {MxOperandLayout::MN_MN,
                             MxOperandLayout::K_K,
                             MxOperandLayout::K_MN,
                             MxOperandLayout::MN_K})
    {
        for(const bool mnContiguousScales : {false, true})
        {
            auto fb = createMxGraphWithLayout(
                layout, DT::FP8_E4M3, DT::HALF, DT::UNSET, mnContiguousScales);
            GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
            EXPECT_NO_THROW(checkMxGraphConstraints(graph))
                << toString(layout) << " mnContiguousScales=" << mnContiguousScales;
        }
    }
}

// The plan hands C to hipBLASLt as a row-major matrix, so a column-major C is declined,
// and hipBLASLt ignores an MX GEMM's leading dimensions (#14290), so is a padded one.
TEST(TestMxGraphConstraints, DeclinesOutputThatIsNotPackedRowMajor)
{
    const std::vector<std::pair<std::vector<int64_t>, std::string>> cases{
        {{48, 1}, "[#14290] MX matmul: output C has a padded leading dimension"},
        {{1, 32}, "output C must be a row-major"}};
    for(const auto& [cStrides, reason] : cases)
    {
        auto fb = createValidMxMatmulGraph({32, 128},
                                           {128, 1},
                                           {128, 32},
                                           {1, 128},
                                           {32, 32},
                                           cStrides,
                                           {32, 4},
                                           {4, 32},
                                           DT::FP8_E4M3,
                                           DT::HALF,
                                           DT::FP8_E8M0,
                                           DT::FP8_E8M0,
                                           DT::FLOAT,
                                           32,
                                           false,
                                           false,
                                           DT::UNSET,
                                           {4, 1},
                                           {1, 4});
        GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
        try
        {
            checkMxGraphConstraints(graph);
            ADD_FAILURE() << "C strides {" << cStrides[0] << ", " << cStrides[1] << "} accepted";
        }
        catch(const HipdnnPluginException& e)
        {
            EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
        }
    }
}

// hipBLASLt reads A and B and writes C as packed matrices (#14290), so the graph-only
// checks decline a padded A or B in either order, and a C whose row stride is not N.
TEST(TestMxGraphConstraints, DeclinesMatricesThatAreNotPacked)
{
    struct Case
    {
        std::string name;
        flatbuffers::FlatBufferBuilder graph;
    };
    std::vector<Case> cases;
    cases.push_back(
        {"K-contiguous A, row stride 136",
         createValidMxMatmulGraph(
             {32, 128}, {136, 1}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32})});
    cases.push_back(
        {"M-contiguous A, column stride 40",
         createValidMxMatmulGraph(
             {32, 128}, {1, 40}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32})});
    cases.push_back(
        {"K-contiguous B, column stride 136",
         createValidMxMatmulGraph(
             {32, 128}, {1, 32}, {128, 32}, {1, 136}, {32, 32}, {32, 1}, {32, 4}, {4, 32})});
    cases.push_back(
        {"N-contiguous B, row stride 40",
         createValidMxMatmulGraph(
             {32, 128}, {1, 32}, {128, 32}, {40, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32})});
    for(const int64_t cRowStride : {16, 0, -32})
    {
        cases.push_back({"C row stride " + std::to_string(cRowStride),
                         createValidMxMatmulGraph({32, 128},
                                                  {1, 32},
                                                  {128, 32},
                                                  {32, 1},
                                                  {32, 32},
                                                  {cRowStride, 1},
                                                  {32, 4},
                                                  {4, 32})});
    }

    for(auto& c : cases)
    {
        GraphWrapper const graph(c.graph.GetBufferPointer(), c.graph.GetSize());
        EXPECT_THROW(checkMxGraphConstraints(graph), HipdnnPluginException) << c.name;
    }
}

// The swizzled scale formats run on hipBLASLt's Tensile kernels, which exist only for
// K-contiguous A and B (#13715). Natural scales take every packed operand order.
TEST(TestMxGraphConstraints, SwizzledScaleFormatsNeedKContiguousOperands)
{
    for(const auto layout : {MxOperandLayout::MN_MN,
                             MxOperandLayout::K_K,
                             MxOperandLayout::K_MN,
                             MxOperandLayout::MN_K})
    {
        auto fb = createMxGraphWithLayout(layout);
        GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
        EXPECT_NO_THROW(checkMxLayoutForScaleFormat(graph, MxScaleFormat::NATURAL))
            << toString(layout);
        for(const auto format : {MxScaleFormat::SWIZZLED_GFX950, MxScaleFormat::SWIZZLED_GFX1250})
        {
            if(layout == MxOperandLayout::K_K)
            {
                EXPECT_NO_THROW(checkMxLayoutForScaleFormat(graph, format)) << toString(format);
                continue;
            }
            try
            {
                checkMxLayoutForScaleFormat(graph, format);
                ADD_FAILURE() << toString(layout) << " accepted for " << toString(format);
            }
            catch(const HipdnnPluginException& e)
            {
                EXPECT_NE(std::string(e.what()).find("(#13715)"), std::string::npos) << e.what();
            }
        }
    }
}

// Alignment: M and N multiples of 16, K a multiple of 128. The reason is checked
// because hipBLASLt itself finds no kernel for some misaligned shapes.
TEST(TestMxGraphConstraints, DeclinesMisalignedShapes)
{
    const auto expectDeclined
        = [](const flatbuffers::FlatBufferBuilder& fb, const std::string& reason) {
              GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
              try
              {
                  checkMxGraphConstraints(graph);
                  ADD_FAILURE() << "accepted; expected: " << reason;
              }
              catch(const HipdnnPluginException& e)
              {
                  EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
              }
          };
    expectDeclined(createValidMxMatmulGraph(
                       {24, 128}, {1, 24}, {128, 32}, {32, 1}, {24, 32}, {32, 1}, {24, 4}, {4, 32}),
                   "M must be divisible by 16");
    expectDeclined(createValidMxMatmulGraph(
                       {32, 128}, {1, 32}, {128, 24}, {24, 1}, {32, 24}, {24, 1}, {32, 4}, {4, 24}),
                   "N must be divisible by 16");
    expectDeclined(createValidMxMatmulGraph(
                       {32, 96}, {1, 32}, {96, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 3}, {3, 32}),
                   "K must be divisible by 128");
}

// rocRoller (gfx950) reads every operand-order combination; the plan transposes a
// scale whose order differs from its operand's.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableAcceptsEveryLayoutOnGfx950)
{
    if(!onArch("gfx950"))
    {
        GTEST_SKIP() << "gfx950-only check";
    }
    for(const auto layout : {MxOperandLayout::MN_MN,
                             MxOperandLayout::K_K,
                             MxOperandLayout::K_MN,
                             MxOperandLayout::MN_K})
    {
        for(const bool mnContiguousScales : {false, true})
        {
            auto fb = createMxGraphWithLayout(
                layout, DT::FP8_E4M3, DT::HALF, DT::UNSET, mnContiguousScales);
            GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
            EXPECT_TRUE(_builder.isApplicable(_handle, graph))
                << toString(layout) << " mnContiguousScales=" << mnContiguousScales;
        }
    }
}

// gfx1250 runs only the gfx1250 swizzled format, so K-contiguous A and B are
// accepted with either scale order (other layouts: SwizzledScaleFormatsNeedKContiguousOperands).
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableAcceptsKContiguousOnGfx1250)
{
    if(!onGfx1250())
    {
        GTEST_SKIP() << "gfx1250-only check";
    }
    for(const bool mnContiguousScales : {false, true})
    {
        auto fb = createMxGraphWithLayout(
            MxOperandLayout::K_K, DT::FP8_E4M3, DT::BFLOAT16, DT::UNSET, mnContiguousScales);
        GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
        EXPECT_TRUE(_builder.isApplicable(_handle, graph))
            << "mnContiguousScales=" << mnContiguousScales;
    }
}

// Non-MX graph (batchnorm) must return false
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsBatchnormGraph)
{
    auto fb = createValidBatchnormInferenceGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// batch > 1 (hipBLASLt requires B==1 for VEC32_UE8M0) — leading batch dim of 2 must
// be rejected. All tensors carry a rank-3 leading batch dim.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsBatchGreaterThan1)
{
    auto fb = createValidMxMatmulGraph({2, 32, 128},
                                       {4096, 1, 32},
                                       {2, 128, 32},
                                       {4096, 32, 1},
                                       {2, 32, 32},
                                       {1024, 32, 1},
                                       {2, 32, 4},
                                       {2, 4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Fused epilogue (4th Pointwise node) is not allowed — extra node → not 3 nodes
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsEpilogue)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       32,
                                       true /*withEpilogue*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// See WORKAROUND_ISSUE_10811 in Workarounds.hpp.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8AWithFp6B)
{
    for(const auto aType : {DT::FP8_E4M3, DT::FP8_E5M2})
    {
        for(const auto bType : {DT::FP6_E2M3, DT::FP6_E3M2})
        {
            auto fb = createValidMxMatmulGraph({32, 128},
                                               {1, 32},
                                               {128, 32},
                                               {32, 1},
                                               {32, 32},
                                               {32, 1},
                                               {32, 4},
                                               {4, 32},
                                               aType,
                                               DT::HALF,
                                               DT::FP8_E8M0,
                                               DT::FP8_E8M0,
                                               DT::FLOAT,
                                               32,
                                               false,
                                               false,
                                               bType /*xTypeB*/);
            GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
            EXPECT_FALSE(_builder.isApplicable(_handle, graph))
                << "A=" << EnumNameDataType(aType) << " B=" << EnumNameDataType(bType);
        }
    }
}

// #10811 only hits hipBLASLt transB = T, i.e. a column-major graph A. A K-contiguous
// (row-major) FP8 A with an FP6 B is correct on gfx950 and is accepted.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableAcceptsKContiguousFp8AWithFp6B)
{
    if(!onArch("gfx950"))
    {
        GTEST_SKIP() << "gfx950-only check (no FP6 MX kernels on gfx1250)";
    }
    for(const auto layout : {MxOperandLayout::K_K, MxOperandLayout::K_MN})
    {
        for(const auto aType : {DT::FP8_E4M3, DT::FP8_E5M2})
        {
            for(const auto bType : {DT::FP6_E2M3, DT::FP6_E3M2})
            {
                auto fb = createMxGraphWithLayout(layout, aType, DT::HALF, bType);
                GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
                EXPECT_TRUE(_builder.isApplicable(_handle, graph))
                    << toString(layout) << " A=" << EnumNameDataType(aType)
                    << " B=" << EnumNameDataType(bType);
            }
        }
    }
}

// Matmul node compute data type must be FP32 (mirrors the plain matmul builder)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFloatComputeType)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::HALF /*computeType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Input shapes must be consistent with the output: C's N-dim (48) differs from
// B's N-dim (32) here. 48 is 16-aligned, so this fails the shape check, not the
// hipBLASLt alignment check.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsShapeMismatch)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 48}, {48, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Virtual/non-virtual contract: the inputs, their scales, and C are real device
// buffers (must be non-virtual); the dequant Y outputs are fused intermediates
// (must be virtual). Each case flips exactly one flag.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::XA);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::XB);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::SCALE_A);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::SCALE_B);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::YA);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::YB);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualOutputC)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::C);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// is_negative_scale selects Y = X * 2^(-scale), which the VEC32_UE8M0 plan does not
// implement, so the builder declines it on either dequantize node. The decline cause
// is asserted from the log because, without a GPU handle, every graph that passes the
// constraint checks is also rejected later by plan construction.
struct NegativeScaleCase
{
    bool negativeScaleA;
    bool negativeScaleB;
};

class TestHipblasltMxMatmulPlanBuilderNegativeScale
    : public TestHipblasltMxMatmulPlanBuilder,
      public ::testing::WithParamInterface<NegativeScaleCase>
{
};

TEST_P(TestHipblasltMxMatmulPlanBuilderNegativeScale, IsApplicableDeclinesNegativeScale)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto fb = createMxGraphWithWrongVirtual(
        MxTensor::NONE, GetParam().negativeScaleA, GetParam().negativeScaleB);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());

    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "is_negative_scale"))
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(Roles,
                         TestHipblasltMxMatmulPlanBuilderNegativeScale,
                         ::testing::Values(NegativeScaleCase{true, false},
                                           NegativeScaleCase{false, true},
                                           NegativeScaleCase{true, true}));

// The schema default (false on both nodes) passes the negative-scale check.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableDoesNotDeclineDefaultScaleSign)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    auto fb = createMxGraphWithWrongVirtual(MxTensor::NONE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());

    (void)_builder.isApplicable(_handle, graph);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "is_negative_scale"))
        << recorder.getRecordedLogsAsString();
}

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableAcceptsDefaultScaleSign)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createMxGraphWithWrongVirtual(MxTensor::NONE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// Each scale must be a packed row- or column-major matrix; the plan prepares a scale
// stored in the other order. The canonical A scale is [32, 4] and the B scale [4, 32].
// The decline cause is asserted from the log because, without a GPU handle, every
// graph that passes the constraint checks is also rejected later by plan construction.
struct ScaleStridesCase
{
    const char* tensorName;
    const char* role;
    std::vector<int64_t> strides;
    bool packed;
};

class TestHipblasltMxMatmulPlanBuilderScaleStrides
    : public TestHipblasltMxMatmulPlanBuilder,
      public ::testing::WithParamInterface<ScaleStridesCase>
{
};

TEST_P(TestHipblasltMxMatmulPlanBuilderScaleStrides, IsApplicableDeclinesScalesThatAreNotPacked)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const auto& param = GetParam();

    auto fb
        = hipdnn_test_sdk::utilities::mutateGraph(createValidMxMatmulGraph(), [&](auto& graphT) {
              for(auto& tensor : graphT.tensors)
              {
                  if(tensor->name == param.tensorName)
                  {
                      tensor->strides = param.strides;
                  }
              }
          });
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());

    (void)_builder.isApplicable(_handle, graph);
    EXPECT_EQ(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                        std::string("MX matmul: ") + param.role
                                            + " scale must be a packed"),
              !param.packed)
        << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(Strides,
                         TestHipblasltMxMatmulPlanBuilderScaleStrides,
                         ::testing::Values(ScaleStridesCase{"scale_a", "A", {4, 1}, true},
                                           ScaleStridesCase{"scale_a", "A", {1, 32}, true},
                                           ScaleStridesCase{"scale_a", "A", {8, 2}, false},
                                           ScaleStridesCase{"scale_a", "A", {4, 0}, false},
                                           ScaleStridesCase{"scale_b", "B", {32, 1}, true},
                                           ScaleStridesCase{"scale_b", "B", {1, 4}, true},
                                           ScaleStridesCase{"scale_b", "B", {64, 2}, false},
                                           ScaleStridesCase{"scale_b", "B", {0, 1}, false}));

// Scale-tensor contract: VEC32_UE8M0 requires the scales to be FP8_E8M0 and to
// declare exactly one element per 32-element operand block (M*(K/32) for A,
// (K/32)*N for B). Each case corrupts exactly one scale's dtype or shape.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeA)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FLOAT /*scaleAType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeB)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FLOAT /*scaleBType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeA)
{
    // scale_a inner dim 5 instead of K/32 = 4
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 5}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeB)
{
    // scale_b outer dim 5 instead of K/32 = 4
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {5, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// A scale with the correct total element count but the K axis split on the wrong
// axis ([K/32, M] instead of [M, K/32]) must be rejected. A total-count-only check
// would accept this and then compute silently-wrong results on the GPU.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsTransposedScaleA)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {4, 32}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsTransposedScaleB)
{
    // scale_b as [N, K/32] instead of [K/32, N]
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {32, 4});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// ===========================================================================
// getWorkspaceSize
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, GetWorkspaceSizeValid)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_NO_THROW(_builder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, GetWorkspaceSizeThrowsOnInvalidGraph)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_THROW(_builder.getWorkspaceSize(_handle, graph), HipdnnPluginException);
}

// ===========================================================================
// buildPlan
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, BuildPlanValid)
{
    SKIP_IF_WORKAROUND_ISSUE_13715();
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    HipdnnEnginePluginExecutionContext ctx;
    EXPECT_NO_THROW(_builder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, BuildPlanThrowsOnInvalidGraph)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    HipdnnEnginePluginExecutionContext ctx;
    EXPECT_THROW(_builder.buildPlan(_handle, graph, ctx), HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}
