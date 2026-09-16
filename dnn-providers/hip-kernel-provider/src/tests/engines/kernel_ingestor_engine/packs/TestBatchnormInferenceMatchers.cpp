// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestBatchnormInferenceMatchers.cpp
 * @brief What hipkernel:BatchnormInference's matchers accept and refuse.
 *
 * The acceptance cases are the shape envelope the kernel was proved over (packed NCHW,
 * packed NHWC, padded x, C=1, unit extents, a non-block-multiple count); the refusals
 * are every precondition the kernel has that the frontend does not enforce. Both
 * directions are load-bearing: an accept-only suite passes for a matcher stuck on true,
 * a refusal-only suite for one stuck on false.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

bool matches(const MatchContext& context)
{
    return matchesGraph(BATCHNORM_INFERENCE, context).has_value();
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: the shapes the kernel was proved over
// ---------------------------------------------------------------------------

/// One graph the matcher must admit, and a readable name for a failing run.
struct GraphCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

class TestBatchnormInferenceGraphMatcherAcceptance : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestBatchnormInferenceGraphMatcherAcceptance, Accepts)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestBatchnormInferenceGraphMatcherAcceptance,
    ::testing::ValuesIn(std::vector<GraphCase>{
        {"PackedNchw", []() { return buildBatchnormInferenceGraph(); }},
        {"PackedNhwc",
         // Layout is not a schema field; NHWC is the same graph with channel-last
         // strides, and the kernel addresses through them rather than assuming an order.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 /*paramDataType=*/std::nullopt,
                                                 /*xDims=*/std::vector<int64_t>{2, 3, 4, 4},
                                                 /*xStridesOverride=*/
                                                 std::vector<int64_t>{48, 1, 12, 3},
                                                 /*yStridesOverride=*/
                                                 std::vector<int64_t>{48, 1, 12, 3});
         }},
        {"NonContiguousX",
         // x is a view into a larger buffer and y is packed: the two stride vectors are
         // read independently, which is the case a shared-stride assumption would break.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 /*paramDataType=*/std::nullopt,
                                                 /*xDims=*/std::vector<int64_t>{2, 3, 4, 4},
                                                 /*xStridesOverride=*/
                                                 std::vector<int64_t>{128, 32, 8, 1});
         }},
        {"UnitExtents",
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{1, 8, 1, 1});
         }},
        {"SingleChannel",
         // C = 1 makes every per-channel operand [1,1,1,1]; the channel term is still
         // read, just always at offset 0.
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{3, 1, 5, 7});
         }},
        {"SmallestPossible",
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{1, 1, 1, 1});
         }},
        {"ElementCountNotABlockMultiple",
         // 5005 elements: the final block is partially populated and the kernel's own
         // bounds guard is what makes that correct.
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{5, 7, 13, 11});
         }},
        {"HalfIoFloatParameters",
         // The pairing every 16-bit corpus graph actually uses, and the reason the KMD
         // carries two dtype fields rather than one. The CPU reference registers
         // (HALF, FLOAT, FLOAT, HALF, FLOAT), so it is numerically provable.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::HALF,
                                                 data_objects::DataType::FLOAT);
         }},
        {"Bfloat16IoFloatParameters",
         // The external headline corpus is entirely this shape: bfloat16 io against
         // float mean/inv_variance/scale/bias, float compute.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::BFLOAT16,
                                                 data_objects::DataType::FLOAT);
         }},
    }),
    [](const ::testing::TestParamInfo<GraphCase>& info) { return info.param.name; });

TEST(TestBatchnormInferenceBinding, BindsAllSixOperandUids)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());

    const auto bound = matchesGraph(BATCHNORM_INFERENCE, fixture.context());
    ASSERT_TRUE(bound.has_value());

    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    EXPECT_EQ(tryGetBoundInt(*bound, BATCHNORM_INFERENCE.inputAToken), BN_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BATCHNORM_INFERENCE.inputBToken), BN_MEAN_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BN_INV_VARIANCE_TOKEN), BN_INV_VARIANCE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BN_SCALE_TOKEN), BN_SCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BN_BIAS_TOKEN), BN_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BATCHNORM_INFERENCE.outputToken), BN_Y_UID);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: refusals
// ---------------------------------------------------------------------------

class TestBatchnormInferenceGraphMatcherRefusal : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestBatchnormInferenceGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestBatchnormInferenceGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphCase>{
        {"InPlaceGraph",
         // y aliasing x is undefined behaviour against the kernel's `__restrict__`, and
         // no run ever exercised it. Refused rather than miscompiled.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 /*aliasYWithX=*/true);
         }},
        {"Rank3",
         // The kernel unravels a flat index into exactly four coordinates. The frontend
         // permits rank >= 2 and the suite ships 3-D batchnorm cases, so this refusal is
         // the difference between a skip and a wrong answer.
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{1, 3, 224});
         }},
        {"Rank5",
         []() {
             return buildBatchnormInferenceGraph(
                 data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{2, 3, 3, 1, 1});
         }},
        {"AllHalfIncludingParameters",
         // io HALF is admitted, but the per-channel operands are HALF here too. No
         // candidate ships a 16-bit parameter type, so this is refused on the parameter
         // dtype -- not, as before the dtype widening, on the io dtype.
         []() { return buildBatchnormInferenceGraph(data_objects::DataType::HALF); }},
        {"MixedIoDtypeBetweenXAndY",
         // One BnIoElement covers both x and y, so whichever tag were baked, one of the
         // two pointers would be accessed at the wrong width. The CPU reference
         // registers (HALF, FLOAT, FLOAT, FLOAT, FLOAT), so this pairing is a real
         // representable graph rather than a constructed one.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::HALF,
                                                 data_objects::DataType::FLOAT,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 /*aliasYWithX=*/false,
                                                 /*xVirtual=*/false,
                                                 /*yVirtual=*/false,
                                                 /*xPassByValue=*/false,
                                                 /*nodeComputeDataType=*/std::nullopt,
                                                 /*yDataTypeOverride=*/
                                                 data_objects::DataType::FLOAT);
         }},
        {"OutputExtentsDisagreeWithInput",
         // One flat index is unravelled against x's dims and applied to both operands.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::vector<int64_t>{24, 8, 2, 1},
                                                 /*yDimsOverride=*/
                                                 std::vector<int64_t>{2, 3, 2, 2});
         }},
        {"PerChannelOperandTooShort",
         // mean[c] is read for c < C; a [1,1,1,1] operand under a 3-channel x is read
         // past its end with nothing raising an error.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 /*paramDimsOverride=*/
                                                 std::vector<int64_t>{1, 1, 1, 1});
         }},
        {"PerChannelOperandNotChannelShaped",
         // [1,C,H,1] is not a per-channel operand, and the kernel would read it with
         // only the channel term.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::vector<int64_t>{1, 3, 4, 1});
         }},
        {"BroadcastStrideOnX",
         // Stride 0 on an axis of extent > 1 reads one element repeatedly. Admitting it
         // on y would additionally race every thread on that axis onto one address.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::vector<int64_t>{0, 16, 4, 1});
         }},
        {"HalfComputeDataType",
         // Float tensors throughout, but the node asks to compute in half. BnCompute is
         // float unconditionally, so serving this would answer a different question than
         // the one the reference executor is given.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 /*nodeComputeDataType=*/
                                                 data_objects::DataType::HALF);
         }},
        {"VirtualOutput",
         // A virtual tensor has no device buffer for launch() to resolve.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 /*aliasYWithX=*/false,
                                                 /*xVirtual=*/false,
                                                 /*yVirtual=*/true);
         }},
        {"PassByValueInput",
         // That variant-pack slot holds a host pointer, not a device one.
         []() {
             return buildBatchnormInferenceGraph(data_objects::DataType::FLOAT,
                                                 std::nullopt,
                                                 std::vector<int64_t>{2, 3, 4, 4},
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 false,
                                                 false,
                                                 false,
                                                 /*xPassByValue=*/true);
         }},
        {"AConvGraph", []() { return buildConvFwdGraph(); }},
        {"APointwiseGraph", []() { return buildPointwiseGraph(); }},
        {"ATwoNodeGraph", []() { return buildTwoNodePointwiseGraph(); }},
    }),
    [](const ::testing::TestParamInfo<GraphCase>& info) { return info.param.name; });

/// The claim the split by graph node type exists to make: each engine's matcher admits
/// only its own node, so a batchnorm graph never reaches the conv or pointwise packs.
TEST(TestBatchnormInferenceGraphMatcher, DoesNotOverlapWithTheOtherShippedEngines)
{
    const GraphFixture batchnormFixture(buildBatchnormInferenceGraph());
    const GraphFixture convFixture(buildConvFwdGraph());
    const GraphFixture pointwiseFixture(buildPointwiseGraph());

    EXPECT_TRUE(matchesGraph(BATCHNORM_INFERENCE, batchnormFixture.context()).has_value());
    EXPECT_FALSE(matchesGraph(CONV_FWD, batchnormFixture.context()).has_value());
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, batchnormFixture.context()).has_value());

    EXPECT_FALSE(matchesGraph(BATCHNORM_INFERENCE, convFixture.context()).has_value());
    EXPECT_FALSE(matchesGraph(BATCHNORM_INFERENCE, pointwiseFixture.context()).has_value());
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestBatchnormInferenceKernelMatcher, AcceptsAKernelWhoseBothDtypesMatchTheGraph)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());

    EXPECT_TRUE(matchesKernel(BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(64)));
}

TEST(TestBatchnormInferenceKernelMatcher, RefusesAKernelBakedForAnotherIoDtype)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());

    EXPECT_FALSE(matchesKernel(
        BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(64, "HALF", "FLOAT")));
}

/// The half of the candidate check that a single-dtype matcher would lose: the kernel's
/// parameter element type is independent of its io element type, and a candidate baked
/// for half parameters would read the four per-channel buffers at half the stride.
TEST(TestBatchnormInferenceKernelMatcher, RefusesAKernelBakedForAnotherParamDtype)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());

    EXPECT_FALSE(matchesKernel(
        BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(64, "FLOAT", "HALF")));
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

/// At one element a 64-thread block leaves 63 lanes idle and a 1024-thread block leaves
/// 1023, so the smallest block wins. A constant score would make the choice arbitrary
/// and nothing else in the fast suite would notice.
TEST(TestBatchnormInferenceScore, PrefersTheBlockThatWastesFewestLanesOnATinyGraph)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph(
        data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{1, 1, 1, 1}));

    EXPECT_GT(scoreKernel(BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(64)),
              scoreKernel(BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(1024)));
}

/// 6422528 elements divides evenly by every shipped block size, so occupancy ties and
/// the tie-break -- fewer blocks for the same work -- decides. This is the direction
/// opposite the case above, which is the point: a scorer that always preferred the
/// smaller block would pass that test and fail this one.
TEST(TestBatchnormInferenceScore, PrefersTheLargerBlockWhenOccupancyTies)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph(
        data_objects::DataType::FLOAT, std::nullopt, std::vector<int64_t>{32, 64, 56, 56}));

    EXPECT_GT(scoreKernel(BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(1024)),
              scoreKernel(BATCHNORM_INFERENCE, fixture.context(), makeBatchnormKernel(64)));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
