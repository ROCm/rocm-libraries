// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormCommon.hpp"

using namespace hip_kernel_provider::batchnorm;

namespace
{

ProblemDescription makeProblem(size_t n,
                               size_t c,
                               size_t h,
                               size_t w,
                               bool isLayoutNHWC,
                               bool isFp32,
                               size_t minWorkgroups = 1,
                               Direction direction = Direction::FORWARD_TRAINING)
{
    return {n, c, h, w, isLayoutNHWC, !isFp32, false, direction, minWorkgroups};
}

} // namespace

TEST(TestProblemDescription, ExposesProblemProperties)
{
    const ProblemDescription problem(32, 64, 8, 16, true, false, true, Direction::BACKWARD, 42);

    EXPECT_EQ(problem.n(), 32);
    EXPECT_EQ(problem.c(), 64);
    EXPECT_EQ(problem.h(), 8);
    EXPECT_EQ(problem.w(), 16);
    EXPECT_TRUE(problem.isLayoutNHWC());
    EXPECT_FALSE(problem.useFp16Mix());
    EXPECT_TRUE(problem.useBfp16Mix());
    EXPECT_FALSE(problem.useFp32());
    EXPECT_EQ(problem.direction(), Direction::BACKWARD);
    EXPECT_EQ(problem.minWorkgroups(), 42);
}

TEST(TestProblemDescription, ComputesDerivedDimensions)
{
    const ProblemDescription problem(
        32, 64, 8, 16, false, false, false, Direction::FORWARD_TRAINING, 1);

    EXPECT_TRUE(problem.useFp32());
    EXPECT_EQ(problem.inCstride(), 128);
    EXPECT_EQ(problem.inNhw(), 4096);
    EXPECT_EQ(problem.inChw(), 8192);
    EXPECT_EQ(problem.inNchw(), 262144);
}

// ============================================================================
// getLocalConfigNHWC - computes local workgroup size for NHWC layout
// ============================================================================

TEST(TestGetLocalConfigNhwcFp32, ProducesSquareWorkgroup)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(makeProblem(1, 128, 16, 16, true, true), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

TEST(TestGetLocalConfigNhwc, MixedPrecisionUsesWiderX)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(makeProblem(1, 128, 16, 16, true, false), 4, x, y);
    EXPECT_EQ(x, 32);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNhwc, HandlesVectorsizeScaling)
{
    size_t x = 0;
    size_t y = 0;

    // With vectorsize of 8, the maxlocalsize should be reduced
    getLocalConfigNHWC(makeProblem(1, 1024, 16, 16, true, true), 8, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNhwc, HandlesIncreasedMinWorkgroups)
{
    size_t xSmallWG = 0;
    size_t ySmallWG = 0;
    getLocalConfigNHWC(makeProblem(1, 64, 64, 64, true, true, 10), 1, xSmallWG, ySmallWG);

    // With high minWorkgroups requirement, the workgroup size should be reduced
    size_t xLargeWG = 0;
    size_t yLargeWG = 0;
    getLocalConfigNHWC(makeProblem(1, 64, 64, 64, true, true, 10000), 1, xLargeWG, yLargeWG);

    EXPECT_EQ(xSmallWG, xLargeWG);
    EXPECT_LT(yLargeWG, ySmallWG);
}

TEST(TestGetLocalConfigNhwc, HandlesSmallChannels)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(makeProblem(1, 2, 16, 16, true, true), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);
}

TEST(TestGetLocalConfigNhwc, HandlesSmallInitialMaxLocalsize)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(makeProblem(1, 128, 16, 16, true, true), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

// ====================================================================================================
// getSpatialMultipleConfig - computes workgroup configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestGetSpatialMultipleConfig, NhwcReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 3, 16, 16, true, true, 80), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NchwReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 3, 5, false, true, 80), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NhwcCalculatesWorkgroupSize)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 16, 16, true, true, 80), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetSpatialMultipleConfig, NchwHandlesLargeSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 64, 64, false, true, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1024);
}

TEST(TestGetSpatialMultipleConfig, NchwScalesDownForSmallSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 8, 16, false, true, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);

    getSpatialMultipleConfig(makeProblem(1, 64, 4, 8, false, true, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 64);
}

// ========================================================================================
// isSpatialMultipleApplicable - checks if spatial multiple implementation can be used
// ========================================================================================

TEST(TestIsSpatialMultipleApplicable, NhwcVectorAlignmentFail)
{
    // If C is not divisible by vectorsize with NHWC layout, it should fail immediately
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(64, 63, 16, 16, true, true), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, NchwSpatialAlignmentFail)
{
    // If H*W is not divisible by vectorsize with NCHW layout, it should fail immediately
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(64, 64, 3, 5, false, true), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInSpatialDimension)
{
    // The last block of spatial dimension is large enough to hold stashed values
    EXPECT_TRUE(
        isSpatialMultipleApplicable(makeProblem(64, 64, 16, 16, true, true), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInBatchDimension)
{
    // Even if the spatial remainder is small, it works if the batch remainder is large enough
    EXPECT_TRUE(
        isSpatialMultipleApplicable(makeProblem(128, 64, 2, 5, true, true), 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashDoesNotFitInSpatialOrBatchDimension)
{
    // Both spatial and batch remainders are smaller than the required stash values
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(10, 64, 2, 5, true, true), 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, MixedPrecisionOddCFailsOnSmallZ)
{
    // For FP16/BF16, if C is odd, the intermediate results MUST fit in the batch dimension
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(10, 65, 16, 16, true, false), 1, 16, 64, 1, 64));
}

// ====================================================================================================
// useMultiple - checks if spatial multiple implementation should be used based on heuristics
// ====================================================================================================

TEST(TestUseMultiple, BackwardSmallSpatialReturnsFalse)
{
    EXPECT_FALSE(useMultiple(makeProblem(64, 1, 16, 16, false, true, 1, Direction::BACKWARD)));
}

TEST(TestUseMultiple, BackwardLargeProblemReturnsTrue)
{
    EXPECT_TRUE(useMultiple(makeProblem(64, 1, 1024, 1024, false, true, 1, Direction::BACKWARD)));
}

TEST(TestUseMultiple, ForwardTrainingSmallProblemReturnsFalse)
{
    EXPECT_FALSE(useMultiple(makeProblem(2, 1, 8, 8, false, true, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, ForwardTrainingLargeBatchReturnsTrue)
{
    EXPECT_TRUE(
        useMultiple(makeProblem(1024, 1, 32, 32, false, true, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, ForwardTrainingMixedPrecisionHeuristic)
{
    EXPECT_FALSE(
        useMultiple(makeProblem(128, 1, 32, 32, false, false, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, NhwcAlwaysReturnsTrue)
{
    EXPECT_TRUE(useMultiple(makeProblem(2, 1, 8, 8, true, true)));
}

// ====================================================================================================
// getStashMethod - determines the stash method to be used based on problem configuration
// ====================================================================================================

TEST(TestGetStashMethod, ReturnsMethodZeroWhenSpatialFits)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 32, 32, false, true), 32, 1024, 1, 64), 0);
}

TEST(TestGetStashMethod, ReturnsMethodOneWhenBatchStashRequired)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 2, 5, false, true), 32, 64, 1, 64), 1);
}

TEST(TestGetStashMethod, ReturnsMethodTwoForNhwcOddCMixedPrecision)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 65, 32, 32, true, false), 16, 1024, 1, 64), 2);
}

TEST(TestGetStashMethod, MixedPrecisionScalesStashValues)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 4, 5, false, false), 16, 64, 1, 64), 1);
}

// ====================================================================================================
// defaultConfigSpatialSingle - provides default configuration for spatial single implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialSingle, NhwcDefaultVariantOne)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(64, 1, 16, 16, true, false, false, Direction::BACKWARD, 1), config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NchwBackwardSmallSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(32, 1, 16, 16, false, false, false, Direction::BACKWARD, 1), config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NchwBackwardSmallSpatialLargeBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(128, 1, 20, 10, false, false, false, Direction::BACKWARD, 1), config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NchwBackwardMidSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(16, 1, 20, 30, false, false, false, Direction::BACKWARD, 1), config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NchwForwardLargeSpatialOrMixed)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(64, 1, 32, 32, false, true, false, Direction::FORWARD_TRAINING, 1),
        config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NchwForwardSmallDefault)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(10, 1, 8, 8, false, false, false, Direction::FORWARD_TRAINING, 1),
        config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

// ====================================================================================================
// defaultConfigSpatialMultiple - provides default configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialMultiple, NhwcFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(128, 64, 16, 16, true, true, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);
    EXPECT_EQ(config.nelements, 128);
    EXPECT_EQ(config.xlocalsize, 16);
    EXPECT_EQ(config.ylocalsize, 8);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NhwcFallbackConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(64, 25, 16, 16, true, true, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 64);
    EXPECT_EQ(config.xlocalsize, 32);
    EXPECT_EQ(config.ylocalsize, 4);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NchwFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(128, 64, 16, 16, false, true, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 128);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 256);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NchwSingleVectorConfig)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(16, 32, 1, 2, false, true, 80), 1, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 16);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 64);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NchwOddSpatialFallback)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(64, 64, 3, 5, false, true, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 64);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 64);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NoConfigAssignedOnFailure)
{
    KernelConfig config;

    // StashValues requirement 12345 is impossible to satisfy
    defaultConfigSpatialMultiple(makeProblem(1, 1, 1, 1, false, true, 80), 12345, config);
    EXPECT_EQ(config.variant, -1);
}

// ====================================================================================================
// The multi-workgroup reduction path stashes intermediate per-channel results into the
// dx output buffer. The stash field count is how many fields it needs there: 4 when the
// backward pass computes stats (mean, variance, dscale, dbias), 2 when they are supplied
// (dscale, dbias only). The applicability gate selects that path only when that many
// fields fit in dx. The shape {2,3,1,1} NHWC fp32 sits at the boundary: its last block
// holds 2 stash fields but not 4. These CPU tests pin that boundary, so the gate keeps
// rejecting the path for any field count whose stash would not fit in dx.
// ====================================================================================================

TEST(TestBatchnormBwdStashFix, NotSavedValueRejectsAsanShapeFromVariant2)
{
    KernelConfig config;

    // dx at {2,3,1,1} holds 2 stash fields, not 4, so a 4-field request (what a
    // stats-computing backward pass needs) makes the gate reject the reduction path.
    defaultConfigSpatialMultiple(makeProblem(2, 3, 1, 1, true, true), 4, config);
    EXPECT_NE(config.variant, 2);
}

TEST(TestBatchnormBwdStashFix, OldValueAcceptedAsanShapeIntoVariant2)
{
    KernelConfig config;

    // A 2-field request fits dx at this shape, so the gate selects the reduction path.
    // This is the boundary the field count must clear: 2 fits where 4 does not.
    defaultConfigSpatialMultiple(makeProblem(2, 3, 1, 1, true, true), 2, config);
    EXPECT_EQ(config.variant, 2);
}

TEST(TestBatchnormBwdStashFix, ApplicabilityGateFlipsOnStashValues)
{
    // The gate the two tests above reach indirectly. At {2,3,1,1} NHWC fp32 the config
    // resolves to ylocalsize=256, zlocalsize=1, nelements=2, whose last block holds 2
    // stash fields but not 4, so the gate accepts a 2-field request and rejects a
    // 4-field one.
    const auto problem = makeProblem(2, 3, 1, 1, true, true);
    EXPECT_TRUE(isSpatialMultipleApplicable(problem, 1, 2, 256, 1, 2));
    EXPECT_FALSE(isSpatialMultipleApplicable(problem, 1, 4, 256, 1, 2));
}

// ====================================================================================================
// Declared tensor alignment. A declared alignment A guarantees only address % A == 0, so it covers an
// N-byte access exactly when N divides A.
// ====================================================================================================

// Bug caught: alignment treated as a minimum (declared >= need) rather than a multiple guarantee,
// so a 24-byte declaration passes for a 16-byte access.
TEST(TestAlignmentCovers, IsAMultipleGuarantee)
{
    EXPECT_TRUE(alignmentCovers(16, 16));
    EXPECT_TRUE(alignmentCovers(16, 4));
    EXPECT_TRUE(alignmentCovers(24, 8));
    EXPECT_FALSE(alignmentCovers(8, 16));
    EXPECT_FALSE(alignmentCovers(24, 16));
    EXPECT_FALSE(alignmentCovers(4, 8));
}

// Bug caught: a missing declaration (0) or an empty need reported as covered.
TEST(TestAlignmentCovers, ZeroGuaranteesNothing)
{
    EXPECT_FALSE(alignmentCovers(0, 4));
    EXPECT_FALSE(alignmentCovers(16, 0));
}

// Bug caught: the declared-alignment default drifting from the 16 bytes every graph declared before
// plugin API 1.3.0, which would change the choice for graphs that never set an alignment.
TEST(TestSpatialAlignment, DefaultsToSixteenBytes)
{
    const SpatialAlignment alignment;
    EXPECT_EQ(alignment.input, 16U);
    EXPECT_EQ(alignment.output, 16U);
    EXPECT_EQ(alignment.channel, 16U);
}

// Bug caught: the per-channel vector width or the fp32 stash held in the output buffer left out
// of the need, so a kernel that loads float4 from a buffer aligned to 4 bytes is accepted.
TEST(TestSpatialAlignmentNeed, MultipleKernelNeedCoversActivationChannelAndStashWidths)
{
    // NHWC single precision at vector size 4: activations, per-channel values and stash are all
    // 16-byte vectors.
    auto need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, true, true), 4);
    EXPECT_EQ(need.input, 16U);
    EXPECT_EQ(need.output, 16U);
    EXPECT_EQ(need.channel, 16U);

    need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, true, true), 1);
    EXPECT_EQ(need.input, 4U);
    EXPECT_EQ(need.output, 4U);
    EXPECT_EQ(need.channel, 4U);

    // NHWC mixed precision: 8-byte activations, but the fp32 stash in the output is a 16-byte
    // vector.
    need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, true, false), 4);
    EXPECT_EQ(need.input, 8U);
    EXPECT_EQ(need.output, 16U);
    EXPECT_EQ(need.channel, 16U);

    // NCHW: per-channel values are single floats whatever the vector size.
    need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, false, true), 4);
    EXPECT_EQ(need.input, 16U);
    EXPECT_EQ(need.output, 16U);
    EXPECT_EQ(need.channel, 4U);

    need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, false, false), 4);
    EXPECT_EQ(need.input, 8U);
    EXPECT_EQ(need.output, 8U);
    EXPECT_EQ(need.channel, 4U);

    // The stash stays an fp32 value in a 2-byte-element output even at vector size 1.
    need = spatialMultipleAlignmentNeed(makeProblem(8, 64, 4, 4, false, false), 1);
    EXPECT_EQ(need.input, 2U);
    EXPECT_EQ(need.output, 4U);
    EXPECT_EQ(need.channel, 4U);
}

// Bug caught: the fixed-width reads of variant 1 left out of the need, or applied outside their
// layout, direction or the hw >= 4096 threshold, so the single kernel runs on an alignment that
// covers only the element.
TEST(TestSpatialAlignmentNeed, SingleKernelWidensOnlyForNchwVariantOne)
{
    // Backward reads x and dy four elements at a time and writes dx two at a time.
    auto need = spatialSingleAlignmentNeed(
        makeProblem(2, 3, 8, 8, false, true, 1, Direction::BACKWARD), 1);
    EXPECT_EQ(need.input, 16U);
    EXPECT_EQ(need.output, 8U);
    EXPECT_EQ(need.channel, 4U);

    need = spatialSingleAlignmentNeed(makeProblem(2, 3, 8, 8, false, false, 1, Direction::BACKWARD),
                                      1);
    EXPECT_EQ(need.input, 8U);
    EXPECT_EQ(need.output, 4U);

    // Forward training reads x four elements at a time from hw = 4096 on, and writes one.
    need = spatialSingleAlignmentNeed(makeProblem(2, 3, 64, 64, false, true), 1);
    EXPECT_EQ(need.input, 16U);
    EXPECT_EQ(need.output, 4U);
    EXPECT_EQ(need.channel, 4U);

    need = spatialSingleAlignmentNeed(makeProblem(2, 3, 63, 65, false, true), 1);
    EXPECT_EQ(need.input, 4U);
    EXPECT_EQ(need.output, 4U);

    // Variants 0 and 3, and NHWC, only access one element at a time.
    for(const int variant : {0, 3})
    {
        need = spatialSingleAlignmentNeed(
            makeProblem(2, 3, 64, 64, false, true, 1, Direction::BACKWARD), variant);
        EXPECT_EQ(need.input, 4U);
        EXPECT_EQ(need.output, 4U);
    }
    need = spatialSingleAlignmentNeed(makeProblem(2, 3, 64, 64, true, true, 1, Direction::BACKWARD),
                                      1);
    EXPECT_EQ(need.input, 4U);
    EXPECT_EQ(need.output, 4U);
}

// Bug caught: NHWC keeping vector size 4 although the per-channel tensors declare less than a
// float4, instead of falling back to the scalar instance.
TEST(TestDefaultConfigSpatialMultiple, NhwcFallsBackToScalarBelowTheVectorNeed)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, config, SpatialAlignment{16, 16, 8});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 128);
}

// Bug caught: the vector width declined at exactly the declared alignment it needs (off by one on
// the boundary), or accepted a power of two below it.
TEST(TestDefaultConfigSpatialMultiple, NhwcKeepsTheVectorSizeAtExactlyItsNeed)
{
    KernelConfig config;
    const auto problem = makeProblem(128, 64, 16, 16, true, true, 80);

    defaultConfigSpatialMultiple(problem, 32, config, SpatialAlignment{16, 16, 16});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);

    config = KernelConfig{};
    defaultConfigSpatialMultiple(problem, 32, config, SpatialAlignment{8, 16, 16});
    EXPECT_EQ(config.vectorsize, 1);
}

// Bug caught: mixed-precision vector instances chosen because the 8-byte activations are covered,
// ignoring that the fp32 stash in the output needs 16 bytes.
TEST(TestDefaultConfigSpatialMultiple, MixedPrecisionNeedsTheStashWidthInTheOutput)
{
    KernelConfig config;
    const auto problem = makeProblem(128, 64, 16, 16, true, false, 80);

    defaultConfigSpatialMultiple(problem, 32, config, SpatialAlignment{8, 16, 16});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);

    config = KernelConfig{};
    defaultConfigSpatialMultiple(problem, 32, config, SpatialAlignment{8, 8, 16});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
}

// Bug caught: NCHW assigning a configuration whose accesses the declared alignment does not cover,
// instead of leaving the variant unset so the caller can pick another kernel.
TEST(TestDefaultConfigSpatialMultiple, NchwAssignsNothingBelowTheElementAlignment)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(
        makeProblem(128, 64, 16, 16, false, true, 80), 32, config, SpatialAlignment{2, 2, 2});
    EXPECT_EQ(config.variant, -1);

    defaultConfigSpatialMultiple(
        makeProblem(128, 64, 16, 16, false, true, 80), 32, config, SpatialAlignment{4, 4, 4});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
}

// Bug caught: graphs at the default alignment routed differently than before the alignment was
// considered.
TEST(TestSelectSpatialConfig, DefaultAlignmentKeepsTheHeuristicChoice)
{
    KernelConfig config;

    ASSERT_TRUE(selectSpatialConfig(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, SpatialAlignment{}, config));
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);

    // Forward training NCHW at hw = 4096 is a single kernel with wide reads.
    ASSERT_TRUE(selectSpatialConfig(makeProblem(4, 3, 64, 64, false, true), 2, {}, config));
    EXPECT_EQ(config.variant, 1);

    ASSERT_TRUE(selectSpatialConfig(
        makeProblem(2, 3, 17, 241, false, true, 1, Direction::BACKWARD), 2, {}, config));
    EXPECT_EQ(config.variant, 1);

    // A large backward problem prefers spatial multiple.
    ASSERT_TRUE(selectSpatialConfig(
        makeProblem(64, 1, 1024, 1024, false, true, 1, Direction::BACKWARD), 4, {}, config));
    EXPECT_EQ(config.variant, 2);
}

// Bug caught: a half-precision output declaring only its element size given the stash kernels, whose
// fp32 partial sums need 4-byte alignment, instead of the single kernel that stashes nothing.
TEST(TestSelectSpatialConfig, OutputBelowTheStashWidthSelectsTheSingleKernel)
{
    KernelConfig config;
    const auto problem = makeProblem(128, 64, 16, 16, true, false, 80);

    ASSERT_TRUE(selectSpatialConfig(problem, 32, SpatialAlignment{16, 2, 16}, config));
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);

    // At the 4 bytes the stash needs, the scalar stash kernel is back.
    ASSERT_TRUE(selectSpatialConfig(problem, 32, SpatialAlignment{16, 4, 16}, config));
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
}

// Bug caught: the single kernel's fixed four-element x reads run on an alignment that covers only
// narrower accesses, instead of the spatial multiple kernel at a narrower vector size.
TEST(TestSelectSpatialConfig, WideReadsFallBackToSpatialMultipleAtANarrowerVectorSize)
{
    KernelConfig config;
    const auto problem = makeProblem(4, 3, 64, 64, false, true);

    // Exactly the 16 bytes of the four-element read keeps the single kernel.
    ASSERT_TRUE(selectSpatialConfig(problem, 2, SpatialAlignment{16, 4, 4}, config));
    EXPECT_EQ(config.variant, 1);

    ASSERT_TRUE(selectSpatialConfig(problem, 2, SpatialAlignment{8, 16, 16}, config));
    EXPECT_EQ(config.variant, 2);
    EXPECT_LT(config.vectorsize, 4);
}

// Bug caught: a problem that has no applicable narrower kernel still given the wide-read kernel,
// or declined at exactly the alignment the wide reads need.
TEST(TestSelectSpatialConfig, WideReadsAreDeclinedWhenNoNarrowerKernelApplies)
{
    KernelConfig config;
    // hw = 17 * 241 = 4097 reads wide, and a single sample leaves the spatial multiple scalar
    // instance no room to stash.
    const auto problem = makeProblem(1, 2, 17, 241, false, true);

    EXPECT_FALSE(selectSpatialConfig(problem, 2, SpatialAlignment{8, 16, 16}, config));
    EXPECT_EQ(config.variant, -1);

    EXPECT_TRUE(selectSpatialConfig(problem, 2, SpatialAlignment{16, 4, 4}, config));
    EXPECT_EQ(config.variant, 1);
}

// Bug caught: the backward single kernel's read width (four elements of x and dy) or write width
// (two elements of dx) left out, or applied to the wrong tensor group.
TEST(TestSelectSpatialConfig, BackwardSingleKernelNeedsItsReadAndWriteWidths)
{
    KernelConfig config;
    const auto problem = makeProblem(1, 2, 17, 241, false, true, 1, Direction::BACKWARD);

    EXPECT_TRUE(selectSpatialConfig(problem, 2, SpatialAlignment{16, 8, 4}, config));
    EXPECT_EQ(config.variant, 1);

    EXPECT_FALSE(selectSpatialConfig(problem, 2, SpatialAlignment{16, 4, 4}, config));
    EXPECT_FALSE(selectSpatialConfig(problem, 2, SpatialAlignment{8, 8, 4}, config));
}

// Bug caught: no problem may run when a tensor declares less than its element size, however the
// kernel is chosen.
TEST(TestSelectSpatialConfig, BelowTheElementAlignmentNothingRuns)
{
    KernelConfig config;

    EXPECT_FALSE(selectSpatialConfig(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, SpatialAlignment{2, 4, 4}, config));
    EXPECT_FALSE(selectSpatialConfig(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, SpatialAlignment{4, 2, 4}, config));
    EXPECT_FALSE(selectSpatialConfig(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, SpatialAlignment{4, 4, 2}, config));
    EXPECT_TRUE(selectSpatialConfig(
        makeProblem(128, 64, 16, 16, true, true, 80), 32, SpatialAlignment{4, 4, 4}, config));
}

// Bug caught: the decline depending on the device's workgroup budget, so the applicability check,
// which cannot know it, and the plan compile could disagree.
TEST(TestSelectSpatialConfig, NhwcOutcomeDoesNotDependOnMinWorkgroups)
{
    for(const size_t minWorkgroups : {size_t{1}, size_t{80}, size_t{1000}})
    {
        KernelConfig config;
        const auto problem = makeProblem(128, 64, 16, 16, true, true, minWorkgroups);

        EXPECT_TRUE(selectSpatialConfig(problem, 32, SpatialAlignment{4, 4, 4}, config));
        EXPECT_FALSE(selectSpatialConfig(problem, 32, SpatialAlignment{4, 4, 2}, config));
    }
}
