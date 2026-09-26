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
