// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormCommon.hpp"

using namespace hip_kernel_provider::batchnorm;

namespace
{

ProblemDescription makeProblem(size_t batches,
                               size_t channels,
                               size_t spatialSize,
                               bool isLayoutChannelLast,
                               bool isFp32,
                               bool isUpcast,
                               size_t minWorkgroups = 1,
                               Direction direction = Direction::FORWARD_TRAINING)
{
    return {batches,
            channels,
            spatialSize,
            isLayoutChannelLast,
            !isFp32 && !isUpcast,
            false,
            !isFp32 && isUpcast,
            false,
            direction,
            minWorkgroups};
}

} // namespace

TEST(TestProblemDescription, ExposesProblemProperties)
{
    const ProblemDescription problem(
        32, 64, 128, true, false, true, false, false, Direction::BACKWARD, 42);

    EXPECT_EQ(problem.batches(), 32);
    EXPECT_EQ(problem.channels(), 64);
    EXPECT_EQ(problem.spatialSize(), 128);
    EXPECT_TRUE(problem.isLayoutChannelLast());
    EXPECT_FALSE(problem.useFp16Mix());
    EXPECT_TRUE(problem.useBfp16Mix());
    EXPECT_FALSE(problem.useFp16Upcast());
    EXPECT_FALSE(problem.useBfp16Upcast());
    EXPECT_FALSE(problem.useFp32());
    EXPECT_EQ(problem.direction(), Direction::BACKWARD);
    EXPECT_EQ(problem.minWorkgroups(), 42);
}

TEST(TestProblemDescription, ComputesDerivedDimensions)
{
    const ProblemDescription problem(
        32, 64, 128, false, false, false, false, false, Direction::FORWARD_TRAINING, 1);

    EXPECT_TRUE(problem.useFp32());
    EXPECT_EQ(problem.inCstride(), 128);
    EXPECT_EQ(problem.inChannelSliceSize(), 4096);
    EXPECT_EQ(problem.inBatchSize(), 8192);
    EXPECT_EQ(problem.inTensorSize(), 262144);
}

// ============================================================================
// getLocalConfigChannelLast - computes local workgroup size for NLC layout
// ============================================================================

TEST(TestGetLocalConfigNlcFp32, ProducesSquareWorkgroup)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigChannelLast(makeProblem(1, 128, 256, true, true, false), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

TEST(TestGetLocalConfigNlc, MixedPrecisionUsesWiderX)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigChannelLast(makeProblem(1, 128, 256, true, false, false), 4, x, y);
    EXPECT_EQ(x, 32);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNlc, UpcastUsesWiderX)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigChannelLast(makeProblem(1, 128, 256, true, false, true), 4, x, y);
    EXPECT_EQ(x, 32);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNlc, HandlesVectorsizeScaling)
{
    size_t x = 0;
    size_t y = 0;

    // With vectorsize of 8, the maxlocalsize should be reduced
    getLocalConfigChannelLast(makeProblem(1, 1024, 256, true, true, false), 8, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNlc, HandlesIncreasedMinWorkgroups)
{
    size_t xSmallWG = 0;
    size_t ySmallWG = 0;
    getLocalConfigChannelLast(
        makeProblem(1, 64, 4096, true, true, false, 10), 1, xSmallWG, ySmallWG);

    // With high minWorkgroups requirement, the workgroup size should be reduced
    size_t xLargeWG = 0;
    size_t yLargeWG = 0;
    getLocalConfigChannelLast(
        makeProblem(1, 64, 4096, true, true, false, 10000), 1, xLargeWG, yLargeWG);

    EXPECT_EQ(xSmallWG, xLargeWG);
    EXPECT_LT(yLargeWG, ySmallWG);
}

TEST(TestGetLocalConfigNlc, HandlesSmallChannels)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigChannelLast(makeProblem(1, 2, 256, true, true, false), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);
}

TEST(TestGetLocalConfigNlc, HandlesSmallInitialMaxLocalsize)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigChannelLast(makeProblem(1, 128, 256, true, true, false), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

// ====================================================================================================
// getSpatialMultipleConfig - computes workgroup configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestGetSpatialMultipleConfig, NlcReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 3, 256, true, true, false, 80), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NclReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 15, false, true, false, 80), 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NlcCalculatesWorkgroupSize)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 256, true, true, false, 80), 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetSpatialMultipleConfig, NclHandlesLargeSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 4096, false, true, false, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1024);
}

TEST(TestGetSpatialMultipleConfig, NclScalesDownForSmallSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(makeProblem(1, 64, 128, false, true, false, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);

    getSpatialMultipleConfig(makeProblem(1, 64, 32, false, true, false, 80), 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 64);
}

// ========================================================================================
// isSpatialMultipleApplicable - checks if spatial multiple implementation can be used
// ========================================================================================

TEST(TestIsSpatialMultipleApplicable, NlcVectorAlignmentFail)
{
    // If C is not divisible by vectorsize with channel last layout, it should fail immediately
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(64, 63, 256, true, true, false), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, NclSpatialAlignmentFail)
{
    // If H*W is not divisible by vectorsize with NCHW layout, it should fail immediately
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(64, 64, 15, false, true, false), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInSpatialDimension)
{
    // The last block of spatial dimension is large enough to hold stashed values
    EXPECT_TRUE(
        isSpatialMultipleApplicable(makeProblem(64, 64, 256, true, true, false), 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInBatchDimension)
{
    // Even if the spatial remainder is small, it works if the batch remainder is large enough
    EXPECT_TRUE(
        isSpatialMultipleApplicable(makeProblem(128, 64, 10, true, true, false), 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashDoesNotFitInSpatialOrBatchDimension)
{
    // Both spatial and batch remainders are smaller than the required stash values
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(10, 64, 10, true, true, false), 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, MixedPrecisionOddCFailsOnSmallZ)
{
    // For FP16/BF16, if C is odd, the intermediate results MUST fit in the batch dimension
    EXPECT_FALSE(isSpatialMultipleApplicable(
        makeProblem(10, 65, 256, true, false, false), 1, 16, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, UpcastOddCFailsOnSmallZ)
{
    // For FP16/BF16, if C is odd, the intermediate results MUST fit in the batch dimension
    EXPECT_FALSE(
        isSpatialMultipleApplicable(makeProblem(10, 65, 256, true, false, true), 1, 16, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, MixedPrecisionOddVectorizedChannelsFailsOnSmallZ)
{
    // C is even but C / vectorsize is odd (12 / 4 = 3), so the stash goes to the batch
    // dimension, which is too small here (MIOpen #13998)
    EXPECT_FALSE(isSpatialMultipleApplicable(
        makeProblem(10, 12, 256, true, false, false), 4, 16, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, MixedPrecisionEvenVectorizedChannelsPassesOnSmallZ)
{
    // 12 / 2 = 6 is even, so the stash fits in the spatial dimension
    EXPECT_TRUE(isSpatialMultipleApplicable(
        makeProblem(10, 12, 256, true, false, false), 2, 16, 64, 1, 64));
}

// ====================================================================================================
// useMultiple - checks if spatial multiple implementation should be used based on heuristics
// ====================================================================================================

TEST(TestUseMultiple, BackwardSmallSpatialReturnsFalse)
{
    EXPECT_FALSE(useMultiple(makeProblem(64, 1, 256, false, true, false, 1, Direction::BACKWARD)));
}

TEST(TestUseMultiple, BackwardLargeProblemReturnsTrue)
{
    EXPECT_TRUE(
        useMultiple(makeProblem(64, 1, 1048576, false, true, false, 1, Direction::BACKWARD)));
}

TEST(TestUseMultiple, ForwardTrainingSmallProblemReturnsFalse)
{
    EXPECT_FALSE(
        useMultiple(makeProblem(2, 1, 64, false, true, false, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, ForwardTrainingLargeBatchReturnsTrue)
{
    EXPECT_TRUE(useMultiple(
        makeProblem(1024, 1, 1024, false, true, false, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, ForwardTrainingMixedPrecisionHeuristic)
{
    EXPECT_FALSE(useMultiple(
        makeProblem(128, 1, 1024, false, false, false, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, ForwardTrainingUpcastHeuristic)
{
    EXPECT_FALSE(
        useMultiple(makeProblem(128, 1, 1024, false, false, true, 1, Direction::FORWARD_TRAINING)));
}

TEST(TestUseMultiple, NhwcAlwaysReturnsTrue)
{
    EXPECT_TRUE(useMultiple(makeProblem(2, 1, 64, true, true, false)));
}

// ====================================================================================================
// getStashMethod - determines the stash method to be used based on problem configuration
// ====================================================================================================

TEST(TestGetStashMethod, ReturnsMethodZeroWhenSpatialFits)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 1024, false, true, false), 32, 1024, 1, 64, 1), 0);
}

TEST(TestGetStashMethod, ReturnsMethodOneWhenBatchStashRequired)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 10, false, true, false), 32, 64, 1, 64, 1), 1);
}

TEST(TestGetStashMethod, ReturnsMethodTwoForNhwcOddCMixedPrecision)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 65, 1024, true, false, false), 16, 1024, 1, 64, 1), 2);
}

TEST(TestGetStashMethod, ReturnsMethodZeroForNhwcOddCUpcast)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 65, 1024, true, false, true), 16, 1024, 1, 64, 1), 0);
}

TEST(TestGetStashMethod, MixedPrecisionScalesStashValues)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 20, false, false, false), 16, 64, 1, 64, 1), 1);
}

TEST(TestGetStashMethod, UpcastDoesntScaleStashValues)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 64, 20, false, false, true), 16, 64, 1, 64, 1), 0);
}

TEST(TestGetStashMethod, ReturnsMethodTwoForNhwcOddVectorizedChannelsMixedPrecision)
{
    // C is even but C / vectorsize is odd (12 / 4 = 3); must agree with getStashIndex
    // (MIOpen #13998)
    EXPECT_EQ(getStashMethod(makeProblem(64, 12, 1024, true, false, false), 16, 1024, 1, 64, 4), 2);
}

TEST(TestGetStashMethod, ReturnsMethodZeroForNhwcEvenVectorizedChannelsMixedPrecision)
{
    EXPECT_EQ(getStashMethod(makeProblem(64, 12, 1024, true, false, false), 16, 1024, 1, 64, 2), 0);
}

// ====================================================================================================
// defaultConfigSpatialSingle - provides default configuration for spatial single implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialSingle, NlcDefaultVariantOne)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(64, 1, 256, true, false, false, false, false, Direction::BACKWARD, 1),
        config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NclBackwardSmallSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(32, 1, 256, false, false, false, false, false, Direction::BACKWARD, 1),
        config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NclBackwardSmallSpatialLargeBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(128, 1, 200, false, false, false, false, false, Direction::BACKWARD, 1),
        config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NclBackwardMidSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(16, 1, 600, false, false, false, false, false, Direction::BACKWARD, 1),
        config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NclForwardLargeSpatialOrMixed)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(
            64, 1, 1024, false, true, false, false, false, Direction::FORWARD_TRAINING, 1),
        config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NclForwardSmallDefault)
{
    KernelConfig config;

    defaultConfigSpatialSingle(
        ProblemDescription(
            10, 1, 64, false, false, false, false, false, Direction::FORWARD_TRAINING, 1),
        config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

// ====================================================================================================
// defaultConfigSpatialMultiple - provides default configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialMultiple, NlcFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(128, 64, 256, true, true, false, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);
    EXPECT_EQ(config.batchElements, 128);
    EXPECT_EQ(config.xlocalsize, 16);
    EXPECT_EQ(config.ylocalsize, 8);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NlcFallbackConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(64, 25, 256, true, true, false, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.batchElements, 64);
    EXPECT_EQ(config.xlocalsize, 32);
    EXPECT_EQ(config.ylocalsize, 4);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NclFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(128, 64, 256, false, true, false, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.batchElements, 128);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 256);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NclSingleVectorConfig)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(16, 32, 2, false, true, false, 80), 1, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.batchElements, 16);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 64);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NclOddSpatialFallback)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(makeProblem(64, 64, 15, false, true, false, 80), 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.batchElements, 64);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 64);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NoConfigAssignedOnFailure)
{
    KernelConfig config;

    // StashValues requirement 12345 is impossible to satisfy
    defaultConfigSpatialMultiple(makeProblem(1, 1, 1, false, true, false, 80), 12345, config);
    EXPECT_EQ(config.variant, -1);
}

// ====================================================================================================
// The multi-workgroup reduction path stashes intermediate per-channel results into the
// dx output buffer. The stash field count is how many fields it needs there: 4 when the
// backward pass computes stats (mean, variance, dscale, dbias), 2 when they are supplied
// (dscale, dbias only). The applicability gate selects that path only when that many
// fields fit in dx. The shape {2,3,1} NLC fp32 sits at the boundary: its last block
// holds 2 stash fields but not 4. These CPU tests pin that boundary, so the gate keeps
// rejecting the path for any field count whose stash would not fit in dx.
// ====================================================================================================

TEST(TestBatchnormBwdStashFix, NotSavedValueRejectsAsanShapeFromVariant2)
{
    KernelConfig config;

    // dx at {2,3,1} holds 2 stash fields, not 4, so a 4-field request (what a
    // stats-computing backward pass needs) makes the gate reject the reduction path.
    defaultConfigSpatialMultiple(makeProblem(2, 3, 1, true, true, false), 4, config);
    EXPECT_NE(config.variant, 2);
}

TEST(TestBatchnormBwdStashFix, OldValueAcceptedAsanShapeIntoVariant2)
{
    KernelConfig config;

    // A 2-field request fits dx at this shape, so the gate selects the reduction path.
    // This is the boundary the field count must clear: 2 fits where 4 does not.
    defaultConfigSpatialMultiple(makeProblem(2, 3, 1, true, true, false), 2, config);
    EXPECT_EQ(config.variant, 2);
}

TEST(TestBatchnormBwdStashFix, ApplicabilityGateFlipsOnStashValues)
{
    // The gate the two tests above reach indirectly. At {2,3,1} NLC fp32 the config
    // resolves to ylocalsize=256, zlocalsize=1, batchElements=2, whose last block holds 2
    // stash fields but not 4, so the gate accepts a 2-field request and rejects a
    // 4-field one.
    const auto problem = makeProblem(2, 3, 1, true, true, false);
    EXPECT_TRUE(isSpatialMultipleApplicable(problem, 1, 2, 256, 1, 2));
    EXPECT_FALSE(isSpatialMultipleApplicable(problem, 1, 4, 256, 1, 2));
}

// ====================================================================================================
// defaultSpatialConfig - the heuristics' choice as an explicit configuration
// ====================================================================================================

namespace
{

// The launch configuration the training and backward builds chose from the heuristics before
// they took a BatchnormSpatialConfig.
KernelConfig heuristicKernelConfig(const ProblemDescription& problem, unsigned int stashValues)
{
    KernelConfig config;
    if(useMultiple(problem))
    {
        defaultConfigSpatialMultiple(problem, stashValues, config);
        if(config.variant == -1)
        {
            defaultConfigSpatialSingle(problem, config);
        }
    }
    else
    {
        defaultConfigSpatialSingle(problem, config);
    }
    return config;
}

struct DefaultSpatialConfigCase
{
    const char* name;
    ProblemDescription problem;
    unsigned int stashValues;
    BatchnormSpatialConfig expected;
};

} // namespace

TEST(TestDefaultSpatialConfig, MatchesTheHeuristicsOnEveryBranch)
{
    const auto fwd = Direction::FORWARD_TRAINING;
    const auto bwd = Direction::BACKWARD;
    const std::vector<DefaultSpatialConfigCase> cases = {
        // Channel last always tries spatial multiple: vector size 4, then 1, then variant 1
        {"NhwcMultipleVector4",
         makeProblem(128, 64, 256, true, true, false, 80, fwd),
         2,
         {2, 0, 4}},
        {"NhwcMultipleVector1", makeProblem(64, 25, 256, true, true, false, 80, fwd), 2, {2, 0, 1}},
        {"NhwcSingleFallback", makeProblem(1, 3, 1, true, true, false, 1, fwd), 2, {1, 256, 1}},
        {"NhwcBwdSingleFallback", makeProblem(2, 3, 1, true, true, false, 1, bwd), 4, {1, 256, 1}},
        {"NhwcBwdMultipleSavedStats",
         makeProblem(2, 3, 1, true, true, false, 1, bwd),
         2,
         {2, 0, 1}},
        // Forward NCHW, spatial single
        {"FwdVariant3", makeProblem(2, 1, 600, false, true, false, 1, fwd), 2, {3, 1024, 1}},
        {"FwdVariant1LargeSpatial",
         makeProblem(2, 1, 2048, false, true, false, 1, fwd),
         2,
         {1, 1024, 1}},
        {"FwdVariant1Mixed",
         makeProblem(64, 1, 1024, false, false, false, 1, fwd),
         2,
         {1, 1024, 1}},
        {"FwdVariant0Small", makeProblem(10, 1, 64, false, true, false, 1, fwd), 2, {0, 256, 1}},
        {"FwdVariant0Batch256Spatial64",
         makeProblem(256, 1, 64, false, true, false, 1, fwd),
         2,
         {0, 256, 1}},
        {"FwdVariant0Batch256Spatial128",
         makeProblem(256, 1, 128, false, true, false, 1, fwd),
         2,
         {0, 1024, 1}},
        // A channel of 32M values keeps variant 1 for a spatial size beyond the workgroup of
        // variants 0 and 3, for fp32 with N <= 2 and an upcast with N <= 768, which spatial
        // multiple does not take
        {"FwdVariant1LargeSpatialLargeChannel",
         makeProblem(2, 1, 16777216, false, true, false, 1, fwd),
         2,
         {1, 1024, 1}},
        {"FwdVariant1UpcastLargeSpatialLargeChannel",
         makeProblem(32, 2, 1048576, false, false, true, 1, fwd),
         2,
         {1, 1024, 1}},
        // Forward NCHW, spatial multiple: the smallest applicable vector size is kept
        {"FwdNchwMultiple", makeProblem(32, 8, 576, false, true, false, 36, fwd), 2, {2, 0, 1}},
        // Backward NCHW, spatial single
        {"BwdVariant1Fp16MixMidSpatial",
         makeProblem(64, 1, 100, false, false, false, 1, bwd),
         4,
         {1, 256, 1}},
        {"BwdVariant1LargeSpatial",
         makeProblem(1, 3, 50176, false, true, false, 1, bwd),
         4,
         {1, 1024, 1}},
        {"BwdVariant1MidSpatialLargeBatch",
         makeProblem(32, 1, 600, false, true, false, 1, bwd),
         4,
         {1, 1024, 1}},
        {"BwdVariant3MidSpatialSmallBatch",
         makeProblem(16, 1, 600, false, true, false, 1, bwd),
         4,
         {3, 1024, 1}},
        {"BwdVariant3SmallSpatialLargeBatch",
         makeProblem(128, 1, 200, false, true, false, 1, bwd),
         4,
         {3, 256, 1}},
        {"BwdVariant0SmallSpatial",
         makeProblem(32, 1, 256, false, true, false, 1, bwd),
         4,
         {0, 1024, 1}},
        // Backward NCHW, spatial multiple, and its fallback when no vector size applies
        {"BwdNchwMultiple", makeProblem(64, 1, 1048576, false, true, false, 1, bwd), 4, {2, 0, 1}},
        {"BwdNchwMultipleFallback",
         makeProblem(2, 1, 16777217, false, true, false, 1, bwd),
         4,
         {1, 1024, 1}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const auto config = defaultSpatialConfig(testCase.problem, testCase.stashValues);
        EXPECT_EQ(config, testCase.expected);

        // The builds launch what spatialKernelConfig() derives from the configuration, which
        // must be what the heuristics chose.
        const auto derived = spatialKernelConfig(testCase.problem, config);
        const auto heuristic = heuristicKernelConfig(testCase.problem, testCase.stashValues);
        EXPECT_EQ(derived.variant, heuristic.variant);
        EXPECT_EQ(derived.vectorsize, heuristic.vectorsize);
        EXPECT_EQ(derived.xlocalsize, heuristic.xlocalsize);
        EXPECT_EQ(derived.ylocalsize, heuristic.ylocalsize);
        EXPECT_EQ(derived.zlocalsize, heuristic.zlocalsize);
        EXPECT_EQ(derived.batchElements, heuristic.batchElements);
    }
}

TEST(TestSpatialKernelConfig, DerivesTheMultipleGeometryFromTheVectorSize)
{
    const auto problem = makeProblem(128, 64, 256, true, true, false, 80);

    const auto config = spatialKernelConfig(problem, {2, 0, 4});
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);
    EXPECT_EQ(config.xlocalsize, 16);
    EXPECT_EQ(config.ylocalsize, 8);
    EXPECT_EQ(config.zlocalsize, 1);
    EXPECT_EQ(config.batchElements, 128);
}

TEST(TestSpatialKernelConfig, KeepsAUnitGeometryForSingleVariants)
{
    const auto config
        = spatialKernelConfig(makeProblem(128, 64, 256, false, true, false), {3, 256, 1});
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 1);
    EXPECT_EQ(config.zlocalsize, 1);
    EXPECT_EQ(config.batchElements, 1);
}

// ====================================================================================================
// isSpatialConfigApplicable - which configurations can run a problem
// ====================================================================================================

TEST(TestSpatialConfigApplicable, Variants0And3NeedChannelFirstAndSpatialWithinTheWorkgroup)
{
    const auto spatial256 = makeProblem(32, 1, 256, false, true, false, 1, Direction::BACKWARD);
    EXPECT_TRUE(isSpatialConfigApplicable(spatial256, 4, {0, 256, 1}));
    EXPECT_TRUE(isSpatialConfigApplicable(spatial256, 4, {3, 256, 1}));

    const auto spatial600 = makeProblem(32, 1, 600, false, true, false, 1, Direction::BACKWARD);
    EXPECT_FALSE(isSpatialConfigApplicable(spatial600, 4, {0, 256, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(spatial600, 4, {3, 256, 1}));
    EXPECT_TRUE(isSpatialConfigApplicable(spatial600, 4, {0, 1024, 1}));
    EXPECT_TRUE(isSpatialConfigApplicable(spatial600, 4, {3, 1024, 1}));
    EXPECT_TRUE(isSpatialConfigApplicable(spatial600, 4, {1, 256, 1}));

    const auto nhwc = makeProblem(32, 4, 64, true, true, false, 1, Direction::BACKWARD);
    EXPECT_FALSE(isSpatialConfigApplicable(nhwc, 4, {0, 1024, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(nhwc, 4, {3, 1024, 1}));
    EXPECT_TRUE(isSpatialConfigApplicable(nhwc, 4, {1, 1024, 1}));
}

TEST(TestSpatialConfigApplicable, RejectsValuesTheBuildsNeverUse)
{
    const auto problem = makeProblem(32, 4, 64, false, true, false);
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {-1, 1024, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {4, 1024, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {1, 512, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {1, 1024, 2}));
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {2, 0, 3}));
    EXPECT_FALSE(isSpatialConfigApplicable(problem, 2, {2, 0, 8}));
}

TEST(TestSpatialConfigApplicable, Variant2FollowsIsSpatialMultipleApplicable)
{
    // The stash boundary of TestBatchnormBwdStashFix: 2 stash fields fit, 4 do not
    const auto boundary = makeProblem(2, 3, 1, true, true, false);
    EXPECT_TRUE(isSpatialConfigApplicable(boundary, 2, {2, 0, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(boundary, 4, {2, 0, 1}));
    // C = 3 is not a multiple of 4
    EXPECT_FALSE(isSpatialConfigApplicable(boundary, 2, {2, 0, 4}));

    // An odd spatial size vectorizes only by 1
    const auto oddSpatial = makeProblem(64, 64, 15, false, true, false);
    EXPECT_TRUE(isSpatialConfigApplicable(oddSpatial, 2, {2, 0, 1}));
    EXPECT_FALSE(isSpatialConfigApplicable(oddSpatial, 2, {2, 0, 2}));
    EXPECT_FALSE(isSpatialConfigApplicable(oddSpatial, 2, {2, 0, 4}));
}

TEST(TestSpatialConfigApplicable, AcceptsTheDefaultOfEveryProblem)
{
    // Spans the boundaries of useMultiple(), defaultConfigSpatialSingle() and
    // defaultSpatialSingleGroupSize(), including channels of 32M values or more with a spatial
    // size beyond 1024, where the forward heuristics once fell back to variant 0.
    const std::vector<size_t> batches = {1, 2, 3, 31, 32, 64, 65, 255, 256, 257, 768, 769};
    const std::vector<size_t> spatialSizes
        = {1,   35,  60,  61,  100,  150,  151,   160,     161,      200,
           255, 256, 512, 513, 1024, 1025, 43691, 1048576, 16777216, 33554432};
    // fp16 mix, bf16 mix, fp16 upcast, bf16 upcast; none set is fp32
    const std::vector<std::array<bool, 4>> precisions = {{false, false, false, false},
                                                         {true, false, false, false},
                                                         {false, true, false, false},
                                                         {false, false, true, false},
                                                         {false, false, false, true}};
    const std::vector<std::pair<Direction, unsigned int>> directions
        = {{Direction::FORWARD_TRAINING, 2}, {Direction::BACKWARD, 2}, {Direction::BACKWARD, 4}};
    constexpr size_t CHANNELS = 2;

    for(const auto n : batches)
    {
        for(const auto spatialSize : spatialSizes)
        {
            // The kernels index the tensor with 32-bit unsigned arithmetic
            if(n * CHANNELS * spatialSize > std::numeric_limits<unsigned int>::max())
            {
                continue;
            }
            for(const bool channelLast : {false, true})
            {
                for(const auto& precision : precisions)
                {
                    for(const auto& [direction, stashValues] : directions)
                    {
                        const ProblemDescription problem(n,
                                                         CHANNELS,
                                                         spatialSize,
                                                         channelLast,
                                                         precision[0],
                                                         precision[1],
                                                         precision[2],
                                                         precision[3],
                                                         direction,
                                                         36);
                        const auto config = defaultSpatialConfig(problem, stashValues);
                        EXPECT_TRUE(isSpatialConfigApplicable(problem, stashValues, config))
                            << "N " << n << ", spatial size " << spatialSize
                            << (channelLast ? ", channel last" : ", channel first")
                            << ", precision flags " << precision[0] << precision[1] << precision[2]
                            << precision[3] << ", direction " << static_cast<int>(direction)
                            << ", stash values " << stashValues << ": variant " << config.variant
                            << " of " << config.groupSize << " threads, vector size "
                            << config.vectorSize;
                    }
                }
            }
        }
    }
}

// ====================================================================================================
// defaultInferenceConfig / isInferenceConfigApplicable - batchnorm inference vector size
// ====================================================================================================

TEST(TestDefaultInferenceConfig, IsTheLargestVectorSizeDividingTheVectorizedDimension)
{
    EXPECT_EQ(defaultInferenceConfig(true, 16, 7).vectorSize, 4);
    EXPECT_EQ(defaultInferenceConfig(true, 6, 8).vectorSize, 2);
    EXPECT_EQ(defaultInferenceConfig(true, 3, 4).vectorSize, 1);
    EXPECT_EQ(defaultInferenceConfig(false, 3, 8).vectorSize, 4);
    EXPECT_EQ(defaultInferenceConfig(false, 16, 10).vectorSize, 2);
    EXPECT_EQ(defaultInferenceConfig(false, 8, 7).vectorSize, 1);
}

TEST(TestInferenceConfigApplicable, NeedsAVectorSizeDividingTheVectorizedDimension)
{
    EXPECT_TRUE(isInferenceConfigApplicable(true, 16, 7, {4}));
    EXPECT_FALSE(isInferenceConfigApplicable(true, 6, 8, {4}));
    EXPECT_TRUE(isInferenceConfigApplicable(true, 6, 8, {2}));
    EXPECT_TRUE(isInferenceConfigApplicable(true, 3, 4, {1}));
    EXPECT_TRUE(isInferenceConfigApplicable(false, 3, 8, {4}));
    EXPECT_FALSE(isInferenceConfigApplicable(false, 16, 10, {4}));
    EXPECT_TRUE(isInferenceConfigApplicable(false, 16, 10, {2}));
    EXPECT_FALSE(isInferenceConfigApplicable(false, 8, 16, {8}));
    EXPECT_FALSE(isInferenceConfigApplicable(false, 8, 16, {0}));
}
