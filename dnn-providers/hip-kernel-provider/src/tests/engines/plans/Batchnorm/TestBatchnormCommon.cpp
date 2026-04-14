// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>

#include "engines/plans/batchnorm/BatchnormCommon.hpp"

using namespace hip_kernel_provider::batchnorm;

// ============================================================================
// getLocalConfigNHWC - computes local workgroup size for NHWC layout
// ============================================================================

TEST(TestGetLocalConfigNHWC, ForFP32DataType)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(128, 16, 16, true, 1, 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

TEST(TestGetLocalConfigNHWC, IfNotFP32DataType)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(128, 16, 16, false, 1, 4, x, y);
    EXPECT_EQ(x, 32);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNHWC, HandlesVectorsizeScaling)
{
    size_t x = 0;
    size_t y = 0;

    // With vectorsize of 8, the maxlocalsize should be reduced
    getLocalConfigNHWC(1024, 16, 16, true, 1, 8, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetLocalConfigNHWC, HandlesIncreasedMinWorkgroups)
{
    size_t xSmallWG = 0;
    size_t ySmallWG = 0;
    getLocalConfigNHWC(64, 64, 64, true, 10, 1, xSmallWG, ySmallWG);

    // With high minWorkgroups requirement, the workgroup size should be reduced
    size_t xLargeWG = 0;
    size_t yLargeWG = 0;
    getLocalConfigNHWC(64, 64, 64, true, 10000, 1, xLargeWG, yLargeWG);

    EXPECT_EQ(xSmallWG, xLargeWG);
    EXPECT_LT(yLargeWG, ySmallWG);
}

TEST(TestGetLocalConfigNHWC, HandlesSmallChannels)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(2, 16, 16, true, 1, 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);
}

TEST(TestGetLocalConfigNHWC, HandlesSmallInitialMaxLocalsize)
{
    size_t x = 0;
    size_t y = 0;

    getLocalConfigNHWC(128, 16, 16, true, 1, 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 16);
}

// ====================================================================================================
// getSpatialMultipleConfig - computes workgroup configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestGetSpatialMultipleConfig, NHWCReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(3, 16, 16, true, true, 80, 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NCHWReturnsDefaultOnMisalignment)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(64, 3, 5, false, true, 80, 4, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1);
}

TEST(TestGetSpatialMultipleConfig, NHWCCalculatesWorkgroupSize)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(64, 16, 16, true, true, 80, 4, x, y);
    EXPECT_EQ(x, 16);
    EXPECT_EQ(y, 8);
}

TEST(TestGetSpatialMultipleConfig, NCHWHandlesLargeSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(64, 64, 64, false, true, 80, 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 1024);
}

TEST(TestGetSpatialMultipleConfig, NCHWScalesDownForSmallSpatial)
{
    size_t x = 0;
    size_t y = 0;

    getSpatialMultipleConfig(64, 8, 16, false, true, 80, 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 128);

    getSpatialMultipleConfig(64, 4, 8, false, true, 80, 1, x, y);
    EXPECT_EQ(x, 1);
    EXPECT_EQ(y, 64);
}

// ========================================================================================
// isSpatialMultipleApplicable - checks if spatial multiple implementation can be used
// ========================================================================================

TEST(TestIsSpatialMultipleApplicable, NHWCVectorAlignmentFail)
{
    // If C is not divisible by vectorsize with NHWC layout, it should fail immediately
    EXPECT_FALSE(isSpatialMultipleApplicable(64, 63, 16, 16, true, true, 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, NCHWSpatialAlignmentFail)
{
    // If H*W is not divisible by vectorsize with NCHW layout, it should fail immediately
    EXPECT_FALSE(isSpatialMultipleApplicable(64, 64, 3, 5, false, true, 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInSpatialDimension)
{
    // The last block of spatial dimension is large enough to hold stashed values
    EXPECT_TRUE(isSpatialMultipleApplicable(64, 64, 16, 16, true, true, 4, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashFitsInBatchDimension)
{
    // Even if the spatial remainder is small, it works if the batch remainder is large enough
    EXPECT_TRUE(isSpatialMultipleApplicable(128, 64, 2, 5, true, true, 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, StashDoesNotFitInSpatialOrBatchDimension)
{
    // Both spatial and batch remainders are smaller than the required stash values
    EXPECT_FALSE(isSpatialMultipleApplicable(10, 64, 2, 5, true, true, 1, 32, 64, 1, 64));
}

TEST(TestIsSpatialMultipleApplicable, MixedPrecisionOddCFailsOnSmallZ)
{
    // For FP16/BF16, if C is odd, the intermediate results MUST fit in the batch dimension
    EXPECT_FALSE(isSpatialMultipleApplicable(10, 65, 16, 16, true, false, 1, 16, 64, 1, 64));
}

// ====================================================================================================
// useMultiple - checks if spatial multiple implementation should be used based on heuristics
// ====================================================================================================

TEST(TestUseMultiple, BackwardSmallSpatialReturnsFalse)
{
    EXPECT_FALSE(useMultiple(64, 16, 16, false, false, Direction::BACKWARD));
}

TEST(TestUseMultiple, BackwardLargeProblemReturnsTrue)
{
    EXPECT_TRUE(useMultiple(64, 1024, 1024, false, false, Direction::BACKWARD));
}

TEST(TestUseMultiple, ForwardTrainingSmallProblemReturnsFalse)
{
    EXPECT_FALSE(useMultiple(2, 8, 8, false, false, Direction::FORWARD_TRAINING));
}

TEST(TestUseMultiple, ForwardTrainingLargeBatchReturnsTrue)
{
    EXPECT_TRUE(useMultiple(1024, 32, 32, false, false, Direction::FORWARD_TRAINING));
}

TEST(TestUseMultiple, ForwardTrainingMixedPrecisionHeuristic)
{
    EXPECT_FALSE(useMultiple(128, 32, 32, true, false, Direction::FORWARD_TRAINING));
}

TEST(TestUseMultiple, NHWCAlwaysReturnsTrue)
{
    EXPECT_TRUE(useMultiple(2, 8, 8, false, true, Direction::FORWARD_TRAINING));
}

// ====================================================================================================
// getStashMethod - determines the stash method to be used based on problem configuration
// ====================================================================================================

TEST(TestGetStashMethod, ReturnsMethodZeroWhenSpatialFits)
{
    EXPECT_EQ(getStashMethod(false, true, 32, 64, 64, 1024, 1024, 1, 64), 0);
}

TEST(TestGetStashMethod, ReturnsMethodOneWhenBatchStashRequired)
{
    EXPECT_EQ(getStashMethod(false, true, 32, 64, 64, 10, 64, 1, 64), 1);
}

TEST(TestGetStashMethod, ReturnsMethodTwoForNHWCOddCMixedPrecision)
{
    EXPECT_EQ(getStashMethod(true, false, 16, 65, 64, 1024, 1024, 1, 64), 2);
}

TEST(TestGetStashMethod, MixedPrecisionScalesStashValues)
{
    EXPECT_EQ(getStashMethod(false, false, 16, 64, 64, 20, 64, 1, 64), 1);
}

// ====================================================================================================
// defaultConfigSpatialSingle - provides default configuration for spatial single implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialSingle, NHWCDefaultVariantOne)
{
    KernelConfig config;

    defaultConfigSpatialSingle(64, 16, 16, false, false, true, Direction::BACKWARD, config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NCHWBackwardSmallSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(32, 16, 16, false, false, false, Direction::BACKWARD, config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NCHWBackwardSmallSpatialLargeBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(128, 20, 10, false, false, false, Direction::BACKWARD, config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NCHWBackwardMidSpatialSmallBatch)
{
    KernelConfig config;

    defaultConfigSpatialSingle(16, 20, 30, false, false, false, Direction::BACKWARD, config);
    EXPECT_EQ(config.variant, 3);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NCHWForwardLargeSpatialOrMixed)
{
    KernelConfig config;

    defaultConfigSpatialSingle(64, 32, 32, true, false, false, Direction::FORWARD_TRAINING, config);
    EXPECT_EQ(config.variant, 1);
    EXPECT_EQ(config.vectorsize, 1);
}

TEST(TestDefaultConfigSpatialSingle, NCHWForwardSmallDefault)
{
    KernelConfig config;

    defaultConfigSpatialSingle(10, 8, 8, false, false, false, Direction::FORWARD_TRAINING, config);
    EXPECT_EQ(config.variant, 0);
    EXPECT_EQ(config.vectorsize, 1);
}

// ====================================================================================================
// defaultConfigSpatialMultiple - provides default configuration for spatial multiple implementation
// ====================================================================================================

TEST(TestDefaultConfigSpatialMultiple, NHWCFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(128, 64, 16, 16, true, true, 80, 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 4);
    EXPECT_EQ(config.nelements, 128);
    EXPECT_EQ(config.xlocalsize, 16);
    EXPECT_EQ(config.ylocalsize, 8);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NHWCFallbackConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(64, 25, 16, 16, true, true, 80, 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 64);
    EXPECT_EQ(config.xlocalsize, 32);
    EXPECT_EQ(config.ylocalsize, 4);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NCHWFullConfigCheck)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(128, 64, 16, 16, false, true, 80, 32, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 128);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 256);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NCHWSingleVectorConfig)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(16, 32, 1, 2, false, true, 80, 1, config);
    EXPECT_EQ(config.variant, 2);
    EXPECT_EQ(config.vectorsize, 1);
    EXPECT_EQ(config.nelements, 16);
    EXPECT_EQ(config.xlocalsize, 1);
    EXPECT_EQ(config.ylocalsize, 64);
    EXPECT_EQ(config.zlocalsize, 1);
}

TEST(TestDefaultConfigSpatialMultiple, NCHWOddSpatialFallback)
{
    KernelConfig config;

    defaultConfigSpatialMultiple(64, 64, 3, 5, false, true, 80, 32, config);
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
    defaultConfigSpatialMultiple(1, 1, 1, 1, false, true, 80, 12345, config);
    EXPECT_EQ(config.variant, -1);
}
