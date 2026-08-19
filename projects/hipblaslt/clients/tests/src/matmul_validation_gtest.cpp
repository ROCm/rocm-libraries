// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_validation/MatmulValidation.hpp>

#include <gtest/gtest.h>

#include <vector>

TEST(HostValidationMatmulValidation, PointerArrayNormCountsEachBatchOnce)
{
    std::vector<HipHostBuffer> expectedOutput;
    std::vector<HipHostBuffer> observedOutput;
    expectedOutput.reserve(2);
    observedOutput.reserve(2);
    for(int batch = 0; batch < 2; ++batch)
    {
        expectedOutput.emplace_back(HIP_R_32F, 1);
        observedOutput.emplace_back(HIP_R_32F, 1);
        expectedOutput.back().as<float>()[0] = 1.0f;
        observedOutput.back().as<float>()[0] = static_cast<float>(batch + 2);
    }

    std::vector<HipHostBuffer> emptyHostBuffers;
    double                     relativeFrobeniusError = 0.0;
    double                     absoluteTolerance      = 0.0;
    double                     relativeTolerance      = 0.0;
    double                     maximumUlp             = 0.0;
    double                     averageUlp             = 0.0;

    hipblaslt::host_validation::validateMatmulOutputs({
        .options                     = {.compareNorm = true},
        .gemmCount                   = 1,
        .rows                        = std::vector<int64_t>{1},
        .columns                     = std::vector<int64_t>{1},
        .outputLeadingDimensions     = std::vector<int64_t>{1},
        .auxiliaryLeadingDimensions  = std::vector<int64_t>{1},
        .outputBatchStrides          = std::vector<int64_t>{0},
        .auxiliaryBatchStrides       = std::vector<int64_t>{0},
        .batchCounts                 = std::vector<int>{2},
        .biasSizes                   = std::vector<size_t>{0},
        .expectedOutput              = expectedOutput,
        .observedOutput              = observedOutput,
        .expectedMaximum             = emptyHostBuffers,
        .observedMaximum             = emptyHostBuffers,
        .expectedAuxiliary           = emptyHostBuffers,
        .observedAuxiliary           = emptyHostBuffers,
        .expectedBias                = emptyHostBuffers,
        .observedBias                = emptyHostBuffers,
        .absoluteTolerances          = std::vector<double>{0.0},
        .symmetricRelativeTolerances = std::vector<double>{0.0},
        .metrics                     = {.relativeFrobeniusError = relativeFrobeniusError,
                                        .absoluteTolerance      = absoluteTolerance,
                                        .relativeTolerance      = relativeTolerance,
                                        .maximumUlp             = maximumUlp,
                                        .averageUlp             = averageUlp},
        .outputType                  = HIP_R_32F,
        .biasType                    = HIP_R_32F,
        .auxiliaryType               = HIP_R_32F,
        .computeType                 = HIP_R_32F,
        .batchMode                   = HIPBLASLT_BATCH_MODE_POINTER_ARRAY,
    });

    EXPECT_DOUBLE_EQ(relativeFrobeniusError, 3.0);
}
