// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <ostream>
#include <random>
#include <vector>

namespace test_bn_common
{

struct BatchnormTestCase
{
    std::vector<int64_t> dims;
    unsigned int seed;

    BatchnormTestCase(std::vector<int64_t>&& dimsLocal, unsigned int seedLocal)
        : dims(std::move(dimsLocal))
        , seed(seedLocal)
    {
        if(dims.size() < 3 || dims.size() > 5)
        {
            throw std::invalid_argument(
                "dims must be 3D (N, C, L), 4D (N, C, H, W), or 5D (N, C, D, H, W)");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const BatchnormTestCase& tc)
    {
        ss << "(dims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, tc.dims);
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<BatchnormTestCase> getBnFwdInference1dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 224}, seed},
        {{2, 16, 512}, seed}, // Multi-batch
        {{1, 64, 1024}, seed}, // Longer sequence
        {{4, 3, 1}, seed}, // Minimal spatial (L=1)
    };
}

inline std::vector<BatchnormTestCase> getBnFwdInference1dFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{8, 32, 2048}, seed}, // Large sequence
        {{16, 128, 512}, seed}, // Many channels
    };
}

inline std::vector<BatchnormTestCase> getBnBwd1dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 224}, seed},
        {{2, 16, 512}, seed},
        {{1, 64, 1024}, seed},
        {{4, 3, 1}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnBwd1dFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{8, 32, 2048}, seed},
        {{16, 128, 512}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingSmoke1dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 1}, seed}, // B=2, L=1 → B×L=2 > 1
        {{1, 16, 14}, seed}, // B=1, L=14 → B×L=14 > 1
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingFull1dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14}, seed}, // Small batch, moderate length
        {{4, 8, 56}, seed}, // Medium size
        {{2, 64, 128}, seed}, // Larger
    };
}

// This is used for operation tests
inline std::vector<BatchnormTestCase> getBatchnorm2dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed},
        {{2, 3, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdInferenceTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed},
        {{1, 256, 1, 1}, seed},
        {{2, 3, 1, 1}, seed},
        {{32, 1, 14, 14}, seed},
        {{32, 3, 1, 14}, seed},
        {{32, 3, 14, 1}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdInferenceFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 16, 112, 112}, seed},
        {{5, 256, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdInference3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed},
        {{16, 3, 8, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdInference3dFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 16, 8, 112, 112}, seed},
        {{5, 256, 4, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnBwdTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed},
        {{2, 3, 1, 1}, seed},
        {{32, 1, 14, 14}, seed},
        {{32, 3, 1, 14}, seed},
        {{32, 3, 14, 1}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnBwdFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 16, 112, 112}, seed},
        {{5, 256, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnBwd3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed},
        {{16, 3, 8, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnBwd3dFullTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 16, 8, 112, 112}, seed},
        {{5, 256, 4, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingSmoke2dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 1, 1}, seed}, // Minimal case
        {{32, 3, 1, 14}, seed}, // Typical small training case
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingFull2dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed}, // Small batch
        {{2, 3, 1, 1}, seed}, // Edge case: 1x1 spatial
        {{8, 16, 28, 28}, seed}, // Medium size
        {{4, 64, 7, 7}, seed}, // Many channels, smaller spatial
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingSmoke3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed}, // Minimal 3D case
        {{2, 3, 2, 4, 4}, seed}, // Small case with non-1 spatial dims
    };
}

inline std::vector<BatchnormTestCase> getBnFwdTrainingFull3dTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed}, // Minimal case
        {{2, 3, 2, 4, 4}, seed}, // Small case
        {{16, 3, 8, 14, 14}, seed}, // Larger regression case
    };
}

} // namespace test_bn_common
