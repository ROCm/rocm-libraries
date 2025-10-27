// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <vector>

#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>

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
        if(dims.size() != 4 && dims.size() != 5)
        {
            throw std::invalid_argument(
                "dims must be either 4D (N, C, H, W) or 5D (N, C, D, H, W)");
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const BatchnormTestCase& tc)
    {
        using namespace hipdnn_sdk::test_utilities;

        ss << "(dims:";
        vecToStream(ss, tc.dims);
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

inline std::vector<BatchnormTestCase> getBatchnormTestCases4D()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {{1, 3, 5, 5}, seed},
        {{2, 3, 14, 14}, seed},
        {{2, 3, 1, 1}, seed},
        {{32, 1, 14, 14}, seed},
        {{32, 3, 1, 14}, seed},
        {{32, 3, 14, 1}, seed},
        {{64, 64, 112, 112}, seed},
        {{64, 512, 14, 14}, seed},
    };
}

inline std::vector<BatchnormTestCase> getBatchnormTestCases5D()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {{2, 3, 3, 1, 1}, seed},
        {{16, 3, 8, 14, 14}, seed},
    };
}

// for backwards compatibility with op tests
// todo: remove and swap tests
struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w
                  << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

inline std::vector<Batchnorm2dTestCase> getBatchnorm2dTestCases()
{
    unsigned seed = hipdnn_sdk::test_utilities::getGlobalTestSeed();

    return {
        {1, 3, 14, 14, seed},
        {2, 3, 14, 14, seed},
    };
}

} // namespace test_bn_common
