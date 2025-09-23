// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <vector>

namespace test_bn_common
{

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
    unsigned int seed = std::random_device{}();

    return {
        {.n = 1, .c = 3, .h = 14, .w = 14, .seed = seed},
        {.n = 2, .c = 3, .h = 14, .w = 14, .seed = seed},
    };
}

} // namespace test_bn_common
