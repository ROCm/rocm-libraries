// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for bitwise_difference.hpp, which hipblaslt-bench --verify_all_iters uses
// to compare each relaunch's D with the first launch's. See README.md in this directory for the
// HostUnit/smoke_ naming.

#include "bitwise_difference.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
    TEST(HostUnitBitwiseDifference, smoke_IdenticalBuffersDoNotDiffer)
    {
        std::vector<float> a{1, 2, 3, 4};
        std::vector<float> b = a;
        const auto         d = bitwise_difference(a.data(), b.data(), a.size(), sizeof(float));
        EXPECT_EQ(d.count, 0u);
    }

    TEST(HostUnitBitwiseDifference, smoke_CountsElementsAndFindsFirst)
    {
        std::vector<float> a{1, 2, 3, 4, 5, 6};
        std::vector<float> b = a;
        b[4]                 = 7;
        b[2]                 = 8;
        const auto d         = bitwise_difference(a.data(), b.data(), a.size(), sizeof(float));
        EXPECT_EQ(d.count, 2u);
        EXPECT_EQ(d.first, 2u);
    }

    // Two bytes that differ inside one 2-byte element count once, at that element's index.
    TEST(HostUnitBitwiseDifference, smoke_CountsElementsNotBytes)
    {
        std::vector<uint16_t> a{0x0000, 0x1234, 0x0000};
        std::vector<uint16_t> b{0x0000, 0x4321, 0x0000};
        const auto d = bitwise_difference(a.data(), b.data(), a.size(), sizeof(uint16_t));
        EXPECT_EQ(d.count, 1u);
        EXPECT_EQ(d.first, 1u);
    }

    // The comparison is on bits, not values: NaN matches the same NaN, and -0 differs from +0.
    TEST(HostUnitBitwiseDifference, smoke_ComparesBitsNotValues)
    {
        const float        nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> a{nan, 0.0f};
        std::vector<float> b{nan, -0.0f};
        const auto         d = bitwise_difference(a.data(), b.data(), a.size(), sizeof(float));
        EXPECT_EQ(d.count, 1u);
        EXPECT_EQ(d.first, 1u);
    }

    TEST(HostUnitBitwiseDifference, smoke_EmptyBuffersDoNotDiffer)
    {
        EXPECT_EQ(bitwise_difference(nullptr, nullptr, 0, 4).count, 0u);
    }
}
