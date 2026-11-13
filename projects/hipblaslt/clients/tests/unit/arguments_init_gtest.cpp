// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_arguments.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <type_traits>

TEST(HostUnitArguments, smoke_InitSetsEveryField)
{
    Arguments zeroed{};
    Arguments garbage;
    std::memset(static_cast<void*>(&garbage), 0xA5, sizeof(Arguments));
    zeroed.init();
    garbage.init();
    EXPECT_EQ(garbage.batch_offset_a, 0);
    EXPECT_EQ(garbage.bias_stride, 0);
    EXPECT_EQ(garbage.M[1], 0);
    EXPECT_EQ(garbage.M[0], 128);
    EXPECT_EQ(garbage.gsu_vector[1], -1);

    // Compare member values, including array tails; padding bytes are unspecified.
    auto check = [](const auto& actual, const auto& expected) {
        if constexpr(std::is_array_v<std::remove_reference_t<decltype(actual)>>)
        {
            EXPECT_TRUE(std::equal(std::begin(actual), std::end(actual), std::begin(expected)));
        }
        else
        {
            EXPECT_EQ(actual, expected);
        }
    };
#define CHECK_ARGUMENT(NAME)              \
    {                                     \
        SCOPED_TRACE(#NAME);              \
        check(garbage.NAME, zeroed.NAME); \
    }
    FOR_EACH_ARGUMENT(CHECK_ARGUMENT, ;);
#undef CHECK_ARGUMENT
}
