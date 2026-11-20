// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

using hipdnn_test_sdk::detail::gatherStridedAux;
using hipdnn_test_sdk::detail::stridedAuxSpan;

// A stride below 1 with several values would read before or repeatedly at the buffer base.
TEST(TestRaggedTokenBoundaries, StridedAuxSpanRejectsStrideBelowOneForSeveralValues)
{
    EXPECT_THROW(stridedAuxSpan(2, 0, "aux"), std::invalid_argument);
    EXPECT_THROW(stridedAuxSpan(3, -1, "aux"), std::invalid_argument);
    EXPECT_EQ(stridedAuxSpan(3, 2, "aux"), 5u);
}

// A stride only matters between values: no value or one value accepts any stride.
TEST(TestRaggedTokenBoundaries, StridedAuxSpanAcceptsAnyStrideForZeroOrOneValue)
{
    EXPECT_EQ(stridedAuxSpan(0, 0, "aux"), 0u);
    EXPECT_EQ(stridedAuxSpan(0, -3, "aux"), 0u);
    EXPECT_EQ(stridedAuxSpan(1, 0, "aux"), 1u);
    EXPECT_EQ(stridedAuxSpan(1, -3, "aux"), 1u);
}

TEST(TestRaggedTokenBoundaries, GatherStridedAuxRejectsStrideBelowOneForSeveralValues)
{
    const std::vector<int32_t> raw = {7, 8, 9};
    EXPECT_THROW(gatherStridedAux(raw.data(), 2, 0, "aux"), std::invalid_argument);
    EXPECT_THROW(gatherStridedAux(raw.data(), 2, -1, "aux"), std::invalid_argument);
}

TEST(TestRaggedTokenBoundaries, GatherStridedAuxHandlesZeroAndOneValue)
{
    const std::vector<int32_t> raw = {7, 8, 9};
    EXPECT_TRUE(gatherStridedAux(raw.data(), 0, 0, "aux").empty());
    EXPECT_EQ(gatherStridedAux(raw.data(), 1, 0, "aux"), (std::vector<int64_t>{7}));
    EXPECT_EQ(gatherStridedAux(raw.data(), 3, 1, "aux"), (std::vector<int64_t>{7, 8, 9}));
}
