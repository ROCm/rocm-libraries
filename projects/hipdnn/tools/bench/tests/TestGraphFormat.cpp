// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_bench/GraphFormat.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{

std::vector<uint8_t> bytesOf(const std::string& text)
{
    return {text.begin(), text.end()};
}

} // namespace

TEST(TestGraphFormat, AJsonObjectIsRecognizedAfterLeadingWhitespace)
{
    EXPECT_TRUE(hipdnn_bench::isJsonGraph(bytesOf("{\"name\":\"g\"}")));
    EXPECT_TRUE(hipdnn_bench::isJsonGraph(bytesOf("\n{\"name\":\"g\"}")));
    EXPECT_TRUE(hipdnn_bench::isJsonGraph(bytesOf(" \t\r\n  {\"name\":\"g\"}")));
}

TEST(TestGraphFormat, AFlatBufferAndEmptyInputAreNotJson)
{
    // A FlatBuffer opens with a little-endian root offset, here 0x0C then zeros.
    EXPECT_FALSE(hipdnn_bench::isJsonGraph({0x0C, 0x00, 0x00, 0x00, 0x08, 0x00}));
    // An offset of 0x20 is whitespace to JSON, but zeros follow, not an object.
    EXPECT_FALSE(hipdnn_bench::isJsonGraph({0x20, 0x00, 0x00, 0x00, 0x08, 0x00}));
    EXPECT_FALSE(hipdnn_bench::isJsonGraph(bytesOf("   \n ")));
    EXPECT_FALSE(hipdnn_bench::isJsonGraph({}));
}
