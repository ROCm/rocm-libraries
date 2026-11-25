// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestChecksum.cpp
 * @brief xxh64 against reference XXH64 values (seed 0), computed with LLVM's xxHash64, so the
 *        checksum is the standard one rather than a self-comparison.
 */

#include "utilities/Checksum.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hip_kernel_provider::utilities
{
namespace
{

std::uint64_t checksumOf(const std::string& text)
{
    return xxh64(text.data(), text.size());
}

TEST(TestChecksum, MatchesReferenceXxh64Values)
{
    EXPECT_EQ(checksumOf(""), 0xEF46DB3751D8E999ULL);
    EXPECT_EQ(checksumOf("a"), 0xD24EC4F1A98C6E5BULL);
    EXPECT_EQ(checksumOf("abc"), 0x44BC2CF5AD770999ULL);

    // Every tail path: 32-byte stripes, then 8-, 4- and 1-byte remainders.
    std::vector<unsigned char> pattern(4099);
    for(std::size_t i = 0; i < pattern.size(); ++i)
    {
        pattern[i] = static_cast<unsigned char>((i * 131 + std::size_t{3} * 7919) & 0xFF);
    }
    EXPECT_EQ(xxh64(pattern.data(), pattern.size()), 0x940C9D854D837E06ULL);
}

TEST(TestChecksum, ChangesWithAnyFlippedBit)
{
    std::vector<unsigned char> bytes(1000, 0x5A);
    const auto original = xxh64(bytes.data(), bytes.size());
    for(const std::size_t position : {std::size_t{0}, std::size_t{511}, std::size_t{999}})
    {
        auto flipped = bytes;
        flipped[position] ^= 0x01;
        EXPECT_NE(xxh64(flipped.data(), flipped.size()), original) << position;
    }
    EXPECT_NE(xxh64(bytes.data(), bytes.size() - 1), original);
}

} // namespace
} // namespace hip_kernel_provider::utilities
