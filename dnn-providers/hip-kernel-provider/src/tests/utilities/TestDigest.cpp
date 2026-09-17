// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestDigest.cpp
 * @brief The oracle for sha256Hex: published FIPS 180-2 vectors, not a self-comparison.
 *
 * Every other digest case compares the loader's hash against one the packer wrote, which
 * agrees whenever both are wrong the same way.
 */

#include "utilities/Digest.hpp"

#include <gtest/gtest.h>

#include <string>

namespace hip_kernel_provider::utilities
{
namespace
{

std::string hashOf(const std::string& text)
{
    return sha256Hex(text.data(), text.size());
}

TEST(TestDigest, MatchesThePublishedVectors)
{
    // A length-zero message still hashes to a full digest, so the padding block stands alone.
    EXPECT_EQ(hashOf(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hashOf("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(hashOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(TestDigest, MatchesTheLongPublishedVector)
{
    // The only vector spanning many blocks, so the only one exercising the length counter.
    const std::string million(1000000, 'a');
    EXPECT_EQ(hashOf(million), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(TestDigest, ProducesSixtyFourLowercaseHexCharacters)
{
    // The SDK validates the descriptor field as 64 lowercase hex and KpackModuleCache
    // compares by string equality, so any other spelling would fail every load.
    const std::string digest = hashOf("abc");
    ASSERT_EQ(digest.size(), 64U);
    for(const char character : digest)
    {
        EXPECT_TRUE((character >= '0' && character <= '9')
                    || (character >= 'a' && character <= 'f'))
            << "not lowercase hex: '" << character << "' in " << digest;
    }
}

TEST(TestDigest, RespectsTheGivenLength)
{
    const std::string text = "abcdef";
    // Reading past `size` -- the mistake a null-terminator assumption invites -- would make
    // these two agree.
    EXPECT_NE(sha256Hex(text.data(), 3), sha256Hex(text.data(), text.size()));
    EXPECT_EQ(sha256Hex(text.data(), 3), hashOf("abc"));
}

} // namespace
} // namespace hip_kernel_provider::utilities
