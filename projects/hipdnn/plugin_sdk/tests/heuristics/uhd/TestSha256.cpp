// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

/// @file TestSha256.cpp
/// @brief The in-tree SHA-256 behind RFC 0019 §6.5's features_hash. It must agree with Python's
/// `hashlib.sha256`, so it is checked against published FIPS 180-4 vectors, not against itself.
namespace hipdnn_plugin_sdk::uhd
{
namespace
{

TEST(TestIngestorSha256, MatchesThePublishedVectors)
{
    EXPECT_EQ(sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // Many whole blocks ahead of the padded tail.
    EXPECT_EQ(sha256(std::string(1000000, 'a')),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(TestIngestorSha256, HandlesTheLengthsWherePaddingChangesBlockCount)
{
    // 55 bytes fit the 0x80 marker and 8-byte length in one block; 56 need a second.
    EXPECT_EQ(sha256(std::string(55, 'a')),
              "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    EXPECT_EQ(sha256(std::string(56, 'a')),
              "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    EXPECT_EQ(sha256(std::string(64, 'a')),
              "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    EXPECT_EQ(sha256(std::string(119, 'a')),
              "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
    EXPECT_EQ(sha256(std::string(120, 'a')),
              "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
}

TEST(TestIngestorSha256, TheByteAndStringOverloadsAgree)
{
    // The string form hashes a feature signature, the byte form a model artifact.
    const std::string text = "q.N,q.C,kernel.tile_m";
    const std::vector<uint8_t> bytes(text.begin(), text.end());

    EXPECT_EQ(sha256(bytes.data(), bytes.size()), sha256(text));
}

TEST(TestIngestorSha256, EmbeddedNulsAreHashedRatherThanTerminating)
{
    // Truncating at a NUL would let two artifacts sharing a prefix hash identically.
    const std::vector<uint8_t> first{'a', 0, 'b'};
    const std::vector<uint8_t> second{'a', 0, 'c'};

    EXPECT_NE(sha256(first.data(), first.size()), sha256(second.data(), second.size()));
    EXPECT_NE(sha256(first.data(), first.size()), sha256("a"));
}

TEST(TestIngestorSha256, EveryDigestIsSixtyFourLowercaseHexDigits)
{
    // Compared as text against the digest Python wrote.
    for(const auto& input : {std::string(), std::string("abc"), std::string(200, 'z')})
    {
        const auto digest = sha256(input);
        ASSERT_EQ(digest.size(), 64U) << "wrong width for a " << input.size() << "-byte input";
        for(const char c : digest)
        {
            EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
                << "not lowercase hex: " << digest;
        }
    }
}

} // namespace
} // namespace hipdnn_plugin_sdk::uhd
