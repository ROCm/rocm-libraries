// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/Sha256.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

/// @file TestSha256.cpp
/// @brief The in-tree SHA-256, which carries RFC 0019 §6.5's features_hash.
///
/// This is an own implementation, not a library call, and it has one job: agree with Python's
/// `hashlib.sha256` in tools/uhd_gen. The hash is how the runtime decides a model was trained on
/// the feature signature it is about to be handed (§6.3 check 1). An implementation that is
/// self-consistent but wrong fails *every* comparison, so every UHD is rejected and every engine
/// falls back to declared order -- which is a legal ranking, logged as a warning, and otherwise
/// indistinguishable from working.
///
/// The vectors below are the published FIPS 180-4 ones. Checking against a constant rather than
/// against ourselves is the point: only an external reference can catch an implementation that
/// is consistently wrong.
namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

TEST(TestIngestorSha256, MatchesThePublishedVectors)
{
    EXPECT_EQ(sha256(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(TestIngestorSha256, HandlesTheLengthsWherePaddingChangesBlockCount)
{
    // The classic implementation bug. A message of 55 bytes leaves exactly enough room for the
    // 0x80 marker and the 8-byte length; 56 does not, and needs a second block. An
    // off-by-one here is correct for almost every input and wrong for a narrow band of
    // lengths -- which a feature signature can land in without anyone choosing it.
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
    // Both are on the live path: the string form hashes a feature signature, the byte form
    // hashes a model artifact. They must not be able to disagree.
    const std::string text = "q.N,q.C,kernel.tile_m";
    const std::vector<uint8_t> bytes(text.begin(), text.end());

    EXPECT_EQ(sha256(bytes.data(), bytes.size()), sha256(text));
}

TEST(TestIngestorSha256, EmbeddedNulsAreHashedRatherThanTerminating)
{
    // A model artifact is arbitrary bytes. Treating a NUL as the end truncates the input, so
    // two different artifacts sharing a prefix would hash identically -- and the check that
    // exists to notice a swapped model would pass.
    const std::vector<uint8_t> first{'a', 0, 'b'};
    const std::vector<uint8_t> second{'a', 0, 'c'};

    EXPECT_NE(sha256(first.data(), first.size()), sha256(second.data(), second.size()));
    EXPECT_NE(sha256(first.data(), first.size()), sha256("a"));
}

TEST(TestIngestorSha256, EveryDigestIsSixtyFourLowercaseHexDigits)
{
    // The format is half the contract: the value is compared as text against a string Python
    // wrote, so an uppercase or unpadded digest never matches even when the bytes are right.
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
} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
