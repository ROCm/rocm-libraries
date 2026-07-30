// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <string>
#include <unordered_set>

class TestStringUtil : public ::testing::Test
{
};

TEST_F(TestStringUtil, Fnv1aHashDeterministicBehavior)
{
    // Same input should always produce the same output
    const char* testString = "MIOPEN_PLUGIN";
    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash(testString);
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash(testString);
    auto hash3 = hipdnn_data_sdk::utilities::fnv1aHash(testString);

    EXPECT_EQ(hash1, hash2);
    EXPECT_EQ(hash2, hash3);
}

TEST_F(TestStringUtil, Fnv1aHashDifferentStringsDifferentHashes)
{
    // Different strings should produce different hashes
    const std::vector<std::string> testStrings = {"MIOPEN_PLUGIN",
                                                  "VENDOR_FAST_CONV",
                                                  "CPU_REFERENCE_ENGINE",
                                                  "EXAMPLE_PLUGIN_RENAME_THIS",
                                                  "CUSTOM_ENGINE_1",
                                                  "CUSTOM_ENGINE_2",
                                                  "AMD_ROCM_ENGINE"};

    std::unordered_set<uint64_t> hashes;
    for(const auto& str : testStrings)
    {
        auto hash = hipdnn_data_sdk::utilities::fnv1aHash(str);
        // Check for uniqueness
        EXPECT_TRUE(hashes.insert(hash).second) << "Collision detected for string: " << str;
    }
}

TEST_F(TestStringUtil, Fnv1aHashHandlesNullPointer)
{
    // Null pointer should return 0
    const char* nullStr = nullptr;
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash(nullStr);
    EXPECT_EQ(hash, 0u);
}

TEST_F(TestStringUtil, Fnv1aHashHandlesEmptyString)
{
    // Empty string should return 0
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash("");
    EXPECT_EQ(hash, 0u);
}

TEST_F(TestStringUtil, Fnv1aHashStringOverloadsConsistent)
{
    const char* cStr = "TEST_STRING";
    const std::string stdStr = "TEST_STRING";
    const std::string_view strView = "TEST_STRING";

    auto hashCStr = hipdnn_data_sdk::utilities::fnv1aHash(cStr);
    auto hashStdStr = hipdnn_data_sdk::utilities::fnv1aHash(stdStr);
    auto hashStrView = hipdnn_data_sdk::utilities::fnv1aHash(strView);

    EXPECT_EQ(hashCStr, hashStdStr);
    EXPECT_EQ(hashStdStr, hashStrView);
}

TEST_F(TestStringUtil, Fnv1aHashLongStringHandling)
{
    // Test with a very long string
    std::string longStr(1000, 'A');
    longStr += "_SUFFIX";

    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash(longStr);
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash(longStr);

    EXPECT_EQ(hash1, hash2); // Still deterministic

    // Should be different from a shorter string
    auto hashShort = hipdnn_data_sdk::utilities::fnv1aHash("A_SUFFIX");
    EXPECT_NE(hash1, hashShort);
}

TEST_F(TestStringUtil, Fnv1aHashSpecialCharacters)
{
    // Test that special characters are treated as distinct regular characters
    // All strings have same base but different special character in the middle
    const std::vector<std::string> specialStrings = {"STRING_CHAR_END",
                                                     "STRING-CHAR-END",
                                                     "STRING.CHAR.END",
                                                     "STRING:CHAR:END",
                                                     "STRING/CHAR/END",
                                                     "STRING CHAR END",
                                                     "STRING@CHAR@END",
                                                     "STRING#CHAR#END",
                                                     "STRING$CHAR$END",
                                                     "STRING%CHAR%END",
                                                     "STRING&CHAR&END",
                                                     "STRING*CHAR*END",
                                                     "STRING+CHAR+END",
                                                     "STRING=CHAR=END",
                                                     "STRING!CHAR!END"};

    std::unordered_set<uint64_t> hashes;
    for(const auto& str : specialStrings)
    {
        auto hash = hipdnn_data_sdk::utilities::fnv1aHash(str);
        // Each should produce a unique hash
        EXPECT_TRUE(hashes.insert(hash).second) << "Collision detected for: " << str;
    }
}

TEST_F(TestStringUtil, Fnv1aHashCaseSensitivity)
{
    // Hash should be case-sensitive
    auto hashLower = hipdnn_data_sdk::utilities::fnv1aHash("test_string");
    auto hashUpper = hipdnn_data_sdk::utilities::fnv1aHash("TEST_STRING");
    auto hashMixed = hipdnn_data_sdk::utilities::fnv1aHash("Test_String");

    EXPECT_NE(hashLower, hashUpper);
    EXPECT_NE(hashUpper, hashMixed);
    EXPECT_NE(hashLower, hashMixed);
}

TEST_F(TestStringUtil, Fnv1aHashSimilarStringsProduceDifferentHashes)
{
    // Test that similar strings still produce different hashes
    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash("STRING_V1");
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash("STRING_V2");
    auto hash3 = hipdnn_data_sdk::utilities::fnv1aHash("STRING_V11");
    auto hash4 = hipdnn_data_sdk::utilities::fnv1aHash("STRING_V21");

    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash1, hash3);
    EXPECT_NE(hash1, hash4);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash2, hash4);
    EXPECT_NE(hash3, hash4);
}

TEST_F(TestStringUtil, Fnv1aHashKnownValues)
{
    // Test that known strings produce consistent values
    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash("MIOPEN_PLUGIN");
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash("VENDOR_FAST_CONV");

    // Just verify they're non-zero and different from each other
    EXPECT_NE(hash1, 0u);
    EXPECT_NE(hash2, 0u);
    EXPECT_NE(hash1, hash2);
}

TEST_F(TestStringUtil, Fnv1aHashBinaryDeterministicBehavior)
{
    std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04, 0x05};
    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash(data.data(), data.size());
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash(data.data(), data.size());

    EXPECT_EQ(hash1, hash2);
    EXPECT_NE(hash1, 0u);
}

TEST_F(TestStringUtil, Fnv1aHashBinaryHandlesNullPointer)
{
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash(static_cast<const uint8_t*>(nullptr), 10);
    EXPECT_EQ(hash, 0u);
}

TEST_F(TestStringUtil, Fnv1aHashBinaryHandlesZeroSize)
{
    std::vector<uint8_t> data = {0x01};
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash(data.data(), 0);
    EXPECT_EQ(hash, 0u);
}

TEST_F(TestStringUtil, Fnv1aHashBinaryDifferentDataDifferentHashes)
{
    std::vector<uint8_t> data1 = {0x01, 0x02, 0x03};
    std::vector<uint8_t> data2 = {0x01, 0x02, 0x04};
    std::vector<uint8_t> data3 = {0x01, 0x02};

    auto hash1 = hipdnn_data_sdk::utilities::fnv1aHash(data1.data(), data1.size());
    auto hash2 = hipdnn_data_sdk::utilities::fnv1aHash(data2.data(), data2.size());
    auto hash3 = hipdnn_data_sdk::utilities::fnv1aHash(data3.data(), data3.size());

    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash1, hash3);
    EXPECT_NE(hash2, hash3);
}

TEST_F(TestStringUtil, Fnv1aHashBinaryConsistentWithStringForAscii)
{
    // For ASCII data, the binary overload should produce the same hash as the string overload
    // since both iterate byte-by-byte with the same algorithm
    const char* testStr = "hello";
    auto hashStr = hipdnn_data_sdk::utilities::fnv1aHash(testStr);
    auto hashBin = hipdnn_data_sdk::utilities::fnv1aHash(reinterpret_cast<const uint8_t*>(testStr),
                                                         std::strlen(testStr));

    EXPECT_EQ(hashStr, hashBin);
}

// Tests for trim function
TEST_F(TestStringUtil, TrimRemovesLeadingWhitespace)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("  hello"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("\thello"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("\nhello"), "hello");
}

TEST_F(TestStringUtil, TrimRemovesTrailingWhitespace)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("hello  "), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("hello\t"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("hello\n"), "hello");
}

TEST_F(TestStringUtil, TrimRemovesBothEnds)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("  hello  "), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("\t\nhello\r\n"), "hello");
}

TEST_F(TestStringUtil, TrimHandlesEmptyString)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim(""), "");
}

TEST_F(TestStringUtil, TrimHandlesAllWhitespace)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("   "), "");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("\t\n\r"), "");
}

TEST_F(TestStringUtil, TrimPreservesInternalWhitespace)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("  hello world  "), "hello world");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("  hello\tworld  "), "hello\tworld");
}

TEST_F(TestStringUtil, TrimHandlesVariousWhitespaceTypes)
{
    // Test all whitespace characters: space, tab, newline, carriage return, form feed, vertical tab
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim(" \t\n\r\f\vhello \t\n\r\f\v"), "hello");
}

TEST_F(TestStringUtil, TrimNoOpOnTrimmedString)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("hello"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::trim("hello world"), "hello world");
}

// Tests for other StringUtil functions (existing functionality)
TEST_F(TestStringUtil, ToLower)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::toLower("HELLO"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::toLower("HeLLo"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::toLower("hello"), "hello");
    EXPECT_EQ(hipdnn_data_sdk::utilities::toLower(""), "");
}

TEST_F(TestStringUtil, RemoveNewlines)
{
    EXPECT_EQ(hipdnn_data_sdk::utilities::removeNewlines("Hello\nWorld"), "HelloWorld");
    EXPECT_EQ(hipdnn_data_sdk::utilities::removeNewlines("Hello\r\nWorld"), "HelloWorld");
    EXPECT_EQ(hipdnn_data_sdk::utilities::removeNewlines("Hello\rWorld"), "HelloWorld");
    EXPECT_EQ(hipdnn_data_sdk::utilities::removeNewlines("HelloWorld"), "HelloWorld");
}

TEST_F(TestStringUtil, VecToString)
{
    const std::vector<int> intVec = {1, 2, 3, 4};
    EXPECT_EQ(hipdnn_data_sdk::utilities::vecToString(intVec), "[1, 2, 3, 4]");

    const std::vector<double> doubleVec = {1.5, 2.5, 3.5};
    const std::string result = hipdnn_data_sdk::utilities::vecToString(doubleVec);
    EXPECT_TRUE(result.find("1.5") != std::string::npos);
    EXPECT_TRUE(result.find("2.5") != std::string::npos);
    EXPECT_TRUE(result.find("3.5") != std::string::npos);

    const std::vector<int> emptyVec;
    EXPECT_EQ(hipdnn_data_sdk::utilities::vecToString(emptyVec), "[]");
}

TEST_F(TestStringUtil, StringVecToStream)
{
    const std::vector<std::string> strVec = {"hello", "world", "test"};
    std::ostringstream oss;
    hipdnn_data_sdk::utilities::stringVecToStream(oss, strVec);
    EXPECT_EQ(oss.str(), "[\"hello\", \"world\", \"test\"]");

    const std::vector<std::string> singleVec = {"single"};
    std::ostringstream oss2;
    hipdnn_data_sdk::utilities::stringVecToStream(oss2, singleVec);
    EXPECT_EQ(oss2.str(), "[\"single\"]");

    const std::vector<std::string> emptyVec;
    std::ostringstream oss3;
    hipdnn_data_sdk::utilities::stringVecToStream(oss3, emptyVec);
    EXPECT_EQ(oss3.str(), "[]");
}
