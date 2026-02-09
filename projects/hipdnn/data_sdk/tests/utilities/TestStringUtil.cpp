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
    std::vector<std::string> testStrings = {"MIOPEN_PLUGIN",
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
    std::string stdStr = "TEST_STRING";
    std::string_view strView = "TEST_STRING";

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
    std::vector<std::string> specialStrings = {"STRING_CHAR_END",
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
    std::vector<int> intVec = {1, 2, 3, 4};
    EXPECT_EQ(hipdnn_data_sdk::utilities::vecToString(intVec), "[1, 2, 3, 4]");

    std::vector<double> doubleVec = {1.5, 2.5, 3.5};
    std::string result = hipdnn_data_sdk::utilities::vecToString(doubleVec);
    EXPECT_TRUE(result.find("1.5") != std::string::npos);
    EXPECT_TRUE(result.find("2.5") != std::string::npos);
    EXPECT_TRUE(result.find("3.5") != std::string::npos);

    std::vector<int> emptyVec;
    EXPECT_EQ(hipdnn_data_sdk::utilities::vecToString(emptyVec), "[]");
}

TEST_F(TestStringUtil, StringVecToStream)
{
    std::vector<std::string> strVec = {"hello", "world", "test"};
    std::ostringstream oss;
    hipdnn_data_sdk::utilities::stringVecToStream(oss, strVec);
    EXPECT_EQ(oss.str(), "[\"hello\", \"world\", \"test\"]");

    std::vector<std::string> singleVec = {"single"};
    std::ostringstream oss2;
    hipdnn_data_sdk::utilities::stringVecToStream(oss2, singleVec);
    EXPECT_EQ(oss2.str(), "[\"single\"]");

    std::vector<std::string> emptyVec;
    std::ostringstream oss3;
    hipdnn_data_sdk::utilities::stringVecToStream(oss3, emptyVec);
    EXPECT_EQ(oss3.str(), "[]");
}
