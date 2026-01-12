// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineIdHash.hpp>
#include <string>
#include <unordered_set>

class TestEngineIdHash : public ::testing::Test
{
};

TEST_F(TestEngineIdHash, DeterministicBehavior)
{
    // Same input should always produce the same output
    const char* engineName = "MIOPEN_PLUGIN";
    auto id1 = hipdnn_data_sdk::engineNameToId(engineName);
    auto id2 = hipdnn_data_sdk::engineNameToId(engineName);
    auto id3 = hipdnn_data_sdk::engineNameToId(engineName);

    EXPECT_EQ(id1, id2);
    EXPECT_EQ(id2, id3);
}

TEST_F(TestEngineIdHash, DifferentStringsDifferentHashes)
{
    // Different strings should produce different hashes
    std::vector<std::string> engineNames = {"MIOPEN_PLUGIN",
                                            "VENDOR_FAST_CONV",
                                            "CPU_REFERENCE_ENGINE",
                                            "EXAMPLE_PLUGIN_RENAME_THIS",
                                            "CUSTOM_ENGINE_1",
                                            "CUSTOM_ENGINE_2",
                                            "AMD_ROCM_ENGINE"};

    std::unordered_set<int64_t> ids;
    for(const auto& name : engineNames)
    {
        auto id = hipdnn_data_sdk::engineNameToId(name);
        // Check for uniqueness
        EXPECT_TRUE(ids.insert(id).second) << "Collision detected for engine name: " << name;
    }
}

TEST_F(TestEngineIdHash, HandlesNullPointer)
{
    // Null pointer should return 0
    const char* nullStr = nullptr;
    auto id = hipdnn_data_sdk::engineNameToId(nullStr);
    EXPECT_EQ(id, 0);
}

TEST_F(TestEngineIdHash, HandlesEmptyString)
{
    // Empty string should return 0.
    auto id = hipdnn_data_sdk::engineNameToId("");
    EXPECT_EQ(id, 0);
}

TEST_F(TestEngineIdHash, StringOverloadsConsistent)
{
    const char* cStr = "TEST_ENGINE";
    std::string stdStr = "TEST_ENGINE";
    std::string_view strView = "TEST_ENGINE";

    auto idCStr = hipdnn_data_sdk::engineNameToId(cStr);
    auto idStdStr = hipdnn_data_sdk::engineNameToId(stdStr);
    auto idStrView = hipdnn_data_sdk::engineNameToId(strView);

    EXPECT_EQ(idCStr, idStdStr);
    EXPECT_EQ(idStdStr, idStrView);
}

TEST_F(TestEngineIdHash, LongStringHandling)
{
    // Test with a very long string
    std::string longName(1000, 'A');
    longName += "_ENGINE";

    auto id1 = hipdnn_data_sdk::engineNameToId(longName);
    auto id2 = hipdnn_data_sdk::engineNameToId(longName);

    EXPECT_EQ(id1, id2); // Still deterministic

    // Should be different from a shorter string
    auto idShort = hipdnn_data_sdk::engineNameToId("A_ENGINE");
    EXPECT_NE(id1, idShort);
}

TEST_F(TestEngineIdHash, SpecialCharacters)
{
    // Test with various special characters
    std::vector<std::string> specialNames = {"ENGINE_WITH_UNDERSCORE",
                                             "ENGINE-WITH-DASH",
                                             "ENGINE.WITH.DOT",
                                             "ENGINE:WITH:COLON",
                                             "ENGINE/WITH/SLASH",
                                             "ENGINE WITH SPACE",
                                             "ENGINE@WITH@AT",
                                             "ENGINE#WITH#HASH",
                                             "ENGINE$WITH$DOLLAR",
                                             "ENGINE%WITH%PERCENT"};

    std::unordered_set<int64_t> ids;
    for(const auto& name : specialNames)
    {
        auto id = hipdnn_data_sdk::engineNameToId(name);
        // Each should produce a unique ID
        EXPECT_TRUE(ids.insert(id).second) << "Collision detected for: " << name;
    }
}

TEST_F(TestEngineIdHash, CaseSensitivity)
{
    // Hash should be case-sensitive
    auto idLower = hipdnn_data_sdk::engineNameToId("miopen_plugin");
    auto idUpper = hipdnn_data_sdk::engineNameToId("MIOPEN_PLUGIN");
    auto idMixed = hipdnn_data_sdk::engineNameToId("MIOpen_Plugin");

    EXPECT_NE(idLower, idUpper);
    EXPECT_NE(idUpper, idMixed);
    EXPECT_NE(idLower, idMixed);
}

TEST_F(TestEngineIdHash, SimilarNamesProduceDifferentHashes)
{
    // Test that similar names still produce different hashes
    auto id1 = hipdnn_data_sdk::engineNameToId("ENGINE_V1");
    auto id2 = hipdnn_data_sdk::engineNameToId("ENGINE_V2");
    auto id3 = hipdnn_data_sdk::engineNameToId("ENGINE_V11");
    auto id4 = hipdnn_data_sdk::engineNameToId("ENGINE_V21");

    EXPECT_NE(id1, id2);
    EXPECT_NE(id1, id3);
    EXPECT_NE(id1, id4);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id2, id4);
    EXPECT_NE(id3, id4);
}

TEST_F(TestEngineIdHash, KnownEngineValues)
{
    // Test that known engines produce consistent values
    // These values should never change once established
    auto miopenId = hipdnn_data_sdk::engineNameToId("MIOPEN_PLUGIN");
    auto vendorId = hipdnn_data_sdk::engineNameToId("VENDOR_FAST_CONV");

    // Just verify they're non-zero and different from each other
    EXPECT_NE(miopenId, 0);
    EXPECT_NE(vendorId, 0);
    EXPECT_NE(miopenId, vendorId);
}
