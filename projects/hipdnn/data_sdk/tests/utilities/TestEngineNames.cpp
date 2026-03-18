// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace hipdnn_data_sdk::utilities;

class TestEngineNames : public ::testing::Test
{
};

TEST_F(TestEngineNames, MacroGeneratesCorrectConstants)
{
    // Check that the string constants are defined
    EXPECT_STREQ(MIOPEN_ENGINE_NAME, "MIOPEN_ENGINE");
    EXPECT_STREQ(HIPBLASLT_ENGINE_NAME, "HIPBLASLT_ENGINE");
    EXPECT_STREQ(FUSILLI_ENGINE_NAME, "FUSILLI_ENGINE");

    // Check that the ID constants are defined and match the hash function
    EXPECT_EQ(MIOPEN_ENGINE_ID, engineNameToId("MIOPEN_ENGINE"));
    EXPECT_EQ(HIPBLASLT_ENGINE_ID, engineNameToId("HIPBLASLT_ENGINE"));
    EXPECT_EQ(FUSILLI_ENGINE_ID, engineNameToId("FUSILLI_ENGINE"));
}

TEST_F(TestEngineNames, EngineIdToNameMappingConsistent)
{
    // Get the ID to name map
    const auto& idToName = getEngineIdToNameMap();

    // Verify each mapping is consistent
    for(const auto& [id, name] : idToName)
    {
        auto calculatedId = engineNameToId(name);
        EXPECT_EQ(id, calculatedId)
            << "ID mismatch for engine: " << name << " (stored: 0x" << std::hex << id
            << ", calculated: 0x" << calculatedId << std::dec << ")";
    }
}

TEST_F(TestEngineNames, IsEngineNameRegistered)
{
    // Test with known registered names
    EXPECT_TRUE(isEngineNameRegistered(MIOPEN_ENGINE_NAME));
    EXPECT_TRUE(isEngineNameRegistered(HIPBLASLT_ENGINE_NAME));
    EXPECT_TRUE(isEngineNameRegistered(FUSILLI_ENGINE_NAME));

    // Test with unregistered names
    EXPECT_FALSE(isEngineNameRegistered("UNKNOWN_ENGINE"));
    EXPECT_FALSE(isEngineNameRegistered("NOT_REGISTERED"));
    EXPECT_FALSE(isEngineNameRegistered(""));
}

TEST_F(TestEngineNames, GetEngineNameFromId)
{
    // Test with registered engines
    EXPECT_EQ(getEngineNameFromId(MIOPEN_ENGINE_ID), "MIOPEN_ENGINE");
    EXPECT_EQ(getEngineNameFromId(HIPBLASLT_ENGINE_ID), "HIPBLASLT_ENGINE");
    EXPECT_EQ(getEngineNameFromId(FUSILLI_ENGINE_ID), "FUSILLI_ENGINE");

    // Test with non-existent ID - should throw
    int64_t nonExistentId = 0xDEADBEEF;
    EXPECT_THROW(getEngineNameFromId(nonExistentId), std::out_of_range);
}

TEST_F(TestEngineNames, EngineCountMatches)
{
    // Verify that the number of engines in getAllEngineNames matches getEngineIdToNameMap
    const auto& allEngines = getAllEngineNames();
    const auto& idToName = getEngineIdToNameMap();

    EXPECT_EQ(allEngines.size(), idToName.size())
        << "Mismatch between getAllEngineNames and getEngineIdToNameMap sizes";

    // Also verify all names in one are in the other
    for(const auto& name : allEngines)
    {
        auto id = engineNameToId(name);
        EXPECT_NE(idToName.find(id), idToName.end())
            << "Engine '" << name << "' is in getAllEngineNames but not in getEngineIdToNameMap";
    }
}

TEST_F(TestEngineNames, EnsureAllEngineNameToIdsBehaveTheSame)
{
    {
        auto engineIdCString = engineNameToId(MIOPEN_ENGINE_NAME);
        auto engineIdString = engineNameToId(std::string(MIOPEN_ENGINE_NAME));
        auto engineIdStringView = engineNameToId(std::string_view(MIOPEN_ENGINE_NAME));

        EXPECT_EQ(engineIdCString, engineIdString);
        EXPECT_EQ(engineIdCString, engineIdStringView);
    }

    {
        auto engineIdCString = engineNameToId(HIPBLASLT_ENGINE_NAME);
        auto engineIdString = engineNameToId(std::string(HIPBLASLT_ENGINE_NAME));
        auto engineIdStringView = engineNameToId(std::string_view(HIPBLASLT_ENGINE_NAME));

        EXPECT_EQ(engineIdCString, engineIdString);
        EXPECT_EQ(engineIdCString, engineIdStringView);
    }

    {
        auto engineIdCString = engineNameToId(FUSILLI_ENGINE_NAME);
        auto engineIdString = engineNameToId(std::string(FUSILLI_ENGINE_NAME));
        auto engineIdStringView = engineNameToId(std::string_view(FUSILLI_ENGINE_NAME));

        EXPECT_EQ(engineIdCString, engineIdString);
        EXPECT_EQ(engineIdCString, engineIdStringView);
    }
}
