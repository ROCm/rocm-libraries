// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineIdHash.hpp>
#include <hipdnn_plugin_sdk/EngineNames.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

class TestEngineNames : public ::testing::Test
{
};

TEST_F(TestEngineNames, MacroGeneratesCorrectConstants)
{
    // Verify the macro-generated constants exist and are correct
    using namespace hipdnn_plugin_sdk::engine_names;

    // Check that the string constants are defined
    EXPECT_STREQ(MIOPEN_PLUGIN, "MIOPEN_PLUGIN");

    // Check that the ID constants are defined and match the hash function
    EXPECT_EQ(MIOPEN_PLUGIN_ID, hipdnn_data_sdk::engineNameToId("MIOPEN_PLUGIN"));
}

TEST_F(TestEngineNames, EngineIdToNameMappingConsistent)
{
    // Get the ID to name map
    const auto& idToName = hipdnn_plugin_sdk::engine_names::getEngineIdToNameMap();

    // Verify each mapping is consistent
    for(const auto& [id, name] : idToName)
    {
        auto calculatedId = hipdnn_data_sdk::engineNameToId(name.data());
        EXPECT_EQ(id, calculatedId)
            << "ID mismatch for engine: " << name << " (stored: 0x" << std::hex << id
            << ", calculated: 0x" << calculatedId << std::dec << ")";
    }
}

TEST_F(TestEngineNames, IsEngineNameRegistered)
{
    // Test with known registered names
    EXPECT_TRUE(hipdnn_plugin_sdk::engine_names::isEngineNameRegistered(
        hipdnn_plugin_sdk::engine_names::MIOPEN_PLUGIN));

    // Test with unregistered names
    EXPECT_FALSE(hipdnn_plugin_sdk::engine_names::isEngineNameRegistered("UNKNOWN_ENGINE"));
    EXPECT_FALSE(hipdnn_plugin_sdk::engine_names::isEngineNameRegistered("NOT_REGISTERED"));
    EXPECT_FALSE(hipdnn_plugin_sdk::engine_names::isEngineNameRegistered(""));
}

TEST_F(TestEngineNames, GetEngineNameFromId)
{
    using namespace hipdnn_plugin_sdk::engine_names;

    // Test with registered engines
    EXPECT_EQ(hipdnn_plugin_sdk::engine_names::getEngineNameFromId(MIOPEN_PLUGIN_ID),
              "MIOPEN_PLUGIN");

    // Test with non-existent ID
    int64_t nonExistentId = 0xDEADBEEF;
    EXPECT_EQ(hipdnn_plugin_sdk::engine_names::getEngineNameFromId(nonExistentId), "");
}

TEST_F(TestEngineNames, EngineCountMatches)
{
    // Verify that the number of engines in getAllEngineNames matches getEngineIdToNameMap
    const auto& allEngines = hipdnn_plugin_sdk::engine_names::getAllEngineNames();
    const auto& idToName = hipdnn_plugin_sdk::engine_names::getEngineIdToNameMap();

    EXPECT_EQ(allEngines.size(), idToName.size())
        << "Mismatch between getAllEngineNames and getEngineIdToNameMap sizes";

    // Also verify all names in one are in the other
    for(const auto& name : allEngines)
    {
        auto id = hipdnn_data_sdk::engineNameToId(name.data());
        EXPECT_NE(idToName.find(id), idToName.end())
            << "Engine '" << name << "' is in getAllEngineNames but not in getEngineIdToNameMap";
    }
}
