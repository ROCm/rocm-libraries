// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace hipdnn_data_sdk::utilities;

// Test engines defined with the macro to verify expansion
// These use unique names to avoid collision with production engines
namespace test_macro_expansion
{
// Single-argument form: identifier and display name are the same
HIPDNN_REGISTER_ENGINE(TEST_MACRO_SINGLE_ARG_ENGINE)

// Two-argument form: identifier differs from display name
HIPDNN_REGISTER_ENGINE(TEST_MACRO_DUAL_ARG_ENGINE, "CustomDisplayName")

// Two-argument form where identifier and display name happen to match (like MIOPEN_ENGINE)
HIPDNN_REGISTER_ENGINE(TEST_MACRO_MATCHING_ARGS_ENGINE, "TEST_MACRO_MATCHING_ARGS_ENGINE")
} // namespace test_macro_expansion

class TestEngineNames : public ::testing::Test
{
};

TEST_F(TestEngineNames, MacroGeneratesCorrectConstants)
{
    // Check that the string constants are defined
    EXPECT_STREQ(MIOPEN_ENGINE_NAME, "MIOPEN_ENGINE");
    EXPECT_STREQ(HIPBLASLT_ENGINE_NAME, "HIPBLASLT_ENGINE");

    // Check that the ID constants are defined and match the hash function
    EXPECT_EQ(MIOPEN_ENGINE_ID, engineNameToId("MIOPEN_ENGINE"));
    EXPECT_EQ(HIPBLASLT_ENGINE_ID, engineNameToId("HIPBLASLT_ENGINE"));
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

    // Test with non-existent ID - should throw
    const int64_t nonExistentId = 0xDEADBEEF;
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

TEST_F(TestEngineNames, RegistrarSucceedsForNewUniqueName)
{
    // Registering a brand-new unique engine name should not throw
    EXPECT_NO_THROW(EngineRegistrar{"TEST_UNIQUE_ENGINE_FOR_REGISTRAR"});

    // Verify it was registered
    EXPECT_TRUE(isEngineNameRegistered("TEST_UNIQUE_ENGINE_FOR_REGISTRAR"));
    auto id = engineNameToId("TEST_UNIQUE_ENGINE_FOR_REGISTRAR");
    EXPECT_EQ(getEngineNameFromId(id), "TEST_UNIQUE_ENGINE_FOR_REGISTRAR");
}

TEST_F(TestEngineNames, RegistrarThrowsOnDuplicateName)
{
    // These names are already in the map from HIPDNN_REGISTER_ENGINE static initialization.
    // Re-registering should throw to catch accidental duplicate engine definitions.
    const std::string_view miopenName{MIOPEN_ENGINE_NAME};
    EXPECT_THROW(EngineRegistrar{miopenName}, std::runtime_error);
}

TEST_F(TestEngineNames, RegistrarDetectsCollision)
{
    // Simulate a collision by inserting a fake entry into the ID map
    // with the same ID that "COLLISION_TEST_ENGINE" would generate,
    // but mapped to a different name
    auto collisionId = engineNameToId("COLLISION_TEST_ENGINE");
    detail::getMutableEngineIdToNameMap()[collisionId] = "SOME_OTHER_ENGINE";

    // Now registering "COLLISION_TEST_ENGINE" should throw because the ID
    // is already taken by "SOME_OTHER_ENGINE"
    EXPECT_THROW(EngineRegistrar{"COLLISION_TEST_ENGINE"}, std::runtime_error);

    // Clean up: remove the fake entry so other tests aren't affected
    detail::getMutableEngineIdToNameMap().erase(collisionId);
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
}

TEST_F(TestEngineNames, MacroSingleArgGeneratesCorrectName)
{
    // Single-argument form: _NAME should be the stringified identifier
    EXPECT_STREQ(test_macro_expansion::TEST_MACRO_SINGLE_ARG_ENGINE_NAME,
                 "TEST_MACRO_SINGLE_ARG_ENGINE");
}

TEST_F(TestEngineNames, MacroSingleArgGeneratesCorrectId)
{
    // Single-argument form: _ID should be computed from the stringified identifier
    EXPECT_EQ(test_macro_expansion::TEST_MACRO_SINGLE_ARG_ENGINE_ID,
              engineNameToId("TEST_MACRO_SINGLE_ARG_ENGINE"));
}

TEST_F(TestEngineNames, MacroSingleArgRegistersEngine)
{
    // Single-argument form: engine should be registered with the stringified identifier
    EXPECT_TRUE(isEngineNameRegistered("TEST_MACRO_SINGLE_ARG_ENGINE"));
    EXPECT_EQ(getEngineNameFromId(test_macro_expansion::TEST_MACRO_SINGLE_ARG_ENGINE_ID),
              "TEST_MACRO_SINGLE_ARG_ENGINE");
}

TEST_F(TestEngineNames, MacroDualArgGeneratesCorrectName)
{
    // Two-argument form: _NAME should be the custom display name, not the identifier
    EXPECT_STREQ(test_macro_expansion::TEST_MACRO_DUAL_ARG_ENGINE_NAME, "CustomDisplayName");
}

TEST_F(TestEngineNames, MacroDualArgGeneratesCorrectId)
{
    // Two-argument form: _ID should be computed from the custom display name
    EXPECT_EQ(test_macro_expansion::TEST_MACRO_DUAL_ARG_ENGINE_ID,
              engineNameToId("CustomDisplayName"));
    // Verify it's NOT computed from the identifier
    EXPECT_NE(test_macro_expansion::TEST_MACRO_DUAL_ARG_ENGINE_ID,
              engineNameToId("TEST_MACRO_DUAL_ARG_ENGINE"));
}

TEST_F(TestEngineNames, MacroDualArgRegistersWithDisplayName)
{
    // Two-argument form: engine should be registered with the display name, not the identifier
    EXPECT_TRUE(isEngineNameRegistered("CustomDisplayName"));
    EXPECT_FALSE(isEngineNameRegistered("TEST_MACRO_DUAL_ARG_ENGINE"));
    EXPECT_EQ(getEngineNameFromId(test_macro_expansion::TEST_MACRO_DUAL_ARG_ENGINE_ID),
              "CustomDisplayName");
}

TEST_F(TestEngineNames, MacroMatchingArgsEquivalentToSingleArg)
{
    // Two-argument form with matching args should behave the same as single-argument form
    EXPECT_STREQ(test_macro_expansion::TEST_MACRO_MATCHING_ARGS_ENGINE_NAME,
                 "TEST_MACRO_MATCHING_ARGS_ENGINE");
    EXPECT_EQ(test_macro_expansion::TEST_MACRO_MATCHING_ARGS_ENGINE_ID,
              engineNameToId("TEST_MACRO_MATCHING_ARGS_ENGINE"));
    EXPECT_TRUE(isEngineNameRegistered("TEST_MACRO_MATCHING_ARGS_ENGINE"));
}
