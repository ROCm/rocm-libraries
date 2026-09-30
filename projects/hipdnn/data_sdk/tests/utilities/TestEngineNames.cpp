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

TEST_F(TestEngineNames, RegisteredRocKEEngineHasStableNameAndId)
{
    EXPECT_STREQ(ROCKE_ENGINE_NAME, "ROCKE_ENGINE");
    EXPECT_EQ(ROCKE_ENGINE_ID, engineNameToId("ROCKE_ENGINE"));
    EXPECT_TRUE(isEngineNameRegistered(ROCKE_ENGINE_NAME));
    EXPECT_EQ(getEngineNameFromId(ROCKE_ENGINE_ID), "ROCKE_ENGINE");
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

TEST_F(TestEngineNames, EngineNameOrIdParsesRegisteredName)
{
    EXPECT_EQ(engineNameOrIdToId(MIOPEN_ENGINE_NAME), MIOPEN_ENGINE_ID);
}

TEST_F(TestEngineNames, EngineNameOrIdPrefersRegisteredNumericName)
{
    [[maybe_unused]] static const EngineRegistrar s_numericEngine("8675309");

    EXPECT_TRUE(isEngineNameRegistered("8675309"));
    EXPECT_EQ(engineNameOrIdToId("8675309"), engineNameToId("8675309"));
    EXPECT_NE(engineNameOrIdToId("8675309"), 8675309);
}

TEST_F(TestEngineNames, EngineNameOrIdParsesDecimalId)
{
    EXPECT_EQ(engineNameOrIdToId("12345"), 12345);
    EXPECT_EQ(engineNameOrIdToId("-18"), -18);
}

TEST_F(TestEngineNames, EngineNameOrIdParsesPlusSignedDecimalId)
{
    EXPECT_EQ(engineNameOrIdToId("+12345"), 12345);
}

TEST_F(TestEngineNames, EngineNameOrIdTreatsBarePlusAsName)
{
    EXPECT_EQ(engineNameOrIdToId("+"), engineNameToId("+"));
}

TEST_F(TestEngineNames, EngineNameOrIdParsesHexId)
{
    EXPECT_EQ(engineNameOrIdToId("0x0000000000003039"), 12345);
    EXPECT_EQ(engineNameOrIdToId("0xffffffffffffffee"), -18);
}

TEST_F(TestEngineNames, EngineNameOrIdTreatsLeadingWhitespaceAsName)
{
    EXPECT_EQ(engineNameOrIdToId(" 12345"), engineNameToId(" 12345"));
    EXPECT_NE(engineNameOrIdToId(" 12345"), 12345);
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

TEST_F(TestEngineNames, EngineNameOrIdToIdParsesPositiveDecimal)
{
    // A plain decimal serialized ID round-trips exactly.
    EXPECT_EQ(engineNameOrIdToId("12345"), 12345);
}

TEST_F(TestEngineNames, EngineNameOrIdToIdRecoversNegativeIdFromSignedHex)
{
    // The frontend serializes an unregistered negative ID (e.g. test plugin -18)
    // as two's-complement unsigned hex. strtoull + reinterpret must recover -18.
    EXPECT_EQ(engineNameOrIdToId("0xffffffffffffffee"), static_cast<int64_t>(-18));
}

TEST_F(TestEngineNames, EngineNameOrIdToIdParsesPositiveHex)
{
    EXPECT_EQ(engineNameOrIdToId("0x10"), 16);
    EXPECT_EQ(engineNameOrIdToId("0X1A2B"), 0x1A2B);
}

TEST_F(TestEngineNames, EngineNameOrIdToIdParsesPlainDecimalNegative)
{
    EXPECT_EQ(engineNameOrIdToId("-18"), static_cast<int64_t>(-18));
}

TEST_F(TestEngineNames, EngineNameOrIdToIdHashesOrdinaryName)
{
    // An ordinary, non-numeric engine name falls through to FNV-1a, matching
    // engineNameToId so registered engines still resolve.
    EXPECT_EQ(engineNameOrIdToId("MIOPEN_ENGINE"), engineNameToId("MIOPEN_ENGINE"));
}

TEST_F(TestEngineNames, EngineNameOrIdToIdTreatsWhitespaceAsNameCharacter)
{
    // Whitespace is a valid name character: a string is only treated as numeric
    // when it parses fully with no whitespace anywhere. Leading, trailing, and
    // internal whitespace all force the FNV-1a name path (matching engineNameToId
    // on the untrimmed string), and an all-whitespace or empty string is a name.
    EXPECT_EQ(engineNameOrIdToId(" 123"), engineNameToId(" 123"));
    EXPECT_EQ(engineNameOrIdToId("\t123"), engineNameToId("\t123"));
    EXPECT_EQ(engineNameOrIdToId("\n123"), engineNameToId("\n123"));
    EXPECT_EQ(engineNameOrIdToId("123 "), engineNameToId("123 "));
    EXPECT_EQ(engineNameOrIdToId("1 2 3"), engineNameToId("1 2 3"));
    EXPECT_EQ(engineNameOrIdToId("   "), engineNameToId("   "));
    EXPECT_EQ(engineNameOrIdToId(""), engineNameToId(""));
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
