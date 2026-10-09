// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPattern.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

/**
 * @file TestUedFixtureParity.cpp
 * @brief The runtime side of the UED parity contract.
 *
 * The descriptor packager and the runtime loader each validate a UED. The shared fixture
 * corpus (the `ued_fixtures` JSON files, each `{"description", "valid", "ued" | "ued_text"}`)
 * states one verdict per document, and the packager's tests hold it to the same corpus, so
 * a document one side accepts and the other refuses fails one of the two suites. The JSON
 * Schema published for authors is held to the loader's key lists and limits here as well.
 *
 * Each invalid fixture also names the error it must raise, so it cannot pass because of an
 * unrelated defect. The text is a plain substring, matched literally:
 *  - `error` when the runtime and the packager share the wording.
 *  - `runtime_error` and `packager_error` when they word it differently. This test reads
 *    `runtime_error` and checks the loader's error log for it.
 * A valid fixture carries none of these keys.
 *
 * The corpus and schema are staged under `share/hipdnn_plugin_sdk/` beside the test binary's
 * `bin/` directory, in the build tree and in an install alike.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

struct UedFixture
{
    std::string name;
    std::filesystem::path path;
};

std::filesystem::path sharedDataDirectory()
{
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / ".." / "share"
           / "hipdnn_plugin_sdk";
}

std::filesystem::path fixtureDirectory()
{
    return sharedDataDirectory() / "test_fixtures" / "ued";
}

/// Every `*.json` file in the corpus, sorted so the parameter order is stable. Returns an empty
/// list if the directory is missing. FixtureCorpusHoldsBothVerdicts reports that as a failure.
std::vector<UedFixture> listFixtures()
{
    std::vector<UedFixture> fixtures;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(fixtureDirectory(), error))
    {
        if(entry.path().extension() != ".json")
        {
            continue;
        }
        // gtest parameter names admit only alphanumerics and underscores.
        auto name = entry.path().stem().string();
        std::replace_if(
            name.begin(), name.end(), [](unsigned char c) { return std::isalnum(c) == 0; }, '_');
        fixtures.push_back(UedFixture{name, entry.path()});
    }
    std::sort(fixtures.begin(), fixtures.end(), [](const UedFixture& a, const UedFixture& b) {
        return a.name < b.name;
    });
    return fixtures;
}

nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    return nlohmann::json::parse(file);
}

/// The text the loader reads for a fixture. Uses `ued_text` verbatim when present, otherwise
/// serializes `ued`. `ued_text` holds documents plain JSON cannot, such as a repeated key.
std::string uedText(const nlohmann::json& fixture)
{
    const auto text = fixture.find("ued_text");
    return text != fixture.end() ? text->get<std::string>() : fixture.at("ued").dump(2);
}

std::set<std::string> keysOf(const nlohmann::json& object)
{
    std::set<std::string> keys;
    for(const auto& item : object.items())
    {
        keys.insert(item.key());
    }
    return keys;
}

std::set<std::string> stringsOf(const nlohmann::json& array)
{
    return array.get<std::set<std::string>>();
}

template <size_t N>
std::set<std::string> setOf(const std::array<std::string_view, N>& values)
{
    return std::set<std::string>(values.begin(), values.end());
}

nlohmann::json readSchema()
{
    return readJson(sharedDataDirectory() / "schemas" / "universal_engine_descriptor.schema.json");
}

/// The schema object describing one `graph_match.nodes` entry.
const nlohmann::json& patternNodeSchema(const nlohmann::json& schema)
{
    return schema.at("properties").at("graph_match").at("properties").at("nodes").at("items");
}

/// The `{"one_of": [...]}` arm of a node's `op`. The other arm is a bare string.
const nlohmann::json& opcodeSetSchema(const nlohmann::json& schema)
{
    for(const auto& arm : patternNodeSchema(schema).at("properties").at("op").at("oneOf"))
    {
        if(arm.value("type", "") == "object")
        {
            return arm;
        }
    }
    throw std::runtime_error("the schema's 'op' has no object arm");
}

class TestUedFixtureParity : public ::testing::TestWithParam<UedFixture>
{
};

} // namespace

/// Loads each fixture through the file entry point, so it gets every check an installed
/// descriptor gets, from duplicate-key rejection through pattern compilation. No native
/// symbols are registered, so native fixtures are checked for structure only.
TEST_P(TestUedFixtureParity, RuntimeAgreesWithTheFixtureVerdict)
{
    const auto fixture = readJson(GetParam().path);
    ASSERT_TRUE(fixture.contains("valid") && fixture.at("valid").is_boolean()) << GetParam().path;
    ASSERT_NE(fixture.contains("ued"), fixture.contains("ued_text"))
        << GetParam().path << " must carry exactly one of 'ued' and 'ued_text'";
    const bool expected = fixture.at("valid").get<bool>();

    const bool sharedError = fixture.contains("error");
    const bool splitError = fixture.contains("runtime_error") || fixture.contains("packager_error");
    ASSERT_EQ(sharedError || splitError, !expected)
        << GetParam().path
        << (expected ? " is valid and must not name an error"
                     : " is invalid and must name the error it raises");
    ASSERT_FALSE(sharedError && splitError)
        << GetParam().path << " must carry 'error' or the 'runtime_error' and 'packager_error' "
        << "pair, not both";
    if(splitError)
    {
        ASSERT_TRUE(fixture.contains("runtime_error") && fixture.contains("packager_error"))
            << GetParam().path << " must carry both 'runtime_error' and 'packager_error'";
    }
    const std::string expectedError = sharedError ? fixture.at("error").get<std::string>()
                                                  : fixture.value("runtime_error", std::string());
    ASSERT_EQ(expectedError.empty(), expected) << GetParam().path;

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const hipdnn_test_sdk::utilities::ScopedDirectory dir
        = hipdnn_test_sdk::utilities::claimScratchDirectory("ued_fixture_parity");
    std::ofstream(dir.path() / "fixture.ued.json", std::ios::binary) << uedText(fixture);

    const auto catalog = loadDescriptorCatalog(dir.path());

    EXPECT_EQ(catalog.engines.size() == 1, expected) << fixture.value("description", "") << "\n"
                                                     << recorder.getRecordedLogsAsString();
    if(!expected)
    {
        EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, expectedError))
            << "expected an error containing: " << expectedError << "\n"
            << recorder.getRecordedLogsAsString();
    }
}

INSTANTIATE_TEST_SUITE_P(Fixtures,
                         TestUedFixtureParity,
                         ::testing::ValuesIn(listFixtures()),
                         [](const ::testing::TestParamInfo<UedFixture>& info) {
                             return info.param.name;
                         });

/// Guards against a missing corpus, or one with a single verdict, letting the parameterized
/// test pass without comparing anything.
TEST(TestUedFixtureParity, FixtureCorpusHoldsBothVerdicts)
{
    size_t valid = 0;
    size_t invalid = 0;
    for(const auto& fixture : listFixtures())
    {
        if(readJson(fixture.path).at("valid").get<bool>())
        {
            ++valid;
        }
        else
        {
            ++invalid;
        }
    }
    EXPECT_GT(valid, 0u) << fixtureDirectory();
    EXPECT_GT(invalid, 0u) << fixtureDirectory();
}

TEST(TestUedFixtureParity, SchemaTopLevelKeysMatchTheLoader)
{
    const auto schema = readSchema();

    EXPECT_EQ(keysOf(schema.at("properties")), setOf(UED_KEYS));
    EXPECT_EQ(stringsOf(schema.at("required")), setOf(UED_REQUIRED_KEYS));
}

/// The loader requires exactly one arm. The schema expresses this as a oneOf of single-key
/// `required` arms, with no top-level `required` list.
TEST(TestUedFixtureParity, SchemaGraphMatchArmsMatchTheLoader)
{
    const auto schema = readSchema();
    const auto& graphMatch = schema.at("properties").at("graph_match");

    EXPECT_EQ(keysOf(graphMatch.at("properties")), setOf(GRAPH_MATCH_KEYS));
    EXPECT_FALSE(graphMatch.contains("required"));

    std::set<std::string> arms;
    for(const auto& arm : graphMatch.at("oneOf"))
    {
        const auto required = stringsOf(arm.at("required"));
        ASSERT_EQ(required.size(), 1u) << arm.dump();
        arms.insert(*required.begin());
    }
    EXPECT_EQ(graphMatch.at("oneOf").size(), GRAPH_MATCH_KEYS.size());
    EXPECT_EQ(arms, setOf(GRAPH_MATCH_KEYS));
}

TEST(TestUedFixtureParity, SchemaPatternNodeKeysMatchTheParser)
{
    const auto schema = readSchema();
    const auto& node = patternNodeSchema(schema);

    EXPECT_EQ(keysOf(node.at("properties")), setOf(PATTERN_NODE_KEYS));
    EXPECT_EQ(stringsOf(node.at("required")), setOf(PATTERN_NODE_REQUIRED_KEYS));
    EXPECT_EQ(keysOf(opcodeSetSchema(schema).at("properties")), setOf(PATTERN_OPCODE_SET_KEYS));
}

TEST(TestUedFixtureParity, SchemaPatternLimitsMatchTheParser)
{
    const auto schema = readSchema();
    const auto& nodes = schema.at("properties").at("graph_match").at("properties").at("nodes");

    EXPECT_EQ(nodes.at("maxItems").get<size_t>(), MAX_PATTERN_NODES);
    EXPECT_EQ(opcodeSetSchema(schema).at("properties").at("one_of").at("maxItems").get<size_t>(),
              MAX_PATTERN_OPCODE_SET);
}

/// Each object the runtime closes to unknown keys allows, in the schema, exactly the
/// extension keys that detail::isExtensionKey accepts.
TEST(TestUedFixtureParity, SchemaExtensionKeysMatchTheLoader)
{
    const auto schema = readSchema();
    const std::array<const nlohmann::json*, 4> closedObjects{
        &schema,
        &schema.at("properties").at("graph_match"),
        &patternNodeSchema(schema),
        &opcodeSetSchema(schema)};
    const std::array<std::string_view, 8> probes{
        "x-build", "x-", "_note", "_", "provenance", "provenance_extra", "xbuild", "note"};

    for(const auto* object : closedObjects)
    {
        ASSERT_EQ(object->at("additionalProperties"), false) << object->dump();
        std::vector<std::regex> patterns;
        const auto patternProperties = object->value("patternProperties", nlohmann::json::object());
        for(const auto& item : patternProperties.items())
        {
            patterns.emplace_back(item.key(), std::regex::ECMAScript);
        }
        for(const auto probe : probes)
        {
            const std::string key(probe);
            const bool schemaAllows
                = std::any_of(patterns.begin(), patterns.end(), [&key](const std::regex& pattern) {
                      return std::regex_search(key, pattern);
                  });
            EXPECT_EQ(schemaAllows, hipdnn_plugin_sdk::ingestor::detail::isExtensionKey(probe))
                << "key '" << key << "' in " << object->value("description", object->dump());
        }
    }
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
