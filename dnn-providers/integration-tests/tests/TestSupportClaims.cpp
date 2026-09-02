// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ScratchDirectory.hpp"
#include "harness/bundle/SupportClaims.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

using hipdnn_integration_tests::bundle::loadSupportClaims;
using hipdnn_integration_tests::bundle::loadSweepSupportClaims;
using hipdnn_integration_tests::bundle::parseSupportClaimsJson;
using hipdnn_integration_tests::bundle::parseSweepSupportClaimsJson;
using hipdnn_integration_tests::bundle::SupportClaimLocator;
using hipdnn_integration_tests::bundle::supportJsonPath;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// Builds one §5.4 claim group JSON object: {"cases": [...], "support": {arch: [platforms]}}.
nlohmann::json makeSweepGroup(const std::vector<std::string>& cases,
                              const std::string& arch,
                              const std::vector<std::string>& platforms)
{
    nlohmann::json group;
    group["cases"] = nlohmann::json(cases);
    group["support"][arch] = nlohmann::json(platforms);
    return group;
}

// Unique-per-test scratch directory. Keyed on pid, clock and a counter rather than
// the source line: a line-keyed name is identical across two concurrent runs of this
// binary, and reusing it requires a remove_all() that deletes the other run's
// fixture. See ScratchDirectory.hpp.
ScopedDirectory makeScopedTestDir(const std::string& prefix)
{
    return hipdnn_integration_tests::scratch::makeDir(prefix + "_");
}

} // namespace

// ---------------------------------------------------------------------------
// parseSupportClaimsJson — single-graph shape
// ---------------------------------------------------------------------------

TEST(TestParseSupportClaimsJson, ValidParseBuildsModelAndIsClaimedMatchesListedEntry)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux", "windows"});

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_EQ(claims.version, 1);
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "windows"));
}

TEST(TestParseSupportClaimsJson, IsClaimedFalseForOmittedPlatform)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "windows"));
}

TEST(TestParseSupportClaimsJson, IsClaimedFalseForAbsentArch)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx90a", "linux"));
}

TEST(TestParseSupportClaimsJson, IsClaimedFalseForAbsentEngine)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array({"linux"});

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("OTHER_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSupportClaimsJson, EmptyClaimsMapIsValidWithNoClaims)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"] = nlohmann::json::object();

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_EQ(claims.version, 1);
    EXPECT_TRUE(claims.claims.empty());
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSupportClaimsJson, AbsentClaimsMapIsValidWithNoClaims)
{
    nlohmann::json json;
    json["version"] = 1;

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_EQ(claims.version, 1);
    EXPECT_TRUE(claims.claims.empty());
}

TEST(TestParseSupportClaimsJson, EmptyPlatformArrayIsLegal)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]["gfx942"] = nlohmann::json::array();

    const auto claims = parseSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSupportClaimsJson, ThrowsOnMissingVersion)
{
    nlohmann::json json;
    json["claims"] = nlohmann::json::object();
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsOnUnsupportedVersion)
{
    nlohmann::json json;
    json["version"] = 2;
    json["claims"] = nlohmann::json::object();
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsOnNonIntegerVersion)
{
    nlohmann::json json;
    json["version"] = "1";
    json["claims"] = nlohmann::json::object();
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsWhenClaimsIsNotAnObject)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"] = nlohmann::json::array();
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsWhenArchValueIsNotAnArray)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"]["gfx942"] = "linux";
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsOnInvalidPlatformToken)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"]["gfx942"] = nlohmann::json::array({"macos"});
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSupportClaimsJson, ThrowsWhenJsonIsNotAnObject)
{
    const nlohmann::json json = nlohmann::json::array();
    EXPECT_THROW(parseSupportClaimsJson(json), std::runtime_error);
}

// ---------------------------------------------------------------------------
// supportJsonPath
// ---------------------------------------------------------------------------

TEST(TestSupportJsonPath, DerivesSupportPathFromBundleJson)
{
    EXPECT_EQ(supportJsonPath("dir/Small.json"), std::filesystem::path("dir/Small.support.json"));
}

// ---------------------------------------------------------------------------
// loadSupportClaims — reader
// ---------------------------------------------------------------------------

TEST(TestLoadSupportClaims, ReturnsNulloptWhenSidecarDoesNotExist)
{
    const ScopedDirectory dir = makeScopedTestDir("test_support_claims");
    std::ofstream(dir.path() / "Bundle.json") << "{}";

    const auto claims = loadSupportClaims(dir.path() / "Bundle.json");
    EXPECT_FALSE(claims.has_value());
}

TEST(TestLoadSupportClaims, ThrowsOnUnparseableFile)
{
    const ScopedDirectory dir = makeScopedTestDir("test_support_claims");
    std::ofstream(dir.path() / "Bundle.json") << "{}";
    std::ofstream(dir.path() / "Bundle.support.json") << "{not valid json";

    EXPECT_THROW(loadSupportClaims(dir.path() / "Bundle.json"), std::runtime_error);
}

TEST(TestLoadSupportClaims, LoadsValidSidecar)
{
    const ScopedDirectory dir = makeScopedTestDir("test_support_claims");
    std::ofstream(dir.path() / "Bundle.json") << "{}";
    std::ofstream(dir.path() / "Bundle.support.json")
        << R"({"version": 1, "claims": {"E": {"gfx942": ["linux"]}}})";

    const auto claims = loadSupportClaims(dir.path() / "Bundle.json");
    ASSERT_TRUE(claims.has_value());
    EXPECT_TRUE(claims->isClaimed("E", "gfx942", "linux"));
}

// ---------------------------------------------------------------------------
// parseSweepSupportClaimsJson — template-sweep shape
// ---------------------------------------------------------------------------

TEST(TestParseSweepSupportClaimsJson, ValidParseGroupsCasesAndIsClaimedMatchesListedEntry)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a", "case_b"}, "gfx942", {"linux"})});

    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_EQ(claims.version, 1);
    EXPECT_TRUE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("case_b", "MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSweepSupportClaimsJson, IsClaimedFalseForCaseOutsideAnyGroup)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"linux"})});

    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("case_z", "MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSweepSupportClaimsJson, IsClaimedFalseForPlatformNotInGroup)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"linux"})});

    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_FALSE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "windows"));
}

TEST(TestParseSweepSupportClaimsJson, DuplicateCaseIdWithinOneEngineThrows)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"linux"}),
                                 makeSweepGroup({"case_a"}, "gfx90a", {"linux"})});

    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, SameCaseIdUnderTwoDifferentEnginesIsOk)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["MIOPEN_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"linux"})});
    json["claims"]["HIP_KERNEL_ENGINE"]
        = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"windows"})});

    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_TRUE(claims.isClaimed("case_a", "MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_TRUE(claims.isClaimed("case_a", "HIP_KERNEL_ENGINE", "gfx942", "windows"));
    EXPECT_FALSE(claims.isClaimed("case_a", "HIP_KERNEL_ENGINE", "gfx942", "linux"));
}

TEST(TestParseSweepSupportClaimsJson, EmptyClaimsMapIsValid)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"] = nlohmann::json::object();

    const auto claims = parseSweepSupportClaimsJson(json);
    EXPECT_EQ(claims.version, 1);
    EXPECT_TRUE(claims.claims.empty());
}

TEST(TestParseSweepSupportClaimsJson, ThrowsOnMissingVersion)
{
    nlohmann::json json;
    json["claims"] = nlohmann::json::object();
    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, ThrowsOnUnsupportedVersion)
{
    nlohmann::json json;
    json["version"] = 2;
    json["claims"] = nlohmann::json::object();
    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, ThrowsWhenGroupMissingCases)
{
    nlohmann::json group;
    group["support"]["gfx942"] = nlohmann::json::array({"linux"});

    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"] = nlohmann::json::array({group});

    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, ThrowsWhenGroupHasEmptyCasesArray)
{
    nlohmann::json group;
    group["cases"] = nlohmann::json::array();
    group["support"]["gfx942"] = nlohmann::json::array({"linux"});

    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"] = nlohmann::json::array({group});

    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, ThrowsWhenGroupMissingSupport)
{
    nlohmann::json group;
    group["cases"] = nlohmann::json::array({"case_a"});

    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"] = nlohmann::json::array({group});

    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

TEST(TestParseSweepSupportClaimsJson, ThrowsOnInvalidPlatformTokenInGroupSupport)
{
    nlohmann::json json;
    json["version"] = 1;
    json["claims"]["E"] = nlohmann::json::array({makeSweepGroup({"case_a"}, "gfx942", {"macos"})});

    EXPECT_THROW(parseSweepSupportClaimsJson(json), std::runtime_error);
}

// ---------------------------------------------------------------------------
// loadSweepSupportClaims — reader
// ---------------------------------------------------------------------------

TEST(TestLoadSweepSupportClaims, ReturnsNulloptWhenSupportJsonDoesNotExist)
{
    const ScopedDirectory dir = makeScopedTestDir("test_sweep_support_claims");

    const auto claims = loadSweepSupportClaims(dir.path());
    EXPECT_FALSE(claims.has_value());
}

TEST(TestLoadSweepSupportClaims, LoadsValidSupportJson)
{
    const ScopedDirectory dir = makeScopedTestDir("test_sweep_support_claims");
    std::ofstream(dir.path() / "support.json")
        << R"({"version": 1, "claims": {"E": [{"cases": ["case_a"], )"
           R"("support": {"gfx942": ["linux"]}}]}})";

    const auto claims = loadSweepSupportClaims(dir.path());
    ASSERT_TRUE(claims.has_value());
    EXPECT_TRUE(claims->isClaimed("case_a", "E", "gfx942", "linux"));
}

TEST(TestLoadSweepSupportClaims, ThrowsOnUnparseableFile)
{
    const ScopedDirectory dir = makeScopedTestDir("test_sweep_support_claims");
    std::ofstream(dir.path() / "support.json") << "{not valid json";

    EXPECT_THROW(loadSweepSupportClaims(dir.path()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// SupportClaimLocator — struct tests
// ---------------------------------------------------------------------------

TEST(TestSupportClaimLocator, SingleGraphIsNotSweep)
{
    const SupportClaimLocator target{"/some/Bundle.support.json", {}, "Bundle.json"};
    EXPECT_FALSE(target.isSweep());
    EXPECT_TRUE(target.caseId.empty());
}

TEST(TestSupportClaimLocator, SweepCaseIsSweep)
{
    const SupportClaimLocator target{"/some/support.json", "case_42", "sweep.json#case_42"};
    EXPECT_TRUE(target.isSweep());
    EXPECT_EQ(target.caseId, "case_42");
}

TEST(TestSupportClaimLocator, DefaultConstructedIsNotSweep)
{
    const SupportClaimLocator target{};
    EXPECT_FALSE(target.isSweep());
    EXPECT_TRUE(target.sidecarPath.empty());
}

// NOLINTEND(readability-identifier-naming)
