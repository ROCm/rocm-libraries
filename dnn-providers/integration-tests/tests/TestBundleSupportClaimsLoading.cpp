// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for how IntegrationTestBundle wires RFC 0015 support claims into
// the loader: single-graph {Name}.support.json and template-sweep support.json
// both populate `IntegrationTestBundle::supportClaims`, and §6.2's hard
// pre-commit error (a claim exists but enforcement_level is missing/invalid)
// fails the load rather than silently defaulting to Full.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include <nlohmann/json.hpp>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

using namespace hipdnn_integration_tests::bundle;
using hipdnn_integration_tests::EnforcementLevel;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// A minimal, schema-valid batchnorm-inference graph -- no placeholders, so
// it is valid verbatim as either a direct bundle's {Name}.json or a sweep's
// graph.template.json.
const char* const kMinimalGraphJson
    = R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
      R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
      R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
      R"("compute_data_type": "float", "name": ""}], "tensors": [)"
      R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
      R"("data_type": "float", "virtual": false}, )"
      R"({"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
      R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
      R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";

std::filesystem::path makeTempDir(const std::string& label)
{
    auto path
        = std::filesystem::temp_directory_path() / (label + "_" + std::to_string(std::rand()));
    std::filesystem::remove_all(path);
    return path;
}

} // namespace

// ---------------------------------------------------------------------------
// Direct bundle
// ---------------------------------------------------------------------------

TEST(TestBundleSupportClaimsLoading, DirectBundleWithNoSupportJsonIsNotClaimBearing)
{
    const ScopedDirectory dir(makeTempDir("direct_bundle"));
    const auto jsonPath = dir.path() / "Small.json";
    std::ofstream(jsonPath) << kMinimalGraphJson;
    std::ofstream(dir.path() / "Small.meta.json") << R"({"format_version": 1})";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    EXPECT_FALSE(bundle.supportClaims.has_value());
    EXPECT_EQ(bundle.metadata.enforcementLevel, EnforcementLevel::Full);
}

TEST(TestBundleSupportClaimsLoading, DirectBundleWithSupportJsonButNoEnforcementLevelFailsToLoad)
{
    const ScopedDirectory dir(makeTempDir("direct_bundle"));
    const auto jsonPath = dir.path() / "Small.json";
    std::ofstream(jsonPath) << kMinimalGraphJson;
    std::ofstream(dir.path() / "Small.meta.json") << R"({"format_version": 1})";
    std::ofstream(dir.path() / "Small.support.json")
        << R"({"version": 1, "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux"]}}})";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_ENFORCEMENT_LEVEL);
}

TEST(TestBundleSupportClaimsLoading,
     DirectBundleWithSupportJsonAndExplicitEnforcementLevelLoadsWithClaims)
{
    const ScopedDirectory dir(makeTempDir("direct_bundle"));
    const auto jsonPath = dir.path() / "Small.json";
    std::ofstream(jsonPath) << kMinimalGraphJson;
    std::ofstream(dir.path() / "Small.meta.json")
        << R"({"format_version": 1, "enforcement_level": "applicability"})";
    std::ofstream(dir.path() / "Small.support.json")
        << R"({"version": 1, "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux"]}}})";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    ASSERT_TRUE(bundle.supportClaims.has_value());
    EXPECT_TRUE(bundle.supportClaims->isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_EQ(bundle.metadata.enforcementLevel, EnforcementLevel::Applicability);
}

TEST(TestBundleSupportClaimsLoading, DirectBundleWithEmptyClaimsMapIsStillClaimBearing)
{
    // §5.3: an empty `claims` map is a legal, present support.json -- still
    // claim-bearing for §6.2's hard-error purposes, even though it asserts
    // nothing.
    const ScopedDirectory dir(makeTempDir("direct_bundle"));
    const auto jsonPath = dir.path() / "Small.json";
    std::ofstream(jsonPath) << kMinimalGraphJson;
    std::ofstream(dir.path() / "Small.meta.json") << R"({"format_version": 1})";
    std::ofstream(dir.path() / "Small.support.json") << R"({"version": 1, "claims": {}})";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_ENFORCEMENT_LEVEL);
}

// ---------------------------------------------------------------------------
// Template-sweep case
// ---------------------------------------------------------------------------

namespace
{

DiscoveredBundle makeSweepDiscovered(const std::filesystem::path& sweepDir,
                                     const std::string& caseId)
{
    DiscoveredBundle discovered;
    discovered.jsonPath = sweepDir / "sweep.json";
    discovered.suiteName = "TestSuite";
    discovered.testName = caseId;
    discovered.sweep = SweepCase{sweepDir / "graph.template.json", caseId};
    return discovered;
}

} // namespace

TEST(TestBundleSupportClaimsLoading, SweepCaseNotNamedInAnySupportGroupIsNotClaimBearing)
{
    const ScopedDirectory dir(makeTempDir("sweep_bundle"));
    std::ofstream(dir.path() / "graph.template.json") << kMinimalGraphJson;

    nlohmann::json caseJson;
    caseJson["id"] = "case_a";
    caseJson["values"] = nlohmann::json::object();
    caseJson["metadata"] = {{"format_version", 1}};
    nlohmann::json sweepJson;
    sweepJson["version"] = 1;
    sweepJson["cases"] = nlohmann::json::array({caseJson});
    std::ofstream(dir.path() / "sweep.json") << sweepJson.dump();

    // support.json exists, but names only "case_b" -- "case_a" is untouched.
    std::ofstream(dir.path() / "support.json")
        << R"({"version": 1, "claims": {"MIOPEN_ENGINE": [)"
           R"({"cases": ["case_b"], "support": {"gfx942": ["linux"]}}]}})";

    auto result = loadIntegrationTestBundle(makeSweepDiscovered(dir.path(), "case_a"));
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    EXPECT_FALSE(bundle.supportClaims.has_value());
    // Not claim-bearing for this case -> enforcement_level default is legal
    // even though it was never specified.
    EXPECT_EQ(bundle.metadata.enforcementLevel, EnforcementLevel::Full);
}

TEST(TestBundleSupportClaimsLoading, SweepCaseNamedInGroupWithoutEnforcementLevelFailsToLoad)
{
    const ScopedDirectory dir(makeTempDir("sweep_bundle"));
    std::ofstream(dir.path() / "graph.template.json") << kMinimalGraphJson;

    nlohmann::json caseJson;
    caseJson["id"] = "case_a";
    caseJson["values"] = nlohmann::json::object();
    caseJson["metadata"] = {{"format_version", 1}}; // no enforcement_level
    nlohmann::json sweepJson;
    sweepJson["version"] = 1;
    sweepJson["cases"] = nlohmann::json::array({caseJson});
    std::ofstream(dir.path() / "sweep.json") << sweepJson.dump();

    std::ofstream(dir.path() / "support.json")
        << R"({"version": 1, "claims": {"MIOPEN_ENGINE": [)"
           R"({"cases": ["case_a"], "support": {"gfx942": ["linux"]}}]}})";

    auto result = loadIntegrationTestBundle(makeSweepDiscovered(dir.path(), "case_a"));
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_ENFORCEMENT_LEVEL);
}

TEST(TestBundleSupportClaimsLoading,
     SweepCaseNamedInGroupWithExplicitEnforcementLevelLoadsWithProjectedClaims)
{
    const ScopedDirectory dir(makeTempDir("sweep_bundle"));
    std::ofstream(dir.path() / "graph.template.json") << kMinimalGraphJson;

    nlohmann::json caseJson;
    caseJson["id"] = "case_a";
    caseJson["values"] = nlohmann::json::object();
    caseJson["metadata"] = {{"format_version", 1}, {"enforcement_level", "buildable"}};
    nlohmann::json sweepJson;
    sweepJson["version"] = 1;
    sweepJson["cases"] = nlohmann::json::array({caseJson});
    std::ofstream(dir.path() / "sweep.json") << sweepJson.dump();

    std::ofstream(dir.path() / "support.json")
        << R"({"version": 1, "claims": {"MIOPEN_ENGINE": [)"
           R"({"cases": ["case_a"], "support": {"gfx942": ["linux"]}}]}})";

    auto result = loadIntegrationTestBundle(makeSweepDiscovered(dir.path(), "case_a"));
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    ASSERT_TRUE(bundle.supportClaims.has_value());
    EXPECT_TRUE(bundle.supportClaims->isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_EQ(bundle.metadata.enforcementLevel, EnforcementLevel::Buildable);
}

// NOLINTEND(readability-identifier-naming)
