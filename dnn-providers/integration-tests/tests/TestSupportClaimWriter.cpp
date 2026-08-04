// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for RFC 0015 §9's write-tool data model
// (SupportClaimWriter.hpp): the single-graph and template-sweep
// flatten/overlay/(re-group) merge logic, canonical JSON serialization, and
// the ClaimObservationCollector's grouping + empty-write guard. Everything
// here is GTest-free/handle-free at the production-code level -- these tests
// exercise it directly, with no bundle, harness, or backend involved.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "harness/bundle/SupportClaimWriter.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::bundle;

namespace
{

// Creates a fresh, unique scratch directory under the system temp dir and
// removes it (recursively) on destruction, so file-writing tests never leak
// state into each other or the real filesystem.
class ScopedTempDir
{
public:
    ScopedTempDir()
    {
        static std::atomic<int> counter{0};
        _path = std::filesystem::temp_directory_path()
                / ("hipdnn_scw_test_"
                   + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                   + std::to_string(counter.fetch_add(1)));
        std::filesystem::create_directories(_path);
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(_path, ec);
    }

    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    std::filesystem::path _path;
};

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

} // namespace

// ---------------------------------------------------------------------------
// applyClaimObservations (single-graph)
// ---------------------------------------------------------------------------

TEST(TestApplyClaimObservations, SupportedObservationAddsPlatform)
{
    SupportClaims existing;
    existing.version = 1;

    auto result = applyClaimObservations(
        existing, {{"MIOPEN_ENGINE", "gfx942", "linux", /*supported=*/true}});

    EXPECT_TRUE(result.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST(TestApplyClaimObservations, DeclinedObservationRemovesExistingPlatformAndPrunesEmpties)
{
    SupportClaims existing;
    existing.version = 1;
    existing.claims["MIOPEN_ENGINE"]["gfx942"] = {"linux"};

    auto result = applyClaimObservations(
        existing, {{"MIOPEN_ENGINE", "gfx942", "linux", /*supported=*/false}});

    EXPECT_FALSE(result.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    // The arch and engine are both now empty and must be pruned entirely,
    // not left behind as an empty container.
    EXPECT_EQ(result.claims.find("MIOPEN_ENGINE"), result.claims.end());
}

TEST(TestApplyClaimObservations, UnobservedSiblingCellsAreUntouched)
{
    SupportClaims existing;
    existing.version = 1;
    existing.claims["MIOPEN_ENGINE"]["gfx942"] = {"linux", "windows"};
    existing.claims["MIOPEN_ENGINE"]["gfx90a"] = {"linux"};
    existing.claims["HIP_KERNEL_ENGINE"]["gfx942"] = {"linux"};

    // Only observe one cell this run.
    auto result = applyClaimObservations(
        existing, {{"MIOPEN_ENGINE", "gfx942", "windows", /*supported=*/false}});

    EXPECT_TRUE(result.isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
    EXPECT_FALSE(result.isClaimed("MIOPEN_ENGINE", "gfx942", "windows"));
    EXPECT_TRUE(result.isClaimed("MIOPEN_ENGINE", "gfx90a", "linux"));
    EXPECT_TRUE(result.isClaimed("HIP_KERNEL_ENGINE", "gfx942", "linux"));
}

TEST(TestApplyClaimObservations, ZeroObservationsReproducesExistingExactly)
{
    SupportClaims existing;
    existing.version = 1;
    existing.claims["MIOPEN_ENGINE"]["gfx942"] = {"linux", "windows"};

    auto result = applyClaimObservations(existing, {});

    EXPECT_EQ(toCanonicalJson(result).dump(), toCanonicalJson(existing).dump());
}

// ---------------------------------------------------------------------------
// applySweepClaimObservations (template-sweep flatten/overlay/re-group)
// ---------------------------------------------------------------------------

TEST(TestApplySweepClaimObservations, CasesWithIdenticalObservedSupportShareOneGroup)
{
    SweepSupportClaims existing;
    existing.version = 1;

    std::vector<SweepClaimObservation> observations = {
        {"case_b", "MIOPEN_ENGINE", "gfx942", "linux", true},
        {"case_a", "MIOPEN_ENGINE", "gfx942", "linux", true},
    };

    auto result = applySweepClaimObservations(existing, observations);

    ASSERT_EQ(result.claims.count("MIOPEN_ENGINE"), 1u);
    const auto& groups = result.claims.at("MIOPEN_ENGINE");
    ASSERT_EQ(groups.size(), 1u);
    // Canonical ordering: cases sorted lexicographically within the group.
    EXPECT_EQ(groups[0].cases, (std::vector<std::string>{"case_a", "case_b"}));
}

TEST(TestApplySweepClaimObservations, CasesWithDifferingSupportSplitAndGroupsOrderByFirstCaseId)
{
    SweepSupportClaims existing;
    existing.version = 1;

    std::vector<SweepClaimObservation> observations = {
        {"zzz_case", "MIOPEN_ENGINE", "gfx942", "linux", true},
        {"aaa_case", "MIOPEN_ENGINE", "gfx90a", "linux", true},
    };

    auto result = applySweepClaimObservations(existing, observations);

    const auto& groups = result.claims.at("MIOPEN_ENGINE");
    ASSERT_EQ(groups.size(), 2u);
    // Two distinct support footprints -> two groups, ordered by first case id.
    EXPECT_EQ(groups[0].cases, (std::vector<std::string>{"aaa_case"}));
    EXPECT_EQ(groups[1].cases, (std::vector<std::string>{"zzz_case"}));
}

TEST(TestApplySweepClaimObservations, DecliningTheOnlyPlatformDropsTheCaseEntirely)
{
    SweepSupportClaims existing;
    existing.version = 1;
    existing.claims["MIOPEN_ENGINE"]
        = {SweepClaimGroup{{"small_fp16"}, ArchPlatformMap{{"gfx942", {"linux"}}}}};

    auto result = applySweepClaimObservations(
        existing, {{"small_fp16", "MIOPEN_ENGINE", "gfx942", "linux", /*supported=*/false}});

    // The case's only claimed platform was declined -- it must not appear in
    // any group for this engine any more (an empty claim is not a claim).
    auto it = result.claims.find("MIOPEN_ENGINE");
    if(it != result.claims.end())
    {
        for(const auto& group : it->second)
        {
            EXPECT_EQ(std::find(group.cases.begin(), group.cases.end(), "small_fp16"),
                      group.cases.end());
        }
    }
}

TEST(TestApplySweepClaimObservations, UnobservedEngineArrayIsByteIdenticalAfterOverlay)
{
    SweepSupportClaims existing;
    existing.version = 1;
    existing.claims["HIP_KERNEL_ENGINE"]
        = {SweepClaimGroup{{"case_x"}, ArchPlatformMap{{"gfx942", {"linux"}}}}};
    existing.claims["MIOPEN_ENGINE"]
        = {SweepClaimGroup{{"case_x"}, ArchPlatformMap{{"gfx942", {"linux"}}}}};

    // Only observe MIOPEN_ENGINE this run.
    auto result = applySweepClaimObservations(
        existing, {{"case_x", "MIOPEN_ENGINE", "gfx90a", "windows", /*supported=*/true}});

    ASSERT_EQ(result.claims.count("HIP_KERNEL_ENGINE"), 1u);
    EXPECT_EQ(toCanonicalJson(result).at("claims").at("HIP_KERNEL_ENGINE"),
              toCanonicalJson(existing).at("claims").at("HIP_KERNEL_ENGINE"));
}

// ---------------------------------------------------------------------------
// Canonical JSON formatting
// ---------------------------------------------------------------------------

TEST(TestToCanonicalJson, SortsEngineAndArchKeys)
{
    SupportClaims claims;
    claims.version = 1;
    claims.claims["ZZZ_ENGINE"]["gfx942"] = {"windows", "linux"};
    claims.claims["AAA_ENGINE"]["gfx942"] = {"linux"};

    const auto dumped = toCanonicalJson(claims).dump(2);
    // AAA_ENGINE must precede ZZZ_ENGINE, and platforms within an arch must
    // be sorted ("linux" before "windows").
    EXPECT_LT(dumped.find("AAA_ENGINE"), dumped.find("ZZZ_ENGINE"));
    EXPECT_LT(dumped.find("\"linux\""), dumped.find("\"windows\""));
}

TEST(TestToCanonicalJson, EmptyClaimsMapIsLegal)
{
    SupportClaims claims;
    claims.version = 1;

    const auto json = toCanonicalJson(claims);
    ASSERT_TRUE(json.at("claims").is_object());
    EXPECT_TRUE(json.at("claims").empty());
}

// ---------------------------------------------------------------------------
// File writing: idempotency (zero diff on an unchanged re-run) and locked
// file error reporting.
// ---------------------------------------------------------------------------

TEST(TestWriteSupportClaimsFile, RewritingIdenticalClaimsProducesByteIdenticalFile)
{
    ScopedTempDir dir;
    const auto path = dir.path() / "Foo.support.json";

    SupportClaims claims;
    claims.version = 1;
    claims.claims["MIOPEN_ENGINE"]["gfx942"] = {"linux"};

    writeSupportClaimsFile(path, claims);
    const auto firstWrite = readFile(path);

    writeSupportClaimsFile(path, claims);
    const auto secondWrite = readFile(path);

    EXPECT_EQ(firstWrite, secondWrite);
    // Canonical JSON ends in exactly one trailing newline.
    ASSERT_FALSE(firstWrite.empty());
    EXPECT_EQ(firstWrite.back(), '\n');
    EXPECT_NE(firstWrite[firstWrite.size() - 2], '\n');
}

TEST(TestWriteSupportClaimsFile, ThrowsNamingTheFileWhenParentIsUnwritable)
{
    // A path whose parent does not exist and cannot be created (a regular
    // file standing where a directory component is expected) reproduces the
    // "locked/unwritable file" contract without needing OS-level ACLs.
    ScopedTempDir dir;
    const auto blockerFile = dir.path() / "not_a_directory";
    {
        std::ofstream(blockerFile) << "x";
    }
    const auto path = blockerFile / "Foo.support.json";

    SupportClaims claims;
    claims.version = 1;

    try
    {
        writeSupportClaimsFile(path, claims);
        FAIL() << "expected an exception";
    }
    catch(const std::exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(path.filename().string()), std::string::npos)
            << e.what();
    }
}

// ---------------------------------------------------------------------------
// ClaimObservationCollector
// ---------------------------------------------------------------------------

class TestClaimObservationCollector : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ClaimObservationCollector::get().reset();
    }

    void TearDown() override
    {
        ClaimObservationCollector::get().reset();
    }
};

TEST_F(TestClaimObservationCollector, EmptyByDefault)
{
    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

TEST_F(TestClaimObservationCollector, RecordMakesItNonEmpty)
{
    ClaimObservationCollector::get().record(
        ClaimWriteTarget{"/tmp/Foo.json", std::nullopt}, "MIOPEN_ENGINE", "gfx942", "linux", true);
    EXPECT_FALSE(ClaimObservationCollector::get().empty());
}

TEST_F(TestClaimObservationCollector, WriteAllCreatesFreshSingleGraphSidecar)
{
    ScopedTempDir dir;
    const auto graphJsonPath = dir.path() / "Foo.json";

    ClaimObservationCollector::get().record(
        ClaimWriteTarget{graphJsonPath, std::nullopt}, "MIOPEN_ENGINE", "gfx942", "linux", true);

    auto written = ClaimObservationCollector::get().writeAll();

    ASSERT_EQ(written.size(), 1u);
    const auto sidecarPath = dir.path() / "Foo.support.json";
    EXPECT_EQ(written[0], sidecarPath);
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    auto loaded = loadSupportClaims(graphJsonPath);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->isClaimed("MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST_F(TestClaimObservationCollector, WriteAllCreatesFreshSweepSidecarGroupedByCase)
{
    ScopedTempDir dir;
    // writeAll's sweep path only writes the support.json sidecar -- it does
    // not require a real sweep.json/graph.template.json to exist alongside
    // it (that cross-check is the pre-commit validator's job, not the write
    // tool's).
    ClaimObservationCollector::get().record(
        ClaimWriteTarget{dir.path(), "small_fp16"}, "MIOPEN_ENGINE", "gfx942", "linux", true);

    auto written = ClaimObservationCollector::get().writeAll();

    ASSERT_EQ(written.size(), 1u);
    const auto sidecarPath = dir.path() / "support.json";
    EXPECT_EQ(written[0], sidecarPath);

    auto loaded = loadSweepSupportClaims(dir.path());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->isClaimed("small_fp16", "MIOPEN_ENGINE", "gfx942", "linux"));
}

TEST_F(TestClaimObservationCollector, ResetClearsRecordedObservations)
{
    ClaimObservationCollector::get().record(
        ClaimWriteTarget{"/tmp/Foo.json", std::nullopt}, "MIOPEN_ENGINE", "gfx942", "linux", true);
    ClaimObservationCollector::get().reset();
    EXPECT_TRUE(ClaimObservationCollector::get().empty());
}

// ---------------------------------------------------------------------------
// EnginePassContext
// ---------------------------------------------------------------------------

TEST(TestEnginePassContext, DefaultsToUnset)
{
    EnginePassContext::get().clear();
    EXPECT_FALSE(EnginePassContext::get().name().has_value());
    EXPECT_FALSE(EnginePassContext::get().id().has_value());
}

TEST(TestEnginePassContext, SetThenClearRoundTrips)
{
    EnginePassContext::get().set("MIOPEN_ENGINE", 42);
    EXPECT_EQ(EnginePassContext::get().name(), "MIOPEN_ENGINE");
    EXPECT_EQ(EnginePassContext::get().id(), 42);

    EnginePassContext::get().clear();
    EXPECT_FALSE(EnginePassContext::get().name().has_value());
    EXPECT_FALSE(EnginePassContext::get().id().has_value());
}

// NOLINTEND(readability-identifier-naming)
