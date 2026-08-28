// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Enforcement lifecycle inside TestBody(): the claim query runs before anything can
// short-circuit, and an accepted claim is only published as confirmed support once
// the engine has actually run the graph green.
//
// These drive the real TestBody() through a deviceless harness. The claim query is
// stubbed at the observeSupportForBundle() seam — the seam exists precisely because
// the real one needs a handle — so the assertions cover the routing and the
// two-phase commit, not the backend.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/TestConfig.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/SupportClaimReport.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_ID = 11;
const std::string ENGINE_NAME = "ENGINE_UNDER_TEST";
constexpr float K_OUTPUT_VALUE = 3.5f;
constexpr int64_t K_OUTPUT_UID = 5;
constexpr size_t K_OUTPUT_ELEMS = 120;

LoadedEngine makeEngineUnderTest()
{
    return LoadedEngine{ENGINE_ID, ENGINE_NAME};
}

SupportResult makeVerdict(SupportVerdict verdict, std::string engineName = ENGINE_NAME)
{
    SupportResult result;
    result.verdict = verdict;
    result.bundlePath = "test/bundle";
    result.engineName = std::move(engineName);
    result.arch = "gfx942";
    result.platform = "linux";
    result.detail = "stub";
    return result;
}

// Drives the real TestBody() with a canned observation and a stubbed engine.
class EnforcementHarness : public IntegrationBundleVerificationHarness
{
public:
    EnforcementHarness(VerificationMode mode, SupportObservation observation, bool engineSucceeds)
        : IntegrationBundleVerificationHarness(/*requiresDevice=*/false, makeEngineUnderTest())
        , _mode(mode)
        , _observation(std::move(observation))
        , _engineSucceeds(engineSucceeds)
    {
    }

    using IntegrationBundleVerificationHarness::SetUp;
    using IntegrationBundleVerificationHarness::TestBody;

    bool comparisonReached() const
    {
        return _comparisonReached;
    }

protected:
    VerificationMode getVerificationMode() const override
    {
        return _mode;
    }

    bool isEnforcingSupportClaims() const override
    {
        return true;
    }

    SupportObservation observeSupportForBundle() override
    {
        return _observation;
    }

    void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack) override
    {
        _comparisonReached = true;
        auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
        std::fill(
            ptr, ptr + K_OUTPUT_ELEMS, _engineSucceeds ? K_OUTPUT_VALUE : K_OUTPUT_VALUE + 100.0f);
    }

    void runReferenceExecutor(ReferenceExecutorType, std::unordered_map<int64_t, void*>&) override
    {
    }

    std::unique_ptr<IReferenceGraphExecutor> makeReferenceExecutor(ReferenceExecutorType) override
    {
        return nullptr;
    }

    void applyMetadataGuards() const override {}

    void enforceAtLevel(EnforcementLevel) override
    {
        skipUnverifiable("enforceAtLevel stubbed (deviceless)");
    }

private:
    VerificationMode _mode;
    SupportObservation _observation;
    bool _engineSucceeds;
    bool _comparisonReached = false;
};

class TestSupportClaimEnforcement : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        // The harness reads tolerances and skip lists through the TestConfig
        // singleton, which throws until somebody initializes it. Another suite may
        // already have; initializing twice is itself an error, so ask first.
        if(!TestConfig::isInitialized())
        {
            TestConfig::initialize(TestConfigOptions{});
        }

        supportClaimCoverage() = {};
        SupportClaimVerdicts::get().clear();

        auto path
            = std::filesystem::temp_directory_path()
              / ("claim_enforcement_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    void TearDown() override
    {
        supportClaimCoverage() = {};
        SupportClaimVerdicts::get().clear();
    }

    static void writeBundleFiles(const std::filesystem::path& dir,
                                 const std::string& name,
                                 bool includeGoldenOutput)
    {
        std::filesystem::create_directories(dir);
        std::ofstream(dir / (name + ".json"))
            << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
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

        std::ofstream(dir / (name + ".meta.json"))
            << R"({"format_version": 1, "operation": "BatchnormInference"})";

        const auto basePath = (dir / name).string();
        const auto writeFloatBin = [&](int64_t uid, size_t elems, float value) {
            const std::vector<float> data(elems, value);
            std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin",
                              std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size() * sizeof(float)));
        };

        writeFloatBin(0, 120, 0.0f);
        writeFloatBin(1, 3, 0.0f);
        writeFloatBin(2, 3, 0.0f);
        writeFloatBin(3, 3, 0.0f);
        writeFloatBin(4, 3, 0.0f);

        if(includeGoldenOutput)
        {
            writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE);
        }
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        const auto dir = _tempDir / name;
        writeBundleFiles(dir, name, includeGoldenOutput);
        auto result = loadIntegrationTestBundle(dir / (name + ".json"));
        EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
        return std::make_shared<IntegrationTestBundle>(
            std::move(std::get<IntegrationTestBundle>(result)));
    }

    // A locator whose sidecar really exists, so shouldEnforceClaims() is satisfied.
    SupportClaimLocator makeLocator() const
    {
        const auto path = _tempDir / "Bundle.support.json";
        std::ofstream(path) << R"({"version": 1, "claims": {}})";

        SupportClaimLocator locator;
        locator.sidecarPath = path;
        locator.diagnosticPath = "test/bundle";
        return locator;
    }

    void run(VerificationMode mode,
             SupportObservation observation,
             bool includeGoldenOutput,
             bool engineSucceeds,
             ::testing::TestPartResultArray* results,
             bool* comparisonReached = nullptr)
    {
        EnforcementHarness harness(mode, std::move(observation), engineSucceeds);
        harness.setBundle(loadBundle("Bundle", includeGoldenOutput), "test/bundle", makeLocator());

        {
            const ::testing::ScopedFakeTestPartResultReporter reporter(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
            harness.SetUp();
            harness.TestBody();
        }

        if(comparisonReached != nullptr)
        {
            *comparisonReached = harness.comparisonReached();
        }
    }

    static bool anyFailed(const ::testing::TestPartResultArray& results)
    {
        for(int i = 0; i < results.size(); ++i)
        {
            if(results.GetTestPartResult(i).failed())
            {
                return true;
            }
        }
        return false;
    }

    static SupportObservation observed(std::vector<SupportResult> results)
    {
        return SupportObservation{/*sidecarChecked=*/true, std::move(results)};
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Coverage: the query happens above every short-circuit in runComparison().
//
// This is the regression for the original defect. Before the hoist, a FULL bundle
// with no golden blob under --verification-mode=golden returned at the mode
// dispatch, never queried its claims, contributed nothing to the summary, and the
// run still exited 0 as long as one other graph had been queried.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, GoldenModeWithoutDataStillQueriesClaims)
{
    ::testing::TestPartResultArray results;
    bool comparisonReached = false;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/false,
        /*engineSucceeds=*/true,
        &results,
        &comparisonReached);

    EXPECT_FALSE(comparisonReached);
    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 1u);
}

TEST_F(TestSupportClaimEnforcement, NonFullBundleStillQueriesClaims)
{
    ::testing::TestPartResultArray results;

    EnforcementHarness harness(VerificationMode::AUTO,
                               observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
                               /*engineSucceeds=*/true);
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::APPLICABILITY;
    harness.setBundle(bundle, "test/bundle", makeLocator());

    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 1u);
}

// A sidecar that was read but produced no enforceable verdict still counts as
// coverage; deriving the counter from the verdict vector would report a gap the run
// cannot close.
TEST_F(TestSupportClaimEnforcement, EvaluatedSidecarWithNoVerdictsStillCountsAsQueried)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::AUTO, observed({}), true, true, &results);

    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 0u);
}

// The per-graph invariant. The run-level guard only fires when nothing anywhere was
// queried, so a partial gap needs its own signal.
TEST_F(TestSupportClaimEnforcement, UnqueriedSidecarFailsTheTest)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::AUTO,
        SupportObservation{/*sidecarChecked=*/false, {}},
        true,
        true,
        &results);

    EXPECT_TRUE(anyFailed(results));
    EXPECT_EQ(supportClaimCoverage().graphsQueried, 0u);
}

// ---------------------------------------------------------------------------
// A broken claim is terminal before the comparison runs.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, BrokenClaimFailsBeforeReachingTheEngine)
{
    ::testing::TestPartResultArray results;
    bool comparisonReached = false;
    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::CLAIM_BROKEN)}),
        true,
        true,
        &results,
        &comparisonReached);

    EXPECT_TRUE(anyFailed(results));
    // Running the comparison after a decline only stacks a sentinel-buffer diff on
    // top of the real diagnostic.
    EXPECT_FALSE(comparisonReached);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
}

TEST_F(TestSupportClaimEnforcement, ErroredQueryFailsBeforeReachingTheEngine)
{
    ::testing::TestPartResultArray results;
    bool comparisonReached = false;
    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::QUERY_ERRORED)}),
        true,
        true,
        &results,
        &comparisonReached);

    EXPECT_TRUE(anyFailed(results));
    EXPECT_FALSE(comparisonReached);
}

// ---------------------------------------------------------------------------
// Two-phase commit: an accepted claim is an observation about the ranked list, not
// a verification. Only execution can promote it.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, AcceptedBecomesConfirmedWhenTheTestPasses)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_CONFIRMED), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_ACCEPTED), 0u);
}

// The promotion policy itself. Kept off the TestBody() path on purpose:
// ScopedFakeTestPartResultReporter diverts failures and skips away from the
// enclosing test's result, so HasFailure()/IsSkipped() cannot observe a stubbed
// engine's outcome from inside a fixture. The mapping is the part with the
// decisions in it, so it is pinned directly.
TEST_F(TestSupportClaimEnforcement, PromotionMapsOutcomeToVerdict)
{
    // Ran and green — the only combination that is evidence of working support.
    EXPECT_EQ(promoteAcceptedClaim(/*exercised=*/true, /*passed=*/true),
              SupportVerdict::CLAIM_CONFIRMED);

    // The defect this phase exists for: the engine accepted the graph and then got
    // it wrong. Publishing that cell as satisfied support would feed a support
    // matrix a combination the same run just proved does not work.
    EXPECT_EQ(promoteAcceptedClaim(/*exercised=*/true, /*passed=*/false),
              SupportVerdict::CLAIM_FAILED_IN_USE);

    // Never ran, so the run has no evidence either way; the claim stays where the
    // query left it rather than being confirmed by a test that did nothing.
    EXPECT_EQ(promoteAcceptedClaim(/*exercised=*/false, /*passed=*/true),
              SupportVerdict::CLAIM_ACCEPTED);
    EXPECT_EQ(promoteAcceptedClaim(/*exercised=*/false, /*passed=*/false),
              SupportVerdict::CLAIM_ACCEPTED);
}

// Only the engine this test drove can be promoted. Another engine's claim was
// adjudicated from the same ranked list but never executed, so the run has no
// evidence either way about it.
TEST_F(TestSupportClaimEnforcement, OnlyTheEngineUnderTestIsPromoted)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED),
                  makeVerdict(SupportVerdict::CLAIM_ACCEPTED, "OTHER_ENGINE")}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_CONFIRMED), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_ACCEPTED), 1u);
}

// Non-promotable verdicts pass through the commit untouched.
TEST_F(TestSupportClaimEnforcement, UnclaimedSupportIsRecordedUnchanged)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT)}),
        true,
        true,
        &results);

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
}

// Every observed verdict reaches the report exactly once, including on the terminal
// failure path — the report is the run's dashboard and must not lose rows.
TEST_F(TestSupportClaimEnforcement, EveryVerdictIsRecordedExactlyOnce)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::CLAIM_BROKEN),
                  makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT, "OTHER_ENGINE")}),
        true,
        true,
        &results);

    EXPECT_EQ(SupportClaimVerdicts::get().total(), 2u);
}

// NOLINTEND(readability-identifier-naming)
