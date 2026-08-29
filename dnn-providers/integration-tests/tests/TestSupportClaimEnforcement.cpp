// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Enforcement lifecycle inside TestBody(): the claim query runs before anything can
// short-circuit, and an accepted claim is only published as confirmed support once
// the engine has actually run the graph green.
//
// These drive the real TestBody() through a deviceless harness. The graph and the
// claim verdicts are stubbed at the openGraph() and adjudicateClaims() seams — they
// exist precisely because the real ones need a handle — so the assertions cover the
// routing and the two-phase commit, not the backend.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/ReferenceCapabilityError.hpp"
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

// What the stubbed reference executor does when the fallback chain reaches it.
enum class RefBehavior
{
    MATCHES, ///< writes what the engine wrote, so the comparison passes
    CAPABILITY_MISS, ///< no oracle can run this op — the chain ends in a skip
    ERRORS, ///< the reference itself is broken, which is not the engine's fault
};

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

    void setRefBehavior(RefBehavior behavior)
    {
        _refBehavior = behavior;
    }

    // Stands in for a real enforcement rung, which needs a device to reach.
    void setEnforceOutcome(VerificationOutcome outcome)
    {
        _enforceOutcome = std::move(outcome);
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

    // No graph, no device: the session carries only what the decisions read.
    GraphSession openGraph() override
    {
        GraphSession session;
        session.engines.accepted = true;
        return session;
    }

    SupportObservation adjudicateClaims(const GraphSession&) override
    {
        return _observation;
    }

    void executeGraphThroughEngine(GraphSession& /*session*/,
                                   std::unordered_map<int64_t, void*>& variantPack) override
    {
        _comparisonReached = true;
        auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
        std::fill(
            ptr, ptr + K_OUTPUT_ELEMS, _engineSucceeds ? K_OUTPUT_VALUE : K_OUTPUT_VALUE + 100.0f);
    }

    void runReferenceExecutor(ReferenceExecutorType,
                              std::unordered_map<int64_t, void*>& variantPack) override
    {
        switch(_refBehavior)
        {
        case RefBehavior::CAPABILITY_MISS:
            throw ReferenceCapabilityError("stub: unsupported op");
        case RefBehavior::ERRORS:
            throw std::runtime_error("stub: reference exploded");
        case RefBehavior::MATCHES:
        default:
        {
            auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
            std::fill(ptr, ptr + K_OUTPUT_ELEMS, K_OUTPUT_VALUE);
            return;
        }
        }
    }

    std::unique_ptr<IReferenceGraphExecutor> makeReferenceExecutor(ReferenceExecutorType) override
    {
        return nullptr;
    }

    void applyMetadataGuards() const override {}

    VerificationOutcome enforceAtLevel(EnforcementLevel, GraphSession& /*session*/) override
    {
        if(_enforceOutcome.has_value())
        {
            return *_enforceOutcome;
        }
        return unverifiable("enforceAtLevel stubbed (deviceless)");
    }

private:
    VerificationMode _mode;
    SupportObservation _observation;
    bool _engineSucceeds;
    bool _comparisonReached = false;
    RefBehavior _refBehavior = RefBehavior::MATCHES;
    std::optional<VerificationOutcome> _enforceOutcome;
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

    // Runs a fully configured harness against a bundle, with the reporter in place.
    void drive(EnforcementHarness& harness,
               const std::shared_ptr<IntegrationTestBundle>& bundle,
               ::testing::TestPartResultArray* results)
    {
        harness.setBundle(bundle, "test/bundle", makeLocator());

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        harness.TestBody();
    }

    void run(VerificationMode mode,
             SupportObservation observation,
             bool includeGoldenOutput,
             bool engineSucceeds,
             ::testing::TestPartResultArray* results,
             bool* comparisonReached = nullptr)
    {
        EnforcementHarness harness(mode, std::move(observation), engineSucceeds);
        drive(harness, loadBundle("Bundle", includeGoldenOutput), results);

        if(comparisonReached != nullptr)
        {
            *comparisonReached = harness.comparisonReached();
        }
    }

    // The single verdict recorded for this run, for tests that care which one it is.
    static SupportVerdict onlyVerdict()
    {
        const auto& all = SupportClaimVerdicts::get().all();
        EXPECT_EQ(all.size(), 1u);
        return all.empty() ? SupportVerdict::CLAIM_BROKEN : all.front().verdict;
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
        return SupportObservation{SidecarState::CHECKED, std::move(results)};
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
    drive(harness, bundle, &results);

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

// The per-graph check. The run-level guard only fires when nothing anywhere was
// queried, so a partial gap needs its own signal.
TEST_F(TestSupportClaimEnforcement, UnqueriedSidecarFailsTheTest)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::AUTO, SupportObservation{SidecarState::NONE, {}}, true, true, &results);

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
// a verification. Only reaching the depth the bundle declares can promote it.
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

// The engine ran the graph green and the comparison caught a mismatch. The claim
// held — the engine did accept the graph — but the cell must not be published as
// working support.
TEST_F(TestSupportClaimEnforcement, MismatchDemotesTheClaimToFailedInUse)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/false,
        &results);

    EXPECT_TRUE(anyFailed(results));
    EXPECT_EQ(onlyVerdict(), SupportVerdict::CLAIM_FAILED_IN_USE);
}

// The regression this rework exists for. The engine executed the graph, then the
// fallback chain ran out of oracles and the test skipped. Nothing verified the
// outputs, so confirming the cell would publish support on the strength of a run
// that compared nothing.
TEST_F(TestSupportClaimEnforcement, ExecutedWithoutAnOracleStaysAccepted)
{
    ::testing::TestPartResultArray results;

    EnforcementHarness harness(VerificationMode::AUTO,
                               observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
                               /*engineSucceeds=*/true);
    harness.setRefBehavior(RefBehavior::CAPABILITY_MISS);
    drive(harness, loadBundle("Bundle", /*includeGoldenOutput=*/false), &results);

    EXPECT_TRUE(harness.comparisonReached()) << "the engine must have run for this to be the case";
    EXPECT_FALSE(anyFailed(results));
    EXPECT_EQ(onlyVerdict(), SupportVerdict::CLAIM_ACCEPTED);
}

// A reference executor that blew up makes the run red without saying anything about
// the engine. Demoting the claim there would print "do not publish this cell" over
// somebody else's defect.
TEST_F(TestSupportClaimEnforcement, ReferenceErrorDoesNotDemoteTheClaim)
{
    ::testing::TestPartResultArray results;

    EnforcementHarness harness(VerificationMode::CPU,
                               observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
                               /*engineSucceeds=*/true);
    harness.setRefBehavior(RefBehavior::ERRORS);
    drive(harness, loadBundle("Bundle", /*includeGoldenOutput=*/false), &results);

    EXPECT_TRUE(anyFailed(results)) << "a broken reference must still fail the test";
    EXPECT_EQ(onlyVerdict(), SupportVerdict::CLAIM_ACCEPTED);
}

// Confirmation is measured against the depth the bundle declares, not against a
// fixed "outputs were compared". A buildable bundle whose plans compiled has done
// everything it promises, so its claim is confirmed rather than stuck at accepted.
TEST_F(TestSupportClaimEnforcement, BuildableBundleConfirmsAtItsOwnDepth)
{
    ::testing::TestPartResultArray results;

    EnforcementHarness harness(VerificationMode::AUTO,
                               observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
                               /*engineSucceeds=*/true);
    harness.setEnforceOutcome(VerificationOutcome::passed(VerificationDepth::BUILDABLE));
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::BUILDABLE;
    drive(harness, bundle, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_EQ(onlyVerdict(), SupportVerdict::CLAIM_CONFIRMED);
}

// An enforcement rung that could not even establish applicability leaves the claim
// exactly where the ranked-list query put it.
TEST_F(TestSupportClaimEnforcement, UnreachedEnforcementRungStaysAccepted)
{
    ::testing::TestPartResultArray results;

    EnforcementHarness harness(VerificationMode::AUTO,
                               observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
                               /*engineSucceeds=*/true);
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::BUILDABLE;
    drive(harness, bundle, &results);

    EXPECT_EQ(onlyVerdict(), SupportVerdict::CLAIM_ACCEPTED);
}

// The promotion policy itself is pinned in TestSupportVerdict.cpp, where it needs no
// fixture; these cases cover the combinations the deviceless harness can actually
// drive end to end.

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

// Positive drift keeps its verdict — there is no claim to promote — but picks up how
// far the run got, which is what separates "this cell works, write it down" from
// "the ranked list said so and nothing tried it".
TEST_F(TestSupportClaimEnforcement, UnclaimedSupportKeepsItsVerdictAndGainsTheDepth)
{
    ::testing::TestPartResultArray results;
    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
    EXPECT_NE(SupportClaimVerdicts::get().all().front().detail.find("verified"), std::string::npos);
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
