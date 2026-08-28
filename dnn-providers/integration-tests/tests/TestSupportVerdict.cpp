// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// One lane tests one engine, so every verdict here is a two-bit decision: does the
// sidecar promise *this* engine for the running (arch, platform[, case]), and is
// this engine in the ranked list the query returned.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::EnforcementLevel;
using hipdnn_integration_tests::bundle::baseArchToken;
using hipdnn_integration_tests::bundle::FailureOrigin;
using hipdnn_integration_tests::bundle::isFailure;
using hipdnn_integration_tests::bundle::LoadedEngine;
using hipdnn_integration_tests::bundle::observeSupport;
using hipdnn_integration_tests::bundle::promoteAcceptedClaim;
using hipdnn_integration_tests::bundle::requiredDepth;
using hipdnn_integration_tests::bundle::SidecarState;
using hipdnn_integration_tests::bundle::SupportClaimLocator;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;
using hipdnn_integration_tests::bundle::VerificationDepth;
using hipdnn_integration_tests::bundle::VerificationOutcome;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t UNDER_TEST_ID = 42;
constexpr int64_t OTHER_ID = 7;

const std::string UNDER_TEST = "ENGINE_UNDER_TEST";
const std::string OTHER_ENGINE = "OTHER_ENGINE";
const std::string ARCH = "gfx942";
const std::string PLAT = "linux";

// A plausible stand-in for what a broken backend actually puts in err_msg.
const std::string QUERY_MSG = "hipdnnBackendFinalize failed: HIPDNN_STATUS_INTERNAL_ERROR";

const LoadedEngine ENGINE{UNDER_TEST_ID, UNDER_TEST};

ScopedDirectory makeScopedTestDir(const std::string& prefix)
{
    auto path
        = std::filesystem::temp_directory_path()
          / (prefix + "_"
             + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
    std::filesystem::remove_all(path);
    return {path};
}

// Single-graph sidecar claiming `engines` for (arch, platform).
SupportClaimLocator writeSidecar(const ScopedDirectory& dir,
                                 const std::vector<std::string>& engines,
                                 const std::string& arch = ARCH,
                                 const std::string& platform = PLAT)
{
    nlohmann::json claims = nlohmann::json::object();
    for(const auto& engine : engines)
    {
        claims[engine] = {{arch, nlohmann::json::array({platform})}};
    }
    const auto path = dir.path() / "Bundle.support.json";
    std::ofstream(path) << nlohmann::json{{"version", 1}, {"claims", claims}}.dump();

    SupportClaimLocator locator;
    locator.sidecarPath = path;
    locator.diagnosticPath = "dir/Bundle.json";
    return locator;
}

// Sweep sidecar where `engine` claims `cases`; the locator selects `caseId`.
SupportClaimLocator writeSweepSidecar(const ScopedDirectory& dir,
                                      const std::string& engine,
                                      const std::vector<std::string>& cases,
                                      const std::string& caseId)
{
    nlohmann::json group;
    group["cases"] = cases;
    group["support"] = {{ARCH, nlohmann::json::array({PLAT})}};

    const auto path = dir.path() / "support.json";
    std::ofstream(path) << nlohmann::json{{"version", 1},
                                          {"claims", {{engine, nlohmann::json::array({group})}}}}
                               .dump();

    SupportClaimLocator locator;
    locator.sidecarPath = path;
    locator.caseId = caseId;
    locator.diagnosticPath = "dir/sweep.json#" + caseId;
    return locator;
}

} // namespace

// ---------------------------------------------------------------------------
// No sidecar → nothing read. The only case that must leave `sidecar` at NONE,
// because it is the only one where there was no promise to look at.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, NoSidecarPathEvaluatesNothing)
{
    const auto observation = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, {}, ENGINE, ARCH, PLAT);
    EXPECT_EQ(observation.sidecar, SidecarState::NONE);
    EXPECT_TRUE(observation.results.empty());
    EXPECT_FALSE(observation.hasApplicableClaim());
}

TEST(TestSupportVerdict, MissingSidecarFileEvaluatesNothing)
{
    SupportClaimLocator locator;
    locator.sidecarPath = "/no/such/file.support.json";
    locator.diagnosticPath = "Bundle.json";

    const auto observation
        = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);
    EXPECT_EQ(observation.sidecar, SidecarState::NONE);
    EXPECT_TRUE(observation.results.empty());
}

// ---------------------------------------------------------------------------
// Claimed: the ranked list decides whether the promise held.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimedAndInRankedListIsAccepted)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation
        = observeSupport(ErrorCode::OK, {OTHER_ID, UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::CLAIM_ACCEPTED);
    EXPECT_EQ(observation.results.front().engineName, UNDER_TEST);
    EXPECT_TRUE(observation.hasApplicableClaim());
    EXPECT_FALSE(isFailure(observation.results.front().verdict));
}

TEST(TestSupportVerdict, ClaimedButAbsentFromRankedListIsBroken)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation = observeSupport(ErrorCode::OK, {OTHER_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::CLAIM_BROKEN);
    EXPECT_TRUE(isFailure(observation.results.front().verdict));
}

TEST(TestSupportVerdict, GraphNotSupportedWithAClaimIsBroken)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation
        = observeSupport(ErrorCode::GRAPH_NOT_SUPPORTED, {}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::CLAIM_BROKEN);
}

// ---------------------------------------------------------------------------
// Unresolved query: the ranked list cannot be trusted, so a claim can only be
// QUERY_ERRORED. Reporting CLAIM_BROKEN would assert a decline nobody observed.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimedWithUnresolvedQueryIsErrored)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation = observeSupport(
        ErrorCode::HEURISTIC_QUERY_FAILED, {}, locator, ENGINE, ARCH, PLAT, QUERY_MSG);

    ASSERT_EQ(observation.results.size(), 1u);
    const auto& r = observation.results.front();
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
    EXPECT_TRUE(isFailure(r.verdict));
    EXPECT_EQ(r.queryStatus, ErrorCode::HEURISTIC_QUERY_FAILED);
    EXPECT_EQ(r.queryMessage, QUERY_MSG);
}

// Unresolved status dominates ranked-list membership: even with the id present,
// the query was not trustworthy.
TEST(TestSupportVerdict, UnresolvedWithIdPresentIsStillErrored)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation = observeSupport(
        ErrorCode::HEURISTIC_QUERY_FAILED, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::QUERY_ERRORED);
}

// An unresolved query also cannot produce drift — that would publish support
// derived from a list nobody read.
TEST(TestSupportVerdict, UnresolvedQueryWithNoClaimReportsNothing)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {});

    const auto observation = observeSupport(
        ErrorCode::HEURISTIC_QUERY_FAILED, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    EXPECT_EQ(observation.sidecar, SidecarState::CHECKED);
    EXPECT_TRUE(observation.results.empty());
}

// Fail-closed: an unknown ErrorCode is unresolved, not "resolved and empty".
TEST(TestSupportVerdict, UnknownErrorCodeTreatedAsUnresolved)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation = observeSupport(
        static_cast<ErrorCode>(9999), {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::QUERY_ERRORED);
}

// A resolved query has nothing to explain — the message is not stored.
TEST(TestSupportVerdict, ResolvedQueryDoesNotStoreMessage)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation = observeSupport(
        ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT, "should be dropped");

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_TRUE(observation.results.front().queryMessage.empty());
}

// ---------------------------------------------------------------------------
// Unclaimed: supported is drift, unsupported is nothing at all.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, SupportedWithoutAClaimIsDrift)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {});

    const auto observation
        = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::UNCLAIMED_SUPPORT);
    EXPECT_FALSE(isFailure(observation.results.front().verdict));
    // Drift is not a promise.
    EXPECT_FALSE(observation.hasApplicableClaim());
}

// Neither claimed nor supported carries no information and must not be recorded —
// that is what keeps the verdict count proportional to what was promised.
TEST(TestSupportVerdict, NeitherClaimedNorSupportedProducesNoRecord)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {});

    const auto observation = observeSupport(ErrorCode::OK, {OTHER_ID}, locator, ENGINE, ARCH, PLAT);

    EXPECT_EQ(observation.sidecar, SidecarState::CHECKED);
    EXPECT_TRUE(observation.results.empty());
}

// ---------------------------------------------------------------------------
// Another engine's claim is another lane's business. It must not be decided here —
// this run cannot execute that engine, so it has no basis to pass or fail it. The
// sidecar is still read, so coverage still counts.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimForAnotherEngineIsNotDecidedHere)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {OTHER_ENGINE});

    const auto observation = observeSupport(ErrorCode::OK, {}, locator, ENGINE, ARCH, PLAT);

    EXPECT_EQ(observation.sidecar, SidecarState::CHECKED);
    EXPECT_TRUE(observation.results.empty());
    EXPECT_FALSE(observation.hasApplicableClaim());
}

TEST(TestSupportVerdict, ClaimForAnotherEngineDoesNotSuppressOurDrift)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {OTHER_ENGINE});

    const auto observation
        = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::UNCLAIMED_SUPPORT);
    EXPECT_EQ(observation.results.front().engineName, UNDER_TEST);
}

// ---------------------------------------------------------------------------
// Arch / platform scoping
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimForAnotherArchDoesNotApply)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST}, "gfx90a", PLAT);

    const auto observation
        = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::UNCLAIMED_SUPPORT);
    EXPECT_FALSE(observation.hasApplicableClaim());
}

// Read in full, promised nothing here, and the engine does not support it either:
// zero verdicts, but the sidecar was still consulted. This is the pair that must
// not be collapsed — coverage says yes, applicability says no.
TEST(TestSupportVerdict, ClaimForAnotherPlatformIsCheckedButNotApplicable)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST}, ARCH, "windows");

    const auto observation = observeSupport(ErrorCode::OK, {}, locator, ENGINE, ARCH, PLAT);

    EXPECT_EQ(observation.sidecar, SidecarState::CHECKED);
    EXPECT_FALSE(observation.hasApplicableClaim());
    EXPECT_TRUE(observation.results.empty());
}

// ---------------------------------------------------------------------------
// Sweep dispatch: caseId selects which claim group applies.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, SweepClaimAppliesToNamedCase)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSweepSidecar(dir, UNDER_TEST, {"case_one", "case_two"}, "case_one");

    const auto observation = observeSupport(ErrorCode::OK, {}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    EXPECT_EQ(observation.results.front().verdict, SupportVerdict::CLAIM_BROKEN);
    EXPECT_EQ(observation.results.front().bundlePath, "dir/sweep.json#case_one");
}

TEST(TestSupportVerdict, SweepClaimDoesNotApplyToUnnamedCase)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSweepSidecar(dir, UNDER_TEST, {"case_one"}, "case_three");

    const auto observation = observeSupport(ErrorCode::OK, {}, locator, ENGINE, ARCH, PLAT);

    EXPECT_EQ(observation.sidecar, SidecarState::CHECKED);
    EXPECT_FALSE(observation.hasApplicableClaim());
    EXPECT_TRUE(observation.results.empty());
}

// ---------------------------------------------------------------------------
// Malformed sidecars surface as exceptions, not silent empty observations.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, UnparseableSidecarThrows)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto path = dir.path() / "Bundle.support.json";
    std::ofstream(path) << "{ this is not json";

    SupportClaimLocator locator;
    locator.sidecarPath = path;
    locator.diagnosticPath = "dir/Bundle.json";

    EXPECT_THROW(observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Result metadata
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ResultCarriesLocatorAndCellMetadata)
{
    const auto dir = makeScopedTestDir("verdict");
    const auto locator = writeSidecar(dir, {UNDER_TEST});

    const auto observation
        = observeSupport(ErrorCode::OK, {UNDER_TEST_ID}, locator, ENGINE, ARCH, PLAT);

    ASSERT_EQ(observation.results.size(), 1u);
    const auto& r = observation.results.front();
    EXPECT_EQ(r.bundlePath, "dir/Bundle.json");
    EXPECT_EQ(r.engineName, UNDER_TEST);
    EXPECT_EQ(r.arch, ARCH);
    EXPECT_EQ(r.platform, PLAT);
    EXPECT_FALSE(r.detail.empty());
}

// ---------------------------------------------------------------------------
// Phase-2 promotion. CLAIM_FAILED_IN_USE is not a claim failure: the claim held,
// and the run is already red from whatever actually broke.
// ---------------------------------------------------------------------------

// Confirmation is measured against the depth the bundle declares, so the policy
// takes the outcome and that depth rather than a pair of booleans.
TEST(TestSupportVerdict, PromotionMapsOutcomeToVerdict)
{
    const auto verified = VerificationDepth::VERIFIED;

    // Reached the declared depth — the only thing that is evidence of working
    // support.
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::passed(verified), verified),
              SupportVerdict::CLAIM_CONFIRMED);

    // A shallower bundle is satisfied by a shallower run; a deeper one is not.
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::passed(VerificationDepth::BUILDABLE),
                                   VerificationDepth::BUILDABLE),
              SupportVerdict::CLAIM_CONFIRMED);
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::passed(VerificationDepth::APPLICABLE),
                                   VerificationDepth::BUILDABLE),
              SupportVerdict::CLAIM_ACCEPTED);

    // Short of the declared depth the run has no evidence either way — including the
    // case that motivated this shape: the engine executed the graph and then found
    // no oracle to compare it against.
    EXPECT_EQ(promoteAcceptedClaim(
                  VerificationOutcome::skipped(VerificationDepth::EXECUTED, "no oracle"), verified),
              SupportVerdict::CLAIM_ACCEPTED);
    EXPECT_EQ(
        promoteAcceptedClaim(
            VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "declined"), verified),
        SupportVerdict::CLAIM_ACCEPTED);

    // The engine accepted the graph and then got it wrong. Publishing that cell as
    // satisfied support would feed a support matrix a combination the same run just
    // proved does not work.
    EXPECT_EQ(promoteAcceptedClaim(
                  VerificationOutcome::failed(verified, FailureOrigin::COMPARISON, {}), verified),
              SupportVerdict::CLAIM_FAILED_IN_USE);
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                                               FailureOrigin::ENGINE,
                                                               "build failed"),
                                   verified),
              SupportVerdict::CLAIM_FAILED_IN_USE);

    // Failures that are not the engine's leave the claim alone. The run is already
    // red for the real reason, and demoting here would print "do not publish this
    // cell" over somebody else's defect.
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::failed(VerificationDepth::EXECUTED,
                                                               FailureOrigin::ORACLE,
                                                               "ref exploded"),
                                   verified),
              SupportVerdict::CLAIM_ACCEPTED);
    EXPECT_EQ(promoteAcceptedClaim(VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                                               FailureOrigin::HARNESS,
                                                               "no golden data"),
                                   verified),
              SupportVerdict::CLAIM_ACCEPTED);
}

// requiredDepth() is the one place enforcement_level becomes a depth; a level that
// silently demanded less would confirm cells nothing checked.
TEST(TestSupportVerdict, EnforcementLevelMapsToRequiredDepth)
{
    EXPECT_EQ(requiredDepth(EnforcementLevel::APPLICABILITY), VerificationDepth::APPLICABLE);
    EXPECT_EQ(requiredDepth(EnforcementLevel::BUILDABLE), VerificationDepth::BUILDABLE);
    EXPECT_EQ(requiredDepth(EnforcementLevel::FULL), VerificationDepth::VERIFIED);
}

TEST(TestSupportVerdict, OnlyBrokenAndErroredAreFailures)
{
    EXPECT_TRUE(isFailure(SupportVerdict::CLAIM_BROKEN));
    EXPECT_TRUE(isFailure(SupportVerdict::QUERY_ERRORED));

    EXPECT_FALSE(isFailure(SupportVerdict::CLAIM_ACCEPTED));
    EXPECT_FALSE(isFailure(SupportVerdict::CLAIM_CONFIRMED));
    EXPECT_FALSE(isFailure(SupportVerdict::CLAIM_FAILED_IN_USE));
    EXPECT_FALSE(isFailure(SupportVerdict::UNCLAIMED_SUPPORT));
}

// Fail-closed: a verdict nobody has taught isFailure() about is a failure.
TEST(TestSupportVerdict, UnknownVerdictIsFailure)
{
    EXPECT_TRUE(isFailure(static_cast<SupportVerdict>(9999)));
}

// ---------------------------------------------------------------------------
// toString / formatVerdictMessage
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ToStringCoversAllValues)
{
    using hipdnn_integration_tests::bundle::toString;
    EXPECT_STREQ(toString(SupportVerdict::CLAIM_BROKEN), "CLAIM_BROKEN");
    EXPECT_STREQ(toString(SupportVerdict::QUERY_ERRORED), "QUERY_ERRORED");
    EXPECT_STREQ(toString(SupportVerdict::CLAIM_ACCEPTED), "CLAIM_ACCEPTED");
    EXPECT_STREQ(toString(SupportVerdict::CLAIM_CONFIRMED), "CLAIM_CONFIRMED");
    EXPECT_STREQ(toString(SupportVerdict::CLAIM_FAILED_IN_USE), "CLAIM_FAILED_IN_USE");
    EXPECT_STREQ(toString(SupportVerdict::UNCLAIMED_SUPPORT), "UNCLAIMED_SUPPORT");
    EXPECT_STREQ(toString(static_cast<SupportVerdict>(9999)), "UNKNOWN");
}

TEST(TestSupportVerdict, FormatVerdictMessageContainsAllFields)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    SupportResult r;
    r.verdict = SupportVerdict::CLAIM_BROKEN;
    r.bundlePath = "path/to/bundle";
    r.engineName = "MY_ENGINE";
    r.arch = "gfx90a";
    r.platform = "linux";
    r.detail = "detail";

    const auto msg = formatVerdictMessage(r);
    EXPECT_NE(msg.find("CLAIM_BROKEN"), std::string::npos);
    EXPECT_NE(msg.find("path/to/bundle"), std::string::npos);
    EXPECT_NE(msg.find("MY_ENGINE"), std::string::npos);
    EXPECT_NE(msg.find("gfx90a"), std::string::npos);
    EXPECT_NE(msg.find("linux"), std::string::npos);
    EXPECT_EQ(msg.find("query:"), std::string::npos);
}

TEST(TestSupportVerdict, FormatVerdictMessageIncludesQueryMessage)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    SupportResult r;
    r.verdict = SupportVerdict::QUERY_ERRORED;
    r.queryMessage = QUERY_MSG;

    const auto msg = formatVerdictMessage(r);
    EXPECT_NE(msg.find(QUERY_MSG), std::string::npos);
    EXPECT_NE(msg.find("query:"), std::string::npos);
}

// ---------------------------------------------------------------------------
// baseArchToken strips target features, keeps base arch
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, BaseArchTokenStripsFeatures)
{
    EXPECT_EQ(baseArchToken("gfx942:sramecc+:xnack-"), "gfx942");
}

TEST(TestSupportVerdict, BaseArchTokenIdempotentOnBare)
{
    EXPECT_EQ(baseArchToken("gfx942"), "gfx942");
}

TEST(TestSupportVerdict, BaseArchTokenEmptyInput)
{
    EXPECT_EQ(baseArchToken(""), "");
}

TEST(TestSupportVerdict, BaseArchTokenColonOnly)
{
    EXPECT_EQ(baseArchToken(":"), "");
}

// NOLINTEND(readability-identifier-naming)
