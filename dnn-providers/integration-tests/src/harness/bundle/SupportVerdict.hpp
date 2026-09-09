// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/VerificationOutcome.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Outcome for one graph, decided against the single engine under test.
///
/// One lane tests one engine, so the question is a two-bit decision: does the
/// sidecar promise this engine for this (arch, platform[, case]), and is this engine
/// in the ranked list `get_ranked_engine_ids` returned. Claims naming *other*
/// engines belong to those engines' lanes; a run that cannot execute them has no
/// basis to pass or fail them, and the static inventory covers engines with no lane
/// at all.
///
/// CLAIM_ACCEPTED is taken from the query alone, before the graph is built or run.
/// It says the engine advertises support, not that the graph works — only a run that
/// reaches the depth the bundle's enforcement_level declares can promote it. See
/// IntegrationBundleVerificationHarness::TestBody().
enum class SupportVerdict
{
    CLAIM_BROKEN, ///< claimed, but absent from the ranked list — FAIL
    QUERY_ERRORED, ///< claimed, but the query did not resolve — FAIL
    CLAIM_ACCEPTED, ///< claimed and in the ranked list; not exercised by this test
    CLAIM_CONFIRMED, ///< accepted, and the engine ran the graph green
    CLAIM_FAILED_IN_USE, ///< accepted, but the engine failed the graph
    UNCLAIMED_SUPPORT, ///< in the ranked list with no claim — positive drift
};

const char* toString(SupportVerdict verdict);

/// Fail-closed: unknown/future verdicts are failures by default.
bool isFailure(SupportVerdict verdict);

/// Phase-2 promotion for a CLAIM_ACCEPTED verdict, once the test outcome is known.
///
///   outcome  — what the test body did: how far it got, and who broke if anything
///   required — the depth this bundle's enforcement_level asks for
///
/// A claim is confirmed when the run got as far as its bundle asks. So a
/// `buildable` bundle whose plans compiled is confirmed, while a `full` bundle that
/// ran the graph and then found no oracle is not — nothing checked its output, and
/// confirmed is the column a support matrix publishes.
///
/// Kept out of the harness so it can be tested without GTest result state, which a
/// fake part-result reporter hides.
SupportVerdict promoteAcceptedClaim(const VerificationOutcome& outcome, VerificationDepth required);

struct SupportResult
{
    SupportVerdict verdict;
    std::string bundlePath;
    std::string engineName;
    std::string arch;
    std::string platform;
    std::string detail;

    hipdnn_frontend::ErrorCode queryStatus = hipdnn_frontend::ErrorCode::OK;
    std::string queryMessage;
};

/// Did this graph have a sidecar, and did we read it?
///
/// An enum, not a bool, so nobody works it out from `results`. A sidecar we read in
/// full can still leave zero verdicts behind: it may claim only another arch,
/// another platform, another sweep case, or engines this build does not load. Those
/// runs did everything they could and must count as covered, but `results.empty()`
/// cannot tell them apart from "there was no sidecar" — the one case that must not
/// count.
///
/// Registration counts the same thing when it seeds `graphsWithClaims` (does the
/// file exist?), which is what keeps `withClaims >= queried` true.
enum class SidecarState : uint8_t
{
    /// No sidecar file. Normal and fine — claims are optional, and a missing one
    /// means "nobody said", not "not supported". A sidecar that exists but does not
    /// parse throws instead of landing here.
    NONE,
    /// The sidecar was read and checked against this run's engine, arch and
    /// platform. Says nothing about what was in it: a sidecar that promised nothing
    /// still counts as checked.
    CHECKED,
    /// The graph never opened, so there was no ranked list to check a sidecar
    /// against. Distinct from NONE because the coverage check reads NONE as "a
    /// sidecar exists and nothing looked at it" — a harness bug — and this is not
    /// that: the run is already failing on the graph itself, and naming an
    /// enforcement gap on top would send a reader after a bug that is not there.
    NOT_QUERIED,
};

/// What one graph's sidecar had to say, and whether we got to read it.
struct SupportObservation
{
    SidecarState sidecar = SidecarState::NONE;
    std::vector<SupportResult> results; ///< claimed engines, plus positive drift

    /// Did the sidecar promise anything about the arch, platform and case this run
    /// is on?
    ///
    /// Safe to work out from `results`, unlike `sidecar`: this one *should* be false
    /// when the sidecar covers only another arch, platform, or sweep case, or claims
    /// nothing at all. Every verdict except UNCLAIMED_SUPPORT names an engine the
    /// sidecar claimed, so having none means nothing was promised here.
    bool hasApplicableClaim() const
    {
        return std::any_of(results.begin(), results.end(), [](const SupportResult& r) {
            return r.verdict != SupportVerdict::UNCLAIMED_SUPPORT;
        });
    }
};

/// The verdict for one cell, from the three facts that decide it.
///
///   claimed  — the sidecar names this engine for this arch/platform (and case)
///   resolved — the ranked-engine query returned an answer we can believe
///   accepted — this engine is in the ranked list
///
/// nullopt means there is nothing to record: neither claimed nor accepted carries
/// no information, and recording it would make the verdict count say more than what
/// was actually promised.
///
/// Split out and returned rather than pushed so the whole table can be read — and
/// tested — in one place, instead of being reconstructed from nested conditions at
/// the point a result is built.
std::optional<SupportVerdict> chooseVerdict(bool claimed, bool resolved, bool accepted);

/// The human-readable reason behind `verdict`, for the report's detail column.
std::string verdictDetail(SupportVerdict verdict, hipdnn_frontend::ErrorCode status);

/// Decide this graph's claim for the engine under test, from one ranked-engine
/// query.
///
/// `engines` is the result of the single `Graph::get_ranked_engine_ids()` call this
/// test makes; `arch` / `platform` are passed in rather than read from TestConfig,
/// so this stays a pure function.
///
/// Returns at most one result: the claim's verdict if the sidecar names this engine
/// for this cell, otherwise UNCLAIMED_SUPPORT if the engine takes the graph anyway,
/// otherwise nothing — the sidecar was still read, so `sidecar` is CHECKED either
/// way.
///
/// Throws std::runtime_error if the sidecar exists but cannot be opened or parsed.
SupportObservation observeSupport(const RankedEngines& engines,
                                  const SupportClaimLocator& locator,
                                  const LoadedEngine& engineUnderTest,
                                  std::string_view arch,
                                  std::string_view platform);

/// Should the comparison be skipped because a claim already failed?
///
/// nullopt when the claims held and the test may run. Otherwise the outcome that
/// stands in for the comparison, carrying every failing verdict's message.
///
/// A broken claim means the engine will not take the graph, so running the
/// comparison would execute nothing, leave the NaN sentinel outputs untouched, and
/// print a full tensor diff on top of the real message.
std::optional<VerificationOutcome> claimBlocked(const SupportObservation& observation);

/// The verdicts to publish for this test, given what the run achieved.
///
/// Only the engine this test drove can be promoted: an accepted claim is a reading
/// of the ranked list taken before the graph was built or run, and only execution
/// can turn it into confirmed support or knock it back. Other engines' verdicts pass
/// through untouched — this run never executed them, so it has no evidence either
/// way. Positive drift for the engine under test keeps its verdict but picks up how
/// far the run got.
///
/// Every input verdict comes back exactly once, so the report cannot lose rows.
std::vector<SupportResult> finalizeClaims(std::vector<SupportResult> results,
                                          std::string_view engineUnderTest,
                                          const VerificationOutcome& outcome,
                                          VerificationDepth required);

/// "gfx942:sramecc+:xnack-" -> "gfx942"
std::string baseArchToken(std::string_view fullArch);

std::string formatVerdictMessage(const SupportResult& result);

} // namespace hipdnn_integration_tests::bundle
