// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

struct LoadedEngine;

/// Outcome for one graph, adjudicated against the single engine under test.
///
/// One lane tests one engine, so the question is a two-bit decision: does the
/// sidecar promise this engine for this (arch, platform[, case]), and is this engine
/// in the ranked list `get_ranked_engine_ids` returned. Claims naming *other*
/// engines belong to those engines' lanes; a run that cannot execute them has no
/// basis to pass or fail them, and the static inventory covers engines with no lane
/// at all.
///
/// CLAIM_ACCEPTED is taken from the query alone, before the graph is built or run.
/// It says the engine advertises support, not that the graph works — only execution
/// can promote it. See IntegrationBundleVerificationHarness::TestBody().
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
///   exercised — the engine under test actually ran this graph (the body neither
///               skipped nor bailed before execution)
///   passed    — the test finished green
///
/// Split out from the harness so the policy is testable without GTest result state,
/// which a fake part-result reporter intercepts and hides.
SupportVerdict promoteAcceptedClaim(bool exercised, bool passed);

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

/// What one graph's sidecar had to say, and whether it was consulted at all.
///
/// `sidecarChecked` names the sidecar, not the claims, on purpose. It promises the
/// file existed, parsed, and was adjudicated — nothing about the contents. A legal
/// sidecar may claim nothing at all, or claim only another arch/platform/case, and
/// still set it. It is the run-time counterpart of the registration-time predicate
/// that seeds graphsWithClaims (`exists(sidecarPathFor(disc))`), so the two count
/// the same population and `withClaims >= queried` stays meaningful.
///
/// It MUST NOT be derived from `results`. Once the neither-claimed-nor-supported
/// quadrant stops being emitted, a fully adjudicated sidecar can yield zero
/// verdicts for exactly the reasons above. Those runs did everything they could and
/// must count as covered; `results.empty()` cannot tell them apart from "there was
/// no sidecar", which is the one case that must not count. Deriving it is what made
/// the coverage guard fail healthy runs before this split existed.
struct SupportObservation
{
    bool sidecarChecked = false; ///< a sidecar existed and was adjudicated
    std::vector<SupportResult> results; ///< claimed engines, plus positive drift

    /// Whether the sidecar actually promised anything about the cell this run is on.
    ///
    /// Safe to derive — unlike sidecarChecked, this one *should* be false when the
    /// sidecar covers only another arch, platform, or sweep case, or claims nothing
    /// at all. Every verdict except UNCLAIMED_SUPPORT names an engine the sidecar
    /// claimed, so their absence is exactly "nothing was promised here".
    bool hasApplicableClaim() const
    {
        return std::any_of(results.begin(), results.end(), [](const SupportResult& r) {
            return r.verdict != SupportVerdict::UNCLAIMED_SUPPORT;
        });
    }
};

/// Adjudicate this graph's claim for the engine under test, from one ranked-engine
/// query.
///
/// `errorCode` / `rankedIds` come from one `Graph::get_ranked_engine_ids()` call.
/// `arch` / `platform` are passed in rather than read from TestConfig so this stays
/// a pure function.
///
/// Yields at most one result: the claim's verdict if the sidecar names this engine
/// for this cell, otherwise UNCLAIMED_SUPPORT if the engine accepts the graph
/// anyway, otherwise nothing — the sidecar was still read, so `sidecarChecked` is
/// set either way.
///
/// Throws std::runtime_error if the sidecar exists but cannot be opened or parsed.
SupportObservation observeSupport(hipdnn_frontend::ErrorCode errorCode,
                                  const std::vector<int64_t>& rankedIds,
                                  const SupportClaimLocator& locator,
                                  const LoadedEngine& engineUnderTest,
                                  std::string_view arch,
                                  std::string_view platform,
                                  std::string_view queryMessage = {});

/// "gfx942:sramecc+:xnack-" -> "gfx942"
std::string baseArchToken(std::string_view fullArch);

std::string formatVerdictMessage(const SupportResult& result);

} // namespace hipdnn_integration_tests::bundle
