// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportVerdict.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

const char* toString(SupportVerdict verdict)
{
    switch(verdict)
    {
    case SupportVerdict::CLAIM_BROKEN:
        return "CLAIM_BROKEN";
    case SupportVerdict::QUERY_ERRORED:
        return "QUERY_ERRORED";
    case SupportVerdict::CLAIM_ACCEPTED:
        return "CLAIM_ACCEPTED";
    case SupportVerdict::CLAIM_CONFIRMED:
        return "CLAIM_CONFIRMED";
    case SupportVerdict::CLAIM_FAILED_IN_USE:
        return "CLAIM_FAILED_IN_USE";
    case SupportVerdict::UNCLAIMED_SUPPORT:
        return "UNCLAIMED_SUPPORT";
    default:
        return "UNKNOWN";
    }
}

// Fail-closed: unknown/future verdicts are failures by default.
//
// CLAIM_FAILED_IN_USE is deliberately not a claim failure. The claim held — the
// engine did accept the graph — and the run is already red from whatever actually
// broke. Failing it a second time here would double-report one defect and bury the
// real diagnostic under a claim message.
bool isFailure(SupportVerdict verdict)
{
    switch(verdict)
    {
    case SupportVerdict::CLAIM_ACCEPTED:
    case SupportVerdict::CLAIM_CONFIRMED:
    case SupportVerdict::CLAIM_FAILED_IN_USE:
    case SupportVerdict::UNCLAIMED_SUPPORT:
        return false;
    default:
        return true;
    }
}

SupportVerdict promoteAcceptedClaim(const VerificationOutcome& outcome, VerificationDepth required)
{
    if(outcome.status == OutcomeStatus::FAILED)
    {
        // Only the engine's own failures are evidence against the engine. A
        // reference executor that errored, or a bundle whose golden data is not
        // pulled, makes the run red without saying anything about the claim —
        // demoting it there would publish "do not use this cell" over a defect that
        // lives somewhere else entirely.
        const bool engineIsAtFault = outcome.origin == FailureOrigin::ENGINE
                                     || outcome.origin == FailureOrigin::COMPARISON;
        return engineIsAtFault ? SupportVerdict::CLAIM_FAILED_IN_USE
                               : SupportVerdict::CLAIM_ACCEPTED;
    }

    // Short of the bundle's declared depth the run has no evidence either way, so
    // leaving it accepted is the honest answer; confirming it would publish support
    // that nothing verified.
    return outcome.depth >= required ? SupportVerdict::CLAIM_CONFIRMED
                                     : SupportVerdict::CLAIM_ACCEPTED;
}

namespace
{

// isResolved() lives in GraphSession.hpp: the executor and the enforcement rungs
// need the same answer, and one graph cannot be "unresolved" to the claim verdict
// and "declined" to everything else.

SupportResult makeResult(SupportVerdict verdict,
                         const SupportClaimLocator& locator,
                         std::string_view engineName,
                         std::string_view arch,
                         std::string_view platform,
                         std::string detail,
                         hipdnn_frontend::ErrorCode errorCode,
                         std::string_view queryMessage)
{
    SupportResult result;
    result.verdict = verdict;
    result.bundlePath = locator.diagnosticPath;
    result.engineName = std::string(engineName);
    result.arch = std::string(arch);
    result.platform = std::string(platform);
    result.detail = std::move(detail);
    result.queryStatus = errorCode;

    // Kept only where it can explain something the detail cannot: an unresolved
    // query has a code but no reason, and QUERY_ERRORED is a FAIL, so this is the
    // one place a human sees the backend's own words.
    if(!isResolved(errorCode))
    {
        result.queryMessage = std::string(queryMessage);
    }
    return result;
}

} // namespace

std::string baseArchToken(std::string_view fullArch)
{
    const auto pos = fullArch.find(':');
    if(pos == std::string_view::npos)
    {
        return std::string(fullArch);
    }
    return std::string(fullArch.substr(0, pos));
}

std::string formatVerdictMessage(const SupportResult& result)
{
    std::ostringstream os;
    os << "\nSupport claim " << toString(result.verdict) << "\n"
       << "  bundle:   " << result.bundlePath << "\n"
       << "  engine:   " << result.engineName << "\n"
       << "  arch:     " << result.arch << "\n"
       << "  platform: " << result.platform << "\n"
       << "  detail:   " << result.detail << "\n";

    if(!result.queryMessage.empty())
    {
        os << "  query:    " << result.queryMessage << "\n";
    }
    return os.str();
}

std::optional<SupportVerdict> chooseVerdict(bool claimed, bool resolved, bool accepted)
{
    if(!claimed)
    {
        if(!accepted || !resolved)
        {
            // Nothing was promised, and either nothing was offered or the ranked list
            // cannot be believed. Recording drift off a query nobody could read would
            // be inventing the one fact this row exists to report.
            return std::nullopt;
        }
        // Supported but not written down. Not a failure; it is how an engine that
        // gained support gets noticed so the sidecar can be updated.
        return SupportVerdict::UNCLAIMED_SUPPORT;
    }

    if(!resolved)
    {
        // The ranked list cannot be believed, so acceptance is unknown. Reporting a
        // decline here would state a fact nobody read.
        return SupportVerdict::QUERY_ERRORED;
    }

    if(accepted)
    {
        return SupportVerdict::CLAIM_ACCEPTED;
    }

    return SupportVerdict::CLAIM_BROKEN;
}

std::string verdictDetail(SupportVerdict verdict, hipdnn_frontend::ErrorCode status)
{
    switch(verdict)
    {
    case SupportVerdict::QUERY_ERRORED:
        return "sidecar claims support, but query returned " + hipdnn_frontend::to_string(status);
    case SupportVerdict::CLAIM_BROKEN:
        return "sidecar claims support, but engine not in ranked list (status="
               + hipdnn_frontend::to_string(status) + ")";
    case SupportVerdict::CLAIM_ACCEPTED:
        return "engine in ranked list";
    case SupportVerdict::UNCLAIMED_SUPPORT:
        return "engine supports this graph but has no claim in the sidecar";
    default:
        // CONFIRMED and FAILED_IN_USE are written by finalizeClaims(), which knows
        // what the run achieved; observeSupport() never produces them.
        return "unexpected verdict from the ranked-engine query";
    }
}

SupportObservation observeSupport(const RankedEngines& engines,
                                  const SupportClaimLocator& locator,
                                  const LoadedEngine& engineUnderTest,
                                  std::string_view arch,
                                  std::string_view platform)
{
    if(locator.sidecarPath.empty() || !std::filesystem::exists(locator.sidecarPath))
    {
        return {};
    }

    const std::string archToken(arch);
    const std::string platformToken(platform);
    const std::string& engineName = engineUnderTest.name;

    // Does the sidecar promise *this* engine for this cell? One lane tests one
    // engine, so another engine's claim is another lane's business — enforcing it
    // here would report a verdict this run has no way to act on.
    bool claimed = false;
    if(locator.isSweep())
    {
        claimed = loadSweepSupportClaimsFromPath(locator.sidecarPath)
                      .isClaimed(locator.caseId, engineName, archToken, platformToken);
    }
    else
    {
        claimed = loadSupportClaimsFromPath(locator.sidecarPath)
                      .isClaimed(engineName, archToken, platformToken);
    }

    SupportObservation observation;
    observation.sidecar = SidecarState::CHECKED;

    // `accepted` is the same answer the executor and the enforcement rungs act on,
    // taken once at the query.
    const auto verdict
        = chooseVerdict(claimed, isResolved(engines.status.get_code()), engines.accepted);
    if(verdict.has_value())
    {
        observation.results.push_back(makeResult(*verdict,
                                                 locator,
                                                 engineName,
                                                 arch,
                                                 platform,
                                                 verdictDetail(*verdict, engines.status.get_code()),
                                                 engines.status.get_code(),
                                                 engines.status.get_message()));
    }

    return observation;
}

std::optional<VerificationOutcome> claimBlocked(const SupportObservation& observation)
{
    std::string aggregate;
    for(const auto& result : observation.results)
    {
        if(isFailure(result.verdict))
        {
            // Gathered up rather than failed on sight, so one FAIL() names every
            // failing verdict instead of the first one hiding the rest.
            aggregate += formatVerdictMessage(result);
        }
    }

    if(aggregate.empty())
    {
        return std::nullopt;
    }

    // CLAIM_BROKEN and QUERY_ERRORED both mean the engine will not take this graph:
    // the first because it is not in the ranked list, the second because the list
    // cannot be trusted at all. Either way the engine is what stands between this
    // bundle and a verdict, so the outcome names it as the cause.
    return VerificationOutcome::failed(
        VerificationDepth::NOT_REACHED, FailureOrigin::ENGINE, std::move(aggregate));
}

std::vector<SupportResult> finalizeClaims(std::vector<SupportResult> results,
                                          std::string_view engineUnderTest,
                                          const VerificationOutcome& outcome,
                                          VerificationDepth required)
{
    for(auto& record : results)
    {
        if(record.engineName != engineUnderTest)
        {
            // Another engine's claim, decided from the same ranked list but never run
            // here. The run has no evidence either way, so it passes through.
            continue;
        }

        if(record.verdict == SupportVerdict::CLAIM_ACCEPTED)
        {
            record.verdict = promoteAcceptedClaim(outcome, required);
            record.detail = describeOutcome(outcome, required);
        }
        else if(record.verdict == SupportVerdict::UNCLAIMED_SUPPORT)
        {
            // Drift is worth more when the graph actually ran: it is the difference
            // between "add this cell to the sidecar" and "the ranked list said so and
            // nothing tried it".
            record.detail += " (" + describeOutcome(outcome, required) + ")";
        }
        else
        {
            // A failing verdict already carries its own detail, and the outcome that
            // reports it is built from these same records.
            continue;
        }

        // describeOutcome() says how far the run got; the outcome's message says what
        // to do about it — the frontend's error text, or which oracle was missing.
        // Without it the summary is a tally nobody can act on.
        if(!outcome.message.empty())
        {
            record.detail += ": " + outcome.message;
        }
    }
    return results;
}

} // namespace hipdnn_integration_tests::bundle
