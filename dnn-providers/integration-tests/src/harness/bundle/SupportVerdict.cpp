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

#include "harness/bundle/LoadedEngineTable.hpp"
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

// Whitelist: only these two codes mean the query resolved and we can trust the
// ranked list. Everything else (including future enum values) is unresolved —
// fail closed toward "cannot evaluate", never toward a false "declined".
bool isResolved(hipdnn_frontend::ErrorCode code)
{
    return code == hipdnn_frontend::ErrorCode::OK
           || code == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
}

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

SupportObservation observeSupport(hipdnn_frontend::ErrorCode errorCode,
                                  const std::vector<int64_t>& rankedIds,
                                  const SupportClaimLocator& locator,
                                  const LoadedEngine& engineUnderTest,
                                  std::string_view arch,
                                  std::string_view platform,
                                  std::string_view queryMessage)
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
    const bool claimed = locator.isSweep()
                             ? loadSweepSupportClaimsFromPath(locator.sidecarPath)
                                   .isClaimed(locator.caseId, engineName, archToken, platformToken)
                             : loadSupportClaimsFromPath(locator.sidecarPath)
                                   .isClaimed(engineName, archToken, platformToken);

    // An unresolved status means the ranked list cannot be trusted, so a claim lands
    // on QUERY_ERRORED rather than a false CLAIM_BROKEN.
    const bool resolved = isResolved(errorCode);
    const bool accepted
        = resolved
          && std::find(rankedIds.begin(), rankedIds.end(), engineUnderTest.id) != rankedIds.end();

    SupportObservation observation;
    observation.sidecar = SidecarState::CHECKED;

    if(claimed && !resolved)
    {
        observation.results.push_back(makeResult(SupportVerdict::QUERY_ERRORED,
                                                 locator,
                                                 engineName,
                                                 arch,
                                                 platform,
                                                 "sidecar claims support, but query returned "
                                                     + hipdnn_frontend::to_string(errorCode),
                                                 errorCode,
                                                 queryMessage));
    }
    else if(claimed)
    {
        observation.results.push_back(
            makeResult(accepted ? SupportVerdict::CLAIM_ACCEPTED : SupportVerdict::CLAIM_BROKEN,
                       locator,
                       engineName,
                       arch,
                       platform,
                       accepted ? "engine in ranked list"
                                : "sidecar claims support, but engine not in ranked list (status="
                                      + hipdnn_frontend::to_string(errorCode) + ")",
                       errorCode,
                       queryMessage));
    }
    else if(accepted)
    {
        // Supported but not written down. Not a failure; it is how an engine that
        // gained support gets noticed so the sidecar can be updated.
        observation.results.push_back(
            makeResult(SupportVerdict::UNCLAIMED_SUPPORT,
                       locator,
                       engineName,
                       arch,
                       platform,
                       "engine supports this graph but has no claim in the sidecar",
                       errorCode,
                       queryMessage));
    }

    return observation;
}

} // namespace hipdnn_integration_tests::bundle
