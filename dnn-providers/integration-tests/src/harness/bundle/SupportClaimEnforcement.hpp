// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/SupportClaims.hpp"

// Pure decision logic for RFC 0015 §5.2 ("verdict meaning"), §7.1 ("per-graph
// outcomes"), and the engine-attribution machinery of §8. Nothing here touches
// GTest, a hipdnnHandle_t, or a hipdnn_frontend::graph::Graph — every function
// takes already-observed data (a support query's Error + ranked id list, the
// bundle's claims, the loaded engine names) and returns a verdict for the
// caller (IntegrationBundleVerificationHarness) to turn into GTest FAIL/report
// calls. This split is what makes the ladder's decision logic unit-testable
// without a real backend/plugin.
namespace hipdnn_integration_tests::bundle
{

// Splits a raw gcnArchName into the arch token support.json keys use — the
// prefix before the first ':' (RFC 0015 §5.1). e.g.
// "gfx942:sramecc+:xnack-" -> "gfx942". A token with no ':' is returned as-is.
inline std::string archToken(const std::string& rawArch)
{
    const auto pos = rawArch.find(':');
    return pos == std::string::npos ? rawArch : rawArch.substr(0, pos);
}

// True iff `status` is a *resolved* support query (RFC 0015 §5.2): status OK
// or GRAPH_NOT_SUPPORTED both mean the harness successfully determined
// support (an empty ranked list under GRAPH_NOT_SUPPORTED is the legitimate
// "no engine accepts it" answer). Any other error status (backend error, OOM,
// device lost, build failure) is *unresolved* — no per-engine verdict is
// available for any engine.
inline bool isResolvedSupportQuery(const hipdnn_frontend::Error& status)
{
    return status.get_code() == hipdnn_frontend::ErrorCode::OK
           || status.get_code() == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
}

// One engine's two-step verdict (RFC 0015 §5.2): the status code separates
// *resolved* from *unknown*; ranked-list membership separates *supported*
// from *declined* within a resolved query.
enum class EngineVerdict
{
    Supported,
    Declined,
    Unknown,
};

// Classifies a single engine's verdict from one support query. `engineId` is
// the claimed engine's numeric id (hipdnn_data_sdk::utilities::engineNameToId);
// `rankedEngineIds` is the id list returned by get_ranked_engine_ids()
// alongside `status`.
inline EngineVerdict classifyEngineVerdict(const hipdnn_frontend::Error& status,
                                           const std::vector<int64_t>& rankedEngineIds,
                                           int64_t engineId)
{
    if(!isResolvedSupportQuery(status))
    {
        return EngineVerdict::Unknown;
    }
    const bool present = std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId)
                         != rankedEngineIds.end();
    return present ? EngineVerdict::Supported : EngineVerdict::Declined;
}

// One claimed engine whose verdict was NOT Supported — the only rows a caller
// needs to act on (RFC 0015 §7.1: Declined -> claim-broken FAIL, Unknown ->
// errored-before-assert FAIL). Supported claims are silently fine and are not
// returned.
struct ClaimCheckResult
{
    std::string engine;
    EngineVerdict verdict; // Declined or Unknown; never Supported
};

// Evaluates every LOADED engine that the bundle's claims list as supported for
// (arch, platform) against one support query's outcome (RFC 0015 §5.2, §7.1,
// §7.3). An engine not loaded in the current run, or not claimed for this
// (arch, platform), is simply skipped -- "not enforced", never a verdict.
// Multi-engine by construction: every claimed+loaded engine is attributed
// independently from the same query, so one engine's Declined never taints
// another's Supported.
inline std::vector<ClaimCheckResult>
    evaluateClaimedEngines(const std::vector<std::string>& loadedEngineNames,
                           const std::string& arch,
                           const std::string& platform,
                           const SupportClaims& claims,
                           const hipdnn_frontend::Error& status,
                           const std::vector<int64_t>& rankedEngineIds)
{
    std::vector<ClaimCheckResult> results;
    for(const auto& engine : loadedEngineNames)
    {
        if(!claims.isClaimed(engine, arch, platform))
        {
            continue;
        }

        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(engine);
        const auto verdict = classifyEngineVerdict(status, rankedEngineIds, engineId);
        if(verdict != EngineVerdict::Supported)
        {
            results.push_back({engine, verdict});
        }
    }
    return results;
}

// Loaded engines that a *resolved* query reports as supported but that the
// bundle's claims do NOT list for (arch, platform) — the "supported but
// unclaimed" rows for the end-of-run informational summary (RFC 0015 §7.1).
// Returns empty for an unresolved query: support is unknown, never treated as
// an unclaimed-support observation either way.
inline std::vector<std::string>
    findUnclaimedSupportedEngines(const std::vector<std::string>& loadedEngineNames,
                                  const std::string& arch,
                                  const std::string& platform,
                                  const SupportClaims& claims,
                                  const hipdnn_frontend::Error& status,
                                  const std::vector<int64_t>& rankedEngineIds)
{
    std::vector<std::string> result;
    if(!isResolvedSupportQuery(status))
    {
        return result;
    }

    for(const auto& engine : loadedEngineNames)
    {
        if(claims.isClaimed(engine, arch, platform))
        {
            continue;
        }

        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(engine);
        if(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId)
           != rankedEngineIds.end())
        {
            result.push_back(engine);
        }
    }
    return result;
}

} // namespace hipdnn_integration_tests::bundle
