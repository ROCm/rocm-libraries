// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportVerdict.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

const char* toString(SupportVerdict verdict)
{
    switch(verdict)
    {
    case SupportVerdict::NO_SIDECAR:
        return "NO_SIDECAR";
    case SupportVerdict::SATISFIED:
        return "SATISFIED";
    case SupportVerdict::CLAIM_BROKEN:
        return "CLAIM_BROKEN";
    case SupportVerdict::QUERY_ERRORED:
        return "QUERY_ERRORED";
    case SupportVerdict::ENGINE_NOT_LOADED:
        return "ENGINE_NOT_LOADED";
    case SupportVerdict::NOT_ENFORCED:
        return "NOT_ENFORCED";
    case SupportVerdict::UNCLAIMED_SUPPORT:
        return "UNCLAIMED_SUPPORT";
    default:
        return "UNKNOWN";
    }
}

// Fail-closed: unknown/future verdicts are failures by default.
bool isFailure(SupportVerdict verdict)
{
    switch(verdict)
    {
    case SupportVerdict::SATISFIED:
    case SupportVerdict::NO_SIDECAR:
    case SupportVerdict::NOT_ENFORCED:
    case SupportVerdict::UNCLAIMED_SUPPORT:
    case SupportVerdict::ENGINE_NOT_LOADED:
        return false;
    default:
        return true;
    }
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

bool isInRankedList(const std::vector<int64_t>& rankedIds, int64_t engineId)
{
    return std::find(rankedIds.begin(), rankedIds.end(), engineId) != rankedIds.end();
}

} // namespace

SupportResult evaluateSupport(hipdnn_frontend::ErrorCode errorCode,
                              const std::vector<int64_t>& rankedIds,
                              int64_t engineId,
                              bool claimed,
                              bool hasSidecar,
                              const std::string& bundlePath,
                              const std::string& engineName,
                              const std::string& arch,
                              const std::string& platform,
                              std::string_view queryMessage)
{
    SupportResult result;
    result.bundlePath = bundlePath;
    result.engineName = engineName;
    result.arch = arch;
    result.platform = platform;

    const bool resolved = isResolved(errorCode);

    // Recorded ahead of every branch, including the NO_SIDECAR short-circuit below:
    // the verdict may not consult the query, but the caller did make it, and the
    // observation is true regardless of what the sidecar says. The message is kept
    // only on the unresolved branch — see SupportResult::queryMessage.
    result.queryStatus = errorCode;
    if(!resolved)
    {
        result.queryMessage = std::string(queryMessage);
    }

    if(!hasSidecar)
    {
        result.verdict = SupportVerdict::NO_SIDECAR;
        result.detail = "no support.json beside this graph";
        return result;
    }

    const bool supported = resolved && isInRankedList(rankedIds, engineId);

    if(claimed)
    {
        if(!resolved)
        {
            result.verdict = SupportVerdict::QUERY_ERRORED;
            result.detail = "sidecar claims support, but query returned "
                            + hipdnn_frontend::to_string(errorCode);
        }
        else if(supported)
        {
            result.verdict = SupportVerdict::SATISFIED;
            result.detail = "engine in ranked list";
        }
        else
        {
            result.verdict = SupportVerdict::CLAIM_BROKEN;
            result.detail = "sidecar claims support, but engine not in ranked list (status="
                            + hipdnn_frontend::to_string(errorCode) + ")";
        }
    }
    else
    {
        if(supported)
        {
            result.verdict = SupportVerdict::UNCLAIMED_SUPPORT;
            result.detail = "engine supports this graph but has no claim in the sidecar";
        }
        else
        {
            // Do not assert the ranked list was read when it wasn't: `supported`
            // is `resolved && isInRankedList(...)`, so an unresolved query lands
            // here too, and saying "not in ranked list" would state an
            // unverified fact.
            result.verdict = SupportVerdict::NOT_ENFORCED;
            result.detail = resolved ? "unclaimed, engine not in ranked list"
                                     : "unclaimed, and query did not resolve ("
                                           + hipdnn_frontend::to_string(errorCode) + ")";
        }
    }

    return result;
}

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

    // Only ever set on an unresolved query, which is also the case where `detail`
    // can say least: it has the code but not the reason. QUERY_ERRORED is a FAIL,
    // so this is the one place a human sees the backend's own words.
    if(!result.queryMessage.empty())
    {
        os << "  query:    " << result.queryMessage << "\n";
    }
    return os.str();
}

std::vector<SupportResult> observeAllSupport(hipdnn_frontend::ErrorCode errorCode,
                                             const std::vector<int64_t>& rankedIds,
                                             const SupportClaimLocator& locator,
                                             const std::vector<LoadedEngine>& loadedEngines,
                                             std::string_view queryMessage)
{
    if(locator.sidecarPath.empty() || !std::filesystem::exists(locator.sidecarPath))
    {
        return {};
    }

    std::optional<SupportClaims> singleClaims;
    std::optional<SweepSupportClaims> sweepClaims;
    if(locator.isSweep())
    {
        sweepClaims = loadSweepSupportClaimsFromPath(locator.sidecarPath);
    }
    else
    {
        singleClaims = loadSupportClaimsFromPath(locator.sidecarPath);
    }

    const std::string arch = baseArchToken(TestConfig::get().getCurrentArch());
    const std::string platform = currentPlatform();

    const std::set<std::string> claimedNames
        = locator.isSweep() ? sweepClaims->claimedEngineNames(locator.caseId, arch, platform)
                            : singleClaims->claimedEngineNames(arch, platform);

    std::set<std::string> loadedNames;
    std::vector<SupportResult> results;
    for(const auto& engine : loadedEngines)
    {
        loadedNames.insert(engine.name);

        const bool claimed = claimedNames.count(engine.name) != 0;

        auto result = evaluateSupport(errorCode,
                                      rankedIds,
                                      engine.id,
                                      claimed,
                                      /*hasSidecar=*/true,
                                      locator.diagnosticPath,
                                      engine.name,
                                      arch,
                                      platform,
                                      queryMessage);

        results.push_back(std::move(result));
    }

    for(const auto& name : claimedNames)
    {
        if(loadedNames.count(name) == 0)
        {
            SupportResult r;
            r.verdict = SupportVerdict::ENGINE_NOT_LOADED;
            r.bundlePath = locator.diagnosticPath;
            r.engineName = name;
            r.arch = arch;
            r.platform = platform;
            r.detail = "sidecar claims support, but engine is not loaded in the runtime";
            results.push_back(std::move(r));
        }
    }

    return results;
}

} // namespace hipdnn_integration_tests::bundle
