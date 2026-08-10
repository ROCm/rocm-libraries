// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportVerdict.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

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
    case SupportVerdict::NOT_ENFORCED:
        return "NOT_ENFORCED";
    case SupportVerdict::UNCLAIMED_SUPPORT:
        return "UNCLAIMED_SUPPORT";
    default:
        return "UNKNOWN";
    }
}

bool isFailure(SupportVerdict verdict)
{
    return verdict == SupportVerdict::CLAIM_BROKEN || verdict == SupportVerdict::QUERY_ERRORED;
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

SupportResult checkSupportClaim(hipdnn_frontend::ErrorCode errorCode,
                                const std::vector<int64_t>& rankedIds,
                                int64_t engineId,
                                const std::filesystem::path& bundlePath,
                                std::string_view queryMessage)
{
    const std::string engineName(TestConfig::get().getEngineName());
    const std::string arch = baseArchToken(TestConfig::get().getCurrentArch());
    const std::string platform = currentPlatform();

    bool hasSidecar = false;
    bool claimed = false;

    if(!bundlePath.empty())
    {
        const auto claims = loadSupportClaims(bundlePath);
        hasSidecar = claims.has_value();
        claimed = hasSidecar && claims->isClaimed(engineName, arch, platform);
    }

    return evaluateSupport(errorCode,
                           rankedIds,
                           engineId,
                           claimed,
                           hasSidecar,
                           bundlePath.string(),
                           engineName,
                           arch,
                           platform,
                           queryMessage);
}

std::vector<SupportResult> checkAllSupportClaims(hipdnn_frontend::ErrorCode errorCode,
                                                 const std::vector<int64_t>& rankedIds,
                                                 const SupportClaimLocator& locator,
                                                 const std::vector<LoadedEngine>& loadedEngines,
                                                 std::string_view queryMessage)
{
    if(locator.sidecarPath.empty() || !std::filesystem::exists(locator.sidecarPath))
    {
        return {};
    }

    std::ifstream file(locator.sidecarPath);
    if(!file)
    {
        throw std::runtime_error("Could not open support claims file: "
                                 + locator.sidecarPath.string());
    }

    auto json = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if(json.is_discarded())
    {
        throw std::runtime_error("support.json is not parseable JSON: "
                                 + locator.sidecarPath.string());
    }

    std::optional<SupportClaims> singleClaims;
    std::optional<SweepSupportClaims> sweepClaims;
    if(locator.isSweep())
    {
        sweepClaims = parseSweepSupportClaimsJson(json, locator.sidecarPath.string());
    }
    else
    {
        singleClaims = parseSupportClaimsJson(json, locator.sidecarPath.string());
    }

    const std::string arch = baseArchToken(TestConfig::get().getCurrentArch());
    const std::string platform = currentPlatform();

    std::vector<SupportResult> results;
    for(const auto& engine : loadedEngines)
    {
        const bool claimed
            = locator.isSweep()
                  ? sweepClaims->isClaimed(locator.caseId, engine.name, arch, platform)
                  : singleClaims->isClaimed(engine.name, arch, platform);

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

        if(result.verdict != SupportVerdict::NOT_ENFORCED)
        {
            results.push_back(std::move(result));
        }
    }

    return results;
}

} // namespace hipdnn_integration_tests::bundle
