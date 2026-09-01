// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaims.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace hipdnn_integration_tests::bundle
{

namespace
{

// support.json is machine-owned and committed (RFC 0015 §5.1/§5.3): a version
// this loader does not understand, or a shape that does not match the schema,
// must fail the run loudly rather than silently degrade — unlike
// BundleMetadata's optional+WARN handling of human-authored, fully-optional
// metadata.
constexpr int K_SUPPORTED_SCHEMA_VERSION = 1;

const std::set<std::string>& validPlatformTokens()
{
    static const std::set<std::string> s_tokens = {"linux", "windows"};
    return s_tokens;
}

std::string withSource(std::string_view source, const std::string& message)
{
    if(source.empty())
    {
        return message;
    }
    return std::string(source) + ": " + message;
}

int parseVersion(const nlohmann::json& json, std::string_view source)
{
    if(!json.contains("version") || !json.at("version").is_number_integer())
    {
        throw std::runtime_error(withSource(source, "missing or invalid 'version'"));
    }

    const int version = json.at("version").get<int>();
    if(version != K_SUPPORTED_SCHEMA_VERSION)
    {
        throw std::runtime_error(
            withSource(source, "unsupported version " + std::to_string(version)));
    }
    return version;
}

// Shared by both claim shapes: `supportObj` is an arch -> [platforms] object,
// e.g. {"gfx942": ["linux", "windows"]}. An empty platform array is legal (an
// arch with no claimed platforms).
ArchPlatformMap parseArchPlatformMap(const nlohmann::json& supportObj, std::string_view source)
{
    if(!supportObj.is_object())
    {
        throw std::runtime_error(withSource(source, "support map must be an object"));
    }

    ArchPlatformMap archMap;
    for(const auto& [arch, platformsJson] : supportObj.items())
    {
        if(!platformsJson.is_array())
        {
            throw std::runtime_error(
                withSource(source, "arch '" + arch + "' value must be an array of platforms"));
        }

        std::set<std::string> platforms;
        for(const auto& platformJson : platformsJson)
        {
            if(!platformJson.is_string()
               || validPlatformTokens().count(platformJson.get<std::string>()) == 0)
            {
                std::string message = "arch '";
                message += arch;
                message += "' has invalid platform token ";
                message += platformJson.dump();
                message += " (expected one of: linux, windows)";
                throw std::runtime_error(withSource(source, message));
            }
            platforms.insert(platformJson.get<std::string>());
        }
        archMap[arch] = std::move(platforms);
    }
    return archMap;
}

} // namespace

bool SupportClaims::isClaimed(const std::string& engine,
                              const std::string& arch,
                              const std::string& platform) const
{
    const auto engineIt = claims.find(engine);
    if(engineIt == claims.end())
    {
        return false;
    }

    const auto archIt = engineIt->second.find(arch);
    if(archIt == engineIt->second.end())
    {
        return false;
    }

    return archIt->second.count(platform) != 0;
}

std::set<std::string> SupportClaims::claimedEngineNames(const std::string& arch,
                                                        const std::string& platform) const
{
    std::set<std::string> names;
    for(const auto& [engine, archMap] : claims)
    {
        const auto archIt = archMap.find(arch);
        if(archIt != archMap.end() && archIt->second.count(platform) != 0)
        {
            names.insert(engine);
        }
    }
    return names;
}

bool SweepSupportClaims::isClaimed(const std::string& caseId,
                                   const std::string& engine,
                                   const std::string& arch,
                                   const std::string& platform) const
{
    const auto engineIt = claims.find(engine);
    if(engineIt == claims.end())
    {
        return false;
    }

    for(const auto& group : engineIt->second)
    {
        if(std::find(group.cases.begin(), group.cases.end(), caseId) == group.cases.end())
        {
            continue;
        }

        const auto archIt = group.support.find(arch);
        if(archIt == group.support.end())
        {
            return false;
        }
        return archIt->second.count(platform) != 0;
    }

    return false;
}

std::set<std::string> SweepSupportClaims::claimedEngineNames(const std::string& caseId,
                                                             const std::string& arch,
                                                             const std::string& platform) const
{
    std::set<std::string> names;
    for(const auto& [engine, groups] : claims)
    {
        for(const auto& group : groups)
        {
            if(std::find(group.cases.begin(), group.cases.end(), caseId) == group.cases.end())
            {
                continue;
            }
            const auto archIt = group.support.find(arch);
            if(archIt != group.support.end() && archIt->second.count(platform) != 0)
            {
                names.insert(engine);
            }
        }
    }
    return names;
}

SupportClaims parseSupportClaimsJson(const nlohmann::json& json, std::string_view source)
{
    if(!json.is_object())
    {
        throw std::runtime_error(withSource(source, "support claims JSON is not an object"));
    }

    SupportClaims result;
    result.version = parseVersion(json, source);

    if(json.contains("claims"))
    {
        if(!json.at("claims").is_object())
        {
            throw std::runtime_error(withSource(source, "'claims' must be an object"));
        }

        for(const auto& [engine, archObj] : json.at("claims").items())
        {
            result.claims[engine] = parseArchPlatformMap(archObj, source);
        }
    }

    return result;
}

SweepSupportClaims parseSweepSupportClaimsJson(const nlohmann::json& json, std::string_view source)
{
    if(!json.is_object())
    {
        throw std::runtime_error(withSource(source, "support claims JSON is not an object"));
    }

    SweepSupportClaims result;
    result.version = parseVersion(json, source);

    if(json.contains("claims"))
    {
        if(!json.at("claims").is_object())
        {
            throw std::runtime_error(withSource(source, "'claims' must be an object"));
        }

        for(const auto& [engine, groupsJson] : json.at("claims").items())
        {
            if(!groupsJson.is_array())
            {
                throw std::runtime_error(withSource(
                    source, "engine '" + engine + "' claims must be an array of groups"));
            }

            // §5.4: every case is claimed at most once per engine, across all
            // of that engine's groups. The sibling sweep.json cross-check
            // (that these ids actually exist there) is out of scope here — it
            // needs sweep.json and belongs to the downstream verifier.
            std::set<std::string> seenCaseIds;
            std::vector<SweepClaimGroup> groups;
            for(const auto& groupJson : groupsJson)
            {
                if(!groupJson.is_object())
                {
                    throw std::runtime_error(
                        withSource(source, "engine '" + engine + "' has a non-object claim group"));
                }
                if(!groupJson.contains("cases") || !groupJson.at("cases").is_array()
                   || groupJson.at("cases").empty())
                {
                    throw std::runtime_error(withSource(
                        source,
                        "engine '" + engine + "' claim group missing non-empty 'cases' array"));
                }
                if(!groupJson.contains("support"))
                {
                    throw std::runtime_error(withSource(
                        source, "engine '" + engine + "' claim group missing 'support' object"));
                }

                SweepClaimGroup group;
                for(const auto& caseIdJson : groupJson.at("cases"))
                {
                    if(!caseIdJson.is_string())
                    {
                        throw std::runtime_error(withSource(
                            source,
                            "engine '" + engine + "' claim group has a non-string case id"));
                    }

                    auto caseId = caseIdJson.get<std::string>();
                    if(!seenCaseIds.insert(caseId).second)
                    {
                        std::string message = "case '";
                        message += caseId;
                        message += "' claimed twice for engine '";
                        message += engine;
                        message += "'";
                        throw std::runtime_error(withSource(source, message));
                    }
                    group.cases.push_back(std::move(caseId));
                }

                group.support = parseArchPlatformMap(groupJson.at("support"), source);
                groups.push_back(std::move(group));
            }

            result.claims[engine] = std::move(groups);
        }
    }

    return result;
}

std::filesystem::path supportJsonPath(const std::filesystem::path& bundleJsonPath)
{
    return bundleJsonPath.parent_path() / (bundleJsonPath.stem().string() + ".support.json");
}

namespace
{

nlohmann::json readJsonFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if(!file)
    {
        throw std::runtime_error("Could not open support claims file: " + path.string());
    }

    auto json = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if(json.is_discarded())
    {
        throw std::runtime_error("support.json is not parseable JSON: " + path.string());
    }
    return json;
}

} // namespace

SupportClaims loadSupportClaimsFromPath(const std::filesystem::path& sidecarPath)
{
    return parseSupportClaimsJson(readJsonFile(sidecarPath), sidecarPath.string());
}

SweepSupportClaims loadSweepSupportClaimsFromPath(const std::filesystem::path& sidecarPath)
{
    return parseSweepSupportClaimsJson(readJsonFile(sidecarPath), sidecarPath.string());
}

std::optional<SupportClaims> loadSupportClaims(const std::filesystem::path& bundleJsonPath)
{
    const auto path = supportJsonPath(bundleJsonPath);
    if(!std::filesystem::exists(path))
    {
        return std::nullopt;
    }
    return loadSupportClaimsFromPath(path);
}

std::optional<SweepSupportClaims> loadSweepSupportClaims(const std::filesystem::path& sweepDir)
{
    const auto path = sweepDir / "support.json";
    if(!std::filesystem::exists(path))
    {
        return std::nullopt;
    }
    return loadSweepSupportClaimsFromPath(path);
}

} // namespace hipdnn_integration_tests::bundle
