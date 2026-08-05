// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimWriter.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace hipdnn_integration_tests::bundle
{

namespace
{

// Must match SupportClaims.cpp's kSupportedSchemaVersion -- a freshly
// (re)written sidecar is always current, whatever version the on-disk file
// it was merged with happened to declare.
constexpr int kCurrentSchemaVersion = 1;

std::vector<std::string> sortedVector(const std::set<std::string>& values)
{
    // std::set already iterates in sorted order; this just changes container
    // type for JSON array serialization.
    return {values.begin(), values.end()};
}

} // namespace

SupportClaims applyClaimObservations(SupportClaims existing,
                                     const std::vector<ClaimObservation>& observations)
{
    for(const auto& obs : observations)
    {
        auto& archMap = existing.claims[obs.engine];
        if(obs.supported)
        {
            archMap[obs.arch].insert(obs.platform);
            continue;
        }

        auto archIt = archMap.find(obs.arch);
        if(archIt == archMap.end())
        {
            continue;
        }
        archIt->second.erase(obs.platform);
        if(archIt->second.empty())
        {
            archMap.erase(archIt);
        }
    }

    for(auto it = existing.claims.begin(); it != existing.claims.end();)
    {
        it = it->second.empty() ? existing.claims.erase(it) : std::next(it);
    }

    existing.version = kCurrentSchemaVersion;
    return existing;
}

SweepSupportClaims
    applySweepClaimObservations(SweepSupportClaims existing,
                                const std::vector<SweepClaimObservation>& observations)
{
    // 1. Flatten: engine -> case id -> ArchPlatformMap. std::map keeps case
    // ids in sorted order for free, which step 4 relies on.
    std::map<std::string, std::map<std::string, ArchPlatformMap>> flat;
    for(const auto& [engine, groups] : existing.claims)
    {
        for(const auto& group : groups)
        {
            for(const auto& caseId : group.cases)
            {
                flat[engine][caseId] = group.support;
            }
        }
    }

    // 2. Overlay every observed cell.
    for(const auto& obs : observations)
    {
        auto& archMap = flat[obs.engine][obs.caseId];
        if(obs.supported)
        {
            archMap[obs.arch].insert(obs.platform);
            continue;
        }

        auto archIt = archMap.find(obs.arch);
        if(archIt == archMap.end())
        {
            continue;
        }
        archIt->second.erase(obs.platform);
        if(archIt->second.empty())
        {
            archMap.erase(archIt);
        }
    }

    // 3. Drop cells left with no claimed platforms at all -- an empty claim
    // is the same as "this case was never named" (RFC 0015 §5.4).
    for(auto& [engine, cases] : flat)
    {
        for(auto it = cases.begin(); it != cases.end();)
        {
            it = it->second.empty() ? cases.erase(it) : std::next(it);
        }
    }

    // 4. Re-group canonically: cases sharing an identical ArchPlatformMap
    // collapse into one group; groups are ordered by their first case id.
    SweepSupportClaims result;
    result.version = kCurrentSchemaVersion;
    for(auto& [engine, cases] : flat)
    {
        if(cases.empty())
        {
            continue;
        }

        std::vector<SweepClaimGroup> groups;
        for(auto& casePair : cases)
        {
            const auto& caseId = casePair.first;
            const auto& archMap = casePair.second;
            auto groupIt
                = std::find_if(groups.begin(), groups.end(), [&](const SweepClaimGroup& g) {
                      return g.support == archMap;
                  });
            if(groupIt != groups.end())
            {
                groupIt->cases.push_back(caseId);
            }
            else
            {
                groups.push_back(SweepClaimGroup{{caseId}, archMap});
            }
        }

        // `cases` (the flattened map) already yields ids in ascending order,
        // so each group's `cases` vector is built sorted -- sort explicitly
        // anyway so canonical ordering never depends on iteration order.
        for(auto& group : groups)
        {
            std::sort(group.cases.begin(), group.cases.end());
        }
        std::sort(
            groups.begin(), groups.end(), [](const SweepClaimGroup& a, const SweepClaimGroup& b) {
                return a.cases.front() < b.cases.front();
            });

        result.claims[engine] = std::move(groups);
    }

    return result;
}

nlohmann::json toCanonicalJson(const SupportClaims& claims)
{
    nlohmann::json claimsJson = nlohmann::json::object();
    for(const auto& [engine, archMap] : claims.claims)
    {
        nlohmann::json archJson = nlohmann::json::object();
        for(const auto& [arch, platforms] : archMap)
        {
            archJson[arch] = sortedVector(platforms);
        }
        claimsJson[engine] = std::move(archJson);
    }

    nlohmann::json j;
    j["version"] = claims.version;
    j["claims"] = std::move(claimsJson);
    return j;
}

nlohmann::json toCanonicalJson(const SweepSupportClaims& claims)
{
    nlohmann::json claimsJson = nlohmann::json::object();
    for(const auto& [engine, groups] : claims.claims)
    {
        nlohmann::json groupsJson = nlohmann::json::array();
        for(const auto& group : groups)
        {
            nlohmann::json supportJson = nlohmann::json::object();
            for(const auto& [arch, platforms] : group.support)
            {
                supportJson[arch] = sortedVector(platforms);
            }

            nlohmann::json groupJson;
            groupJson["cases"] = group.cases;
            groupJson["support"] = std::move(supportJson);
            groupsJson.push_back(std::move(groupJson));
        }
        claimsJson[engine] = std::move(groupsJson);
    }

    nlohmann::json j;
    j["version"] = claims.version;
    j["claims"] = std::move(claimsJson);
    return j;
}

namespace
{

void writeJsonFile(const std::filesystem::path& path, const nlohmann::json& json)
{
    try
    {
        if(path.has_parent_path())
        {
            std::filesystem::create_directories(path.parent_path());
        }

        std::ofstream file(path, std::ios::trunc);
        if(!file)
        {
            throw std::runtime_error("Could not open support claims file for writing: "
                                     + path.string());
        }

        file << json.dump(2) << "\n";
        if(!file)
        {
            throw std::runtime_error("Failed while writing support claims file: " + path.string());
        }
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        // Re-thrown as std::runtime_error naming the actual sidecar path
        // (RFC 0015 §9.2: "reports a clear error naming the file") -- the
        // raw filesystem_error only names whichever path component the OS
        // call failed on (e.g. a blocked parent directory), not necessarily
        // this sidecar file itself.
        throw std::runtime_error("Could not write support claims file: " + path.string() + " ("
                                 + e.what() + ")");
    }
}

} // namespace

void writeSupportClaimsFile(const std::filesystem::path& path, const SupportClaims& claims)
{
    writeJsonFile(path, toCanonicalJson(claims));
}

void writeSweepSupportClaimsFile(const std::filesystem::path& path,
                                 const SweepSupportClaims& claims)
{
    writeJsonFile(path, toCanonicalJson(claims));
}

// ---------------------------------------------------------------------------
// ClaimObservationCollector
// ---------------------------------------------------------------------------

ClaimObservationCollector& ClaimObservationCollector::get()
{
    static ClaimObservationCollector instance;
    return instance;
}

void ClaimObservationCollector::record(const ClaimWriteTarget& target,
                                       const std::string& engine,
                                       const std::string& arch,
                                       const std::string& platform,
                                       bool supported)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _records.push_back({target, engine, arch, platform, supported});
}

bool ClaimObservationCollector::empty() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _records.empty();
}

void ClaimObservationCollector::reset()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _records.clear();
}

std::vector<std::filesystem::path> ClaimObservationCollector::writeAll() const
{
    std::vector<Record> records;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        records = _records;
    }

    // Bucket by anchor path -- a direct bundle's {Name}.json path never
    // collides with a sweep directory path, so string identity alone is a
    // sufficient (and stable, for iteration order) bucket key.
    std::map<std::string, std::vector<Record>> byAnchor;
    for(auto& record : records)
    {
        byAnchor[record.target.anchorPath.string()].push_back(record);
    }

    std::vector<std::filesystem::path> written;
    written.reserve(byAnchor.size());
    for(const auto& anchorEntry : byAnchor)
    {
        const auto& bucket = anchorEntry.second;
        const auto& anchorPath = bucket.front().target.anchorPath;
        if(bucket.front().target.isSweepCase())
        {
            auto existingOpt = loadSweepSupportClaims(anchorPath);
            const bool fileExisted = existingOpt.has_value();
            std::vector<SweepClaimObservation> observations;
            observations.reserve(bucket.size());
            for(const auto& record : bucket)
            {
                observations.push_back({*record.target.sweepCaseId,
                                        record.engine,
                                        record.arch,
                                        record.platform,
                                        record.supported});
            }
            auto merged = applySweepClaimObservations(
                std::move(existingOpt).value_or(SweepSupportClaims{}), observations);

            // A brand-new sidecar (no prior file) that ends up with zero
            // claims after this run's observations would still flip the
            // sibling case(s) it covers into "claim-bearing" per the loader
            // (a support.json's mere existence requires an explicit
            // enforcement_level -- RFC 0015 §6.2), for zero enforcement
            // value (there is nothing to enforce with no claims). Only
            // write it when refreshing a file that already existed (where
            // that requirement was already in force) or the merge produced
            // at least one real claim.
            if(!fileExisted && merged.claims.empty())
            {
                continue;
            }

            const auto sidecarPath = anchorPath / "support.json";
            writeSweepSupportClaimsFile(sidecarPath, merged);
            written.push_back(sidecarPath);
        }
        else
        {
            auto existingOpt = loadSupportClaims(anchorPath);
            const bool fileExisted = existingOpt.has_value();
            std::vector<ClaimObservation> observations;
            observations.reserve(bucket.size());
            for(const auto& record : bucket)
            {
                observations.push_back(
                    {record.engine, record.arch, record.platform, record.supported});
            }
            auto merged = applyClaimObservations(std::move(existingOpt).value_or(SupportClaims{}),
                                                 observations);

            // Same rationale as the sweep branch above.
            if(!fileExisted && merged.claims.empty())
            {
                continue;
            }

            const auto sidecarPath = supportJsonPath(anchorPath);
            writeSupportClaimsFile(sidecarPath, merged);
            written.push_back(sidecarPath);
        }
    }

    return written;
}

// ---------------------------------------------------------------------------
// EnginePassContext
// ---------------------------------------------------------------------------

EnginePassContext& EnginePassContext::get()
{
    static EnginePassContext instance;
    return instance;
}

void EnginePassContext::set(std::string engineName, int64_t engineId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _name = std::move(engineName);
    _id = engineId;
}

void EnginePassContext::clear()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _name.reset();
    _id.reset();
}

std::optional<std::string> EnginePassContext::name() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _name;
}

std::optional<int64_t> EnginePassContext::id() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _id;
}

} // namespace hipdnn_integration_tests::bundle
