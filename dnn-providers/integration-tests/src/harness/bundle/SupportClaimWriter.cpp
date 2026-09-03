// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimWriter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

void overlaySingleGraphCell(SupportClaims& existing,
                            const std::string& engineName,
                            const std::string& arch,
                            const std::string& platform,
                            bool engineIsSupported)
{
    if(engineIsSupported)
    {
        existing.claims[engineName][arch].insert(platform);
    }
    else
    {
        auto engineIt = existing.claims.find(engineName);
        if(engineIt == existing.claims.end())
        {
            return;
        }
        auto archIt = engineIt->second.find(arch);
        if(archIt == engineIt->second.end())
        {
            return;
        }
        archIt->second.erase(platform);
        if(archIt->second.empty())
        {
            engineIt->second.erase(archIt);
        }
        if(engineIt->second.empty())
        {
            existing.claims.erase(engineIt);
        }
    }
}

// Per-case support: engine -> caseId -> ArchPlatformMap
using FlatSweepMap = std::map<std::string, std::map<std::string, ArchPlatformMap>>;

FlatSweepMap flattenSweepClaims(const SweepSupportClaims& existing)
{
    FlatSweepMap flat;
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
    return flat;
}

SweepSupportClaims regroupSweepClaims(const FlatSweepMap& flat, int version)
{
    SweepSupportClaims result;
    result.version = version;

    for(const auto& [engine, caseMap] : flat)
    {
        // Bucket cases by identical support footprint. ArchPlatformMap is a
        // std::map of std::sets, so it is directly usable as a map key — two
        // cases land in the same bucket exactly when their support is equal.
        std::map<ArchPlatformMap, std::vector<std::string>> casesByFootprint;

        for(const auto& [caseId, supportMap] : caseMap)
        {
            if(supportMap.empty())
            {
                continue;
            }
            casesByFootprint[supportMap].push_back(caseId);
        }

        std::vector<SweepClaimGroup> groups;
        for(auto& [supportMap, cases] : casesByFootprint)
        {
            std::sort(cases.begin(), cases.end());
            groups.push_back({std::move(cases), supportMap});
        }

        // Order groups by their first case id.
        std::sort(
            groups.begin(), groups.end(), [](const SweepClaimGroup& a, const SweepClaimGroup& b) {
                return a.cases.front() < b.cases.front();
            });

        if(!groups.empty())
        {
            result.claims[engine] = std::move(groups);
        }
    }

    return result;
}

// "" is a legal JSON key, so a cell with an empty engine, arch, or platform
// serializes without complaint and lands an entry in a checked-in sidecar that
// no enforcement run can ever match or clear. An empty arch is the realistic
// one: it is what a failed device probe leaves behind in the policy.
std::string cellDefect(const ObservedSupportCell& cell)
{
    if(cell.claimLocator.sidecarPath.empty())
    {
        return "empty sidecar path";
    }
    if(cell.engineName.empty())
    {
        return "empty engine name";
    }
    if(cell.arch.empty())
    {
        return "empty arch";
    }
    if(cell.platform.empty())
    {
        return "empty platform";
    }
    return {};
}

bool writeIfChanged(const std::filesystem::path& filePath,
                    const std::string& newContent,
                    WriteSummary& summary)
{
    if(std::filesystem::exists(filePath))
    {
        std::ifstream existingFile(filePath);
        if(existingFile)
        {
            const std::string existingContent((std::istreambuf_iterator<char>(existingFile)),
                                              std::istreambuf_iterator<char>());
            if(existingContent == newContent)
            {
                ++summary.filesUnchanged;
                return true;
            }
        }
    }

    std::ofstream outputFile(filePath);
    if(!outputFile)
    {
        summary.errors.push_back("could not open for writing: " + filePath.string());
        return false;
    }
    outputFile << newContent;
    if(!outputFile)
    {
        summary.errors.push_back("write failed: " + filePath.string());
        return false;
    }
    ++summary.filesWritten;
    return true;
}

} // namespace

WriteSummary writeObservedSupportClaims(const std::vector<ObservedSupportCell>& observations)
{
    WriteSummary summary;

    // One sidecar file's worth of work. isSweep is a property of the bundle, not
    // of the engine queried, so every observation landing here should agree on
    // it; `skip` records the cases where they did not, or where an observation
    // was too malformed to place.
    struct SidecarTarget
    {
        bool isSweep = false;
        bool skip = false;
        std::vector<ObservedSupportCell> observations;
    };

    // Keyed on the normalized path so that two spellings of one file ("d/x.json"
    // and "d/./x.json") cannot become two targets. Two targets for one file would
    // slip past the isSweep check below and write the file twice, the second
    // write overlaying a set of claims the first had already replaced.
    std::map<std::filesystem::path, SidecarTarget> targetsBySidecarPath;
    for(const auto& observation : observations)
    {
        auto& target
            = targetsBySidecarPath[observation.claimLocator.sidecarPath.lexically_normal()];

        if(target.observations.empty())
        {
            target.isSweep = observation.claimLocator.isSweep();
        }
        else if(target.isSweep != observation.claimLocator.isSweep())
        {
            // Single-graph and sweep sidecars have incompatible shapes, so one of
            // the two readings of this file is wrong. Which one is not knowable
            // here, and writing under either would destroy the other's structure.
            summary.errors.push_back(
                "refusing to write a sidecar observed as both single-graph and sweep: "
                + observation.claimLocator.sidecarPath.string());
            target.skip = true;
        }

        const std::string defect = cellDefect(observation);
        if(!defect.empty())
        {
            summary.errors.push_back("refusing to write from a malformed observation (" + defect
                                     + "): '" + observation.claimLocator.diagnosticPath + "'");
            target.skip = true;
        }

        target.observations.push_back(observation);
    }

    // Every sidecar reached below is written from a complete, well-formed
    // observation set. A sidecar nothing observed is never a key here, which is
    // the RFC §9.2 empty-write guard: an absent observation set cannot reach the
    // file, let alone null a claim in it.
    for(const auto& [sidecarPath, target] : targetsBySidecarPath)
    {
        if(target.skip)
        {
            continue;
        }

        const auto& fileObservations = target.observations;

        if(target.isSweep)
        {
            SweepSupportClaims existing;
            existing.version = 1;
            if(std::filesystem::exists(sidecarPath))
            {
                try
                {
                    // Read the file we are about to write, by the same rule the
                    // enforcer reads it. Re-deriving the path from the directory
                    // would let the read and the write name different files, and
                    // a read that misses returns an empty claim set -- which
                    // overlays into a write that erases everything in it.
                    existing = loadSweepSupportClaimsFromPath(sidecarPath);
                }
                catch(const std::exception& e)
                {
                    summary.errors.push_back("refusing to overwrite unparseable sidecar: "
                                             + sidecarPath.string() + ": " + e.what());
                    continue;
                }
            }

            auto flat = flattenSweepClaims(existing);

            for(const auto& obs : fileObservations)
            {
                const auto& caseId = obs.claimLocator.caseId;
                if(obs.engineIsSupported)
                {
                    flat[obs.engineName][caseId][obs.arch].insert(obs.platform);
                }
                else
                {
                    auto engineIt = flat.find(obs.engineName);
                    if(engineIt == flat.end())
                    {
                        continue;
                    }
                    auto caseIt = engineIt->second.find(caseId);
                    if(caseIt == engineIt->second.end())
                    {
                        continue;
                    }
                    auto archIt = caseIt->second.find(obs.arch);
                    if(archIt == caseIt->second.end())
                    {
                        continue;
                    }
                    archIt->second.erase(obs.platform);
                    if(archIt->second.empty())
                    {
                        caseIt->second.erase(archIt);
                    }
                    if(caseIt->second.empty())
                    {
                        engineIt->second.erase(caseIt);
                    }
                    if(engineIt->second.empty())
                    {
                        flat.erase(engineIt);
                    }
                }
            }

            const auto regrouped = regroupSweepClaims(flat, existing.version);
            const auto jsonContent = dumpCanonical(toJson(regrouped));
            writeIfChanged(sidecarPath, jsonContent, summary);
        }
        else
        {
            SupportClaims existing;
            existing.version = 1;
            if(std::filesystem::exists(sidecarPath))
            {
                std::ifstream existingFile(sidecarPath);
                if(existingFile)
                {
                    auto json
                        = nlohmann::json::parse(existingFile, nullptr, /*allow_exceptions=*/false);
                    if(json.is_discarded())
                    {
                        summary.errors.push_back("refusing to overwrite unparseable sidecar: "
                                                 + sidecarPath.string());
                        continue;
                    }
                    try
                    {
                        existing = parseSupportClaimsJson(json, sidecarPath.string());
                    }
                    catch(const std::exception& e)
                    {
                        summary.errors.push_back("refusing to overwrite unparseable sidecar: "
                                                 + sidecarPath.string() + ": " + e.what());
                        continue;
                    }
                }
            }

            for(const auto& obs : fileObservations)
            {
                overlaySingleGraphCell(
                    existing, obs.engineName, obs.arch, obs.platform, obs.engineIsSupported);
            }

            const auto jsonContent = dumpCanonical(toJson(existing));
            writeIfChanged(sidecarPath, jsonContent, summary);
        }
    }

    return summary;
}

} // namespace hipdnn_integration_tests::bundle
