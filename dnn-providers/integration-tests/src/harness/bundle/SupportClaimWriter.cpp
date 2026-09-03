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

// Both sidecar shapes are the same claim set underneath: a single-graph bundle
// is the one-case sweep. Flattening to this on read, and only re-imposing the
// on-disk shape on write, keeps the whole middle of the writer shape-agnostic --
// one overlay, not one per format.
//
//   engine -> caseId ("" for single-graph) -> arch -> platforms
using FlatClaims = std::map<std::string, std::map<std::string, ArchPlatformMap>>;

// The parsers reject any version but 1, so a claim set that survived a read is
// version 1 and a claim set we invented starts there.
constexpr int K_SUPPORT_CLAIMS_VERSION = 1;

// "" is a legal JSON key, so an observation with an empty engine, arch, or
// platform serializes without complaint and lands an entry in a checked-in
// sidecar that no enforcement run can ever match or clear. An empty arch is the
// realistic one: it is what a failed device probe leaves behind in the policy.
std::string observationDefect(const ObservedGraphSupport& observation)
{
    if(observation.claimLocator.sidecarPath.empty())
    {
        return "empty sidecar path";
    }
    if(observation.engineName.empty())
    {
        return "empty engine name";
    }
    if(observation.arch.empty())
    {
        return "empty arch";
    }
    if(observation.platform.empty())
    {
        return "empty platform";
    }
    return {};
}

// Returns the reason this whole sidecar must be left alone, or "" to proceed.
// One bad observation condemns the file rather than itself: the write is an
// overlay onto checked-in claims, so applying the good half of a set we do not
// trust can erase a claim that is still true.
std::string sidecarDefect(const std::vector<ObservedGraphSupport>& observations)
{
    const bool isSweep = observations.front().claimLocator.isSweep();

    for(const auto& observation : observations)
    {
        // Single-graph and sweep sidecars have incompatible shapes, so one of
        // the two readings of this file is wrong. Which one is not knowable
        // here, and writing under either would destroy the other's structure.
        if(observation.claimLocator.isSweep() != isSweep)
        {
            return "refusing to write a sidecar observed as both single-graph and sweep: "
                   + observation.claimLocator.sidecarPath.string();
        }

        const std::string defect = observationDefect(observation);
        if(!defect.empty())
        {
            return "refusing to write from a malformed observation (" + defect + "): '"
                   + observation.claimLocator.diagnosticPath + "'";
        }
    }

    return {};
}

// Reads the file we are about to write, by the same rule the enforcer reads it.
// Re-deriving the path from the directory would let the read and the write name
// different files, and a read that misses returns an empty claim set -- which
// overlays into a write that erases everything in it.
//
// Absent file yields an empty set. Throws if the file exists but will not parse.
FlatClaims readFlatClaims(const std::filesystem::path& sidecarPath, bool isSweep)
{
    FlatClaims flat;
    if(!std::filesystem::exists(sidecarPath))
    {
        return flat;
    }

    if(isSweep)
    {
        for(const auto& [engine, groups] : loadSweepSupportClaimsFromPath(sidecarPath).claims)
        {
            for(const auto& group : groups)
            {
                // An empty footprint claims nothing; dropping it here is what
                // lets an empty FlatClaims mean exactly "nothing to write".
                if(group.support.empty())
                {
                    continue;
                }
                for(const auto& caseId : group.cases)
                {
                    flat[engine][caseId] = group.support;
                }
            }
        }
    }
    else
    {
        for(const auto& [engine, support] : loadSupportClaimsFromPath(sidecarPath).claims)
        {
            if(!support.empty())
            {
                flat[engine][std::string{}] = support;
            }
        }
    }

    return flat;
}

// Overlays one observation onto the loaded claims. True inserts; false erases,
// because a resolved decline is the authoring signal that retires a stale claim.
void applyObservation(FlatClaims& claims, const ObservedGraphSupport& observation)
{
    const std::string& caseId = observation.claimLocator.caseId;

    if(observation.engineIsSupported)
    {
        claims[observation.engineName][caseId][observation.arch].insert(observation.platform);
        return;
    }

    auto engineIt = claims.find(observation.engineName);
    if(engineIt == claims.end())
    {
        return;
    }
    auto caseIt = engineIt->second.find(caseId);
    if(caseIt == engineIt->second.end())
    {
        return;
    }
    auto archIt = caseIt->second.find(observation.arch);
    if(archIt == caseIt->second.end())
    {
        return;
    }

    // Prune every container the erase emptied. A left-behind empty engine or
    // case would serialize as an empty object, which reads back as "claimed
    // nothing here" instead of "no claim", and keeps the file alive.
    archIt->second.erase(observation.platform);
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
        claims.erase(engineIt);
    }
}

// Re-imposes the on-disk shape. Sweeps group cases that share a footprint, so a
// 500-case sweep where every case behaves alike stays one group rather than 500
// repeats -- which is why the grouped format cannot be edited in place.
std::string serializeFlatClaims(const FlatClaims& flat, bool isSweep)
{
    if(!isSweep)
    {
        SupportClaims out;
        out.version = K_SUPPORT_CLAIMS_VERSION;
        for(const auto& [engine, byCase] : flat)
        {
            const auto caseIt = byCase.find(std::string{});
            if(caseIt != byCase.end())
            {
                out.claims[engine] = caseIt->second;
            }
        }
        return dumpCanonical(toJson(out));
    }

    SweepSupportClaims out;
    out.version = K_SUPPORT_CLAIMS_VERSION;
    for(const auto& [engine, byCase] : flat)
    {
        // ArchPlatformMap is a std::map of std::sets, so it is directly usable
        // as a key -- two cases land in the same bucket exactly when their
        // support is equal. byCase is a std::map, so cases arrive sorted.
        std::map<ArchPlatformMap, std::vector<std::string>> casesByFootprint;
        for(const auto& [caseId, support] : byCase)
        {
            casesByFootprint[support].push_back(caseId);
        }

        std::vector<SweepClaimGroup> groups;
        groups.reserve(casesByFootprint.size());
        for(auto& [support, cases] : casesByFootprint)
        {
            groups.push_back({std::move(cases), support});
        }

        // Order groups by their first case id.
        std::sort(
            groups.begin(), groups.end(), [](const SweepClaimGroup& a, const SweepClaimGroup& b) {
                return a.cases.front() < b.cases.front();
            });

        if(!groups.empty())
        {
            out.claims[engine] = std::move(groups);
        }
    }

    return dumpCanonical(toJson(out));
}

enum class WriteOutcome
{
    Written,
    Unchanged,
    OpenFailed,
    WriteFailed,
};

// Byte-compares before writing so that "no support change" leaves no mtime bump
// and no git diff -- the property that lets CI run the tool and check `git diff`.
WriteOutcome writeIfChanged(const std::filesystem::path& filePath, const std::string& newContent)
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
                return WriteOutcome::Unchanged;
            }
        }
    }

    std::ofstream outputFile(filePath);
    if(!outputFile)
    {
        return WriteOutcome::OpenFailed;
    }
    outputFile << newContent;
    if(!outputFile)
    {
        return WriteOutcome::WriteFailed;
    }
    return WriteOutcome::Written;
}

// Groups observations by the file they land in. Keyed on the normalized path so
// that two spellings of one file ("d/x.json" and "d/./x.json") cannot become two
// targets and write it twice, the second write overlaying claims the first had
// already replaced.
std::map<std::filesystem::path, std::vector<ObservedGraphSupport>>
    groupBySidecarPath(const std::vector<ObservedGraphSupport>& observations)
{
    std::map<std::filesystem::path, std::vector<ObservedGraphSupport>> bySidecarPath;
    for(const auto& observation : observations)
    {
        bySidecarPath[observation.claimLocator.sidecarPath.lexically_normal()].push_back(
            observation);
    }
    return bySidecarPath;
}

} // namespace

WriteSummary writeObservedSupportClaims(const std::vector<ObservedGraphSupport>& observations)
{
    WriteSummary summary;

    // A sidecar nothing observed is never a key here, which is the RFC §9.2
    // empty-write guard: an absent observation set cannot reach the file, let
    // alone null a claim in it. Each target is independent -- a refusal below
    // skips one file and leaves the rest of the run to finish.
    for(const auto& [sidecarPath, sidecarObservations] : groupBySidecarPath(observations))
    {
        if(const std::string defect = sidecarDefect(sidecarObservations); !defect.empty())
        {
            summary.errors.push_back(defect);
            ++summary.filesSkipped;
            continue;
        }

        const bool isSweep = sidecarObservations.front().claimLocator.isSweep();

        FlatClaims claims;
        try
        {
            claims = readFlatClaims(sidecarPath, isSweep);
        }
        catch(const std::exception& e)
        {
            summary.errors.push_back("refusing to overwrite unparseable sidecar: "
                                     + sidecarPath.string() + ": " + e.what());
            ++summary.filesSkipped;
            continue;
        }

        for(const auto& observation : sidecarObservations)
        {
            applyObservation(claims, observation);
        }

        // Every engine declined and there is no file to correct: writing one
        // would check in a sidecar that claims nothing.
        if(claims.empty() && !std::filesystem::exists(sidecarPath))
        {
            ++summary.filesSkipped;
            continue;
        }

        switch(writeIfChanged(sidecarPath, serializeFlatClaims(claims, isSweep)))
        {
        case WriteOutcome::Written:
            ++summary.filesWritten;
            break;
        case WriteOutcome::Unchanged:
            ++summary.filesUnchanged;
            break;
        case WriteOutcome::OpenFailed:
            summary.errors.push_back("could not open for writing: " + sidecarPath.string());
            ++summary.filesSkipped;
            break;
        case WriteOutcome::WriteFailed:
            summary.errors.push_back("write failed: " + sidecarPath.string());
            ++summary.filesSkipped;
            break;
        }
    }

    return summary;
}

} // namespace hipdnn_integration_tests::bundle
