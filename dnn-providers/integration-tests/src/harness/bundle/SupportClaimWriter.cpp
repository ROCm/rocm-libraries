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

using EngineName = ObservedGraphSupport::EngineName;
using CaseId = ObservedGraphSupport::CaseId;

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

// Both sidecar shapes are the same claim set underneath: a single-graph bundle
// is the one-case sweep. Flattening to this on load, and only re-imposing the
// on-disk shape on serialize, keeps the whole middle of the writer shape-agnostic
// -- one overlay, not one per format (RFC §9.2: flatten → overlay → regroup).
class ClaimSet
{
public:
    static ClaimSet load(const std::filesystem::path& sidecarPath, bool isSweep)
    {
        ClaimSet result;
        if(!std::filesystem::exists(sidecarPath))
        {
            return result;
        }

        if(isSweep)
        {
            for(const auto& [engine, groups] : loadSweepSupportClaimsFromPath(sidecarPath).claims)
            {
                for(const auto& group : groups)
                {
                    if(group.support.empty())
                    {
                        continue;
                    }
                    for(const auto& caseId : group.cases)
                    {
                        result._cells[engine][caseId] = group.support;
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
                    result._cells[engine][CaseId{}] = support;
                }
            }
        }

        return result;
    }

    void apply(const ObservedGraphSupport& observation)
    {
        const CaseId& caseId = observation.claimLocator.caseId;

        if(observation.engineIsSupported)
        {
            _cells[observation.engineName][caseId][observation.arch].insert(observation.platform);
            return;
        }

        auto engineIt = _cells.find(observation.engineName);
        if(engineIt == _cells.end())
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
            _cells.erase(engineIt);
        }
    }

    std::string serialize(bool isSweep) const
    {
        if(!isSweep)
        {
            SupportClaims out;
            out.version = K_SUPPORT_CLAIMS_VERSION;
            for(const auto& [engine, byCase] : _cells)
            {
                const auto caseIt = byCase.find(CaseId{});
                if(caseIt != byCase.end())
                {
                    out.claims[engine] = caseIt->second;
                }
            }
            return dumpCanonical(toJson(out));
        }

        SweepSupportClaims out;
        out.version = K_SUPPORT_CLAIMS_VERSION;
        for(const auto& [engine, byCase] : _cells)
        {
            std::map<ArchPlatformMap, std::vector<CaseId>> casesByFootprint;
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

            std::sort(groups.begin(),
                      groups.end(),
                      [](const SweepClaimGroup& a, const SweepClaimGroup& b) {
                          return a.cases.front() < b.cases.front();
                      });

            if(!groups.empty())
            {
                out.claims[engine] = std::move(groups);
            }
        }

        return dumpCanonical(toJson(out));
    }

    bool empty() const
    {
        return _cells.empty();
    }

private:
    std::map<EngineName, std::map<CaseId, ArchPlatformMap>> _cells;
};

enum class WriteOutcome
{
    Written,
    Unchanged,
    OpenFailed,
    WriteFailed,
};

WriteOutcome writeIfChanged(const std::filesystem::path& filePath, const std::string& newContent)
{
    if(std::filesystem::exists(filePath))
    {
        std::ifstream existingFile(filePath, std::ios::binary);
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

    auto tempPath = filePath;
    tempPath += ".tmp";

    {
        std::ofstream outputFile(tempPath, std::ios::binary);
        if(!outputFile)
        {
            return WriteOutcome::OpenFailed;
        }
        outputFile << newContent;
        outputFile.close();
        if(!outputFile)
        {
            std::filesystem::remove(tempPath);
            return WriteOutcome::WriteFailed;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tempPath, filePath, ec);
    if(ec)
    {
        std::filesystem::remove(tempPath);
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

        ClaimSet claims;
        try
        {
            claims = ClaimSet::load(sidecarPath, isSweep);
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
            claims.apply(observation);
        }
        summary.observationsApplied += sidecarObservations.size();

        // Every engine declined and there is no file to correct: writing one
        // would check in a sidecar that claims nothing.
        if(claims.empty() && !std::filesystem::exists(sidecarPath))
        {
            ++summary.filesSkipped;
            continue;
        }

        switch(writeIfChanged(sidecarPath, claims.serialize(isSweep)))
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
        default:
            break;
        }
    }

    return summary;
}

} // namespace hipdnn_integration_tests::bundle
