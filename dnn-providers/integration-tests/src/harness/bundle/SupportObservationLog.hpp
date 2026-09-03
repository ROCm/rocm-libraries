// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

// What the machine said about one cell, before any sidecar is consulted
// (RFC 0015 §5.2). Three values, because "the query broke" is not a third
// shade of no: DECLINED is a fact about the engine, UNKNOWN is a fact about
// the run. Collapsing them to a bool is what would let a driver fault erase a
// claim, so the distinction is carried in the type rather than in a comment.
enum class ObservedSupport
{
    SUPPORTED, ///< query resolved, engine in the ranked list
    DECLINED, ///< query resolved, engine absent
    UNKNOWN, ///< query did not resolve; evidence of nothing
};

inline const char* toString(ObservedSupport support)
{
    switch(support)
    {
    case ObservedSupport::SUPPORTED:
        return "supported";
    case ObservedSupport::DECLINED:
        return "declined";
    case ObservedSupport::UNKNOWN:
        return "unknown";
    default:
        return "<invalid-observed-support>";
    }
}

// One (graph, engine, arch, platform) cell of observed support, as seen on the
// hardware the run happened on.
//
// Not SupportVerdict.hpp's SupportObservation, which is the enforcing side: what a
// sidecar promised and whether we got to read it. This one is the authoring side --
// what the engines actually took, with no sidecar involved.
struct ObservedSupportCell
{
    // Says which sidecar this cell belongs in and, for a sweep, which case
    // within it. Carried rather than re-derived: the writer must agree with the
    // enforcer about the target file, and this is the value the enforcer used.
    SupportClaimLocator claimLocator;
    std::string engineName;
    std::string arch;
    std::string platform;
    ObservedSupport support = ObservedSupport::UNKNOWN;
    // The rung this bundle is checked at. Not used to decide the observation —
    // it is carried so the harvest emitter can stamp it on the record without
    // needing the bundle back, long after the test object is gone.
    EnforcementLevel enforcementLevel = EnforcementLevel::FULL;

    bool engineIsSupported() const
    {
        return support == ObservedSupport::SUPPORTED;
    }
    bool isResolved() const
    {
        return support != ObservedSupport::UNKNOWN;
    }
};

// Process-wide log of support observations. Internally a keyed map with upsert
// semantics (SUPPORTED > DECLINED > UNKNOWN), so duplicate records for the same
// cell are collapsed in-place rather than accumulated. Populated during
// --write-support-claims and --emit-support-observations runs and drained once
// after RUN_ALL_TESTS().
class SupportObservationLog
{
public:
    static SupportObservationLog& get()
    {
        static SupportObservationLog s_instance;
        return s_instance;
    }

    SupportObservationLog(const SupportObservationLog&) = delete;
    SupportObservationLog& operator=(const SupportObservationLog&) = delete;
    SupportObservationLog(SupportObservationLog&&) = delete;
    SupportObservationLog& operator=(SupportObservationLog&&) = delete;

    // Upsert: higher verdict wins (SUPPORTED > DECLINED > UNKNOWN).
    void record(const ObservedSupportCell& observation)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        CellKey key{observation.claimLocator.sidecarPath,
                    observation.claimLocator.caseId,
                    observation.engineName,
                    observation.arch,
                    observation.platform};
        CellValue val{observation.support,
                      observation.enforcementLevel,
                      observation.claimLocator.diagnosticPath};

        auto it = _cells.find(key);
        if(it == _cells.end() || verdictRank(val.support) > verdictRank(it->second.support))
        {
            _cells.insert_or_assign(std::move(key), std::move(val));
        }
    }

    // Bridge: reconstruct the flat vector that SupportClaimWriter consumes.
    std::vector<ObservedSupportCell> all() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        std::vector<ObservedSupportCell> result;
        result.reserve(_cells.size());
        for(const auto& [key, val] : _cells)
        {
            result.push_back(ObservedSupportCell{
                SupportClaimLocator{key.sidecarPath, key.caseId, val.diagnosticPath},
                key.engineName,
                key.arch,
                key.platform,
                val.support,
                val.enforcementLevel});
        }
        return result;
    }

    bool empty() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _cells.empty();
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _cells.clear();
    }

    // Build one snapshot JSON per unique (arch, platform) target in the log.
    // Multi-GPU runs produce multiple snapshots; single-GPU runs (the common
    // case) produce exactly one.  Returns an empty vector when the log is
    // empty, so callers need not check empty() first (avoids TOCTOU).
    // `bundleRoot` turns absolute sidecar paths into relative bundle keys
    // (e.g. "quick/Conv/Default").
    std::vector<nlohmann::json> toSnapshotJsons(const std::filesystem::path& bundleRoot) const
    {
        // Snapshot the cells under the lock, then release before doing
        // filesystem work (std::filesystem::relative is a syscall).
        std::map<CellKey, CellValue> snapshot;
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            snapshot = _cells;
        }

        // Group cells by target — preserves map ordering within each group.
        std::map<std::pair<std::string, std::string>, nlohmann::json> targetObs;
        for(const auto& [key, val] : snapshot)
        {
            std::error_code ec;
            const auto relative
                = std::filesystem::relative(key.sidecarPath.parent_path(), bundleRoot, ec);
            const std::string bundle = (ec || relative.empty() || *relative.begin() == "..")
                                           ? key.sidecarPath.parent_path().generic_string()
                                           : relative.generic_string();

            // Sweep sidecars are "support.json"; single-graph sidecars are
            // "<Graph>.support.json". Strip the double extension to recover the
            // graph stem the Python harvester needs for multi-graph disambiguation.
            const auto sidecarStem = key.sidecarPath.stem();
            const auto graphStem = std::filesystem::path(sidecarStem).stem();
            const std::string graphStr = graphStem.generic_string();

            nlohmann::json obs;
            obs["bundle"] = bundle;
            obs["case_id"]
                = key.caseId.empty() ? nlohmann::json(nullptr) : nlohmann::json(key.caseId);
            obs["engine"] = key.engineName;
            obs["graph"]
                = (graphStr == "support") ? nlohmann::json(nullptr) : nlohmann::json(graphStr);
            obs["verdict"] = toString(val.support);
            obs["enforcement_level"] = toString(val.enforcementLevel);

            targetObs[{key.arch, key.platform}].push_back(std::move(obs));
        }

        std::vector<nlohmann::json> snapshots;
        snapshots.reserve(targetObs.size());
        for(auto& [target, observations] : targetObs)
        {
            nlohmann::json snap;
            snap["schema_version"] = 1;
            snap["target"] = {{"arch", target.first}, {"platform", target.second}};
            snap["observations"] = std::move(observations);
            snapshots.push_back(std::move(snap));
        }
        return snapshots;
    }

private:
    SupportObservationLog() = default;

    struct CellKey
    {
        std::filesystem::path sidecarPath;
        std::string caseId;
        std::string engineName;
        std::string arch;
        std::string platform;

        bool operator<(const CellKey& rhs) const
        {
            return std::tie(sidecarPath, caseId, engineName, arch, platform)
                   < std::tie(rhs.sidecarPath, rhs.caseId, rhs.engineName, rhs.arch, rhs.platform);
        }
    };

    struct CellValue
    {
        ObservedSupport support = ObservedSupport::UNKNOWN;
        EnforcementLevel enforcementLevel = EnforcementLevel::FULL;
        std::string diagnosticPath;
    };

    static int verdictRank(ObservedSupport s)
    {
        switch(s)
        {
        case ObservedSupport::SUPPORTED:
            return 2;
        case ObservedSupport::DECLINED:
            return 1;
        default:
            return 0;
        }
    }

    mutable std::mutex _mutex;
    std::map<CellKey, CellValue> _cells;
};

} // namespace hipdnn_integration_tests::bundle
