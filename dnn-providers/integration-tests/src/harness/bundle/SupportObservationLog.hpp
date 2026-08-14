// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>
#include <vector>

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
        return "unknown";
    }
}

// One (graph, engine, arch, platform) cell of observed support, as seen on the
// hardware the run happened on.
struct SupportObservation
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

// Process-wide log of support observations. Populated during
// --write-support-claims runs and drained once after RUN_ALL_TESTS().
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

    void record(SupportObservation observation)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _observations.push_back(std::move(observation));
    }

    std::vector<SupportObservation> all() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _observations;
    }

    bool empty() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _observations.empty();
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _observations.clear();
    }

private:
    SupportObservationLog() = default;

    mutable std::mutex _mutex;
    std::vector<SupportObservation> _observations;
};

} // namespace hipdnn_integration_tests::bundle
