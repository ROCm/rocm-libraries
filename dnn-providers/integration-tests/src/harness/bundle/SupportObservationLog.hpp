// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

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
    bool engineIsSupported = false; // resolved + in ranked list
};

// Process-wide log of resolved support observations. Populated during
// --write-support-claims runs and drained once after RUN_ALL_TESTS() to
// produce .support.json sidecars. Only resolved queries (OK or
// GRAPH_NOT_SUPPORTED) are recorded; unresolved queries are not
// observations of "unsupported" and must never null an existing claim.
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
