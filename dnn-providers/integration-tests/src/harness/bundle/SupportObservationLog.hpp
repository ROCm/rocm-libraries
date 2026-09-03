// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

// One resolved answer to "did this engine take this graph", for the arch and
// platform the run happened on.
//
// Not SupportVerdict.hpp's SupportObservation, which is the enforcing side: what a
// sidecar promised and whether we got to read it. This one is the authoring side --
// what the engines actually took, with no sidecar involved.
struct ObservedGraphSupport
{
    // Says which sidecar this observation belongs in and, for a sweep, which case
    // within it. Carried rather than re-derived: the writer must agree with the
    // enforcer about the target file, and this is the value the enforcer used.
    SupportClaimLocator claimLocator;
    std::string engineName;
    std::string arch;
    std::string platform;

    // The engine was in the ranked list for this graph. False means the engine
    // resolved and declined -- it does not mean "we could not tell". The writer
    // reads false as "erase this claim", so recording an unknown as false
    // deletes a true claim; see the log's precondition below.
    bool engineIsSupported = false;
};

// Process-wide log of resolved support observations. Populated during
// --write-support-claims runs and drained once after RUN_ALL_TESTS() to
// produce .support.json sidecars.
//
// Precondition on every recorded observation: the query that produced it resolved (OK
// or GRAPH_NOT_SUPPORTED). An unresolved query is not an observation of
// "unsupported" and must never null an existing claim. The type cannot express
// this -- the guard is the early return in
// IntegrationBundleVerificationHarness::observeSupportOnly(), which is the only
// production caller of record().
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

    void record(ObservedGraphSupport observation)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _observations.push_back(std::move(observation));
    }

    std::vector<ObservedGraphSupport> all() const
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
    std::vector<ObservedGraphSupport> _observations;
};

} // namespace hipdnn_integration_tests::bundle
