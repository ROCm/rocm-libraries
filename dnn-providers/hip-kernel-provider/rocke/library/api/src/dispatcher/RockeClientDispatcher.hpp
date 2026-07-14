// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <mutex>
#include <optional>

#include "dispatcher/AotCatalog.hpp"
#include "dispatcher/AotInstance.hpp"
#include "dispatcher/SdpaProblem.hpp"

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{
class IGraph;
}

namespace rocke_client
{
struct RockeClientHandle;
}

namespace rocke_client::dispatcher
{

// Owns the per-device AOT catalog map and performs kernel selection:
// graph -> normalized SdpaProblem -> filter catalog candidates -> winner.
//
// PRODUCTION (default ctor): starts with an empty catalog map. The first
// call to selectInstance/selectForArch for a given (deviceId, arch) pair
// triggers a lazy call to AotCatalog::loadForDevice, which reads the
// installed bundle from the plugin directory (or from the TEST-ONLY env
// override ROCKE_CLIENT_AOT_BUNDLE_DIR). Subsequent calls return the
// cached result. All selection methods are noexcept.
//
// TEST INJECTION (AotCatalog ctor): an injected catalog is returned for
// every (deviceId, arch) query without any HIP or filesystem calls. This
// lets TestRockeClientDispatcher and TestRockeClientEngine exercise the
// full graph->problem->select path without a GPU.
class RockeClientDispatcher
{
public:
    // Production: empty catalog map, lazy per-device load.
    RockeClientDispatcher();

    // Test injection: seeds a fixed catalog returned for all device queries.
    explicit RockeClientDispatcher(AotCatalog catalog);

    // Pure selection over an already-normalized problem (no HIP).
    // Uses the catalog for device 0 keyed on problem.arch.
    // Returns the first instance satisfying all constraints (stable order).
    std::optional<AotInstance> select(const SdpaProblem& problem) const;

    // Full path used by the engine: translate graph, detect device arch from
    // handle's stream, retrieve the per-device catalog lazily, then select.
    // Never throws.
    std::optional<AotInstance> selectInstance(
        const RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

    // Test/explicit-arch seam: run translate + select using device 0's catalog
    // for the given arch, bypassing HIP device detection.
    // Lets the full graph -> problem -> select path be exercised without a GPU.
    // Never throws.
    std::optional<AotInstance> selectForArch(
        const std::string& arch,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

    // Graph accept/reject: whether any AOT instance can serve this graph.
    bool isApplicable(
        const RockeClientHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) const noexcept;

private:
    // Get or lazily load the catalog for (deviceId, arch). Thread-safe via
    // _catalogMutex. Once loaded (or found absent), the result is cached in
    // _catalogsByDevice for the lifetime of this dispatcher.
    const AotCatalog& catalogForDevice(int deviceId, const std::string& arch) const;

    // Core selection over a provided catalog snapshot (no HIP, no locks).
    static std::optional<AotInstance> selectFromCatalog(const AotCatalog& catalog,
                                                        const SdpaProblem& problem);

    mutable std::map<int, AotCatalog> _catalogsByDevice;
    mutable std::mutex _catalogMutex;

    // Non-empty only when constructed via the injection ctor. When set,
    // catalogForDevice() returns this catalog for every (deviceId, arch) query
    // without calling loadForDevice or touching the filesystem/HIP.
    std::optional<AotCatalog> _injectedCatalog;
};

} // namespace rocke_client::dispatcher
