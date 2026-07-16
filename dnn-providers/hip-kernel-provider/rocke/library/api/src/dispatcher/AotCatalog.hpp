// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "dispatcher/AotInstance.hpp"

namespace rocke_client::dispatcher
{

// AotCatalog: the set of AOT-built kernel instances the dispatcher selects from.
//
// Constructed empty (default) or from explicit instances (test injection).
// Production catalogs are populated lazily per device by loadForDevice(), called
// from RockeClientDispatcher::catalogForDevice() on the first selection for that
// device: it resolves the per-arch bundle installed beside the plugin, parses
// its manifest into AotInstances, and returns them for selection. The kpack
// HSACO is NOT loaded here -- the winning instance's module load is deferred to
// plan construction (RockeClientPlan), so an unselected bundle costs nothing but
// a manifest parse. A missing bundle yields an empty catalog and the engine
// declines every graph for that arch.
//
// The selection logic (candidatesFor) is independent of the on-disk format and
// is exercised in unit tests via catalogs constructed from fixture instances.
class AotCatalog
{
public:
    AotCatalog() = default;
    explicit AotCatalog(std::vector<AotInstance> instances);

    // Resolve and parse the AOT catalog for the given HIP device arch.
    //
    // Reachable from the noexcept selectInstance path (via catalogForDevice), so
    // it MUST NOT THROW: every failure is a WARN/ERROR log and yields an empty
    // catalog. It resolves the plugin directory, locates the per-arch bundle
    // manifest (aotManifestPath), and parses its entries into AotInstances.
    //
    // The catalog is a function of ARCH ONLY: the kpack module load moves to
    // plan construction (which runs on the stream's device), so no device id is
    // needed here.
    static AotCatalog loadForDevice(const std::string& arch);

    // Instances whose op and arch match, in stable (insertion) order.
    std::vector<std::reference_wrapper<const AotInstance>>
        candidatesFor(const std::string& op, const std::string& arch) const;

    bool empty() const
    {
        return _instances.empty();
    }

    std::size_t size() const
    {
        return _instances.size();
    }

private:
    std::vector<AotInstance> _instances;
};

// Parse every per-arch rocke_client_<arch>.json bundle manifest found under
// `root` (one per-arch subdirectory) into AotInstances. A malformed or
// unreadable bundle is logged and skipped rather than discarding the whole set;
// a missing directory yields an empty vector. Exposed for CPU unit testing of
// the manifest parser without a loaded plugin.
std::vector<AotInstance> loadManifestsFromDirectory(const std::filesystem::path& root);

} // namespace rocke_client::dispatcher
