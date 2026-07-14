// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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
// device. Until kpack packaging + manifest ship in an installed bundle,
// loadForDevice() returns an empty catalog and the engine declines every graph.
//
// TEMPORARY PROBE: today this class exists to *prove* the AOT load path
// end-to-end, not to run kernels. loadForDevice() does a real
// kpack -> hipModuleLoadData -> hipModuleGetFunction, logs a marker, unloads,
// and returns an empty catalog; the integration tests assert on those markers,
// not on real results.
//
// TODO(AICK-1484): replace this probe with real plans. AotCatalog becomes
// a plain `Catalog` -- a simple collection of the candidate instances (plus the
// metadata needed to build a plan) that merely *presents options*. The
// dispatcher picks the winner and, based on its kind (AOT today; JIT later),
// hands off to plan construction; the kpack extraction + module load move there,
// after selection. Catalogs are scoped per (op, arch), NOT one global catalog
// per arch -- e.g. separate catalogs for SDPA@gfx950, Conv@gfx950, and
// SDPA@gfx942, since kernel parameters need not overlap across op or arch.
// Instances become concrete op+arch-specific types deriving from a common base.
// The log-marker integration tests are then replaced by real E2E tests
// (graph submit + kernel launch + result validation).
class AotCatalog
{
public:
    AotCatalog() = default;
    explicit AotCatalog(std::vector<AotInstance> instances);

    // Resolve and load the AOT catalog for the given HIP device and arch.
    //
    // Reachable from the noexcept selectInstance path (via catalogForDevice), so
    // it MUST NOT THROW: every failure is a loud ERROR log (the
    // AOT_PROBE_LOAD_* markers, for test observability) and yields an empty
    // catalog. "Fail-loud" means ERROR log, not exception.
    //
    // TODO(AICK-1484): drop the `deviceId` parameter and the
    // hipSetDevice + hipModuleLoadData + unload it guards -- all throwaway. The
    // catalog is conceptually a function of ARCH ONLY; hipSetDevice exists only
    // so the probe load runs on the right device. Under AICK-1484 the real
    // load moves to plan construction and the catalog is built from arch alone
    // (see class doc). Today we load+unload one kernel purely to prove the path.
    static AotCatalog loadForDevice(int deviceId, const std::string& arch);

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

} // namespace rocke_client::dispatcher
