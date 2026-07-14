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

// The set of AOT-built kernel instances the dispatcher can select from.
//
// An AotCatalog is constructed either as empty (default) or from explicit
// instances (test injection). Production catalogs are populated lazily per
// device by AotCatalog::loadForDevice(), called from
// RockeClientDispatcher::catalogForDevice() on the first selection attempt for
// that device. Until the kpack packaging + manifest land in an installed bundle,
// loadForDevice() returns an empty catalog and the engine declines every graph.
class AotCatalog
{
public:
    AotCatalog() = default;
    explicit AotCatalog(std::vector<AotInstance> instances);

    // Resolve and load the AOT catalog for the given HIP device and arch.
    //
    // Called from RockeClientDispatcher::catalogForDevice() which is reachable
    // from the noexcept selectInstance path. Therefore this function MUST NOT
    // THROW: all errors are logged (using the AOT_SKELETON_LOAD_FAILED/OK
    // for test observability) and result in an empty catalog being returned.
    // "Fail-loud" means ERROR log, not exception.
    //
    // hipSetDevice(deviceId) is called before hipModuleLoadData to ensure the
    // code object is loaded on the correct device; the prior active device is
    // saved and restored.
    //
    // TODO(kpack-fastfollow): parse real instances (compile_spec /
    // selection.batch / attribute_constraints) into the returned catalog,
    // defer hipModuleLoad to plan-construction, and wire selection + execution
    // + KERNEL LAUNCH + RESULT VALIDATION. Today we load+unload one kernel
    // purely to prove the kpack->hipModuleLoad->GetFunction path and return an
    // empty catalog so the engine stays inert (no graphs selected/executed).
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
