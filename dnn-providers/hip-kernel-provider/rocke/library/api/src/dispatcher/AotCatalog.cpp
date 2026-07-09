// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/AotCatalog.hpp"

#include <utility>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace rocke_client::dispatcher
{

AotCatalog::AotCatalog(std::vector<AotInstance> instances)
    : _instances(std::move(instances))
{
}

AotCatalog AotCatalog::loadDefault()
{
    // TODO(kpack): populate from the installed AOT catalog once PR #8866's kpack
    // packaging + install rules exist. See AotCatalog.hpp for the required steps.
    // Until then there are no runtime instances, so the engine declines every
    // graph. This is the intended Phase-1 no-op.
    HIPDNN_PLUGIN_LOG_INFO(
        "rocke-client dispatcher: no AOT catalog available yet (kpack not landed); "
        "engine will decline all graphs");
    return AotCatalog{};
}

std::vector<std::reference_wrapper<const AotInstance>>
    AotCatalog::candidatesFor(const std::string& op, const std::string& arch) const
{
    std::vector<std::reference_wrapper<const AotInstance>> candidates;
    for(const auto& instance : _instances)
    {
        if(instance.op == op && instance.arch == arch)
        {
            candidates.emplace_back(instance);
        }
    }
    return candidates;
}

const char* defaultArtifactRoot()
{
    // Plugin-relative root for the installed rocKE AOT bundles. They install
    // next to the loaded engine plugin at
    //   <plugin_dir>/hip_kernel_provider/rocke/<arch>/
    // holding rocke_client_<arch>.kpack, its rocke_client_<arch>.json sidecar,
    // and rocke's per-arch kernel-selection heuristics. loadDefault() resolves
    // <plugin_dir> from the loaded plugin and appends the device <arch>. Nothing
    // reads it yet (Phase-1 catalog is empty); the dispatcher-wiring work does.
    return "hip_kernel_provider/rocke";
}

} // namespace rocke_client::dispatcher
