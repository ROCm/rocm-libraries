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
    // under a generic per-arch content container next to the loaded engine
    // plugin at
    //   <plugin_dir>/arch_content/rocke/<arch>/
    // holding rocke_client_<arch>.kpack, its rocke_client_<arch>.json sidecar,
    // and rocke's per-arch kernel-selection heuristics. loadDefault() resolves
    // <plugin_dir> from the loaded plugin and appends the device <arch>. The
    // "arch_content" container is generic (other engines get sibling subdirs,
    // e.g. arch_content/aiter/, arch_content/asm/) and, unlike a
    // "hip_kernel_provider/" directory, does not collide with the
    // hip_kernel_provider(.dll/.so) plugin file in the same engines dir.
    return "arch_content/rocke";
}

} // namespace rocke_client::dispatcher
