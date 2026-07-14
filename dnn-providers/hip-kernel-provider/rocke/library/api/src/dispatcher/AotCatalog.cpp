// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/AotCatalog.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <nlohmann/json.hpp>
#include <rocm_kpack/kpack.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "dispatcher/AotSkeletonMarkers.hpp"
#include "dispatcher/KpackModuleLoader.hpp"
#include "dispatcher/PluginModuleDir.hpp"

namespace rocke_client::dispatcher
{

namespace
{

// Bare GFX arch string for device 0 (e.g. "gfx942"), or "" when no device is
// available. Mirrors deviceArch() in RockeClientDispatcher.cpp.
std::string deviceArch()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    std::string arch{props.gcnArchName};
    const auto colon = arch.find(':'); // strip "gfx942:sramecc+:xnack-"
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

} // namespace

AotCatalog::AotCatalog(std::vector<AotInstance> instances)
    : _instances(std::move(instances))
{
}

AotCatalog AotCatalog::loadDefault()
{
    // --- Step 1: resolve the plugin's own directory ---
    // currentPluginDirectory() uses dladdr/GetModuleHandleExA to locate this
    // DSO, so the result is always relative to the installed plugin, not to
    // the executable's working directory.
    std::filesystem::path pluginDir;
    try
    {
        pluginDir = currentPluginDirectory();
    }
    catch(const std::exception& ex)
    {
        HIPDNN_PLUGIN_LOG_INFO("rocke-client: cannot resolve plugin directory ("
                               << ex.what() << "); declining all graphs");
        return AotCatalog{};
    }

    // --- Step 2: determine the running device's GFX arch ---
    const std::string arch = deviceArch();
    if(arch.empty())
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "rocke-client: no HIP device available; engine will decline all graphs");
        return AotCatalog{};
    }

    // --- Step 3: locate the per-arch bundle ---
    // Expected layout: <plugin_dir>/arch_content/rocke/<arch>/rocke_client_<arch>.{kpack,json}
    const std::filesystem::path bundleDir = pluginDir / defaultArtifactRoot() / arch;
    const std::filesystem::path kpackPath = bundleDir / ("rocke_client_" + arch + ".kpack");
    const std::filesystem::path manifestPath = bundleDir / ("rocke_client_" + arch + ".json");

    // --- Step 4: if bundle absent, decline silently (inert — normal install without bundle) ---
    if(!std::filesystem::exists(kpackPath))
    {
        HIPDNN_PLUGIN_LOG_INFO("rocke-client: no AOT bundle for '"
                               << arch << "' at " << kpackPath.string()
                               << "; engine will decline all graphs");
        return AotCatalog{};
    }

    // --- Step 5: bundle EXISTS — FAIL LOUDLY on any error ---
    // Parse the bundle manifest for toc_key + kernel symbol.
    HIPDNN_PLUGIN_LOG_INFO("rocke-client: loading AOT bundle for '" << arch << "' from "
                                                                    << bundleDir.string());

    std::ifstream manifestFile{manifestPath.string()};
    if(!manifestFile.is_open())
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=manifest_open arch=" << arch
                                                         << " path=" << manifestPath.string());
        throw hipdnn_plugin_sdk::HipdnnPluginException{HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "rocke-client: cannot open bundle manifest: "
                                                           + manifestPath.string()};
    }

    nlohmann::json manifest;
    try
    {
        manifestFile >> manifest;
    }
    catch(const nlohmann::json::parse_error& ex)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=manifest_parse arch=" << arch
                                                         << " error=" << ex.what());
        throw hipdnn_plugin_sdk::HipdnnPluginException{HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "rocke-client: cannot parse bundle manifest "
                                                           + manifestPath.string() + ": "
                                                           + ex.what()};
    }

    std::string tocKey;
    std::string symbol;
    try
    {
        tocKey = manifest.at("entries").at(0).at("toc_key").get<std::string>();
        symbol = manifest.at("entries").at(0).at("symbol").get<std::string>();
    }
    catch(const nlohmann::json::exception& ex)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=manifest_fields arch=" << arch
                                                         << " error=" << ex.what());
        throw hipdnn_plugin_sdk::HipdnnPluginException{
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "rocke-client: manifest missing required fields (entries[0].toc_key/symbol) in "
                + manifestPath.string() + ": " + ex.what()};
    }

    // E2E load: kpack_open -> kpack_get_kernel -> hipModuleLoadData ->
    // hipModuleGetFunction. All cleanup (kpack_free_kernel, kpack_close) is
    // handled inside loadKernelFromKpack regardless of outcome.
    const KpackLoadResult loaded = loadKernelFromKpack(kpackPath.string(), tocKey, arch, symbol);

    if(loaded.kpackError != KPACK_SUCCESS)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED
                                << " step=kpack_extract arch=" << arch
                                << " kpack_error=" << static_cast<int>(loaded.kpackError)
                                << " toc_key=" << tocKey << " kpack=" << kpackPath.string());
        throw hipdnn_plugin_sdk::HipdnnPluginException{
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "rocke-client: kpack extraction failed (kpack_error_t "
                + std::to_string(static_cast<int>(loaded.kpackError)) + ") for toc_key=" + tocKey
                + " arch=" + arch + " in " + kpackPath.string()};
    }

    if(loaded.hipError != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED
                                << " step=hip_module_load arch=" << arch << " hip_error="
                                << static_cast<int>(loaded.hipError) << " symbol=" << symbol);
        throw hipdnn_plugin_sdk::HipdnnPluginException{
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "rocke-client: hipModuleLoad/GetFunction failed (hipError_t "
                + std::to_string(static_cast<int>(loaded.hipError)) + ") for symbol=" + symbol
                + " arch=" + arch};
    }

    // --- Step 6: prove complete; unload immediately (skeleton only) ---
    // loadKernelFromKpack returns valid handles on success; caller owns them.
    static_cast<void>(hipModuleUnload(loaded.module));

    // Stable, greppable marker so integration tests can assert the full
    // kpack->hipModuleLoad->GetFunction path was actually exercised.
    HIPDNN_PLUGIN_LOG_INFO(AOT_SKELETON_LOAD_OK << " arch=" << arch << " symbol=" << symbol
                                                << " kpack=" << kpackPath.string());

    // TODO(kpack-fastfollow): SKELETON ONLY — proves the AOT load path end-to-end.
    // TODO(kpack-fastfollow): parse real instances (compile_spec / selection.batch /
    // attribute_constraints) into the catalog, defer hipModuleLoad to plan-construction,
    // and wire selection + execution + KERNEL LAUNCH + RESULT VALIDATION here.
    // Today we load+unload one kernel purely to prove the wiring and do not run it.
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
