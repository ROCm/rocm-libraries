// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/AotCatalog.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <nlohmann/json.hpp>
#include <rocm_kpack/kpack.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "dispatcher/AotBundlePaths.hpp"
#include "dispatcher/AotSkeletonMarkers.hpp"
#include "dispatcher/KpackModuleLoader.hpp"
#include "dispatcher/PluginModuleDir.hpp"

namespace rocke_client::dispatcher
{

AotCatalog::AotCatalog(std::vector<AotInstance> instances)
    : _instances(std::move(instances))
{
}

AotCatalog AotCatalog::loadForDevice(int deviceId, const std::string& arch)
{
    // No-throw contract: all errors produce an ERROR log and return empty.
    // Do NOT throw — this may be called from the noexcept selectInstance path.
    try
    {
        // --- Step 1: resolve the plugin's directory ---
        const std::filesystem::path pluginDir = currentPluginDirectory();

        // --- Step 2: locate the per-arch bundle ---
        const std::filesystem::path kpackPath = aotKpackPath(pluginDir, arch);
        const std::filesystem::path manifestPath = aotManifestPath(pluginDir, arch);

        if(!std::filesystem::exists(kpackPath))
        {
            HIPDNN_PLUGIN_LOG_INFO("rocke-client: no AOT bundle for '"
                                   << arch << "' at " << kpackPath.string()
                                   << "; engine will decline all graphs");
            return AotCatalog{};
        }

        // --- Step 3: bundle EXISTS — fail loudly on any error (ERROR log) ---
        HIPDNN_PLUGIN_LOG_INFO("rocke-client: loading AOT bundle for '" << arch << "' from "
                                                                        << kpackPath.string());

        // Parse bundle manifest: entries[0].toc_key + symbol
        std::ifstream manifestFile{manifestPath.string()};
        if(!manifestFile.is_open())
        {
            HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=manifest_open arch=" << arch
                                                             << " path=" << manifestPath.string());
            return AotCatalog{};
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
            return AotCatalog{};
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
            HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=manifest_fields arch="
                                                             << arch << " error=" << ex.what());
            return AotCatalog{};
        }

        // --- Step 4: set device, load HSACO, verify, unload ---
        // TODO(kpack-fastfollow): TEMPORARY smoke-test wiring. This whole
        // set-device + hipModuleLoadData + unload block exists only to prove the
        // full kpack -> hipModuleLoadData -> hipModuleGetFunction path works
        // right now; nothing is launched or retained. Remove once hipModuleLoad
        // is deferred to plan construction (see loadForDevice() header) and the
        // deviceId parameter is dropped (module/device binding leaves AotCatalog).
        // Save and restore the previously active HIP device.
        int prevDevice = 0;
        const bool deviceSaved = (hipGetDevice(&prevDevice) == hipSuccess);
        if(hipSetDevice(deviceId) != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=set_device arch=" << arch
                                                             << " deviceId=" << deviceId);
            return AotCatalog{};
        }

        const KpackLoadResult loaded
            = loadKernelFromKpack(kpackPath.string(), tocKey, arch, symbol);

        if(deviceSaved)
        {
            static_cast<void>(hipSetDevice(prevDevice));
        }

        if(loaded.kpackError != KPACK_SUCCESS)
        {
            HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED
                                    << " step=kpack_extract arch=" << arch
                                    << " kpack_error=" << static_cast<int>(loaded.kpackError)
                                    << " toc_key=" << tocKey << " kpack=" << kpackPath.string());
            return AotCatalog{};
        }

        if(loaded.hipError != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED
                                    << " step=hip_module_load arch=" << arch << " hip_error="
                                    << static_cast<int>(loaded.hipError) << " symbol=" << symbol);
            return AotCatalog{};
        }

        // Unload immediately: this is a skeleton probe, not production use.
        static_cast<void>(hipModuleUnload(loaded.module));

        // Stable, greppable marker for integration-test log capture.
        HIPDNN_PLUGIN_LOG_INFO(AOT_SKELETON_LOAD_OK << " arch=" << arch << " symbol=" << symbol
                                                    << " kpack=" << kpackPath.string());

        // TODO(kpack-fastfollow): SKELETON ONLY — parse real instances
        // (compile_spec / selection.batch / attribute_constraints) into the
        // catalog, defer hipModuleLoad to plan-construction, and wire selection
        // + execution + KERNEL LAUNCH + RESULT VALIDATION.
        // Today we load+unload one kernel purely to prove the wiring.
        return AotCatalog{};
    }
    catch(const std::exception& ex)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=exception arch=" << arch
                                                         << " error=" << ex.what());
        return AotCatalog{};
    }
    catch(...)
    {
        HIPDNN_PLUGIN_LOG_ERROR(AOT_SKELETON_LOAD_FAILED << " step=unknown_exception arch="
                                                         << arch);
        return AotCatalog{};
    }
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

} // namespace rocke_client::dispatcher
