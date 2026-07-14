// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace rocke_client::dispatcher
{

// Return the directory from which loadForDevice() reads the per-arch bundle.
//
// Production path: <pluginDir>/arch_content/rocke/<arch>/
//   where pluginDir is the directory that contains the loaded rocke-client DSO.
//
// TEST-ONLY override: if the environment variable ROCKE_CLIENT_AOT_BUNDLE_DIR
//   is non-empty, return that directory directly (arch is NOT appended — the
//   caller controls per-arch layout via the env value). This lets integration
//   tests stage valid/corrupt bundles beside the test binary and point to them
//   without touching the production arch_content tree.
//
// TODO(AICK-1484): replace the env override with a content root injected into
//   the dispatcher at construction, once real plan-based selection/execution
//   lands, so tests don't need a process-wide env override.
inline std::filesystem::path aotBundleDir(const std::filesystem::path& pluginDir,
                                          const std::string& arch)
{
    const std::string overrideDir
        = hipdnn_data_sdk::utilities::getEnv("ROCKE_CLIENT_AOT_BUNDLE_DIR");
    if(!overrideDir.empty())
    {
        return {overrideDir};
    }
    return pluginDir / "arch_content" / "rocke" / arch;
}

inline std::filesystem::path aotKpackPath(const std::filesystem::path& pluginDir,
                                          const std::string& arch)
{
    return aotBundleDir(pluginDir, arch) / ("rocke_client_" + arch + ".kpack");
}

inline std::filesystem::path aotManifestPath(const std::filesystem::path& pluginDir,
                                             const std::string& arch)
{
    return aotBundleDir(pluginDir, arch) / ("rocke_client_" + arch + ".json");
}

} // namespace rocke_client::dispatcher
