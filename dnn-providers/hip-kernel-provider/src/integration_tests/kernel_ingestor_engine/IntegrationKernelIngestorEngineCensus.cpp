// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
// Defines the backend the frontend's logging calls (ScopedPluginLogCapture).
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <nlohmann/json.hpp>

#include "ScopedPluginLogCapture.hpp"
#include "TestDescriptorRoot.hpp"

/**
 * @file IntegrationKernelIngestorEngineCensus.cpp
 * @brief Load-time engine census: every hip-kernel-provider engine the device's shard of the
 *        production descriptor tree names must be one the loaded plugin reports.
 *
 * The loader drops a whole engine whose descriptor set fails validation and only logs why:
 *   - a kernel id declared twice (a stale pack left by a reinstall);
 *   - a native symbol this build does not register;
 *   - a pack naming a matcher or dispatch descriptor that is absent.
 * Graph tests on the engines that load cannot see an engine that never loaded, so this
 * census compares the shard's engine descriptors (`*.ued.json`) with the engine ids the
 * plugin reports. It fails with each missing engine's name and the reason: the loader's log
 * line when one is captured, else what a structural check of the shard finds.
 */
namespace
{

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR

constexpr std::string_view PROVIDER_ENGINE_PREFIX = "hipkernel:";

std::filesystem::path pluginPath()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    return std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
        / pluginFile);
}

/// The base arch of the current device (features dropped), empty without one.
std::string deviceArch()
{
    int device = 0;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&device) != hipSuccess
       || hipGetDeviceProperties(&properties, device) != hipSuccess)
    {
        return {};
    }
    const std::string name = properties.gcnArchName;
    return name.substr(0, name.find(':'));
}

/// The engine names of every `*.ued.json` under @p shard, with the file that names each.
std::map<std::string, std::filesystem::path> enginesNamedIn(const std::filesystem::path& shard)
{
    std::map<std::string, std::filesystem::path> engines;
    std::error_code error;
    for(auto it = std::filesystem::recursive_directory_iterator(shard, error);
        !error && it != std::filesystem::recursive_directory_iterator();
        it.increment(error))
    {
        const auto file = it->path().filename().string();
        if(file.size() < 9 || file.compare(file.size() - 9, 9, ".ued.json") != 0)
        {
            continue;
        }
        std::ifstream stream(it->path());
        const auto document = nlohmann::json::parse(stream, nullptr, false);
        if(document.is_object() && document.contains("name") && document["name"].is_string())
        {
            engines.emplace(document["name"].get<std::string>(), it->path());
        }
    }
    return engines;
}

/// What a structural check of @p shard finds wrong with engine @p name: a set that does not
/// resolve, or kernel ids its packs declare twice. Native symbols are the plugin's own, so a
/// set that passes here and still did not load points at them.
std::string structuralReason(const std::filesystem::path& shard, const std::string& name)
{
    try
    {
        const auto sets = hipdnn_plugin_sdk::ingestor::resolveDescriptorSets(
            hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(shard));
        const auto set = std::find_if(sets.begin(), sets.end(), [&](const auto& candidate) {
            return candidate.engine.name == name;
        });
        if(set == sets.end())
        {
            return "its descriptor set does not resolve (a pack, matcher or dispatch it names is "
                   "missing or invalid)";
        }
        std::map<std::string, std::string> firstPack;
        for(const auto& pack : set->packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const auto id = hipdnn_plugin_sdk::ingestor::toString(kernel.id);
                const auto [it, inserted] = firstPack.emplace(id, pack.name);
                if(!inserted)
                {
                    return "kernel " + id + " is declared in packs '" + it->second + "' and '"
                           + pack.name + "' (a stale pack left in the tree?)";
                }
            }
        }
        return "its descriptors are well formed, so a native symbol it names is likely "
               "unregistered in this build";
    }
    catch(const std::exception& error)
    {
        return std::string("the shard does not load: ") + error.what();
    }
}

TEST(IntegrationKernelIngestorEngineCensus, EveryEngineOfTheDeviceShardLoads)
{
    const auto arch = deviceArch();
    if(arch.empty())
    {
        GTEST_SKIP() << "no device";
    }
    const auto root = hip_kernel_provider::testing::descriptorSetRoot(
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
    const auto shard = root / arch;
    std::error_code absent;
    if(root.empty() || !std::filesystem::is_directory(shard, absent))
    {
        GTEST_SKIP() << "no production descriptor shard for " << arch << " under " << root;
    }
    std::map<std::string, std::filesystem::path> expected;
    for(const auto& [name, file] : enginesNamedIn(shard))
    {
        if(name.rfind(PROVIDER_ENGINE_PREFIX, 0) == 0)
        {
            expected.emplace(name, file);
        }
    }
    ASSERT_FALSE(expected.empty()) << "the " << arch << " shard names no engine";

    // The plugin reports its engines when asked; the loader logs each drop with its reason.
    int captureTag = 0;
    const hip_kernel_provider::kernel_ingestor_engine::integration::ScopedPluginLogCapture capture(
        &captureTag);
    auto* library = hipdnn_data_sdk::utilities::openLibrary(pluginPath());
    ASSERT_NE(library, nullptr) << pluginPath();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* getAllEngineIds = reinterpret_cast<decltype(&hipdnnEnginePluginGetAllEngineIds)>(
        hipdnn_data_sdk::utilities::getSymbol(library, "hipdnnEnginePluginGetAllEngineIds"));
    ASSERT_NE(getAllEngineIds, nullptr);
    uint32_t count = 0;
    ASSERT_EQ(getAllEngineIds(nullptr, 0, &count), HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> ids(count);
    ASSERT_EQ(getAllEngineIds(ids.data(), count, &count), HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::set<int64_t> loaded(ids.begin(), ids.end());
    const auto logs = capture.recorder().getRecordedLogs();
    hipdnn_data_sdk::utilities::closeLibrary(library);

    std::ostringstream missing;
    int dropped = 0;
    for(const auto& [name, file] : expected)
    {
        if(loaded.count(hipdnn_data_sdk::utilities::engineNameToId(name)) != 0)
        {
            continue;
        }
        ++dropped;
        std::string reason;
        for(const auto& log : logs)
        {
            if(log.message.find("'" + name + "'") != std::string::npos
               && log.message.find("dropping") != std::string::npos)
            {
                reason = log.message;
                break;
            }
        }
        if(reason.empty())
        {
            reason = structuralReason(shard, name);
        }
        missing << "\n  " << name << " (" << file.string() << "): " << reason;
    }
    EXPECT_EQ(dropped, 0) << dropped << " of " << expected.size() << " engines the " << arch
                          << " shard names did not load:" << missing.str();
}

#endif // HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR

} // namespace
