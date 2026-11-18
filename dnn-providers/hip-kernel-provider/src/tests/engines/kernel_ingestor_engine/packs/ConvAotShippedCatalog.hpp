// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/// @file ConvAotShippedCatalog.hpp
/// A production conv AOT bundle a build packed (dgrad or forward), loaded through the real
/// state manager, for the AOT engines' shipped-catalog tests: the arch shard of the
/// production tree, the engine's state manager, knobs and dispatch symbols, and the check
/// that a shipped kernel passes the pack's signature check and loads only on its own
/// device. Needs HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR (the unit test binary's).
namespace hip_kernel_provider::kernel_ingestor_engine::conv_aot_shipped_catalog
{

using StateManager = hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>;

struct ShippedCatalog
{
    std::unique_ptr<StateManager> manager; ///< null when the bundle is not staged
    std::vector<std::string> knobs;
    /// The dispatch symbol of every UDD the bundle's packs name.
    std::vector<std::string> dispatchSymbols;
};

/// The @p arch shard of the production tree this build packed, when it holds the bundle
/// folder rocKE/@p bundleFolder; an empty path otherwise. The whole shard is loaded, not
/// the bundle's folder: a kernel's archive sits at the shard root and must lie inside the
/// tree it was read from.
inline std::filesystem::path packedShard(const std::string& arch, const std::string& bundleFolder)
{
    const auto shard = hip_kernel_provider::testing::descriptorSetRoot(
                           HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR)
                       / arch;
    std::error_code missing;
    return std::filesystem::is_directory(shard / "rocKE" / bundleFolder, missing)
               ? shard
               : std::filesystem::path();
}

/// @p engineName's shipped bundle, folder rocKE/@p bundleFolder of the @p arch shard,
/// loaded once per process. Keyed by architecture and engine: one shard holds several
/// engines' bundles (gfx942's dgrad and forward ones), each with its own state manager.
inline const ShippedCatalog& shippedCatalog(const std::string& arch,
                                            std::string_view engineName,
                                            const std::string& bundleFolder)
{
    static std::map<std::pair<std::string, std::string>, ShippedCatalog> s_loaded;
    const auto key = std::make_pair(arch, std::string(engineName));
    const auto found = s_loaded.find(key);
    if(found != s_loaded.end())
    {
        return found->second;
    }
    // Filed only once loaded: a load that throws is retried by the next test rather than
    // read back as an unstaged bundle.
    ShippedCatalog loaded;
    const auto shard = packedShard(arch, bundleFolder);
    if(!shard.empty())
    {
        registerNativeIngestorSymbols();
        auto sets = hipdnn_plugin_sdk::ingestor::resolveDescriptorSets(
            hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(shard));
        const auto set = std::find_if(sets.begin(), sets.end(), [&](const auto& candidate) {
            return candidate.engine.name == engineName;
        });
        if(set == sets.end())
        {
            throw std::runtime_error("the staged " + arch + " bundle holds no "
                                     + std::string(engineName));
        }
        loaded.knobs = set->engine.knobs;
        for(const auto& dispatch : set->dispatches)
        {
            loaded.dispatchSymbols.push_back(dispatch.dispatchSymbol);
        }
        const auto graphMatch = set->engine.graphMatchNativeSymbol;
        loaded.manager
            = hipdnn_plugin_sdk::ingestor::makeStateManager<Handle>(std::move(*set), graphMatch);
    }
    return s_loaded.emplace(key, std::move(loaded)).first->second;
}

/// Why a shipped-catalog test skips when the bundle is not staged.
inline std::string notStaged(const std::string& arch, std::string_view engineName)
{
    return "this configuration packed no " + arch + " production tree, so the shipped "
           + std::string(engineName) + " bundle is not staged beside this binary";
}

/// Prepares @p kernel through @p handler: requireSignatureMatch compares the ABI the pack
/// marshals (names, kinds, sizes) with what hkp_pack recorded before any device is
/// needed, and with a device of the kernel's architecture (@p loadable) the code object
/// then loads. Without one, the load must be what fails, never a message in
/// @p forbidden.
inline void expectPreparesOnItsDeviceOnly(
    const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>& handler,
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
    const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
    bool loadable,
    std::initializer_list<std::string_view> forbidden)
{
    try
    {
        EXPECT_NE(handler.prepare(context, bound, kernel), nullptr);
        EXPECT_TRUE(loadable) << "loaded a code object without a device of its architecture";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_FALSE(loadable) << message;
        // The expected load failure on another device leaves its HIP error set; clear it,
        // or the HIP error listener fails the test for it.
        (void)hipGetLastError();
        (void)hipExtGetLastError();
        for(const auto text : forbidden)
        {
            EXPECT_EQ(message.find(text), std::string::npos) << message;
        }
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_aot_shipped_catalog
