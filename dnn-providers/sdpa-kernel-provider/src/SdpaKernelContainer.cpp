// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelContainer.hpp"
#include "SdpaKernelEngine.cpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <ranges>

namespace sdpa_kernel_provider
{

// ============================================================================
// Engine Registration
// ============================================================================
// Use HIPDNN_REGISTER_ENGINE to register engine names here when adding engines.
// This will:
// 1. Create _NAME and _ID constants for the engine
// 2. Detect hash collisions with other registered engines
//
// Example:
// HIPDNN_REGISTER_ENGINE(SDPA_KERNEL_ENGINE, "SDPA_KERNEL_ENGINE")
// ============================================================================

// Comma separated list of all engine classes
#define ENGINE_TYPES SdpaKernelEngine

namespace detail
{
template <class... Ts>
std::array<int64_t, sizeof...(Ts)> engineIds()
{
    return {Ts::staticId()...};
}

template <class... Ts>
std::vector<std::unique_ptr<
    hipdnn_plugin_sdk::IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>>>
    createEngines()
{
}
}

const auto& engineIds()
{
    static auto s_engineIds = detail::engineIds<ENGINE_TYPES>();
    return s_engineIds;
}

std::vector<std::unique_ptr<
    hipdnn_plugin_sdk::IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>>>
    SdpaKernelContainer::getEngines()
{
    return {std::unique_ptr<SdpaKernelEngine>(
        new SdpaKernelEngine({std::make_unique<SdpaKernelPlanBuilder>()}))};
}

uint32_t SdpaKernelContainer::copyEngineIds(int64_t* engineIds,
                                            uint32_t maxEngines,
                                            uint32_t& numEngines)
{
    static std::vector<int64_t> s_allEngineIds = []() {
        auto idRange = getEngines()
                       | ranges::views::transform([](const auto& engine) { return engine->id(); });
        return std::vector<int64_t>(idRange.begin(), idRange.end());
    }();

    if(maxEngines == 0)
    {
        numEngines = s_allEnginesIds.size();
        return numEngines;
    }

    numEngines = std::min(maxEngines, s_allEnginesIds.size());
    std::ranges::copy_n(s_allEnginesIds, numEngines, engineIds);

    return allEngines.size();
}

SdpaKernelContainer::SdpaKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating SdpaKernelContainer");

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>>(
        getEngines());
}

SdpaKernelContainer::~SdpaKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying SdpaKernelContainer");
}

hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>&
    SdpaKernelContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace sdpa_kernel_provider
