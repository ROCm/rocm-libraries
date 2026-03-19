// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelContainer.hpp"
#include "SdpaKernelEngine.hpp"
#include "SdpaKernelHelpers.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <ranges>

namespace sdpa_kernel_provider
{

/**
 * @brief ENGINE_TYPES expands to a comma separated list of all engine types to be loaded into the container.
 * A defaultPlanBuilders() specialization must be present for each engine type.
 *  **/
// TODO: Consider moving this macro and defaultPlanBuilders to a separate header file
#define ENGINE_TYPES SdpaKernelEngine

using PlanBuilderList = std::vector<std::unique_ptr<IPlanBuilder>>;

/**
 * @brief Default list of plan builders associated with the engine used as a template argument.
 */
template <class T>
PlanBuilderList defaultPlanBuilders();

template <>
PlanBuilderList defaultPlanBuilders<SdpaKernelEngine>()
{
    return makeVector<std::unique_ptr<IPlanBuilder>>(std::make_unique<SdpaKernelPlanBuilder>());
}

namespace detail
{
template <class... Ts>
std::array<int64_t, sizeof...(Ts)> engineIdArray()
{
    return {Ts::staticId()...};
}

template <class... Ts>
std::vector<std::unique_ptr<IEngine>> createEngines()
{
    return makeVector<std::unique_ptr<IEngine>>(std::make_unique<Ts>(defaultPlanBuilders<Ts>())...);
}
}

/**
 * Array of all engine ids for the engines listed in ENGINE_TYPES
 */
const auto& engineIdArray()
{
    static auto s_engineIds = detail::engineIdArray<ENGINE_TYPES>();
    return s_engineIds;
}

/**
 * @brief Creates vector of engines for every engine class in the ENGINE_TYPES macro. These are the
 * engines assigned to the SdpaKernelContainer on construction
 *
 * Each entry of ENGINE_TYPES requires a corresponding specialization of defaultPlanBuilders or it
 * will fail to compile.
 */
auto createEngines()
{
    return detail::createEngines<ENGINE_TYPES>();
}

uint32_t SdpaKernelContainer::copyEngineIds(int64_t* engineIds,
                                            uint32_t maxEngines,
                                            uint32_t& numEngines)
{
    const auto& allEngineIds = engineIdArray();

    auto totalEngines = static_cast<uint32_t>(allEngineIds.size());
    if(maxEngines == 0 || engineIds == nullptr)
    {
        numEngines = totalEngines;
        return numEngines;
    }

    numEngines = std::min(maxEngines, totalEngines);
    std::copy_n(allEngineIds.data(), numEngines, engineIds);

    return totalEngines;
}

SdpaKernelContainer::SdpaKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating SdpaKernelContainer");

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle,
                                                                       SdpaKernelSettings,
                                                                       SdpaKernelContext>>();

    auto engines = createEngines();

    for(auto& engine : engines)
    {
        _engineManager->addEngine(std::move(engine));
    }
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
