// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

/// @file MakeEngine.hpp
/// @brief Builds an engine from a descriptor set; nothing here is operation-specific.
namespace hipdnn_plugin_sdk::ingestor
{

/// Takes @p set by value so a caller building both an engine and its state manager
/// builds the set once.
/// @param graphMatchSymbol The engine's `graph_match` native symbol; empty means the
///        engine declares none and binds no tokens.
/// @param describedBy Names the engine in the graph_match resolution failure and in the
///        warning an engine shipping no heuristic gets. Defaulted from @p set, but a
///        caller that already moved `set.engine` out must pass it, or both name nothing.
/// @param engineName The engine's scoped name, used to locate its on-disk
///        winner-cache shard. Defaulted from @p set like @p describedBy -- a caller
///        that already moved `set.engine` out must pass it explicitly, or the state
///        manager gets an empty name and disables its disk cache.
/// @param knobs The UED's declared knobs, carrying RFC 0019 §6.3 check 2 into the
///        heuristic factory. Defaulted from @p set for the same reason as the two above,
///        and for the same reason a caller that already moved `set.engine` out must pass
///        it: read from a moved-from UED the list is empty, and an empty list compares
///        equal to the axes of a model that reads no `$kernel.*` feature -- so the check
///        that is supposed to catch a knob/axis disagreement instead passes vacuously,
///        or, once a model does read one, refuses every model an engine ever ships.
template <typename THandle>
std::unique_ptr<KernelIngestorStateManager<THandle>>
    makeStateManager(DescriptorSet set,
                     const std::string& graphMatchSymbol,
                     std::string describedBy = {},
                     std::string engineName = {},
                     std::vector<std::string> knobs = {})
{
    if(describedBy.empty())
    {
        describedBy = describeDescriptor("engine", set.engine.name, set.engine.id);
    }
    if(engineName.empty())
    {
        engineName = set.engine.name;
    }
    if(knobs.empty())
    {
        knobs = set.engine.knobs;
    }
    auto heuristic
        = makeKernelHeuristic(set.heuristic, describedBy, knobs, set.heuristicsByArch);
    return std::make_unique<KernelIngestorStateManager<THandle>>(
        std::move(set.schema),
        std::move(set.matchers),
        std::move(set.dispatches),
        std::move(set.packs),
        std::move(heuristic),
        graphMatchSymbol,
        describedBy,
        KernelIngestorStateManager<THandle>::DEFAULT_CATALOG_CACHE_CAPACITY,
        std::move(engineName));
}

/// @param deviceResolver Held by reference by the engine; providers use a
///        process-lifetime static.
template <typename THandle, typename TSettings, typename TContext>
std::unique_ptr<IEngine<THandle, TSettings, TContext>>
    makeEngine(DescriptorSet set, const IDeviceResolver<THandle>& deviceResolver)
{
    // Each read of the UED is its own statement, sequenced before the moves below:
    // reading engine/describedBy/engineName inside the same call as a move would be
    // unsequenced and could read an already-moved-from (empty) engine, silently
    // disabling the disk cache.
    auto describedBy = describeDescriptor("engine", set.engine.name, set.engine.id);
    auto engineName = set.engine.name;
    auto knobs = set.engine.knobs;
    auto engine = std::move(set.engine);
    auto graphMatchSymbol = engine.graphMatchNativeSymbol;
    return std::make_unique<GenericEngine<THandle, TSettings, TContext>>(
        std::move(engine),
        makeStateManager<THandle>(std::move(set),
                                  std::move(graphMatchSymbol),
                                  std::move(describedBy),
                                  std::move(engineName),
                                  std::move(knobs)),
        deviceResolver);
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
