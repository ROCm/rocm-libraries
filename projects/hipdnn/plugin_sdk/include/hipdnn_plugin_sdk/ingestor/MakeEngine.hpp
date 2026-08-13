// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <utility>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

/**
 * @file MakeEngine.hpp
 * @brief Builds an engine from a descriptor set.
 *
 * Nothing here is specific to any operation: a descriptor set names its own engine,
 * schema, heuristic, matchers, dispatches, and packs, so one function covers every
 * pack a provider ships and every set a loader will later produce.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief Builds the state manager backing @p set.
 *
 * Takes @p set by value so a caller building both an engine and its state manager
 * builds the set once. @p set's UED is not consumed here; the engine owns it.
 */
template <typename THandle>
std::unique_ptr<KernelIngestorStateManager<THandle>> makeStateManager(DescriptorSet set)
{
    return std::make_unique<KernelIngestorStateManager<THandle>>(
        std::move(set.schema),
        std::move(set.matchers),
        std::move(set.dispatches),
        std::move(set.packs),
        makeKernelHeuristic(set.heuristic));
}

/**
 * @brief Builds the engine @p set describes.
 *
 * @param deviceResolver Answers which device a call is for. Held by reference by the
 *        engine, so it must outlive it; providers use a process-lifetime static.
 */
template <typename THandle, typename TSettings, typename TContext>
std::unique_ptr<IEngine<THandle, TSettings, TContext>>
    makeEngine(DescriptorSet set, const IDeviceResolver<THandle>& deviceResolver)
{
    // Moved out in its own statement, fully sequenced before the move of the remainder.
    auto engine = std::move(set.engine);
    return std::make_unique<GenericEngine<THandle, TSettings, TContext>>(
        std::move(engine), makeStateManager<THandle>(std::move(set)), deviceResolver);
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
