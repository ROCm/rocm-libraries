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

/// @brief Descriptor dependencies published even before the first L1 model is trained.
inline nlohmann::json enginePredictionProvenance(const DescriptorSet& set)
{
    const auto dependency = [](const auto& descriptor) {
        return nlohmann::json{{"id", toString(descriptor.id)},
                              {"revision",
                               std::to_string(descriptor.revision.major) + "."
                                   + std::to_string(descriptor.revision.minor)}};
    };
    auto provenance = nlohmann::json{{"ued", dependency(set.engine)},
                                     {"kmd", dependency(set.schema)},
                                     {"umd", nlohmann::json::array()}};
    for(const auto& matcher : set.matchers)
    {
        provenance["umd"].push_back(dependency(matcher));
    }
    std::sort(provenance["umd"].begin(),
              provenance["umd"].end(),
              [](const auto& lhs, const auto& rhs) { return lhs.at("id") < rhs.at("id"); });
    return provenance;
}

/// @brief Stable selector identity, including ranker provenance but never evaluating L2.
inline std::string engineSelectorRevision(const DescriptorSet& set)
{
    auto selector = enginePredictionProvenance(set);
    selector["selector"] = "generic-untuned-v1";
    selector["graph_match"] = set.engine.graphMatchNativeSymbol;
    selector["knobs"] = set.engine.knobs;
    selector["rankers"] = nlohmann::json::object();
    const auto rankerIdentity = [](const HeuristicDescriptor& descriptor) {
        return nlohmann::json{{"id", toString(descriptor.id)},
                              {"model_hash", descriptor.modelHash},
                              {"features_hash", descriptor.featuresHash},
                              {"adapter", static_cast<int>(descriptor.adapter)},
                              {"native", descriptor.nativeSymbol},
                              {"objective", descriptor.objective},
                              {"transform", descriptor.score.transform}};
    };
    for(const auto& [arch, descriptor] : set.heuristicsByArch)
    {
        selector["rankers"][arch] = rankerIdentity(descriptor);
    }
    if(set.heuristic && set.heuristicsByArch.count("default") == 0)
    {
        selector["rankers"]["default"] = rankerIdentity(*set.heuristic);
    }
    for(const auto& arch : set.unavailableHeuristicArches)
    {
        selector["rankers"][arch] = "unavailable";
    }
    for(const auto& matcher : set.matchers)
    {
        selector["matchers"][toString(matcher.id)] = matcher.matchSymbol;
    }
    for(const auto& dispatch : set.dispatches)
    {
        selector["dispatches"][toString(dispatch.id)] = dispatch.dispatchSymbol;
    }
    for(const auto& pack : set.packs)
    {
        auto& resolvedPack
            = selector["packs"][toString(pack.id) + "/" + nlohmann::json(pack.arch).dump()];
        resolvedPack["dispatch"] = toString(pack.dispatchId);
        for(const auto& kernel : pack.kernels)
        {
            auto& value = resolvedPack["kernels"][toString(kernel.id)];
            value = {{"priority", kernel.priority},
                     {"arch", kernel.arch},
                     {"source_kind", static_cast<int>(kernel.source.kind)},
                     {"entry_point", kernel.source.entryPoint},
                     {"source_file", kernel.source.sourceFile},
                     {"toc_key", kernel.source.tocKey},
                     {"symbol", kernel.source.symbol},
                     {"sha256", kernel.source.sha256}};
            for(const auto& [name, metadata] : kernel.metadata)
            {
                value["metadata"][name] = detail::metadataValueToJson(metadata);
            }
        }
    }
    return "generic-untuned-v1/" + uhd::sha256(selector.dump());
}

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
    auto heuristic = makeKernelHeuristic(
        set.heuristic, describedBy, knobs, set.heuristicsByArch, set.unavailableHeuristicArches);
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
    auto predictions = std::move(set.enginePredictionsByArch);
    auto unavailablePredictionArches = std::move(set.unavailableEnginePredictionArches);
    auto provenance = enginePredictionProvenance(set);
    auto selectorRevision = engineSelectorRevision(set);
    auto engine = std::move(set.engine);
    auto graphMatchSymbol = engine.graphMatchNativeSymbol;
    return std::make_unique<GenericEngine<THandle, TSettings, TContext>>(
        std::move(engine),
        makeStateManager<THandle>(std::move(set),
                                  std::move(graphMatchSymbol),
                                  std::move(describedBy),
                                  std::move(engineName),
                                  std::move(knobs)),
        deviceResolver,
        std::move(predictions),
        std::move(unavailablePredictionArches),
        std::move(selectorRevision),
        std::move(provenance));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
