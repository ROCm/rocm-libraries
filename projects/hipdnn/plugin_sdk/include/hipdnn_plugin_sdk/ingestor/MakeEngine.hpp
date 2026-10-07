// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

namespace detail
{

/// @p adapter as it enters a selector revision or model hash. The codes are fixed here rather
/// than taken from the enumerator's position, so reordering UhdAdapter cannot change the
/// revision a shipped model records. A new adapter takes the next unused code.
inline int selectorCode(UhdAdapter adapter)
{
    switch(adapter)
    {
    case UhdAdapter::STATIC_ORDER:
        return 0;
    case UhdAdapter::NATIVE:
        return 1;
    case UhdAdapter::TREE_DATA:
        return 2;
    case UhdAdapter::TABLE:
        return 3;
    case UhdAdapter::CUSTOM_LIBRARY:
        return 4;
    // Required by -Wswitch-default; reached only by an adapter missing from this table.
    default:
        throw std::logic_error("UHD adapter has no selector code");
    }
}

/// @p kind as it enters a selector revision, fixed for the same reason as the adapter codes.
inline int selectorCode(KernelSourceKind kind)
{
    switch(kind)
    {
    case KernelSourceKind::EMBEDDED_SOURCE:
        return 0;
    case KernelSourceKind::KPACK:
        return 1;
    case KernelSourceKind::HSACO_FILE:
        return 2;
    case KernelSourceKind::ROCKE_BUILDER:
        return 3;
    // Required by -Wswitch-default; reached only by a source kind missing from this table.
    default:
        throw std::logic_error("kernel source kind has no selector code");
    }
}

/// What identifies one ranker to a selector revision and to the model hash.
inline nlohmann::json rankerIdentity(const HeuristicDescriptor& descriptor)
{
    return nlohmann::json{{"id", toString(descriptor.id)},
                          {"model_hash", descriptor.modelHash},
                          {"features_hash", descriptor.featuresHash},
                          {"adapter", selectorCode(descriptor.adapter)},
                          {"native", descriptor.nativeSymbol},
                          {"objective", descriptor.objective},
                          {"metric", descriptor.score.metric},
                          {"transform", descriptor.score.transform}};
}

} // namespace detail

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
    // Per metric, then arch: which model ranks depends on the request's metric as well as the
    // device (RFC 0019 §11.4), so either changing is a different selector.
    for(const auto& [metric, byArch] : set.heuristicsByMetric)
    {
        for(const auto& [arch, descriptor] : byArch)
        {
            selector["rankers"][metric][arch] = detail::rankerIdentity(descriptor);
        }
    }
    // A set built in memory may carry only its default ranker.
    if(set.heuristic && !selector["rankers"][set.heuristic->score.metric].contains("default"))
    {
        selector["rankers"][set.heuristic->score.metric]["default"]
            = detail::rankerIdentity(*set.heuristic);
    }
    for(const auto& [metric, arches] : set.unavailableHeuristicArches)
    {
        for(const auto& arch : arches)
        {
            selector["rankers"][metric][arch] = "unavailable";
        }
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
                     {"source_kind", detail::selectorCode(kernel.source.kind)},
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

/// @brief The engine facts a cached ranking's validity depends on, for `EngineIdentity`.
///
/// Digests every heuristic @p set can resolve: the per-arch choice happens at first
/// rank(), but the cache directory is not arch-keyed. Kernels are excluded; a new kernel is
/// the coverage gate's concern. Empty when the engine ships no heuristic.
inline std::string engineModelHash(const DescriptorSet& set)
{
    // An ordered map, so the digest does not depend on hash-table iteration order. Keyed by
    // metric and arch together: one UHD per (metric, arch key).
    std::map<std::string, nlohmann::json> rankers;
    for(const auto& [metric, byArch] : set.heuristicsByMetric)
    {
        const std::string metricPrefix = metric + "@";
        for(const auto& [arch, descriptor] : byArch)
        {
            rankers.emplace(metricPrefix + arch, detail::rankerIdentity(descriptor));
        }
    }
    if(set.heuristic)
    {
        rankers.emplace(set.heuristic->score.metric + "@default",
                        detail::rankerIdentity(*set.heuristic));
    }
    if(rankers.empty())
    {
        return {};
    }
    return uhd::sha256(nlohmann::json(rankers).dump());
}

/// @brief Whether every ranker @p set can resolve has a content identity.
///
/// False when one names an artifact but carries no digest: none was declared and no bytes
/// were present to digest at load. engineModelHash() cannot version such a model, so the
/// persistent winner cache is declined for the engine (EngineIdentity::contentIdentified).
inline bool engineContentIdentified(const DescriptorSet& set)
{
    const auto identified = [](const HeuristicDescriptor& descriptor) {
        return descriptor.modelArtifactPath.empty() || !descriptor.modelHash.empty();
    };
    for(const auto& byMetric : set.heuristicsByMetric)
    {
        for(const auto& byArch : byMetric.second)
        {
            if(!identified(byArch.second))
            {
                return false;
            }
        }
    }
    return !set.heuristic || identified(*set.heuristic);
}

/// @brief What identifies this engine to the caches that outlive one ranking.
inline EngineIdentity engineIdentity(const DescriptorSet& set)
{
    // Read off the descriptor (the loader keeps it in step with DescriptorSet::heuristic) so
    // a set built in memory still identifies its cache directory.
    std::string uhdId;
    if(set.engine.heuristicId.has_value())
    {
        uhdId = toString(*set.engine.heuristicId);
    }
    else if(set.heuristic)
    {
        uhdId = toString(set.heuristic->id);
    }
    EngineIdentity identity{set.engine.name, set.engine.revision, uhdId, engineModelHash(set)};
    identity.contentIdentified = engineContentIdentified(set);
    return identity;
}

/// Takes @p set by value so a caller building both an engine and its state manager
/// builds the set once.
/// @param graphMatchSymbol The engine's `graph_match` native symbol; empty means the
///        engine declares none and binds no tokens.
/// @param describedBy Names the engine in the graph_match resolution failure and in the
///        warning an engine shipping no heuristic gets. Defaulted from @p set, but a
///        caller that already moved `set.engine` out must pass it, or both name nothing.
/// @param engine The engine's identity, locating its winner-cache shard and versioning its
///        catalog cache. Defaulted from @p set; pass it if `set.engine` was moved out, or
///        the disk cache is disabled.
/// @param knobs The UED's declared knobs for RFC 0019 §6.3 check 2. Defaulted from @p set;
///        pass it if `set.engine` was moved out, or the check passes vacuously.
template <typename THandle>
std::unique_ptr<KernelIngestorStateManager<THandle>>
    makeStateManager(DescriptorSet set,
                     const std::string& graphMatchSymbol,
                     std::string describedBy = {},
                     EngineIdentity engine = {},
                     std::vector<std::string> knobs = {})
{
    if(describedBy.empty())
    {
        describedBy = describeDescriptor("engine", set.engine.name, set.engine.id);
    }
    if(engine.name.empty())
    {
        engine = engineIdentity(set);
    }
    if(knobs.empty())
    {
        knobs = set.engine.knobs;
    }
    // Read before `set.schema` is moved below: a moved-from schema declares no fields and
    // would fail RFC 0019 §6.3 check 2 for every model reading `$kernel.*`.
    std::unordered_set<std::string> kmdFields;
    for(const auto& field : set.schema.fields)
    {
        kmdFields.insert(field.name);
    }
    auto heuristic = makeKernelHeuristic(set.heuristic,
                                         describedBy,
                                         knobs,
                                         kmdFields,
                                         set.heuristicsByMetric,
                                         set.unavailableHeuristicArches);
    return std::make_unique<KernelIngestorStateManager<THandle>>(
        std::move(set.schema),
        std::move(set.matchers),
        std::move(set.dispatches),
        std::move(set.packs),
        std::move(heuristic),
        graphMatchSymbol,
        describedBy,
        KernelIngestorStateManager<THandle>::DEFAULT_CATALOG_CACHE_CAPACITY,
        std::move(engine));
}

/// @param deviceResolver Held by reference by the engine; providers use a
///        process-lifetime static.
template <typename THandle, typename TSettings, typename TContext>
std::unique_ptr<IEngine<THandle, TSettings, TContext>>
    makeEngine(DescriptorSet set, const IDeviceResolver<THandle>& deviceResolver)
{
    // Each read of the UED is its own statement, sequenced before the moves below:
    // reading engine/describedBy/identity inside the same call as a move would be
    // unsequenced and could read an already-moved-from (empty) engine, silently
    // disabling the disk cache.
    auto describedBy = describeDescriptor("engine", set.engine.name, set.engine.id);
    auto identity = engineIdentity(set);
    auto knobs = set.engine.knobs;
    auto predictions = std::move(set.enginePredictionsByMetric);
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
                                  std::move(identity),
                                  std::move(knobs)),
        deviceResolver,
        std::move(predictions),
        std::move(unavailablePredictionArches),
        std::move(selectorRevision),
        std::move(provenance));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
