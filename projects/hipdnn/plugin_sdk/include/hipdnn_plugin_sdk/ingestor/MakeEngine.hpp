// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
#include <memory>
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
                              {"metric", descriptor.score.metric},
                              {"transform", descriptor.score.transform}};
    };
    // Per metric, then arch: which model ranks depends on the request's metric as well as the
    // device (RFC 0019 §11.4), so either changing is a different selector.
    for(const auto& [metric, byArch] : set.heuristicsByMetric)
    {
        for(const auto& [arch, descriptor] : byArch)
        {
            selector["rankers"][metric][arch] = rankerIdentity(descriptor);
        }
    }
    // A set built in memory may carry only its default ranker.
    if(set.heuristic && !selector["rankers"][set.heuristic->score.metric].contains("default"))
    {
        selector["rankers"][set.heuristic->score.metric]["default"]
            = rankerIdentity(*set.heuristic);
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

/// @brief The engine facts a cached ranking's validity depends on, for `EngineIdentity`.
///
/// The model hash is a digest over EVERY heuristic @p set can resolve, not over one of them:
/// which model ranks is decided per architecture at first rank() (RFC 0019 §8.3), long after
/// this runs, and a shard is keyed by arch but a cache directory is not. Hashing all of them
/// means the directory changes when ANY of them changes, which over-invalidates a little --
/// a gfx1151-only model change also retires gfx942's records -- and under-invalidates never.
///
/// The kernels are deliberately NOT hashed in, unlike engineSelectorRevision() above: a pack
/// gaining a kernel is what the coverage gate is for, and retiring every measured ranking on
/// it would cost a GPU sweep to re-learn an order the records already hold.
///
/// Empty when the engine ships no heuristic at all: there is no model content to version,
/// and `winnerCacheShardPath()` renders that as its own directory.
inline std::string engineModelHash(const DescriptorSet& set)
{
    const auto rankerIdentity = [](const HeuristicDescriptor& descriptor) {
        return nlohmann::json{{"id", toString(descriptor.id)},
                              {"model_hash", descriptor.modelHash},
                              {"features_hash", descriptor.featuresHash},
                              {"adapter", static_cast<int>(descriptor.adapter)},
                              {"native", descriptor.nativeSymbol},
                              {"objective", descriptor.objective},
                              {"metric", descriptor.score.metric},
                              {"transform", descriptor.score.transform}};
    };

    // An ordered map, so the digest does not depend on hash-table iteration order. Keyed by
    // metric and arch together: one UHD per (metric, arch key).
    std::map<std::string, nlohmann::json> rankers;
    for(const auto& [metric, byArch] : set.heuristicsByMetric)
    {
        for(const auto& [arch, descriptor] : byArch)
        {
            rankers.emplace(metric + "@" + arch, rankerIdentity(descriptor));
        }
    }
    if(set.heuristic)
    {
        rankers.emplace(set.heuristic->score.metric + "@default", rankerIdentity(*set.heuristic));
    }
    if(rankers.empty())
    {
        return {};
    }
    return uhd::sha256(nlohmann::json(rankers).dump());
}

/// @brief What identifies this engine to the caches that outlive one ranking.
inline EngineIdentity engineIdentity(const DescriptorSet& set)
{
    // The resolved default ranker's id (DescriptorSet::heuristic), which the loader keeps in
    // step with EngineDescriptor::heuristicId; read off the descriptor so a set built in memory
    // without the loader still identifies its cache directory.
    std::string uhdId;
    if(set.engine.heuristicId.has_value())
    {
        uhdId = toString(*set.engine.heuristicId);
    }
    else if(set.heuristic)
    {
        uhdId = toString(set.heuristic->id);
    }
    return EngineIdentity{set.engine.name, set.engine.revision, uhdId, engineModelHash(set)};
}

/// Takes @p set by value so a caller building both an engine and its state manager
/// builds the set once.
/// @param graphMatchSymbol The engine's `graph_match` native symbol; empty means the
///        engine declares none and binds no tokens.
/// @param describedBy Names the engine in the graph_match resolution failure and in the
///        warning an engine shipping no heuristic gets. Defaulted from @p set, but a
///        caller that already moved `set.engine` out must pass it, or both name nothing.
/// @param engine The engine's identity -- scoped name, revision, UHD id and model content
///        hash -- which locates its on-disk winner-cache shard and versions its in-memory
///        catalog cache. Defaulted from @p set like @p describedBy; a caller that already
///        moved `set.engine` out must pass it explicitly (see engineIdentity()), or the
///        state manager gets an empty name and disables its disk cache.
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
    // Read from `set.schema` before the move below hands it to the state manager: RFC 0019
    // §6.3 check 2's first assertion needs the KMD's declared fields, and a moved-from
    // schema declares none -- which would refuse every model that reads a `$kernel.*`
    // feature, exactly the way an empty knob list would.
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
