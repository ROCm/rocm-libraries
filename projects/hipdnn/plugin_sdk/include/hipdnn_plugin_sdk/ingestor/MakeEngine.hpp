// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
    auto provenance = nlohmann::json{{"ued", dependency(set.engine)}};
    if(set.slots.empty())
    {
        provenance["kmd"] = dependency(set.schema);
    }
    else
    {
        // RFC 0021 §9: a slotted engine names every slot's KMD, in launch order, instead of one.
        auto& kmds = provenance["kmds"] = nlohmann::json::array();
        for(const auto& slot : set.slots)
        {
            auto entry = dependency(slot.schema);
            entry["slot"] = slot.name;
            kmds.push_back(std::move(entry));
        }
    }
    provenance["umd"] = nlohmann::json::array();
    for(const auto& matcher : set.matchers)
    {
        provenance["umd"].push_back(dependency(matcher));
    }
    std::sort(provenance["umd"].begin(),
              provenance["umd"].end(),
              [](const auto& lhs, const auto& rhs) { return lhs.at("id") < rhs.at("id"); });
    return provenance;
}

namespace detail
{

/// Writes the rankers of one selection scope (an unslotted engine, or one slot) under
/// @p rankers: per metric, then arch, since which model ranks depends on the request's metric as
/// well as the device (RFC 0019 §11.4), so either changing is a different selector.
inline void describeRankers(
    nlohmann::json& rankers,
    const std::map<std::string, std::map<std::string, HeuristicDescriptor>>& heuristicsByMetric,
    const std::optional<HeuristicDescriptor>& heuristic,
    const std::map<std::string, std::set<std::string>>& unavailableHeuristicArches)
{
    for(const auto& [metric, byArch] : heuristicsByMetric)
    {
        for(const auto& [arch, descriptor] : byArch)
        {
            rankers[metric][arch] = rankerIdentity(descriptor);
        }
    }
    // A set built in memory may carry only its default ranker.
    if(heuristic && !rankers[heuristic->score.metric].contains("default"))
    {
        rankers[heuristic->score.metric]["default"] = rankerIdentity(*heuristic);
    }
    for(const auto& [metric, arches] : unavailableHeuristicArches)
    {
        for(const auto& arch : arches)
        {
            rankers[metric][arch] = "unavailable";
        }
    }
}

/// Writes @p packs under @p out["packs"], each with its dispatch and every kernel's identity,
/// one entry per launch for a multi-launch kernel.
inline void describePacks(nlohmann::json& out, const std::vector<KernelDescriptorPack>& packs)
{
    for(const auto& pack : packs)
    {
        auto& resolvedPack
            = out["packs"][toString(pack.id) + "/" + nlohmann::json(pack.arch).dump()];
        resolvedPack["dispatch"] = toString(pack.dispatchId);
        for(const auto& kernel : pack.kernels)
        {
            auto& value = resolvedPack["kernels"][toString(kernel.id)];
            value = {{"priority", kernel.priority},
                     {"arch", kernel.arch},
                     {"source_kind", selectorCode(kernel.source.kind)},
                     {"entry_point", kernel.source.entryPoint},
                     {"source_file", kernel.source.sourceFile},
                     {"toc_key", kernel.source.tocKey},
                     {"symbol", kernel.source.symbol},
                     {"sha256", kernel.source.sha256}};
            // A multi-launch kernel keeps no `source`: each launch's code object is part of
            // what the selector's ranking was trained against.
            for(const auto& [launch, source] : kernel.sources)
            {
                value["sources"][launch] = {{"toc_key", source.tocKey},
                                            {"symbol", source.symbol},
                                            {"sha256", source.sha256}};
            }
            for(const auto& [name, metadata] : kernel.metadata)
            {
                value["metadata"][name] = metadataValueToJson(metadata);
            }
        }
    }
}

} // namespace detail

/// @brief Stable selector identity, including ranker provenance but never evaluating L2.
///
/// A slotted engine's selector adds a `slots` array in launch order: each slot's name, KMD,
/// knobs, rankers and packs, so renaming, reordering or retuning a slot is a different selector
/// (RFC 0021 §9). An unslotted engine's selector is unchanged.
inline std::string engineSelectorRevision(const DescriptorSet& set)
{
    auto selector = enginePredictionProvenance(set);
    selector["selector"] = "generic-untuned-v1";
    selector["graph_match"] = set.engine.graphMatchNativeSymbol;
    selector["knobs"] = set.engine.knobs;
    selector["rankers"] = nlohmann::json::object();
    detail::describeRankers(
        selector["rankers"], set.heuristicsByMetric, set.heuristic, set.unavailableHeuristicArches);
    for(const auto& matcher : set.matchers)
    {
        selector["matchers"][toString(matcher.id)] = matcher.matchSymbol;
    }
    for(const auto& dispatch : set.dispatches)
    {
        selector["dispatches"][toString(dispatch.id)] = dispatch.dispatchSymbol;
    }
    detail::describePacks(selector, set.packs);
    if(!set.slots.empty())
    {
        selector["shared_workspace"] = set.engine.sharedWorkspaceToken;
        selector["slots"] = nlohmann::json::array();
        for(size_t i = 0; i < set.slots.size(); ++i)
        {
            const auto& slot = set.slots[i];
            nlohmann::json described{{"name", slot.name}, {"rankers", nlohmann::json::object()}};
            // The knobs live on the UED's slot, which a set built in memory may not carry.
            described["knobs"] = i < set.engine.slots.size() ? set.engine.slots[i].knobs
                                                             : std::vector<std::string>{};
            detail::describeRankers(described["rankers"],
                                    slot.heuristicsByMetric,
                                    slot.heuristic,
                                    slot.unavailableHeuristicArches);
            detail::describePacks(described, slot.packs);
            selector["slots"].push_back(std::move(described));
        }
    }
    return "generic-untuned-v1/" + uhd::sha256(selector.dump());
}

namespace detail
{

/// The digest of every ranker one selection scope can resolve, or empty when it has none.
inline nlohmann::json rankersIdentity(
    const std::map<std::string, std::map<std::string, HeuristicDescriptor>>& heuristicsByMetric,
    const std::optional<HeuristicDescriptor>& heuristic)
{
    // An ordered map, so the digest does not depend on hash-table iteration order. Keyed by
    // metric and arch together: one UHD per (metric, arch key).
    std::map<std::string, nlohmann::json> rankers;
    for(const auto& [metric, byArch] : heuristicsByMetric)
    {
        const std::string metricPrefix = metric + "@";
        for(const auto& [arch, descriptor] : byArch)
        {
            rankers.emplace(metricPrefix + arch, rankerIdentity(descriptor));
        }
    }
    if(heuristic)
    {
        rankers.emplace(heuristic->score.metric + "@default", rankerIdentity(*heuristic));
    }
    return rankers.empty() ? nlohmann::json() : nlohmann::json(rankers);
}

/// Whether every ranker of one selection scope has a content identity: a ranker that names an
/// artifact needs a digest of it.
inline bool rankersContentIdentified(
    const std::map<std::string, std::map<std::string, HeuristicDescriptor>>& heuristicsByMetric,
    const std::optional<HeuristicDescriptor>& heuristic)
{
    const auto identified = [](const HeuristicDescriptor& descriptor) {
        return descriptor.modelArtifactPath.empty() || !descriptor.modelHash.empty();
    };
    for(const auto& byMetric : heuristicsByMetric)
    {
        for(const auto& byArch : byMetric.second)
        {
            if(!identified(byArch.second))
            {
                return false;
            }
        }
    }
    return !heuristic || identified(*heuristic);
}

/// Each slot's declared knobs, in the UED's slot order.
inline std::vector<std::vector<std::string>> slotKnobsOf(const EngineDescriptor& engine)
{
    std::vector<std::vector<std::string>> slotKnobs;
    slotKnobs.reserve(engine.slots.size());
    for(const auto& slot : engine.slots)
    {
        slotKnobs.push_back(slot.knobs);
    }
    return slotKnobs;
}

} // namespace detail

/// @brief The engine facts a cached ranking's validity depends on, for `EngineIdentity`.
///
/// Digests every heuristic @p set can resolve: the per-arch choice happens at first
/// rank(), but the cache directory is not arch-keyed. Kernels are excluded; a new kernel is
/// the coverage gate's concern. Empty when an unslotted engine ships no heuristic.
///
/// A slotted engine hashes its slots' rankers in launch order, each under its slot name, so the
/// hash changes when any slot's ranker does and when the slots are reordered or renamed
/// (RFC 0021 §9). It is never empty for a slotted engine, even when no slot has a ranker.
inline std::string engineModelHash(const DescriptorSet& set)
{
    if(set.slots.empty())
    {
        const auto rankers = detail::rankersIdentity(set.heuristicsByMetric, set.heuristic);
        return rankers.is_null() ? std::string{} : uhd::sha256(rankers.dump());
    }

    auto slotRankers = nlohmann::json::array();
    for(const auto& slot : set.slots)
    {
        auto rankers = detail::rankersIdentity(slot.heuristicsByMetric, slot.heuristic);
        slotRankers.push_back(nlohmann::json{{"slot", slot.name}, {"rankers", std::move(rankers)}});
    }
    return uhd::sha256(slotRankers.dump());
}

/// @brief Whether every ranker @p set can resolve has a content identity.
///
/// False when one names an artifact but carries no digest: none was declared and no bytes
/// were present to digest at load. engineModelHash() cannot version such a model, so the
/// persistent winner cache is declined for the engine (EngineIdentity::contentIdentified). For
/// a slotted engine, every slot's rankers are checked.
inline bool engineContentIdentified(const DescriptorSet& set)
{
    if(set.slots.empty())
    {
        return detail::rankersContentIdentified(set.heuristicsByMetric, set.heuristic);
    }
    return std::all_of(set.slots.begin(), set.slots.end(), [](const SlotDescriptorSet& slot) {
        return detail::rankersContentIdentified(slot.heuristicsByMetric, slot.heuristic);
    });
}

/// @brief What identifies this engine to the caches that outlive one ranking.
///
/// The `uhdId` of a slotted engine is the UED id: the engine has one ranker per slot and none
/// of its own (RFC 0021 §9).
inline EngineIdentity engineIdentity(const DescriptorSet& set)
{
    // Read off the descriptor (the loader keeps it in step with DescriptorSet::heuristic) so
    // a set built in memory still identifies its cache directory.
    std::string uhdId;
    if(!set.slots.empty())
    {
        uhdId = toString(set.engine.id);
    }
    else if(set.engine.heuristicId.has_value())
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
///
/// A set with `slots` builds a slotted manager: one KernelSlotInput per slot, with that slot's
/// ranker built against the slot's own KMD fields and knobs, and with `ws.*` tokens kept out of
/// the ranker's features (RFC 0021 §6).
/// @param slotKnobs Each slot's declared knobs, in slot order, for the same check. Defaulted
///        from `set.engine.slots`; pass it if `set.engine` was moved out.
/// @param sharedWorkspaceToken The UED's `shared_workspace` token. Defaulted from @p set; pass
///        it if `set.engine` was moved out, or the engine's token goes unvalidated.
template <typename THandle>
std::unique_ptr<KernelIngestorStateManager<THandle>>
    makeStateManager(DescriptorSet set,
                     const std::string& graphMatchSymbol,
                     std::string describedBy = {},
                     EngineIdentity engine = {},
                     std::vector<std::string> knobs = {},
                     std::vector<std::vector<std::string>> slotKnobs = {},
                     std::string sharedWorkspaceToken = {})
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
    if(!set.slots.empty())
    {
        if(slotKnobs.empty())
        {
            slotKnobs = detail::slotKnobsOf(set.engine);
        }
        if(sharedWorkspaceToken.empty())
        {
            sharedWorkspaceToken = set.engine.sharedWorkspaceToken;
        }
        slotKnobs.resize(set.slots.size());
        std::vector<KernelSlotInput> slots;
        slots.reserve(set.slots.size());
        for(size_t i = 0; i < set.slots.size(); ++i)
        {
            auto& slot = set.slots[i];
            const auto slotFields = slot.schema.fieldNames();
            const auto slotDescription = describedBy + " slot '" + slot.name + "'";
            auto heuristic = makeKernelHeuristic(slot.heuristic,
                                                 slotDescription,
                                                 slotKnobs[i],
                                                 slotFields,
                                                 slot.heuristicsByMetric,
                                                 slot.unavailableHeuristicArches,
                                                 /*reserveWorkspaceTokens=*/true);
            slots.push_back(KernelSlotInput{std::move(slot.name),
                                            std::move(slot.schema),
                                            std::move(slot.packs),
                                            std::move(heuristic)});
        }
        return std::make_unique<KernelIngestorStateManager<THandle>>(
            std::move(slots),
            std::move(set.matchers),
            std::move(set.dispatches),
            graphMatchSymbol,
            std::move(sharedWorkspaceToken),
            describedBy,
            KernelIngestorStateManager<THandle>::DEFAULT_CATALOG_CACHE_CAPACITY,
            std::move(engine));
    }
    // Read before `set.schema` is moved below: a moved-from schema declares no fields and
    // would fail RFC 0019 §6.3 check 2 for every model reading `$kernel.*`.
    const auto kmdFields = set.schema.fieldNames();
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
    auto slotKnobs = detail::slotKnobsOf(set.engine);
    auto sharedWorkspaceToken = set.engine.sharedWorkspaceToken;
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
                                  std::move(knobs),
                                  std::move(slotKnobs),
                                  std::move(sharedWorkspaceToken)),
        deviceResolver,
        std::move(predictions),
        std::move(unavailablePredictionArches),
        std::move(selectorRevision),
        std::move(provenance));
}

/// The engine for @p set over a state manager already built from it, e.g. the one
/// loadValidatedDescriptorSets built while validating that set. @p set supplies the UED,
/// the engine-prediction bindings and the selector identity; its packs already belong to
/// @p stateManager and are not copied.
/// @throws std::invalid_argument @p stateManager is null.
template <typename THandle, typename TSettings, typename TContext>
std::unique_ptr<IEngine<THandle, TSettings, TContext>>
    makeEngine(const DescriptorSet& set,
               std::unique_ptr<KernelIngestorStateManager<THandle>> stateManager,
               const IDeviceResolver<THandle>& deviceResolver)
{
    // GenericEngine dereferences the state manager while constructing, so a null one
    // would be undefined behaviour there rather than an error a caller can catch.
    if(stateManager == nullptr)
    {
        throw std::invalid_argument("kernel ingestor engine requires a state manager");
    }
    return std::make_unique<GenericEngine<THandle, TSettings, TContext>>(
        set.engine,
        std::move(stateManager),
        deviceResolver,
        set.enginePredictionsByMetric,
        set.unavailableEnginePredictionArches,
        engineSelectorRevision(set),
        enginePredictionProvenance(set));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
