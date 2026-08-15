// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/LruCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// What a caller needs to size and launch one selected kernel; copied out so it does
/// not pin the state manager's internals.
template <typename THandle>
struct KernelDispatcher
{
    KernelDefinition kernel;
    const IKernelDispatchHandler<THandle>* handler = nullptr;
};

/// A UMD plus the native function its matchSymbol resolved to at construction.
struct ResolvedMatcher
{
    MatchDescriptor descriptor;
    GraphMatcherFn graphFn = nullptr;
    KernelMatcherFn kernelFn = nullptr;
};

/// A UDD plus the handler its dispatchSymbol resolved to at construction.
template <typename THandle>
struct ResolvedDispatch
{
    DispatchDescriptor descriptor;
    const IKernelDispatchHandler<THandle>* handler = nullptr;
};

/// The engine's view of its own kernels: which apply to a graph, in what order, and
/// how to launch one. Answers isApplicable (unsortedDefinitions non-empty), getDetails
/// (sortedDefinitions), getMaxWorkspaceSize (getDispatchDetails per survivor, max), and
/// initializeExecutionContext (sortedDefinitions().front(), getDispatchDetails).
///
/// Thread-safe: the cache is internally synchronized; matcher and scorer calls run
/// outside the lock.
template <typename THandle>
class KernelIngestorStateManager
{
public:
    /// How many (graph, device) catalogs to retain; eviction costs a rematch, never a
    /// wrong answer.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /// @throws std::invalid_argument bad pack reference, or duplicate metadata tuple.
    /// @throws std::runtime_error a UMD names a match symbol this build does not ship.
    ///
    /// Matcher and dispatch symbols resolve here, eagerly, so a missing one excludes
    /// this engine at construction instead of throwing later from isApplicable().
    KernelIngestorStateManager(MetadataSchema schema,
                               std::vector<MatchDescriptor> matchers,
                               std::vector<DispatchDescriptor> dispatches,
                               std::vector<KernelDescriptorPack> packs,
                               std::shared_ptr<IKernelHeuristic> heuristic,
                               size_t catalogCacheCapacity = DEFAULT_CATALOG_CACHE_CAPACITY)
        : _schema(std::move(schema))
        , _packs(std::move(packs))
        , _heuristic(std::move(heuristic))
        , _catalogCache(catalogCacheCapacity)
    {
        if(_heuristic == nullptr)
        {
            throw std::invalid_argument("kernel ingestor requires a heuristic");
        }

        for(auto& matcher : matchers)
        {
            const auto id = matcher.id;
            const auto description = describeDescriptor("matcher", matcher.name, matcher.id);
            ResolvedMatcher resolved{std::move(matcher), nullptr, nullptr};
            if(resolved.descriptor.scope == MatchScope::GRAPH)
            {
                resolved.graphFn
                    = GraphMatcherRegistry::resolve(resolved.descriptor.matchSymbol, description);
            }
            else
            {
                resolved.kernelFn
                    = KernelMatcherRegistry::resolve(resolved.descriptor.matchSymbol, description);
            }

            if(const auto [it, inserted] = _matchers.emplace(id, std::move(resolved)); !inserted)
            {
                throw std::invalid_argument("duplicate match descriptor id '" + toString(id)
                                            + "' collides with '" + it->second.descriptor.name
                                            + "'");
            }
        }
        for(auto& dispatch : dispatches)
        {
            const auto id = dispatch.id;
            ResolvedDispatch<THandle> resolved{std::move(dispatch), nullptr};
            resolved.handler = DispatchRegistry<THandle>::resolve(
                resolved.descriptor.dispatchSymbol,
                describeDescriptor("dispatch", resolved.descriptor.name, id));

            if(const auto [it, inserted] = _dispatches.emplace(id, std::move(resolved)); !inserted)
            {
                throw std::invalid_argument("duplicate dispatch descriptor id '" + toString(id)
                                            + "' collides with '" + it->second.descriptor.name
                                            + "'");
            }
        }

        validateAndIndexPacks();
    }

    const MetadataSchema& metadataSchema() const
    {
        return _schema;
    }

    /// Every kernel that applies to the graph and device @p context names, unordered.
    std::vector<KernelDefinition> unsortedDefinitions(const MatchContext& context) const
    {
        return catalogFor(context).entries;
    }

    /// The unranked catalog and the state matching bound, from one lookup.
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        return catalogFor(context);
    }

    /// Every kernel that applies to the graph and device @p context names, best first.
    std::vector<KernelDefinition> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalog(context).entries;
    }

    /// The ranked catalog and the state matching bound, from one lookup.
    Catalog sortedCatalog(const MatchContext& context) const
    {
        Catalog catalog = catalogFor(context);
        if(catalog.isSorted)
        {
            return catalog;
        }

        catalog.entries = _heuristic->rank(catalog, context);
        catalog.isSorted = true;

        if(const auto key = cacheKey(context); key.has_value())
        {
            // put, not putIfAbsent: sorted is strictly better than whatever is cached.
            _catalogCache.put(*key, catalog);
        }

        return catalog;
    }

    /// Resolves how to size and launch @p kernel.
    /// @throws std::runtime_error if the kernel's dispatch descriptor is unknown.
    KernelDispatcher<THandle> getDispatchDetails(const KernelDefinition& kernel) const
    {
        auto it = _dispatches.find(kernel.dispatchId);
        if(it == _dispatches.end())
        {
            throw std::runtime_error("kernel '" + toString(kernel.kernelId)
                                     + "' names unknown dispatch descriptor '"
                                     + toString(kernel.dispatchId) + "'");
        }
        return {kernel, it->second.handler};
    }

    /// The distinct values @p field takes across @p kernels, in ranked-first order.
    static std::vector<MetadataValue> knobValues(const std::vector<KernelDefinition>& kernels,
                                                 const std::string& field)
    {
        std::vector<MetadataValue> values;
        for(const auto& kernel : kernels)
        {
            const auto value = kernel.tryGetMetadata(field);
            if(!value.has_value())
            {
                continue;
            }
            if(std::find(values.begin(), values.end(), *value) == values.end())
            {
                values.push_back(*value);
            }
        }
        return values;
    }

private:
    /// Validates every pack's references and builds the KernelDefinition for each of
    /// its kernels. Every field of a definition is context-independent, so this is the
    /// only place they are ever computed: buildCatalog copies them per query rather
    /// than completing each kernel's metadata again on every graph.
    void validateAndIndexPacks()
    {
        // set, not a scanned vector: the check is quadratic otherwise, and the tuple is
        // an ordered map, so it already orders.
        std::set<MetadataValues> seenKeys;

        _definitions.reserve(_packs.size());
        for(const auto& pack : _packs)
        {
            for(const auto& matcherId : pack.matcherIds)
            {
                if(_matchers.find(matcherId) == _matchers.end())
                {
                    throw std::invalid_argument("pack '" + toString(pack.id)
                                                + "' names unknown matcher '" + toString(matcherId)
                                                + "'");
                }
            }
            if(_dispatches.find(pack.dispatchId) == _dispatches.end())
            {
                throw std::invalid_argument("pack '" + toString(pack.id)
                                            + "' names unknown dispatch descriptor '"
                                            + toString(pack.dispatchId) + "'");
            }

            std::vector<KernelDefinition> packDefinitions;
            packDefinitions.reserve(pack.kernels.size());
            for(const auto& kernel : pack.kernels)
            {
                if(kernel.source.kind != KernelSourceKind::EMBEDDED_SOURCE)
                {
                    throw std::invalid_argument(
                        describeDescriptor("kernel", kernel.name, kernel.id)
                        + " declares a source kind this build has no adapter for; only "
                          "EMBEDDED_SOURCE is implemented");
                }

                auto key = completeMetadata(kernel);
                if(!seenKeys.insert(key).second)
                {
                    throw std::invalid_argument(
                        "kernel '" + toString(kernel.id)
                        + "' duplicates the metadata tuple of another kernel under schema '"
                        + _schema.name + "'; the tuple is the catalog key and must be unique");
                }

                packDefinitions.push_back(KernelDefinition{kernel.id,
                                                           pack.id,
                                                           pack.dispatchId,
                                                           kernel.source,
                                                           std::move(key),
                                                           kernel.priority});
            }
            _definitions.push_back(std::move(packDefinitions));
        }
    }

    /// A kernel's metadata values with the KMD's defaults filled in; the completed
    /// tuple, not the descriptor id, is the catalog key.
    MetadataValues completeMetadata(const KernelDescriptor& kernel) const
    {
        MetadataValues complete;

        for(const auto& field : _schema.fields)
        {
            auto it = kernel.metadata.find(field.name);

            if(it == kernel.metadata.end())
            {
                if(!field.defaultValue.has_value())
                {
                    throw std::invalid_argument("kernel '" + toString(kernel.id)
                                                + "' omits metadata field '" + field.name
                                                + "', which declares no default");
                }
                complete.emplace(field.name, *field.defaultValue);
                continue;
            }

            if(metadataTypeOf(it->second) != field.type)
            {
                throw std::invalid_argument("kernel '" + toString(kernel.id)
                                            + "' supplies metadata field '" + field.name
                                            + "' with a value of the wrong type");
            }
            complete.emplace(field.name, it->second);
        }

        for(const auto& [name, value] : kernel.metadata)
        {
            if(complete.find(name) == complete.end())
            {
                throw std::invalid_argument("kernel '" + toString(kernel.id)
                                            + "' supplies metadata field '" + name
                                            + "', which its engine's metadata schema does "
                                              "not declare");
            }
        }

        return complete;
    }

    /// Device comes from the context, not a separate argument, so one device's catalog
    /// never caches under another's key.
    std::optional<CatalogKey> cacheKey(const MatchContext& context) const
    {
        const auto graphId = tryGetGraphId(context.graph);
        if(!graphId.has_value())
        {
            return std::nullopt;
        }
        return CatalogKey{*graphId, context.deviceId};
    }

    Catalog catalogFor(const MatchContext& context) const
    {
        // Nothing below this line can be answered without a device: pack pruning reads
        // the device's arch, matchers read its properties, and a kernel that somehow
        // matched could not be launched. Answered once here rather than in every
        // provider's matchers, where it is easy to leave out and impossible to see
        // missing -- an empty catalog is what those matchers were producing anyway.
        if(context.deviceId == NO_DEVICE)
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: no device resolved; no kernel applies");
            return Catalog{};
        }

        const auto key = cacheKey(context);
        if(key.has_value())
        {
            if(auto cached = _catalogCache.get(*key); cached.has_value())
            {
                HIPDNN_PLUGIN_LOG_TRACE("ingestor: catalog cache hit for device "
                                        << context.deviceId);
                // get() already returned a copy; moving out of that local avoids a
                // second one on the hot path.
                return std::move(*cached);
            }
        }
        else
        {
            HIPDNN_PLUGIN_LOG_TRACE(
                "ingestor: graph carries no identity, so its catalog cannot be cached");
        }

        Catalog catalog = buildCatalog(context);

        if(key.has_value())
        {
            // putIfAbsent: another thread may already have installed a sorted catalog
            // here; overwriting with this unsorted one would discard that ranking.
            _catalogCache.putIfAbsent(*key, catalog);
        }

        return catalog;
    }

    /// One graph-scoped matcher's verdict for one (graph, device); memoized per
    /// matcher so a pack merges only the matchers it lists.
    struct GraphMatcherVerdict
    {
        bool passed = false;
        BoundTokens bound;
    };
    using GraphMatcherMemo
        = std::unordered_map<DescriptorId, GraphMatcherVerdict, DescriptorIdHash>;

    /// Runs every pack's matchers over @p context: arch, then graph-scoped (memoized
    /// across packs), then kernel-scoped. A pruned pack's bindings are never merged.
    Catalog buildCatalog(const MatchContext& context) const
    {
        Catalog catalog;
        GraphMatcherMemo graphVerdicts;

        for(size_t packIndex = 0; packIndex < _packs.size(); ++packIndex)
        {
            const auto& pack = _packs[packIndex];

            if(!archSupports(pack.arch, context.deviceProperties.gcnArchName))
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack "
                                       << toString(pack.id) << " does not support device arch '"
                                       << context.deviceProperties.gcnArchName << "'");
                continue;
            }

            BoundTokens packBound;
            if(!graphLevelMatchersPass(pack, context, graphVerdicts, packBound))
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " declined at a graph-scoped matcher");
                continue;
            }

            mergeBound(catalog.bound, packBound, describeDescriptor("pack", pack.name, pack.id));

            size_t admitted = 0;
            for(const auto& precomputed : _definitions[packIndex])
            {
                // Copied, not rebuilt: every field was settled at construction, and the
                // kernel matcher below reads the definition without mutating it.
                KernelDefinition definition = precomputed;

                if(kernelLevelMatchersPass(pack, context, definition))
                {
                    catalog.entries.push_back(std::move(definition));
                    ++admitted;
                }
            }

            if(admitted == 0)
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack "
                                       << toString(pack.id) << " admitted no kernel of "
                                       << pack.kernels.size() << " at a kernel-scoped matcher");
                continue;
            }

            HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id) << " admitted "
                                                     << admitted << " of " << pack.kernels.size()
                                                     << " kernel(s) after kernel-scoped matching");
        }

        HIPDNN_PLUGIN_LOG_INFO("ingestor: catalog for device "
                               << context.deviceId << " holds " << catalog.entries.size()
                               << " kernel(s) from " << _packs.size() << " pack(s)");
        return catalog;
    }

    /// Folds @p source into @p bound; rejects a token bound to two disagreeing values,
    /// across packs or within one pack's own matchers.
    static void mergeBound(BoundTokens& bound, const BoundTokens& source, const std::string& scope)
    {
        for(const auto& [token, value] : source)
        {
            const auto [it, inserted] = bound.emplace(token, value);
            if(!inserted && !(it->second == value))
            {
                std::string message = scope;
                message += " binds token '";
                message += token;
                message += "' to a value that disagrees with another binding of the same "
                           "token; one token name must mean one thing across an engine";
                throw std::runtime_error(message);
            }
        }
    }

    bool graphLevelMatchersPass(const KernelDescriptorPack& pack,
                                const MatchContext& context,
                                GraphMatcherMemo& graphVerdicts,
                                BoundTokens& packBound) const
    {
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.descriptor.scope != MatchScope::GRAPH)
            {
                continue;
            }

            auto memo = graphVerdicts.find(matcherId);
            if(memo == graphVerdicts.end())
            {
                GraphMatcherVerdict verdict;
                verdict.passed = matcher.graphFn(context, verdict.bound);
                memo = graphVerdicts.emplace(matcherId, std::move(verdict)).first;
            }

            if(!memo->second.passed)
            {
                return false;
            }

            mergeBound(
                packBound,
                memo->second.bound,
                describeDescriptor("matcher", matcher.descriptor.name, matcher.descriptor.id));
        }
        return true;
    }

    bool kernelLevelMatchersPass(const KernelDescriptorPack& pack,
                                 const MatchContext& context,
                                 const KernelDefinition& kernel) const
    {
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.descriptor.scope != MatchScope::KERNEL)
            {
                continue;
            }
            if(!matcher.kernelFn(context, kernel))
            {
                return false;
            }
        }
        return true;
    }

    MetadataSchema _schema;
    std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash> _matchers;
    std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash> _dispatches;
    std::vector<KernelDescriptorPack> _packs;
    /// One entry per pack, parallel to _packs: its kernels' context-independent
    /// definitions, completed once at construction.
    std::vector<std::vector<KernelDefinition>> _definitions;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
