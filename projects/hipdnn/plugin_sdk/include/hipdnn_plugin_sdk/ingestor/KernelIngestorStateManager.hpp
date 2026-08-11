// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

/// What a caller needs to size and launch one selected kernel: the kernel itself, plus
/// the dispatch handler its pack's UDD resolved to. Copied out like KernelDefinition, so
/// holding one does not pin the state manager's internals.
template <typename THandle>
struct KernelDispatcher
{
    KernelDefinition kernel;
    /// Non-owning. Points at a provider-owned handler that must outlive any plan built
    /// from this dispatcher (see NativeRegistry's DispatchRegistry note).
    const IKernelDispatchHandler<THandle>* handler = nullptr;
};

/**
 * @brief The engine's view of its own kernels: which apply to a graph, in what order,
 *        and how to launch one.
 *
 * Holds the descriptor state one engine selects over (its KMD and UHD, and the packs
 * naming it) and answers the three questions hipDNN's four host calls reduce to:
 *
 * | Host call                    | Answered by                                    |
 * |------------------------------|------------------------------------------------|
 * | isApplicable                 | unsortedDefinitions() is non-empty             |
 * | getDetails (knobs)           | sortedDefinitions() — value sets and defaults  |
 * | getMaxWorkspaceSize          | getDispatchDetails() per survivor, then max    |
 * | initializeExecutionContext   | sortedDefinitions().front(), getDispatchDetails() |
 *
 * **Work is done as late as possible and kept.** Applicability only needs to know
 * whether any kernel survived, so it matches but does not rank. Ranking happens on the
 * first call that needs an order, and is cached alongside the catalog it ordered. Both
 * are keyed on (graph, device) — the problem — rather than on the handle, which is a
 * caller-side object whose lifetime says nothing about whether the work is still valid.
 *
 * Thread-safe: the cache is internally synchronized, and matcher and scorer calls are
 * made outside the lock, so one thread matching a graph never blocks another.
 */
template <typename THandle>
class KernelIngestorStateManager
{
public:
    /// How many (graph, device) catalogs to retain. Entries hold ids and metadata values,
    /// not kernels or graphs, so this is generous by design; eviction costs a rematch,
    /// never a wrong answer.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /**
     * @param schema     The engine's KMD. Supplies the defaults completing each kernel's
     *                   metadata tuple, and the field types a value is checked against.
     * @param matchers   Every UMD any of @p packs references, by id.
     * @param dispatches Every UDD any of @p packs references, by id.
     * @param packs      The KDPs naming this engine.
     * @param heuristic  That engine's UHD, resolved to a scorer.
     *
     * @throws std::invalid_argument if a pack names a matcher or dispatch descriptor not
     *         supplied, or if two kernels share a metadata tuple. Both are load-time
     *         validation failures in the real system: a dangling cross-reference cannot
     *         be evaluated, and duplicate catalog keys leave selection with two
     *         indistinguishable candidates and no basis to prefer either.
     */
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
            _matchers.emplace(matcher.id, std::move(matcher));
        }
        for(auto& dispatch : dispatches)
        {
            _dispatches.emplace(dispatch.id, std::move(dispatch));
        }

        validateAndIndexPacks();
    }

    const MetadataSchema& metadataSchema() const
    {
        return _schema;
    }

    /**
     * @brief Every kernel that applies to @p graph on @p deviceId, in no particular order.
     *
     * On a cache hit, returns the cached catalog. Otherwise runs the matchers in pruning
     * order — each pack's graph-scoped matchers first, since one failure disqualifies the
     * whole pack, then the kernel-scoped matchers over whatever survived — and caches the
     * result.
     *
     * A graph carrying no identity cannot be cached (there is no key, and inventing one
     * would alias unrelated graphs), so it is matched fresh every call. That costs time,
     * never correctness.
     */
    std::vector<KernelDefinition> unsortedDefinitions(DeviceId deviceId,
                                                      const MatchContext& context) const
    {
        return catalogFor(deviceId, context).entries;
    }

    /**
     * @brief Every kernel that applies to @p graph on @p deviceId, best first.
     *
     * Returns the cached order if there is one; otherwise ranks the catalog and caches
     * that order. If the catalog itself is missing, it is built first.
     */
    std::vector<KernelDefinition> sortedDefinitions(DeviceId deviceId,
                                                    const MatchContext& context) const
    {
        Catalog catalog = catalogFor(deviceId, context);
        if(catalog.isSorted)
        {
            return catalog.entries;
        }

        catalog.entries = _heuristic->rank(catalog, context);
        catalog.isSorted = true;

        if(const auto key = cacheKey(deviceId, context); key.has_value())
        {
            _catalogCache.put(*key, catalog);
        }

        return catalog.entries;
    }

    /**
     * @brief Resolves how to size and launch @p kernel.
     * @throws std::runtime_error if the kernel's pack is unknown or its UDD names a
     *         symbol the provider does not ship. Both fail closed: after applicability
     *         accepted a graph, hipDNN has already chosen this engine on that promise, so
     *         a missing dispatch is a hard error rather than a silent decline.
     */
    KernelDispatcher<THandle> getDispatchDetails(const KernelDefinition& kernel) const
    {
        auto it = _dispatches.find(kernel.dispatchId);
        if(it == _dispatches.end())
        {
            throw std::runtime_error("kernel '" + toString(kernel.kernelId)
                                     + "' names unknown dispatch descriptor '"
                                     + toString(kernel.dispatchId) + "'");
        }

        return {kernel, DispatchRegistry<THandle>::resolve(it->second.dispatchSymbol)};
    }

    /// @brief The distinct values @p field takes across @p kernels, in ranked-first order.
    ///
    /// A knob offers what the catalog actually implements, never the KMD field's
    /// theoretical range: offering a value no surviving kernel carries would produce a
    /// request nothing can serve.
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
    void validateAndIndexPacks()
    {
        std::vector<MetadataValues> seenKeys;

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

            for(const auto& kernel : pack.kernels)
            {
                auto key = completeMetadata(kernel);
                if(std::find(seenKeys.begin(), seenKeys.end(), key) != seenKeys.end())
                {
                    throw std::invalid_argument(
                        "kernel '" + toString(kernel.id)
                        + "' duplicates the metadata tuple of another kernel under schema '"
                        + _schema.name + "'; the tuple is the catalog key and must be unique");
                }
                seenKeys.push_back(std::move(key));
            }
        }
    }

    /// A kernel's declared values plus the KMD's defaults for every field it omitted.
    /// This completed tuple, not the descriptor id, is what identifies the kernel to
    /// matching and ranking.
    MetadataValues completeMetadata(const KernelDescriptor& kernel) const
    {
        MetadataValues complete = kernel.metadata;

        for(const auto& field : _schema.fields)
        {
            auto it = complete.find(field.name);

            if(it == complete.end())
            {
                // A field with no default is one every kernel must state for itself, so
                // an omission is an authoring error rather than a silent fallback: it
                // would otherwise produce a catalog key the author never wrote.
                if(!field.defaultValue.has_value())
                {
                    throw std::invalid_argument("kernel '" + toString(kernel.id)
                                                + "' omits metadata field '" + field.name
                                                + "', which declares no default");
                }
                complete.emplace(field.name, *field.defaultValue);
                continue;
            }

            // The KMD declares each field's type, so a value of the wrong one is caught
            // here rather than surfacing as a bad_variant_access from a matcher or a
            // scorer, far from the descriptor that caused it.
            if(metadataTypeOf(it->second) != field.type)
            {
                throw std::invalid_argument("kernel '" + toString(kernel.id)
                                            + "' supplies metadata field '" + field.name
                                            + "' with a value of the wrong type");
            }
        }

        return complete;
    }

    std::optional<CatalogKey> cacheKey(DeviceId deviceId, const MatchContext& context) const
    {
        const auto graphId = tryGetGraphId(context.graph);
        if(!graphId.has_value())
        {
            return std::nullopt;
        }
        return CatalogKey{*graphId, deviceId};
    }

    Catalog catalogFor(DeviceId deviceId, const MatchContext& context) const
    {
        const auto key = cacheKey(deviceId, context);
        if(key.has_value())
        {
            if(auto cached = _catalogCache.get(*key); cached.has_value())
            {
                return *cached;
            }
        }

        Catalog catalog = buildCatalog(context);

        if(key.has_value())
        {
            _catalogCache.put(*key, catalog);
        }

        return catalog;
    }

    /**
     * @brief Runs every pack's matchers over @p context, cheapest and broadest first.
     *
     * Graph-scoped matchers read only graph and device facts, so each runs once for the
     * whole pack and one failure disqualifies every kernel in it without any per-kernel
     * work. Only then do the kernel-scoped matchers run, once per surviving kernel.
     *
     * That ordering is the point of splitting matchers by scope: it is what keeps
     * applicability cheap for an engine whose packs do not apply, which is most engines
     * for most graphs.
     */
    Catalog buildCatalog(const MatchContext& context) const
    {
        Catalog catalog;

        for(const auto& pack : _packs)
        {
            if(!graphLevelMatchersPass(pack, context))
            {
                continue;
            }

            for(const auto& kernel : pack.kernels)
            {
                KernelDefinition definition{kernel.id,
                                            pack.id,
                                            pack.dispatchId,
                                            kernel.sourceFile,
                                            kernel.entryPoint,
                                            completeMetadata(kernel),
                                            kernel.priority};

                if(kernelLevelMatchersPass(pack, context, definition))
                {
                    catalog.entries.push_back(std::move(definition));
                }
            }
        }

        return catalog;
    }

    bool graphLevelMatchersPass(const KernelDescriptorPack& pack, const MatchContext& context) const
    {
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.scope != MatchScope::GRAPH)
            {
                continue;
            }
            if(!GraphMatcherRegistry::resolve(matcher.matchSymbol)(context))
            {
                return false;
            }
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
            if(matcher.scope != MatchScope::KERNEL)
            {
                continue;
            }
            if(!KernelMatcherRegistry::resolve(matcher.matchSymbol)(context, kernel))
            {
                return false;
            }
        }
        return true;
    }

    MetadataSchema _schema;
    std::unordered_map<DescriptorId, MatchDescriptor, DescriptorIdHash> _matchers;
    std::unordered_map<DescriptorId, DispatchDescriptor, DescriptorIdHash> _dispatches;
    std::vector<KernelDescriptorPack> _packs;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    /// Mutable because the query methods are logically const: they answer questions about
    /// the descriptor set without changing it, and caching is an optimization invisible
    /// to the caller. The cache is internally synchronized.
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
