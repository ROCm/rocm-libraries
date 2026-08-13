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

/// What a caller needs to size and launch one selected kernel. Copied out, so
/// holding one does not pin the state manager's internals.
template <typename THandle>
struct KernelDispatcher
{
    KernelDefinition kernel;
    /// Non-owning; must outlive any plan built from this dispatcher (see
    /// NativeRegistry's DispatchRegistry).
    const IKernelDispatchHandler<THandle>* handler = nullptr;
};

/// A UMD paired with the native function its matchSymbol resolved to at construction.
/// Exactly one of the two function pointers is set, per the descriptor's scope.
struct ResolvedMatcher
{
    MatchDescriptor descriptor;
    GraphMatcherFn graphFn = nullptr;
    KernelMatcherFn kernelFn = nullptr;
};

/// A UDD paired with the handler its dispatchSymbol resolved to at construction.
template <typename THandle>
struct ResolvedDispatch
{
    DispatchDescriptor descriptor;
    const IKernelDispatchHandler<THandle>* handler = nullptr;
};

/**
 * @brief The engine's view of its own kernels: which apply to a graph, in what order,
 *        and how to launch one.
 *
 * Holds the descriptor state one engine selects over (its KMD and UHD, and the packs
 * naming it) and answers hipDNN's four host calls:
 *
 * | Host call                    | Answered by                                    |
 * |------------------------------|------------------------------------------------|
 * | isApplicable                 | unsortedDefinitions() is non-empty             |
 * | getDetails (knobs)           | sortedDefinitions(): value sets and defaults   |
 * | getMaxWorkspaceSize          | getDispatchDetails() per survivor, then max    |
 * | initializeExecutionContext   | sortedDefinitions().front(), getDispatchDetails() |
 *
 * Ranking is cached alongside the catalog, keyed on (graph, device) rather than the
 * handle.
 *
 * Thread-safe: the cache is internally synchronized; matcher and scorer calls run
 * outside the lock.
 */
template <typename THandle>
class KernelIngestorStateManager
{
public:
    /// How many (graph, device) catalogs to retain. Entries hold ids and metadata
    /// values, not kernels or graphs; eviction costs a rematch, never a wrong answer.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /**
     * @param schema     The engine's KMD. Supplies the defaults completing each kernel's
     *                   metadata tuple, and the field types a value is checked against.
     * @param matchers   Every UMD any of @p packs references, by id.
     * @param dispatches Every UDD any of @p packs references, by id.
     * @param packs      The KDPs naming this engine.
     * @param heuristic  That engine's UHD, resolved to a scorer.
     *
     * @throws std::invalid_argument if a pack names an unknown matcher or dispatch
     *         descriptor, or if two kernels share a metadata tuple (the catalog key
     *         must be unique).
     * @throws std::runtime_error if a UMD names a match symbol this build does not
     *         ship.
     *
     * Eagerly validates every pack and kernel at plugin load (validateAndIndexPacks())
     * and resolves every matcher symbol here, so a descriptor naming a behaviour that
     * does not exist excludes this engine at load rather than throwing from inside a
     * later isApplicable(), which RFC 0017 §8.6 has already turned into a promise by
     * then. Kernel source compilation stays lazy until a graph needs it.
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
            const auto id = matcher.id;
            const auto description = describeDescriptor("matcher", matcher.name, matcher.id);
            // Resolved here rather than per matching call: a descriptor naming a
            // symbol this build does not ship must exclude the engine at load, not
            // throw from inside isApplicable() after the catalog was already promised
            // (RFC 0017 §8.6). Also lifts a mutexed registry lookup out of the
            // per-pack, per-kernel matching loops.
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
                // A pack naming this id would otherwise silently run whichever
                // matcher loaded first.
                throw std::invalid_argument("duplicate match descriptor id '" + toString(id)
                                            + "' collides with '" + it->second.descriptor.name
                                            + "'");
            }
        }
        for(auto& dispatch : dispatches)
        {
            const auto id = dispatch.id;
            // Resolved here for the same reason matchers are, and it closes the last
            // hole in that argument. SymbolScope makes a pack's registration atomic, so
            // a dispatch symbol missing because its pack failed is already handled; what
            // is not is a descriptor naming a symbol no pack registers. That is exactly
            // what the two-file split admits, since the descriptor and native halves
            // agree by string value with no compile-time check. Lazily, such a typo
            // survives load, survives isApplicable, and throws at plan build, past the
            // point RFC 0017 §8.6 made applicability a binding promise.
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

    /**
     * @brief Every kernel that applies to the graph and device @p context names, in no
     *        particular order.
     *
     * Runs pack matchers in pruning order (graph-scoped, then kernel-scoped) and
     * caches the result. A graph with no identity is matched fresh every call.
     */
    std::vector<KernelDefinition> unsortedDefinitions(const MatchContext& context) const
    {
        return catalogFor(context).entries;
    }

    /**
     * @brief The unranked catalog and the state matching bound, from one lookup.
     *
     * Returned together so a caller needing both does not match twice for a graph
     * with no identity to cache under (RFC 0017 §8.1). `bound` is empty when no
     * graph-scoped matcher bound anything.
     */
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        return catalogFor(context);
    }

    /**
     * @brief Every kernel that applies to the graph and device @p context names, best
     *        first.
     *
     * Uses the cached order if present; otherwise ranks and caches it, building the
     * catalog first if needed.
     */
    std::vector<KernelDefinition> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalog(context).entries;
    }

    /**
     * @brief The ranked catalog and the state matching bound, from one lookup.
     *
     * Avoids matching twice for a graph with no identity to cache under.
     */
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
            // put, not putIfAbsent: a sorted catalog is strictly better than whatever
            // is there, since an unsorted entry costs the next reader a rank() and a
            // sorted one is equivalent to this.
            _catalogCache.put(*key, catalog);
        }

        return catalog;
    }

    /**
     * @brief Resolves how to size and launch @p kernel.
     * @throws std::runtime_error if the kernel's dispatch descriptor is unknown or
     *         names a symbol the provider does not ship; fails hard rather than
     *         declining silently.
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

        // Resolved at construction, so this cannot fail here.
        return {kernel, it->second.handler};
    }

    /// @brief The distinct values @p field takes across @p kernels, in ranked-first order.
    ///
    /// Reflects what the catalog implements, not the KMD field's theoretical range.
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
                // The source kind is RFC 0017 §9.1's adapter dispatch point, and only
                // EMBEDDED_SOURCE has an adapter. Rejected here rather than left to the
                // dispatch handler: a kind with no adapter otherwise passes validation,
                // passes matching, and reaches prepare(), which hands a payload meant
                // for another loader to a source compile. That is a throw after
                // applicability already promised the graph (§8.6). A loader reading
                // kinds from a file is the case this exists for.
                if(kernel.source.kind != KernelSourceKind::EMBEDDED_SOURCE)
                {
                    throw std::invalid_argument(
                        describeDescriptor("kernel", kernel.name, kernel.id)
                        + " declares a source kind this build has no adapter for; only "
                          "EMBEDDED_SOURCE is implemented");
                }

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

    /// A kernel's declared metadata values, with the KMD's defaults filled in for
    /// omitted fields. This completed tuple, not the descriptor id, is the catalog key.
    MetadataValues completeMetadata(const KernelDescriptor& kernel) const
    {
        MetadataValues complete;

        for(const auto& field : _schema.fields)
        {
            auto it = kernel.metadata.find(field.name);

            if(it == kernel.metadata.end())
            {
                // A field with no default must be set by every kernel; omission is
                // an authoring error.
                if(!field.defaultValue.has_value())
                {
                    throw std::invalid_argument("kernel '" + toString(kernel.id)
                                                + "' omits metadata field '" + field.name
                                                + "', which declares no default");
                }
                complete.emplace(field.name, *field.defaultValue);
                continue;
            }

            // Caught here rather than surfacing as a bad_variant_access far from the
            // descriptor that caused it.
            if(metadataTypeOf(it->second) != field.type)
            {
                throw std::invalid_argument("kernel '" + toString(kernel.id)
                                            + "' supplies metadata field '" + field.name
                                            + "' with a value of the wrong type");
            }
            complete.emplace(field.name, it->second);
        }

        // Rejects fields the schema doesn't declare (usually a misspelled name);
        // otherwise silently ignored while joining the key.
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

    /// Device comes from the context, not a separate argument, to avoid caching one
    /// device's catalog under another device's key.
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
        const auto key = cacheKey(context);
        if(key.has_value())
        {
            if(auto cached = _catalogCache.get(*key); cached.has_value())
            {
                HIPDNN_PLUGIN_LOG_TRACE("ingestor: catalog cache hit for device "
                                        << context.deviceId);
                return *cached;
            }
        }
        else
        {
            // A graph with no identity is re-matched every call rather than cached.
            HIPDNN_PLUGIN_LOG_TRACE(
                "ingestor: graph carries no identity, so its catalog cannot be cached");
        }

        Catalog catalog = buildCatalog(context);

        if(key.has_value())
        {
            // putIfAbsent, not put: two threads can miss this key together, and while
            // they matched, one of them may have ranked and installed a *sorted*
            // catalog. Overwriting that with this unsorted one is never a wrong answer,
            // since both describe the same kernels, but it throws away the ranking, so a
            // hot key can thrash between sorted and unsorted and never converge. The
            // loser of the race keeps its own copy and discards only the write.
            _catalogCache.putIfAbsent(*key, catalog);
        }

        return catalog;
    }

    /// One graph-scoped matcher's verdict for one (graph, device), keyed by matcher id.
    /// Memoizes per matcher so a pack merges only the matchers it lists.
    struct GraphMatcherVerdict
    {
        bool passed = false;
        BoundTokens bound;
    };
    using GraphMatcherMemo
        = std::unordered_map<DescriptorId, GraphMatcherVerdict, DescriptorIdHash>;

    /**
     * @brief Runs every pack's matchers over @p context, cheapest and broadest first.
     *
     * Packs whose declared architecture excludes this call's device are dropped before
     * any matcher runs. Graph-scoped matchers then run once per (graph, device)
     * regardless of how many packs list them; a failure disqualifies the pack before
     * kernel-scoped matching runs. A pruned pack's graph-scoped bindings are discarded,
     * never merged.
     */
    Catalog buildCatalog(const MatchContext& context) const
    {
        Catalog catalog;
        GraphMatcherMemo graphVerdicts;

        for(const auto& pack : _packs)
        {
            // Arch first, ahead of every matcher. Cheaper, since it skips a mutexed
            // registry resolve, the matcher body, a token merge, and per kernel a
            // KernelDefinition plus a completeMetadata() map allocation -- all provably
            // wasted for a pack that cannot run here, and it scales with kernel count.
            //
            // The correctness argument is the stronger one. Arch enforced inside a
            // native matcher depends on the author remembering to encode it; miss it and
            // the pack matches, wins ranking, and fails at plan build in a wrong-target
            // hipRTC compile -- past isApplicable, so RFC 0017 §8.6's promise is already
            // broken. Declaring it makes that unreachable rather than diligence-dependent.
            if(!archSupports(pack.arch, context.deviceProperties.gcnArchName))
            {
                // Deliberately distinct from the matcher-decline line below: this is a
                // correct, expected decline on a cross-arch install, not a fault.
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack "
                                       << toString(pack.id) << " does not support device arch '"
                                       << context.deviceProperties.gcnArchName << "'");
                continue;
            }

            // Merged into catalog.bound below only if the pack survives.
            BoundTokens packBound;
            if(!graphLevelMatchersPass(pack, context, graphVerdicts, packBound))
            {
                // RFC 0017 §10: surfaces the pack decline reason for operators.
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " declined at a graph-scoped matcher");
                continue;
            }

            mergeBound(catalog.bound, packBound, describeDescriptor("pack", pack.name, pack.id));

            size_t admitted = 0;
            for(const auto& kernel : pack.kernels)
            {
                KernelDefinition definition{kernel.id,
                                            pack.id,
                                            pack.dispatchId,
                                            kernel.source,
                                            completeMetadata(kernel),
                                            kernel.priority};

                if(kernelLevelMatchersPass(pack, context, definition))
                {
                    catalog.entries.push_back(std::move(definition));
                    ++admitted;
                }
            }

            HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id) << " admitted "
                                                     << admitted << " of " << pack.kernels.size()
                                                     << " kernel(s) after kernel-scoped matching");
        }

        // Logged unconditionally so an empty catalog is distinguishable from a
        // missing engine id.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: catalog for device "
                               << context.deviceId << " holds " << catalog.entries.size()
                               << " kernel(s) from " << _packs.size() << " pack(s)");
        return catalog;
    }

    /// Folds @p source into @p bound, rejecting a token bound to two different values.
    ///
    /// One code path for both scopes: across packs, where two packs disagree, and
    /// within one pack, where two of its graph matchers disagree. The invariant the
    /// message states is engine-wide, so enforcing it only at the pack boundary would
    /// leave the narrower conflict resolved silently by matcher declaration order.
    ///
    /// @param scope Names what disagreed, for the message: an operator has to find the
    ///        offending descriptor, and a bare UUID is not something they can grep for.
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

    /**
     * @param packBound Accumulates this pack's graph-scoped bindings; merged by the
     *        caller only if this returns true.
     */
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
                // Evaluated once per matcher; later packs reuse the memoized verdict.
                GraphMatcherVerdict verdict;
                verdict.passed = matcher.graphFn(context, verdict.bound);
                memo = graphVerdicts.emplace(matcherId, std::move(verdict)).first;
            }

            if(!memo->second.passed)
            {
                return false;
            }

            // Merges this matcher's bindings into the pack's scoped view. Two matchers
            // in one pack disagreeing is the same authoring error as two packs
            // disagreeing, and for the reference packs these tokens are tensor uids, so
            // silently taking the first would launch a kernel against the wrong buffer.
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
    std::shared_ptr<IKernelHeuristic> _heuristic;
    /// Mutable because the query methods are logically const; the cache is internally
    /// synchronized.
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
