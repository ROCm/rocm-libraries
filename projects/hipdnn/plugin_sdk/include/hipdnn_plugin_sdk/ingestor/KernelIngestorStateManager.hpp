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
     *
     * **This constructor eagerly walks and validates every pack and kernel, via
     * validateAndIndexPacks(), at plugin load -- a conscious amendment of RFC 0017 §3
     * and §8's "nothing is parsed until a graph needs it".** What stays lazy is the
     * expensive part: a kernel source is not compiled and a heuristic model is not read
     * until a graph actually needs that kernel or that ranking (see
     * NativeKernelHeuristic's doc for the latter). Descriptor *parsing* -- walking the
     * KMD/UMD/UDD/UKD structures this constructor is handed and checking their
     * cross-references and metadata tuples -- is cheap relative to those, and doing it
     * once at load time is what lets every later match, rank, and dispatch assume the
     * descriptor set is internally consistent rather than re-checking it per graph. The
     * cost this trades away is startup latency proportional to descriptor count, which
     * for an in-process pack like this one is negligible; a loader reading many packs
     * from disk is where that cost becomes visible, and if it ever needs to be paid
     * lazily instead, this is the constructor that amendment would have to change.
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
            if(const auto [it, inserted] = _matchers.emplace(id, std::move(matcher)); !inserted)
            {
                // Same defect NativeRegistry.hpp rejects for a duplicate symbol: a pack
                // naming this id would silently run whichever matcher loaded first.
                throw std::invalid_argument("duplicate match descriptor id '" + toString(id)
                                            + "' collides with '" + it->second.name + "'");
            }
        }
        for(auto& dispatch : dispatches)
        {
            const auto id = dispatch.id;
            if(const auto [it, inserted] = _dispatches.emplace(id, std::move(dispatch)); !inserted)
            {
                throw std::invalid_argument("duplicate dispatch descriptor id '" + toString(id)
                                            + "' collides with '" + it->second.name + "'");
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
     * On a cache hit, returns the cached catalog. Otherwise runs the matchers in pruning
     * order — each pack's graph-scoped matchers first, since one failure disqualifies the
     * whole pack, then the kernel-scoped matchers over whatever survived — and caches the
     * result.
     *
     * A graph carrying no identity cannot be cached (there is no key, and inventing one
     * would alias unrelated graphs), so it is matched fresh every call. That costs time,
     * never correctness.
     */
    std::vector<KernelDefinition> unsortedDefinitions(const MatchContext& context) const
    {
        return catalogFor(context).entries;
    }

    /**
     * @brief The unranked catalog and the state matching bound, from one lookup.
     *
     * RFC 0017 §8.1 keeps the bound token state alongside the catalog so that "nothing
     * is re-matched" after applicability: a dispatch handler sizing a workspace reads the
     * values the matcher already resolved rather than walking the graph again with a
     * second notion of what it looks like.
     *
     * Returned together with the entries rather than through a separate accessor because
     * a caller needing both would otherwise match twice for a graph carrying no identity,
     * which is exactly the case with no cache to absorb the repeat.
     *
     * `bound` is empty when no graph-scoped matcher bound anything, which is legal: a
     * pack whose launch geometry is fully determined by kernel metadata has nothing to
     * bind.
     */
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        return catalogFor(context);
    }

    /**
     * @brief Every kernel that applies to the graph and device @p context names, best
     *        first.
     *
     * Returns the cached order if there is one; otherwise ranks the catalog and caches
     * that order. If the catalog itself is missing, it is built first.
     */
    std::vector<KernelDefinition> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalog(context).entries;
    }

    /**
     * @brief The ranked catalog and the state matching bound, from one lookup.
     *
     * A plan build needs both, and asking for them separately means two calls into the
     * cache -- or, for a graph carrying no identity and so no cache entry, two full
     * matching passes. Returning them together makes "match once" hold for the
     * uncacheable case as well, which is the case that can least afford the second pass.
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
            _catalogCache.put(*key, catalog);
        }

        return catalog;
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

    /// A kernel's declared values for the fields its engine's KMD declares, with the
    /// KMD's defaults filled in for the ones it omitted. This completed tuple, not the
    /// descriptor id, is what identifies the kernel to matching and ranking.
    ///
    /// Built from the schema's field list rather than from the kernel's own map, because
    /// the KMD fields are the only per-kernel input selection has: a value the schema
    /// never declared is unreadable to a matcher or a scorer, so admitting one into the
    /// key would let two kernels that selection cannot tell apart both enter the catalog.
    MetadataValues completeMetadata(const KernelDescriptor& kernel) const
    {
        MetadataValues complete;

        for(const auto& field : _schema.fields)
        {
            auto it = kernel.metadata.find(field.name);

            if(it == kernel.metadata.end())
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
            complete.emplace(field.name, it->second);
        }

        // A value the schema does not declare cannot be read by anything downstream, so
        // it is almost always a misspelled field name. Left in place it would be doubly
        // silent: the real field takes its default, and the stray value joins the key.
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

    /// The device comes from the context rather than a separate argument: taking it
    /// twice would let a caller cache one device's catalog under another device's key,
    /// which is a wrong answer rather than a missed hit.
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
            // Worth saying once per call: a graph with no identity is re-matched every
            // time, so an operator seeing repeated matching for what looks like the same
            // graph is seeing this rather than a caching bug.
            HIPDNN_PLUGIN_LOG_TRACE(
                "ingestor: graph carries no identity, so its catalog cannot be cached");
        }

        Catalog catalog = buildCatalog(context);

        if(key.has_value())
        {
            _catalogCache.put(*key, catalog);
        }

        return catalog;
    }

    /// One graph-scoped matcher's verdict for one (graph, device), keyed by matcher id.
    ///
    /// Matchers are shared by id across packs, so the same check would otherwise be
    /// re-run once per pack that lists it. The memo covers what the matcher *bound* as
    /// well as whether it passed, since re-running it to recover its bindings would
    /// defeat the memo entirely.
    ///
    /// `bound` holds only what THIS matcher wrote, isolated from every other matcher's
    /// writes. That isolation is what makes it safe to merge into a per-pack scoped
    /// view one matcher at a time: a pack merges in exactly the matchers it lists, never
    /// a sibling pack's unrelated contribution to some other matcher's memo entry.
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
     * Graph-scoped matchers read only graph and device facts, so each is evaluated once
     * per (graph, device) no matter how many packs list it, and one failure disqualifies
     * every kernel in every pack that lists it without any per-kernel work. Only then do
     * the kernel-scoped matchers run, once per surviving kernel.
     *
     * That ordering, and that sharing, are the point of splitting matchers by scope: the
     * broadly shared checks are what prune the candidate set fast, so an engine whose
     * packs do not apply pays for each distinct check once rather than once per pack.
     *
     * Each pack's graph-scoped bindings are accumulated in a scope-local view and merged
     * into the catalog's shared bound state only when the whole pack survives (see
     * graphLevelMatchersPass()). Bindings a pruned pack's matchers wrote along the way
     * are discarded with that pack's local view rather than leaking into the state a
     * surviving pack's kernel later reads: with one pack this was invisible, because a
     * pruned pack's catalog is empty and nothing reads its bound state, but two packs
     * sharing an engine is the normal case (matchers are shared by id), and the moment a
     * second pack survives, a bare shared `bound` map would hand its kernels tokens that
     * describe a graph shape only the pruned pack's own matcher resolved.
     */
    Catalog buildCatalog(const MatchContext& context) const
    {
        Catalog catalog;
        GraphMatcherMemo graphVerdicts;

        for(const auto& pack : _packs)
        {
            // Scoped to this pack. Merged into catalog.bound below only if every one of
            // this pack's graph-scoped matchers passes; discarded with this pack's
            // failure otherwise. See buildCatalog()'s doc for why that scoping is the
            // fix rather than an optimization.
            BoundTokens packBound;
            if(!graphLevelMatchersPass(pack, context, graphVerdicts, packBound))
            {
                // Logged, not silent: "no kernel matched my graph" is the question a
                // data-driven engine is hardest to answer, because there is no
                // hand-written switch to read. RFC 0017 §10 asks that an operator be
                // able to see why a kernel was not selected, and a pack pruned at the
                // graph level is the coarsest and most common reason.
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " declined at a graph-scoped matcher");
                continue;
            }

            // Merged rather than kept per pack: a token name means the same thing to
            // every pack in an engine (Catalog::bound's doc).
            mergeBound(catalog.bound, packBound, pack.id);

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

        // The summary a caller diagnosing an unexpected decline actually needs: an empty
        // catalog here is what applicability reports as "does not apply", and without
        // this line the only observable is the absence of an engine id.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: catalog for device "
                               << context.deviceId << " holds " << catalog.entries.size()
                               << " kernel(s) from " << _packs.size() << " pack(s)");
        return catalog;
    }

    /// Folds @p packBound into @p bound. Two packs agreeing on a token's value are just
    /// sharing a matcher by id; two packs writing DIFFERENT values under one name is an
    /// authoring error, only detectable here since it depends on a runtime match.
    static void
        mergeBound(BoundTokens& bound, const BoundTokens& packBound, const DescriptorId& packId)
    {
        for(const auto& [token, value] : packBound)
        {
            const auto [it, inserted] = bound.emplace(token, value);
            if(!inserted && !(it->second == value))
            {
                throw std::runtime_error(
                    "pack '" + toString(packId) + "' binds token '" + token
                    + "' to a value that disagrees with another pack's binding of the "
                      "same token; one token name must mean one thing across an engine");
            }
        }
    }

    /**
     * @param packBound Accumulates what THIS pack's own graph-scoped matchers resolved,
     *        isolated from every other pack's contribution (see GraphMatcherVerdict).
     *        The caller merges it into the catalog's shared bound state only when this
     *        function returns true; a pack this function prunes leaves @p packBound to
     *        be discarded unmerged, which is what stops its matchers' bindings from
     *        reaching a kernel in a pack that survives.
     */
    bool graphLevelMatchersPass(const KernelDescriptorPack& pack,
                                const MatchContext& context,
                                GraphMatcherMemo& graphVerdicts,
                                BoundTokens& packBound) const
    {
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.scope != MatchScope::GRAPH)
            {
                continue;
            }

            auto memo = graphVerdicts.find(matcherId);
            if(memo == graphVerdicts.end())
            {
                // Evaluated on the first pack that lists this matcher only. A later pack
                // listing the same matcher reuses both the verdict and the bindings it
                // already wrote from this memo entry, which is the whole point of
                // memoizing across packs -- the native function itself runs once no
                // matter how many packs share the matcher.
                GraphMatcherVerdict verdict;
                verdict.passed
                    = GraphMatcherRegistry::resolve(matcher.matchSymbol)(context, verdict.bound);
                memo = graphVerdicts.emplace(matcherId, std::move(verdict)).first;
            }

            if(!memo->second.passed)
            {
                return false;
            }

            // Merges only THIS matcher's own bindings into the pack's scoped view. A
            // pack that does not list this matcher never merges them; a pack that
            // shares it with another pack that failed elsewhere still merges exactly
            // what this matcher wrote, reused from the memo rather than recomputed.
            packBound.insert(memo->second.bound.begin(), memo->second.bound.end());
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
