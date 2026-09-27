// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/LineStore.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/LruCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>

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
    GraphCriterionFn graphFn = nullptr;
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
/// Thread safety. Three `LruCache`es, each synchronizing internally -- `_catalogCache`,
/// `_calibratedCache`, and `_winnerCache` -- plus `_winnerCacheMutex`, which guards the
/// bookkeeping that sits AROUND the winner cache rather than inside it: the per-shard
/// "already loaded" set and the one-shot failure-log set.
///
/// `_winnerCacheMutex` may be held across an `_winnerCache` call (`mightHaveWinnerFor()`
/// asks both questions as one); the reverse never happens, since an `LruCache` calls no
/// user code, so the ordering is total and the two cannot deadlock. Matchers, the
/// heuristic, and the accessors below all run outside every lock.
///
/// Everything between a lookup and a store is thread-local: `catalogFor` returns a
/// `Catalog` by value and callers mutate that copy, so ordering a catalog touches no
/// shared state. Returning a reference into `_catalogCache` instead would make those
/// mutations a data race. `winnerFor` returns a copy for the same reason.
///
/// Concurrent callers can therefore duplicate work -- two threads may rank the same
/// catalog, or record a ranking for the same key -- and the last store wins. Both
/// stores are equally valid, so the cost is the redundant work, never a wrong answer.
template <typename THandle>
class KernelIngestorStateManager
{
public:
    /// How many (graph, device) catalogs to retain; eviction costs a rematch, never a
    /// wrong answer.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /// How many benchmarked rankings to retain in memory.
    ///
    /// RFC 0019 §9.2 requires the cache to be capacity-bounded: "a working set larger than
    /// the bound re-ranks on eviction. The bound is a tuning parameter, not a correctness
    /// issue." This was a soft warning threshold with no eviction, on the argument that
    /// discarding a measured ranking costs a GPU sweep. That argument does not survive the
    /// on-disk shard: an evicted record is re-read from disk on the next lookup for its arch,
    /// or, if this process already latched that shard, re-measured once and then DISCARDED in
    /// favour of the line already on disk (`writeBackToShard()` adopts on a fresh miss). So
    /// eviction costs at most one redundant sweep and never a lost measurement, while an
    /// unbounded map costs a process that ranks many distinct graphs a monotonically growing
    /// map it can never release.
    static constexpr size_t DEFAULT_WINNER_CACHE_CAPACITY = 4096;

    /// @throws std::invalid_argument bad pack reference, or duplicate metadata tuple.
    /// @throws std::runtime_error a UMD or the engine's graph_match names a symbol this
    /// build does not ship.
    ///
    /// Matcher, dispatch, and graph_match symbols resolve here, eagerly, so a missing
    /// one excludes this engine at construction instead of throwing later from
    /// isApplicable().
    /// @param describedBy Names the engine in the graph_match resolution failure, which
    ///        is the only one of the three that has no descriptor of its own to name:
    ///        the symbol lives on the UED, so without this the diagnostic would carry
    ///        the symbol string alone.
    /// @param engine The engine's own identity (`EngineDescriptor::name`/`revision` plus its
    ///        UHD's id and content hash) -- not `describedBy`, which is a diagnostic string,
    ///        not a path component. It composes the on-disk winner-cache shard path and
    ///        supplies the version half of every `CatalogKey`. Defaults to empty so existing
    ///        direct-construction call sites keep compiling; an empty name disables the disk
    ///        cache rather than sharing one shard across every unnamed instance.
    KernelIngestorStateManager(MetadataSchema schema,
                               std::vector<MatchDescriptor> matchers,
                               std::vector<DispatchDescriptor> dispatches,
                               std::vector<KernelDescriptorPack> packs,
                               std::shared_ptr<IKernelHeuristic> heuristic,
                               const std::string& graphMatchSymbol,
                               const std::string& describedBy = {},
                               size_t catalogCacheCapacity = DEFAULT_CATALOG_CACHE_CAPACITY,
                               EngineIdentity engine = {},
                               size_t winnerCacheCapacity = DEFAULT_WINNER_CACHE_CAPACITY)
        : _schema(std::move(schema))
        , _packs(std::move(packs))
        , _heuristic(std::move(heuristic))
        , _graphMatchFn(graphMatchSymbol.empty()
                            ? nullptr
                            : GraphMatchRegistry::resolve(graphMatchSymbol, describedBy))
        , _catalogCache(catalogCacheCapacity)
        // Same key and same capacity as _catalogCache on purpose: the two answer for the same
        // (graph, device, engine version), so an entry in one is valid exactly when an entry
        // in the other would be. Held beside the catalog rather than inside it because
        // `Catalog` is handed to every matcher, scorer and dispatch formula, and none of them
        // reads a calibrated ranking.
        , _calibratedCache(catalogCacheCapacity)
        , _engine(std::move(engine))
        , _winnerCache(winnerCacheCapacity)
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
                    = GraphCriterionRegistry::resolve(resolved.descriptor.matchSymbol, description);
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

    /// Graph bindings only: the L1 path never constructs or ranks a kernel catalog.
    std::optional<BoundTokens> graphBindings(const MatchContext& context) const
    {
        return _graphMatchFn == nullptr ? std::optional<BoundTokens>(BoundTokens{})
                                        : _graphMatchFn(context);
    }

    /// Calibrated model scores for exact configuration prediction, never cached timings.
    ///
    /// @param full The canonical candidate set: every kernel the matchers admitted for this
    ///        graph, before any knob pin narrows it. Also carries the bound tokens the
    ///        scorer reads.
    /// @param filtered What survived the caller's knob pin and workspace limit, in @p full's
    ///        order. Pass `full.entries` for an unpinned request.
    /// @param modelId Set to the id of the model that produced the scores.
    ///
    /// RFC 0019 §5 step 8 fixes the basis a ranking is decided on: "A scorer error is
    /// resolved against the canonical candidate set -- every kernel the matchers admitted
    /// for this graph, before any knob filter narrows it ... Knob filtering then applies to
    /// the resulting order." §9.2 says the same thing about the cache: "What is cached is
    /// the full applicable catalog's order ... a filtered request restricts the cached order
    /// rather than producing its own."
    ///
    /// Ranking @p filtered directly breaks both, and §5 step 8 names the failure exactly:
    /// "if scoring fails on one candidate, ranking the full catalog first would fall back
    /// for everything, while a knob pin that happened to exclude that candidate would score
    /// normally, and the two paths would disagree on the winner from the same inputs."
    ///
    /// A cold cache ranks the full catalog too, and restricts the result exactly as a warm
    /// one does. Ranking @p filtered directly -- which this did -- makes the basis depend on
    /// whether this particular request happened to pin, and step 8's failure reappears on the
    /// cold path: a scorer that throws on a candidate the pin excludes degrades the unpinned
    /// request to static order while the pinned one ranks by model, from the same inputs. The
    /// cost of getting that right is one ranking of the kernels the pin removed, paid once
    /// per graph, because the full order is memoized whether or not this request was pinned.
    /// That memo is what makes every later request -- pinned or not -- commute with this one.
    std::vector<ScoredKernel> calibratedRanking(const Catalog& full,
                                                const std::vector<KernelDefinition>& filtered,
                                                const MatchContext& context,
                                                std::string& modelId) const
    {
        const auto key = cacheKey(context);
        if(key.has_value())
        {
            if(auto cached = _calibratedCache.get(*key); cached.has_value())
            {
                modelId = std::move(cached->modelId);
                return restrictRanking(cached->ranking, filtered);
            }
        }

        auto ranking = _heuristic->calibratedRanking(full, context, modelId);

        if(key.has_value() && !ranking.empty())
        {
            // Not memoized when empty: an empty ranking is "this heuristic cannot calibrate"
            // (IKernelHeuristic::calibratedRanking's default), a property of the heuristic
            // rather than a result worth a cache entry, and caching it would make a miss and
            // a hit indistinguishable.
            _calibratedCache.put(*key, CalibratedRanking{ranking, modelId});
        }
        return restrictRanking(ranking, filtered);
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

    /// Matching only, in descriptor-ID order: independent of heuristic/winner state.
    /// A page walk must not change order when another caller benchmarks the catalog.
    Catalog enumerableCatalog(const MatchContext& context) const
    {
        auto catalog = catalogFor(context);
        std::sort(catalog.entries.begin(),
                  catalog.entries.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.kernelId < rhs.kernelId; });
        for(size_t i = 1; i < catalog.entries.size(); ++i)
        {
            if(catalog.entries[i - 1].kernelId == catalog.entries[i].kernelId)
            {
                throw std::invalid_argument("Ambiguous candidate id '"
                                            + toString(catalog.entries[i].kernelId) + "'");
            }
        }
        return catalog;
    }

    /// Every kernel that applies to the graph and device @p context names, best first.
    std::vector<KernelDefinition> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalog(context).entries;
    }

    /// The ordered catalog and the state matching bound, from one lookup.
    ///
    /// A benchmarked record covering the whole catalog supplies a measured order and
    /// `rank()` is never called; otherwise the heuristic orders it.
    Catalog sortedCatalog(const MatchContext& context) const
    {
        // Mirrors catalogFor's own reject guard: cacheKey() below only reads graph and
        // device ordinal, not arch, so without this an unresolved-arch context would
        // cache an empty catalog under the SAME key a later, resolved call for this
        // device reuses -- permanently hiding that device's real catalog.
        if(context.deviceId == NO_DEVICE || context.deviceProperties.gcnArchName.empty())
        {
            return catalogFor(context);
        }

        Catalog catalog = catalogFor(context);

        // A measured order is final; a heuristic one is provisional, so this lookup runs
        // again even when the catalog is already sorted -- a sweep can postdate the
        // memoized sort.
        if(catalog.isSorted && catalog.orderedFromRecord)
        {
            return catalog;
        }

        if(auto ordered = orderFromWinnerRecord(catalog.entries, context); ordered.has_value())
        {
            catalog.entries = std::move(*ordered);
            catalog.orderedFromRecord = true;
        }
        else
        {
            if(catalog.isSorted)
            {
                return catalog;
            }
            catalog.entries = _heuristic->rank(catalog, context);
        }
        catalog.isSorted = true;

        if(const auto key = cacheKey(context); key.has_value())
        {
            // put, not putIfAbsent: sorted is strictly better than whatever is cached.
            _catalogCache.put(*key, catalog);
        }

        return catalog;
    }

    /// Records @p record under @p key. Whether an existing on-disk ranking for @p key is
    /// kept or replaced depends on @p cause -- see the full rule below.
    ///
    /// Write-back: if this manager has an engine name, the record is also appended to
    /// the on-disk shard. Unlike the in-memory-only read-through path below, this holds
    /// the shard's own file lock across the whole read-then-append sequence as one
    /// critical section -- `_winnerCacheMutex` alone cannot serialize against a second
    /// process sharing the file. @p cause selects the write-back rule and is required,
    /// not defaulted: an existing on-disk entry for the key is adopted as-is and nothing
    /// is written on a fresh miss; on a coverage-triggered re-benchmark, the new ranking
    /// is always appended and supersedes the old one under the reader's last-line-wins
    /// merge, even if the ranking happens to be unchanged. Any disk failure degrades to
    /// in-memory-only, keeping the measurement just taken, logged once per manager.
    void
        recordWinner(const WinnerKey& key, const WinnerRecord& record, WinnerWriteCause cause) const
    {
        if(record.empty())
        {
            // An all-unusable sweep has nothing to record; an empty record would read
            // as a covered hit for the empty candidate set.
            return;
        }

        if(!key.graph.isUsable())
        {
            // Unkeyable graphs never match on lookup (GraphContentKey::operator==), so
            // storing here would leak memory on an unreachable entry.
            HIPDNN_PLUGIN_LOG_INFO("ingestor: a benchmarked ranking could not be cached "
                                   "because its graph yields no key");
            return;
        }

        WinnerRecord adopted = writeBackToShard(key, record, cause);

        // Bounded, least-recently-used: RFC 0019 §9.2's "the cache is capacity-bounded ...
        // so a working set larger than the bound re-ranks on eviction". LRU rather than a
        // cheaper policy because the access pattern this cache serves is a graph being
        // executed repeatedly -- recency is exactly the signal that says which rankings are
        // still being asked for. See DEFAULT_WINNER_CACHE_CAPACITY for what eviction costs.
        _winnerCache.put(key, std::move(adopted));
    }

    /// The ranking recorded for @p key, or nullopt. Returns a copy, which is also what
    /// refreshes @p key's recency: a lookup is what keeps a record alive under the bound.
    ///
    /// Read-through: an in-memory miss loads @p key's on-disk shard once per shard (a
    /// per-shard "already loaded" flag), taking only `_winnerCacheMutex` to guard that flag
    /// -- unlike write-back, this never holds a file lock across the read.
    std::optional<WinnerRecord> winnerFor(const WinnerKey& key) const
    {
        if(auto found = _winnerCache.get(key); found.has_value())
        {
            return found;
        }

        loadShardIfAbsent(key.device.properties().gcnArchName);

        return _winnerCache.get(key);
    }

    /// How many rankings are held; never more than the manager's winner-cache capacity.
    /// For tests and diagnostics.
    size_t winnerCacheSize() const
    {
        return _winnerCache.size();
    }

    /// Is it worth building a WinnerKey for @p gcnArchName? True if the in-memory cache
    /// holds anything, or this arch's shard has not been attempted yet. Probes the
    /// stripped arch, matching how `loadShardIfAbsent()` latches.
    bool mightHaveWinnerFor(const std::string& gcnArchName) const
    {
        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        // size(), not empty(): LruCache exposes no empty(), and this predicate now runs
        // against the bounded cache rather than the unbounded map it was written for.
        return _winnerCache.size() != 0
               || _loadedWinnerShards.find(std::string(stripArchFeatures(gcnArchName)))
                      == _loadedWinnerShards.end();
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

    /// The integer that addresses @p value for metadata field @p field, or nullopt when the
    /// value is not one this engine's kernels carry.
    ///
    /// An `INT` field addresses itself; every other type is addressed by its index in the
    /// field's ordinal domain (see buildOrdinalDomains). Nullopt rather than a fallback
    /// number: a value outside the domain names no kernel, and inventing an index for it
    /// would silently address the neighbour that happens to sit there.
    std::optional<int64_t> knobOrdinal(const std::string& field, const MetadataValue& value) const
    {
        if(const auto* intValue = std::get_if<int64_t>(&value))
        {
            return *intValue;
        }

        const auto domain = _ordinalDomains.find(field);
        if(domain == _ordinalDomains.end())
        {
            return std::nullopt;
        }
        const auto position = std::find(domain->second.begin(), domain->second.end(), value);
        if(position == domain->second.end())
        {
            return std::nullopt;
        }
        return static_cast<int64_t>(std::distance(domain->second.begin(), position));
    }

    /// Does @p kernel's value for @p field carry the pinned ordinal?
    ///
    /// The comparison a knob filter needs: a kernel that does not carry the field at all
    /// cannot be addressed by it and does not match.
    bool knobMatches(const KernelDefinition& kernel, const std::string& field, int64_t pinned) const
    {
        const auto value = kernel.tryGetMetadata(field);
        if(!value.has_value())
        {
            return false;
        }
        const auto ordinal = knobOrdinal(field, *value);
        return ordinal.has_value() && *ordinal == pinned;
    }

    /// Is @p field addressed by an ordinal rather than by its own value? True exactly for a
    /// non-integer field some kernel carries -- what a knob's description must say, so a
    /// caller reading `dtype in {0,1}` knows those are indices and not values.
    bool isOrdinalKnob(const std::string& field) const
    {
        return _ordinalDomains.find(field) != _ordinalDomains.end();
    }

private:
    /// One memoized full-catalog calibrated ranking. The model id travels with it because
    /// the ranking is meaningless without the provenance the caller reports alongside it,
    /// and re-deriving it would mean calling the heuristic again, which is the cost this
    /// cache exists to avoid.
    struct CalibratedRanking
    {
        std::vector<ScoredKernel> ranking;
        std::string modelId;
    };

    /// @p ranking in its own order, keeping only entries naming a kernel in @p candidates.
    ///
    /// This is what makes filtering and ranking commute (RFC 0019 §5 step 8): the order and
    /// the scores are the full catalog's, and the pin only removes rows. Dropping rather
    /// than reordering matters -- a candidate the pin excluded must not influence the
    /// relative order of two that survived, and it cannot, since the surviving rows keep
    /// their relative positions.
    ///
    /// Linear scan per entry rather than a set: a catalog is tens of kernels, and building
    /// an unordered_set of DescriptorIds costs more than the scan it replaces at that size.
    static std::vector<ScoredKernel>
        restrictRanking(const std::vector<ScoredKernel>& ranking,
                        const std::vector<KernelDefinition>& candidates)
    {
        std::vector<ScoredKernel> restricted;
        restricted.reserve(std::min(ranking.size(), candidates.size()));
        for(const auto& scored : ranking)
        {
            const bool survived = std::any_of(
                candidates.begin(), candidates.end(), [&scored](const KernelDefinition& kernel) {
                    return kernel.kernelId == scored.kernelId;
                });
            if(survived)
            {
                restricted.push_back(scored);
            }
        }
        return restricted;
    }

    /// Validates every pack's references and builds the KernelDefinition for each of
    /// its kernels. Every field of a definition is context-independent, so this is the
    /// only place they are ever computed: buildCatalog copies them per query rather
    /// than completing each kernel's metadata again on every graph.
    void validateAndIndexPacks()
    {
        // Two kernels may share a tuple when no single device can see both -- that is
        // exactly the per-arch shard layout. Uniqueness is therefore per overlapping-arch
        // group, not per engine: the tuple is the catalog key, and a catalog is built for
        // one device. Keyed by the tuple (an ordered map, so it already orders) rather
        // than scanned, which would be quadratic.
        std::map<MetadataValues, std::vector<std::vector<std::string>>> archesClaimingTuple;

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
                if(kernel.source.kind != KernelSourceKind::EMBEDDED_SOURCE
                   && kernel.source.kind != KernelSourceKind::KPACK)
                {
                    // Dropped, not thrown: an unadaptable kernel costs only itself, so its
                    // pack keeps serving whichever siblings this build can dispatch.
                    HIPDNN_PLUGIN_LOG_ERROR(
                        "ingestor: " << describeDescriptor("kernel", kernel.name, kernel.id)
                                     << " declares a source kind this build has no adapter for;"
                                        " only EMBEDDED_SOURCE and KPACK are implemented,"
                                        " dropping it");
                    continue;
                }

                auto key = completeMetadata(kernel);
                // A kernel that declared no arch of its own runs wherever its pack does;
                // one that declared a narrower list claims only that. Claiming by the
                // kernel rather than the pack is what lets two kernels of ONE pack share a
                // tuple under disjoint arch -- one implementation per capability -- while
                // still catching two that a single device would see together.
                std::vector<std::string> kernelArch = kernel.arch.empty() ? pack.arch : kernel.arch;
                // try_emplace, not operator[], only because misc-const-correctness
                // misreads the operator[] form here and demands a const map.
                std::vector<std::vector<std::string>>& claimants
                    = archesClaimingTuple.try_emplace(key).first->second;
                for(const auto& claimed : claimants)
                {
                    if(archOverlaps(claimed, kernelArch))
                    {
                        throw std::invalid_argument(
                            "kernel '" + toString(kernel.id)
                            + "' duplicates the metadata tuple of another kernel under schema '"
                            + _schema.name
                            + "' on an arch both reach; the tuple is the catalog key "
                            + "and must be unique per device");
                    }
                }
                claimants.push_back(kernelArch);

                packDefinitions.push_back(KernelDefinition{kernel.id,
                                                           pack.id,
                                                           pack.dispatchId,
                                                           kernel.source,
                                                           std::move(key),
                                                           kernel.priority,
                                                           std::move(kernelArch),
                                                           kernel.originDirectory,
                                                           kernel.name,
                                                           kernel.treeRoot});
            }
            // Pushed even when every kernel was dropped: _definitions is indexed by pack
            // position, so skipping one entry would bind every later pack's definitions to
            // the wrong pack and run the last index out of range.
            _definitions.push_back(std::move(packDefinitions));
        }

        buildOrdinalDomains();
    }

    /// The value set of every non-integer metadata field, in the order that numbers it.
    ///
    /// RFC 0019 §13.2 wants the knob tuple to equal the metadata tuple, so that every catalog
    /// entry is individually addressable. A knob value is an int64 end to end, so only an
    /// `INT` field addresses itself; the other four types are addressed by an INDEX into this
    /// set. Two kernels differing solely in, say, `dtype` were previously indistinguishable
    /// through the knob surface and collided as one candidate.
    ///
    /// Built here, over completed metadata, for two reasons. Defaults are already filled in,
    /// so the set holds the values the catalog will actually carry rather than the authored
    /// subset. And it spans every pack the engine owns rather than one device's catalog: a
    /// per-catalog set would give one integer different meanings on different graphs, so a
    /// recorded pin would replay against a different kernel.
    ///
    /// `std::variant`'s ordering is the specification: same-type values compare by the
    /// underlying type -- `false` before `true`, numbers numerically, strings by byte (which
    /// is code-point order in UTF-8), and `int_list` lexicographically. The generation tool
    /// derives the same indices from the same descriptors without a table being shipped
    /// (`uhd_gen/addressing.py`), so the two sides agree by construction rather than by
    /// exchange.
    void buildOrdinalDomains()
    {
        for(const auto& field : _schema.fields)
        {
            if(field.type == MetadataType::INT)
            {
                continue;
            }

            std::set<MetadataValue> observed;
            for(const auto& packDefinitions : _definitions)
            {
                for(const auto& definition : packDefinitions)
                {
                    const auto it = definition.metadata.find(field.name);
                    if(it != definition.metadata.end())
                    {
                        observed.insert(it->second);
                    }
                }
            }
            if(!observed.empty())
            {
                _ordinalDomains.emplace(
                    field.name, std::vector<MetadataValue>(observed.begin(), observed.end()));
            }
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
    /// never caches under another's key. The engine revision comes from this manager, not
    /// the context: the catalog is this engine's kernels ranked by this engine's heuristic,
    /// and a context knows nothing about either.
    std::optional<CatalogKey> cacheKey(const MatchContext& context) const
    {
        const auto graphId = tryGetGraphId(context.graph);
        if(!graphId.has_value())
        {
            return std::nullopt;
        }
        return CatalogKey{*graphId, context.deviceId, _engine.version};
    }

    Catalog catalogFor(const MatchContext& context) const
    {
        // Pack pruning and matchers both read the device, so nothing below can be
        // answered without one. Checked here rather than in every provider's matchers,
        // where an omission is invisible.
        if(context.deviceId == NO_DEVICE || context.deviceProperties.gcnArchName.empty())
        {
            const auto* reason = context.deviceId == NO_DEVICE
                                     ? "no device resolved"
                                     : "resolved device reports no gcnArchName";
            HIPDNN_PLUGIN_LOG_INFO("ingestor: " << reason << "; no kernel applies");
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

    /// One graph-scoped criterion's verdict for one (graph, device); memoized per
    /// matcher so a pack's matchers are evaluated only once each.
    struct GraphMatcherVerdict
    {
        bool passed = false;
    };
    using GraphMatcherMemo
        = std::unordered_map<DescriptorId, GraphMatcherVerdict, DescriptorIdHash>;

    /// Runs @p context's graph_match once (lazily, on the first pack that clears the
    /// arch gate; absent means the engine binds nothing and always proceeds with an
    /// empty map) and every pack's UMDs: graph-scoped criteria (memoized across
    /// packs), then kernel-scoped ones.
    Catalog buildCatalog(const MatchContext& context) const
    {
        Catalog catalog;
        GraphMatcherMemo graphVerdicts;
        std::optional<std::optional<BoundTokens>> graphMatch;

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

            if(!graphMatch.has_value())
            {
                graphMatch = _graphMatchFn == nullptr ? std::optional<BoundTokens>(BoundTokens{})
                                                      : _graphMatchFn(context);
                if(graphMatch->has_value())
                {
                    catalog.bound = std::move(**graphMatch);
                }
            }

            if(!graphMatch->has_value())
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: engine declined graph_match for device "
                                       << context.deviceId);
                return catalog;
            }

            if(!graphLevelMatchersPass(pack, context, graphVerdicts, catalog.bound))
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " declined at a graph-scoped matcher");
                continue;
            }

            size_t admitted = 0;
            for(const auto& precomputed : _definitions[packIndex])
            {
                // The pack gate above answered for the pack's own list; a kernel that
                // narrowed itself still has to be asked. Restating it for an unrestricted
                // kernel is one empty-list test, and the alternative -- trusting the pack
                // gate for some kernels and not others -- is the kind of conditional that
                // stops being true the next time this loop changes.
                if(!archSupports(precomputed.arch, context.deviceProperties.gcnArchName))
                {
                    HIPDNN_PLUGIN_LOG_INFO("ingestor: kernel "
                                           << toString(precomputed.kernelId)
                                           << " does not support device arch '"
                                           << context.deviceProperties.gcnArchName << "'");
                    continue;
                }

                // Matched in place and copied only when admitted: every field was settled at
                // construction and the matchers take it by const reference. Copying first cost
                // one full KernelDefinition per kernel per query -- nearly all of an
                // applicability check's time, since almost every kernel is then rejected.
                if(kernelLevelMatchersPass(pack, context, catalog.bound, precomputed))
                {
                    catalog.entries.push_back(precomputed);
                    ++admitted;
                }
            }

            if(admitted == 0)
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " admitted no kernel of "
                                                         << pack.kernels.size()
                                                         << " at the arch gate or a "
                                                            "kernel-scoped matcher");
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

    bool graphLevelMatchersPass(const KernelDescriptorPack& pack,
                                const MatchContext& context,
                                GraphMatcherMemo& graphVerdicts,
                                const BoundTokens& bound) const
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
                verdict.passed = matcher.graphFn(context, bound);
                memo = graphVerdicts.emplace(matcherId, std::move(verdict)).first;
            }

            if(!memo->second.passed)
            {
                return false;
            }
        }
        return true;
    }

    bool kernelLevelMatchersPass(const KernelDescriptorPack& pack,
                                 const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel) const
    {
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.descriptor.scope != MatchScope::KERNEL)
            {
                continue;
            }
            if(!matcher.kernelFn(context, bound, kernel))
            {
                return false;
            }
        }
        return true;
    }

    /// The measured order for @p context's graph and device, or nullopt when no record
    /// covers @p entries and the caller must rank as it always has.
    ///
    /// Full coverage is required: a partial record cannot order the rest, and
    /// interleaving measured with unmeasured entries would not be a valid order.
    std::optional<std::vector<KernelDefinition>>
        orderFromWinnerRecord(const std::vector<KernelDefinition>& entries,
                              const MatchContext& context) const
    {
        // Cheap rejection first: mightHaveWinnerFor() accounts for an on-disk shard this
        // process has not read yet, unlike a bare winnerCacheSize() check.
        if(entries.empty() || !mightHaveWinnerFor(context.deviceProperties.gcnArchName))
        {
            return std::nullopt;
        }

        const WinnerKey key{
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{context.graph},
            DeviceKey{context.deviceProperties}};
        if(!key.graph.isUsable())
        {
            // No bytes to key on; such graphs never match each other either.
            return std::nullopt;
        }

        const auto record = winnerFor(key);
        if(!record.has_value())
        {
            return std::nullopt;
        }

        auto ordered = orderIfFullyCovered(*record, entries);
        if(ordered.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: ordered " << ordered->size()
                                                        << " catalog entries from a benchmarked "
                                                           "record; heuristic ranking skipped");
        }
        return ordered;
    }


    /// Loads the on-disk shard covering @p gcnArchName into `_winnerCache` once, tracked
    /// by `_loadedWinnerShards`. File I/O runs with `_winnerCacheMutex` UNHELD, so a slow
    /// disk read never blocks an unrelated call.
    ///
    /// The latch is keyed on the shard's own arch component, not the raw `gcnArchName`:
    /// `stripArchFeatures()` maps several raw arch strings onto one shard, so keying it
    /// raw would decode the same file once per variant.
    ///
    /// Only a shard that was actually read -- or one deterministically declined, which a
    /// version mismatch is for the life of the process -- is latched. A transient open or
    /// read failure is left unlatched so a later lookup retries, rather than disabling
    /// this arch's cache for the manager's lifetime over one blip.
    void loadShardIfAbsent(const std::string& gcnArchName) const
    {
        const std::string shardArch(stripArchFeatures(gcnArchName));

        if(_engine.name.empty())
        {
            // No engine name means no shard path; mark it loaded so later lookups take
            // the fast in-memory-only path.
            const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
            _loadedWinnerShards.insert(shardArch);
            return;
        }

        {
            const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
            if(_loadedWinnerShards.find(shardArch) != _loadedWinnerShards.end())
            {
                return;
            }
        }

        std::vector<std::pair<WinnerKey, WinnerRecord>> decoded;
        bool attemptSettled = false;
        auto [shard, openStatus] = openWinnerCacheShard(_engine, gcnArchName);
        if(openStatus == hipdnn_data_sdk::utilities::LineStoreStatus::VERSION_MISMATCH)
        {
            // Deterministic for this process: the shard's version line will not change
            // under us, so there is nothing to retry.
            attemptSettled = true;
            logShardFailureOnce(WinnerShardFailureKind::VERSION_MISMATCH, gcnArchName);
        }
        else if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            logShardFailureOnce(WinnerShardFailureKind::OPEN, gcnArchName);
        }
        else
        {
            auto [records, readStatus]
                = hipdnn_data_sdk::utilities::readAllLines(*shard, &decodeWinnerRecordLine);
            if(readStatus == hipdnn_data_sdk::utilities::LineStoreStatus::OK)
            {
                attemptSettled = true;
                decoded = std::move(records);
            }
            else
            {
                logShardFailureOnce(WinnerShardFailureKind::READ, gcnArchName);
            }
        }

        {
            const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
            if(!attemptSettled)
            {
                return;
            }
            if(!_loadedWinnerShards.insert(shardArch).second)
            {
                // Another thread's read-through raced this one and already merged.
                return;
            }
        }

        // Walk in reverse and never overwrite: within the file the last line for a key
        // wins, and a key already in memory was put there by this process's own
        // measurement, which is newer than anything the file can offer.
        //
        // Merged outside `_winnerCacheMutex` now that `_winnerCache` locks itself. The latch
        // above is already claimed, so no second thread reaches this loop for this shard, and
        // a concurrent recordWinner() for one of these keys wins on its own merits --
        // putIfAbsent declines to overwrite it, which is the rule this loop always had.
        for(auto it = decoded.rbegin(); it != decoded.rend(); ++it)
        {
            if(!it->first.graph.isUsable())
            {
                // A key whose graph carries no content is not even equal to itself
                // (GraphContentKey::operator==), so it can never be looked up -- and as
                // a non-reflexive key it would violate the cache's requirements on KeyEqual.
                // recordWinner() rejects these on the write side too.
                continue;
            }
            // A shard holding more lines than the cache's capacity keeps the LAST ones it
            // decodes, which are the file's EARLIEST: reverse iteration means capacity is
            // spent on whichever lines happen to come first. That is arbitrary but never
            // wrong -- an evicted key simply misses and is re-measured, and the shard still
            // holds every line for a later process to read.
            _winnerCache.putIfAbsent(it->first, std::move(it->second));
        }
    }

    /// Write-back: re-reads @p key's shard under its LineStore lock, then applies
    /// @p cause's rule, as one critical section under the shard's own lock (mirrors
    /// `LineStoreLockHelper.cpp`). A fresh miss adopts any existing on-disk record for
    /// @p key as-is and appends nothing; a coverage-triggered re-benchmark always appends
    /// @p record, superseding the old entry under the reader's last-line-wins merge. The
    /// file lock is used rather than `_winnerCacheMutex`, because the racing writer may
    /// be another process.
    ///
    /// A record is never immutable: a shard may hold several lines for one key, and the
    /// reader resolves them last-line-wins (see `loadShardIfAbsent()`), so appending a
    /// newer ranking is how a record is replaced. Without that, a catalog that gains a
    /// kernel could never satisfy the coverage gate again, and every later run would
    /// re-benchmark and discard the result forever.
    ///
    /// @return The on-disk record for @p key, if one already exists AND @p cause is a fresh
    ///     miss -- adopted as-is, so nothing is written; otherwise @p record itself, both
    ///     when it was appended and on every disk failure, since write-back is
    ///     best-effort and the caller's own measurement is the right value to keep in
    ///     memory.
    WinnerRecord
        writeBackToShard(const WinnerKey& key, WinnerRecord record, WinnerWriteCause cause) const
    {
        if(_engine.name.empty())
        {
            return record;
        }

        const auto& gcnArchName = key.device.properties().gcnArchName;
        auto [shard, openStatus] = openWinnerCacheShard(_engine, gcnArchName);
        if(openStatus == hipdnn_data_sdk::utilities::LineStoreStatus::VERSION_MISMATCH)
        {
            logShardFailureOnce(WinnerShardFailureKind::VERSION_MISMATCH, gcnArchName);
            return record;
        }
        if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            logShardFailureOnce(WinnerShardFailureKind::OPEN, gcnArchName);
            return record;
        }

        if(hipdnn_data_sdk::utilities::lockLineStore(*shard)
           != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            logShardFailureOnce(WinnerShardFailureKind::LOCK, gcnArchName);
            return record;
        }

        const auto [existing, readStatus]
            = hipdnn_data_sdk::utilities::readAllLines(*shard, &decodeWinnerRecordLine);
        if(readStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            hipdnn_data_sdk::utilities::unlockLineStore(*shard);
            logShardFailureOnce(WinnerShardFailureKind::READ, gcnArchName);
            return record;
        }

        // Last-line-wins, mirroring the reader's merge order: compare against the LAST
        // line for this key, not the first. Once a shard can hold a superseded line, the
        // first match is stale and comparing against it would wrongly adopt it.
        const WinnerRecord* onDiskWinner = nullptr;
        for(const auto& [existingKey, existingRecord] : existing)
        {
            if(existingKey == key)
            {
                onDiskWinner = &existingRecord;
            }
        }

        if(onDiskWinner != nullptr && cause == WinnerWriteCause::FRESH_MISS)
        {
            // Presence, not equality. Two writers racing the same fresh key measure the same
            // candidates and rank them the same way up to noise, so comparing rankings let both
            // append and the shard kept two lines for one graph. A re-benchmark forced by a record
            // that no longer covers the candidate set is the only writer that supersedes.
            hipdnn_data_sdk::utilities::unlockLineStore(*shard);
            return *onDiskWinner;
        }

        const auto appendStatus
            = hipdnn_data_sdk::utilities::appendLine(*shard, encodeWinnerRecordLine(key, record));
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
        if(appendStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            logShardFailureOnce(WinnerShardFailureKind::APPEND, gcnArchName);
        }
        return record;
    }

    enum class WinnerShardFailureKind
    {
        OPEN,
        LOCK,
        READ,
        APPEND,
        VERSION_MISMATCH,
    };

    /// Logs @p kind once per manager instance, never per call. Every kind here is a
    /// disk-level decline, never a throw; the caller has already fallen back to
    /// in-memory-only behavior by the time this runs.
    void logShardFailureOnce(WinnerShardFailureKind kind, const std::string& gcnArchName) const
    {
        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        if(!_loggedShardFailureKinds.insert(kind).second)
        {
            return;
        }
        switch(kind)
        {
        case WinnerShardFailureKind::OPEN:
            HIPDNN_PLUGIN_LOG_INFO("ingestor: on-disk winner cache could not be opened for "
                                   "engine '"
                                   << _engine.name << "' arch '" << gcnArchName
                                   << "'; on-disk miss, in-memory behavior continues");
            break;
        case WinnerShardFailureKind::LOCK:
            HIPDNN_PLUGIN_LOG_INFO(
                "ingestor: could not lock the on-disk winner cache shard for engine '"
                << _engine.name << "' arch '" << gcnArchName
                << "'; on-disk write-back skipped for this call");
            break;
        case WinnerShardFailureKind::READ:
            HIPDNN_PLUGIN_LOG_INFO("ingestor: on-disk winner cache for engine '"
                                   << _engine.name << "' arch '" << gcnArchName
                                   << "' could not be read; on-disk miss, in-memory behavior "
                                      "continues");
            break;
        case WinnerShardFailureKind::APPEND:
            HIPDNN_PLUGIN_LOG_INFO(
                "ingestor: could not append a benchmarked ranking to the on-disk winner "
                "cache for engine '"
                << _engine.name << "' arch '" << gcnArchName
                << "'; the ranking is kept in-memory only for this process");
            break;
        case WinnerShardFailureKind::VERSION_MISMATCH:
            HIPDNN_PLUGIN_LOG_WARN("ingestor: on-disk winner cache for engine '"
                                   << _engine.name << "' arch '" << gcnArchName
                                   << "' declined: version mismatch; on-disk miss, in-memory "
                                      "behavior continues");
            break;
        default:
            // Unreachable; present because clang-tidy requires an explicit default.
            break;
        }
    }

    MetadataSchema _schema;
    std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash> _matchers;
    std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash> _dispatches;
    std::vector<KernelDescriptorPack> _packs;
    /// One entry per pack, parallel to _packs: its kernels' context-independent
    /// definitions, completed once at construction.
    std::vector<std::vector<KernelDefinition>> _definitions;
    /// Per non-integer metadata field, its value set in index order: the table that makes a
    /// string, bool, float or int_list field addressable by an int64 knob. Fixed at
    /// construction, like _definitions, because the engine's kernels are.
    std::map<std::string, std::vector<MetadataValue>> _ordinalDomains;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    GraphMatchFn _graphMatchFn = nullptr;
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
    /// The full applicable catalog's calibrated ranking, memoized so that a knob-pinned
    /// request restricts it instead of ranking its own subset (RFC 0019 §9.2). Every
    /// request fills it, pinned or not: the ranking is always taken over the full catalog,
    /// so what lands here does not depend on which request arrived first.
    mutable LruCache<CatalogKey, CalibratedRanking, CatalogKeyHash> _calibratedCache;

    /// The engine's identity: its scoped name and revision plus its UHD's id and content
    /// hash. The name locates the on-disk winner-cache shard -- empty disables the disk
    /// cache for this manager entirely -- and the revision is the version half of every
    /// CatalogKey this manager builds.
    EngineIdentity _engine;

    /// Bounded and self-locking (see DEFAULT_WINNER_CACHE_CAPACITY). Its own cache rather
    /// than a share of _catalogCache's: a shared bound would let a burst of benchmarked
    /// rankings evict the catalogs that ordinary lookups depend on, and vice versa.
    mutable LruCache<WinnerKey, WinnerRecord, WinnerKeyHash> _winnerCache;
    /// Guards the bookkeeping AROUND _winnerCache -- the two sets below -- not the cache
    /// itself, which locks internally.
    mutable std::mutex _winnerCacheMutex;
    /// Shard arch components (`stripArchFeatures(gcnArchName)`) whose on-disk shard has
    /// been read, or deterministically declined; see loadShardIfAbsent(). Guarded by
    /// _winnerCacheMutex.
    mutable std::unordered_set<std::string> _loadedWinnerShards;
    /// Failure kinds already logged once (see logShardFailureOnce()); guarded by
    /// _winnerCacheMutex.
    mutable std::unordered_set<WinnerShardFailureKind> _loggedShardFailureKinds;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
