// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
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

/// What a plan needs to size and launch one selected kernel; holds its own copy of the
/// definition so it does not pin the state manager's internals.
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

/// The graph-level half of a catalog build, run up to and including graph_match: what
/// the per-variant half needs to finish it. Packs before `firstPack` were declined at
/// the arch gate and logged as such.
struct CatalogGraphStage
{
    /// The first pack whose arch supports the device; the pack count when none does.
    size_t firstPack = 0;
    /// graph_match's tokens; nullopt when it declined or never ran (no pack supports the
    /// device).
    std::optional<BoundTokens> bound;
    /// Kernels the per-variant half will examine: every kernel, in a pack from
    /// `firstPack` on, whose own arch and whose pack's arch both support the device.
    /// Graph-scoped criteria may still decline some of those packs.
    size_t predictedVariants = 0;
};

/// A completed metadata tuple reduced to its values, in the tuple's own (name-sorted)
/// order. completeMetadata() has already proven every tuple of an engine carries exactly
/// the KMD's field names, so for comparing two of them the names are redundant, and
/// leaving them out keeps hashing and comparing a key to the values alone.
using MetadataTupleValues = std::vector<MetadataValue>;

struct MetadataTupleValuesHash
{
    size_t operator()(const MetadataTupleValues& values) const
    {
        size_t seed = values.size();
        const auto combine = [&seed](size_t hash) {
            seed ^= hash + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        for(const auto& value : values)
        {
            combine(value.index());
            std::visit(
                [&combine](const auto& held) {
                    using Held = std::decay_t<decltype(held)>;
                    if constexpr(std::is_same_v<Held, std::vector<int64_t>>)
                    {
                        combine(held.size());
                        for(const auto element : held)
                        {
                            combine(std::hash<int64_t>{}(element));
                        }
                    }
                    else
                    {
                        combine(std::hash<Held>{}(held));
                    }
                },
                value);
        }
        return seed;
    }
};

/// The engine's view of its own kernels: which apply to a graph, in what order, and
/// how to launch one. Answers isApplicable (unsortedDefinitions non-empty), getDetails
/// (sortedDefinitions), getMaxWorkspaceSize (dispatchHandlerFor per survivor, max), and
/// initializeExecutionContext (sortedDefinitions().front(), getDispatchDetails).
///
/// Thread safety. `_catalogCache`, `_calibratedCache`, and `_winnerCache` are `LruCache`s that
/// synchronize internally; `_winnerCacheMutex` guards the per-shard "loaded" set and the
/// one-shot failure-log set.
///
/// `_winnerCacheMutex` may be held across a `_winnerCache` call, never the reverse (an
/// `LruCache` calls no user code), so they cannot deadlock. Matchers, the heuristic, and the
/// accessors below run outside every lock.
///
/// `finishCatalog` is the staged form of the per-variant loop and may run on a worker
/// thread, concurrently with other engines' `finishCatalog` calls and with the caller
/// thread. A matcher or graph criterion is a plain function of its const arguments and
/// must stay callable from any thread at once: any state it reaches beyond those
/// arguments needs its own synchronization.
///
/// Everything between a lookup and a store is thread-local: `catalogFor` returns a
/// `Catalog` by value and callers mutate that copy, so ordering a catalog touches no
/// shared state. Returning a reference into `_catalogCache` instead would make those
/// mutations a data race. `winnerFor` returns a copy for the same reason. The shared
/// `Catalog::measuredRecord` points to const and is never written after creation.
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

    /// How many benchmarked rankings to retain in memory (RFC 0019 §9.2). An evicted record is
    /// re-read from the on-disk shard, so eviction costs at most one redundant sweep.
    static constexpr size_t DEFAULT_WINNER_CACHE_CAPACITY = 4096;

    /// @throws std::invalid_argument bad pack reference, or two kernels sharing a metadata tuple
    ///         that compete at the same arch tier on a device.
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
    /// @param engine The engine's identity; composes the on-disk winner-cache shard path and
    ///        the version half of every `CatalogKey`. An empty name disables the disk cache.
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
        // Same key and capacity as _catalogCache: both answer for (graph, device, engine
        // version). Kept outside `Catalog` because no matcher, scorer or formula reads it.
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

    // Pinned in place: catalog pointers into `_definitions` rely on a stable address.
    KernelIngestorStateManager(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager& operator=(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager(KernelIngestorStateManager&&) = delete;
    KernelIngestorStateManager& operator=(KernelIngestorStateManager&&) = delete;

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
    /// @param full Every kernel the matchers admitted, before any knob pin; also carries the
    ///        bound tokens the scorer reads.
    /// @param filtered What survived the knob pin and workspace limit, in @p full's order.
    /// @param modelId Set to the id of the model that produced the scores.
    ///
    /// Always ranks @p full and restricts the result to @p filtered (RFC 0019 §5 step 8,
    /// §9.2), so a pinned and an unpinned request cannot disagree on the winner.
    std::vector<ScoredKernel>
        calibratedRanking(const Catalog& full,
                          const std::vector<const KernelDefinition*>& filtered,
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
            // Not memoized when empty: empty means "cannot calibrate", and caching it would
            // make a miss and a hit indistinguishable.
            _calibratedCache.put(*key, CalibratedRanking{ranking, modelId});
        }
        return restrictRanking(ranking, filtered);
    }

    /// The model calibratedRanking() would use for @p metric on @p arch, from bindings alone.
    std::string calibratedModelId(const std::string& metric, const std::string& arch) const
    {
        return _heuristic->calibratedModelId(metric, arch);
    }

    /// Every kernel that applies to the graph and device @p context names, unordered.
    std::vector<const KernelDefinition*> unsortedDefinitions(const MatchContext& context) const
    {
        return catalogFor(context).entries;
    }

    /// The unranked catalog and the state matching bound, from one lookup.
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        return catalogFor(context);
    }

    /// nullopt when the graph supplies no usable content key (no or unverifiable
    /// `bytes()`); such a graph is rematched on every call. Device comes from the
    /// context, not a separate argument, so one device's catalog never caches under
    /// another's key. The engine version comes from this manager. The ranking metric is
    /// part of the key because both orders depend on it (RFC 0019 §11.4); the registry's
    /// own name is stored because the key outlives the request.
    std::optional<CatalogKey> cacheKey(const MatchContext& context) const
    {
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graph{context.graph};
        const auto* metric = hipdnn_data_sdk::utilities::findRankingMetric(context.rankingMetric);
        if(!graph.isUsable() || metric == nullptr)
        {
            return std::nullopt;
        }

        const auto& flatGraph = context.graph.getGraph();
        std::vector<int64_t> tensorUids;
        if(const auto* tensors = flatGraph.tensors(); tensors != nullptr)
        {
            tensorUids.reserve(tensors->size());
            for(const auto* tensor : *tensors)
            {
                tensorUids.push_back(tensor->uid());
            }
        }
        return CatalogKey{std::move(graph),
                          CatalogGraphFields::of(flatGraph),
                          std::move(tensorUids),
                          context.deviceId,
                          _engine.version,
                          metric->name};
    }

    // The staged entry points below answer what `!unsortedDefinitions(context).empty()`
    // answers, in pieces a caller can spread across threads. Each expects a context that
    // already passed catalogFor's device guard, and @p key built by cacheKey() from it.

    /// The cached verdict for @p key, read in place without copying the catalog;
    /// nullopt on a miss.
    std::optional<bool> cachedVerdict(const MatchContext& context, const CatalogKey& key) const
    {
        auto verdict = _catalogCache.visit(
            key, [](const Catalog& catalog) { return !catalog.entries.empty(); });
        if(verdict.has_value())
        {
            HIPDNN_PLUGIN_LOG_TRACE("ingestor: catalog cache hit for device " << context.deviceId);
        }
        return verdict;
    }

    /// The graph-level half of a miss. nullopt when it alone settles the catalog as
    /// empty -- no pack supports the device, or graph_match declined -- which is then
    /// cached exactly as catalogFor would cache it; otherwise what finishCatalog needs.
    std::optional<CatalogGraphStage> beginCatalog(const MatchContext& context,
                                                  const CatalogKey& key) const
    {
        CatalogGraphStage stage = matchGraph(context);
        if(stage.firstPack == _packs.size())
        {
            finishCatalog(context, key, std::move(stage));
            return std::nullopt;
        }
        if(!stage.bound.has_value())
        {
            _catalogCache.putIfAbsent(key, Catalog{});
            return std::nullopt;
        }
        return stage;
    }

    /// The per-variant half of a miss: completes @p stage's catalog, caches it as
    /// catalogFor would, and returns whether any kernel applies. Makes no HIP calls.
    bool finishCatalog(const MatchContext& context,
                       const CatalogKey& key,
                       CatalogGraphStage stage) const
    {
        Catalog catalog = matchVariants(context, std::move(stage));
        const bool applicable = !catalog.entries.empty();
        // putIfAbsent, as in catalogFor: a racing writer may already hold a sorted one.
        _catalogCache.putIfAbsent(key, std::move(catalog));
        return applicable;
    }

    /// Whether any kernel applies to a graph that yields no usable key; matched in full
    /// and never cached, as catalogFor does for such a graph.
    bool applicableUncached(const MatchContext& context) const
    {
        return !catalogFor(context, std::nullopt).entries.empty();
    }

    /// Matching only, in descriptor-ID order, independent of heuristic/winner state, so a
    /// page walk keeps its order when another caller benchmarks the catalog. Ids are unique
    /// within one device's catalog (validateAndIndexPacks).
    Catalog enumerableCatalog(const MatchContext& context) const
    {
        auto catalog = catalogFor(context);
        std::sort(catalog.entries.begin(),
                  catalog.entries.end(),
                  [](const KernelDefinition* lhs, const KernelDefinition* rhs) {
                      return lhs->kernelId < rhs->kernelId;
                  });
        return catalog;
    }

    /// Every kernel that applies to the graph and device @p context names, best first.
    std::vector<const KernelDefinition*> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalog(context).entries;
    }

    /// The ordered catalog and the state matching bound, from one lookup.
    ///
    /// A benchmarked record covering the whole catalog supplies a measured order and
    /// `rank()` is never called; otherwise the heuristic orders it.
    Catalog sortedCatalog(const MatchContext& context) const
    {
        // Mirrors catalogFor's own reject guard: cacheKey() below reads graph content
        // and device ordinal, not arch, so without this an unresolved-arch context would
        // cache an empty catalog under the SAME key a later, resolved call for this
        // device reuses -- permanently hiding that device's real catalog.
        if(context.deviceId == NO_DEVICE || context.deviceProperties.gcnArchName.empty())
        {
            return catalogFor(context);
        }

        const auto key = cacheKey(context);
        Catalog catalog = measuredCatalogFor(context, key);
        if(catalog.isSorted)
        {
            return catalog;
        }

        catalog.entries = _heuristic->rank(catalog, context);
        catalog.isSorted = true;
        if(key.has_value())
        {
            // put, not putIfAbsent: sorted is strictly better than whatever is cached.
            _catalogCache.put(*key, catalog);
        }
        return catalog;
    }

    /// The catalog in measured order, with `measuredRecord` set, when a benchmarked record
    /// covers all of it; otherwise the catalog as cached. Never calls `rank()`.
    ///
    /// Shared by plan build and configuration prediction so they agree that measurement
    /// outranks estimate (RFC 0019 §5 step 9). An adopted record is cached with its order, so
    /// later callers do not depend on the winner cache still holding it.
    Catalog measuredCatalog(const MatchContext& context) const
    {
        // Same guard as sortedCatalog: an unresolved device must not cache under a key a
        // later, resolved call for this device reuses.
        if(context.deviceId == NO_DEVICE || context.deviceProperties.gcnArchName.empty())
        {
            return catalogFor(context);
        }
        return measuredCatalogFor(context, cacheKey(context));
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

        // LRU (RFC 0019 §9.2): a graph executed repeatedly keeps its ranking.
        _winnerCache.put(key, std::move(adopted));
    }

    /// The ranking recorded for @p key, or nullopt, as a copy. A lookup refreshes @p key's
    /// recency.
    ///
    /// Read-through: an in-memory miss loads @p key's on-disk shard once per shard (a
    /// per-shard flag guarded by `_winnerCacheMutex`); it never holds a file lock.
    std::optional<WinnerRecord> winnerFor(const WinnerKey& key) const
    {
        if(auto found = _winnerCache.get(key); found.has_value())
        {
            return found;
        }

        loadShardIfAbsent(key.device.properties().gcnArchName);

        return _winnerCache.get(key);
    }

    /// How many rankings are held (at most the winner-cache capacity).
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
        // size(), not empty(): LruCache exposes no empty().
        return _winnerCache.size() != 0
               || _loadedWinnerShards.find(std::string(stripArchFeatures(gcnArchName)))
                      == _loadedWinnerShards.end();
    }

    /// The handler that sizes and launches @p kernel; looked up in place, without
    /// copying the definition.
    /// @throws std::runtime_error if the kernel's dispatch descriptor is unknown.
    const IKernelDispatchHandler<THandle>* dispatchHandlerFor(const KernelDefinition& kernel) const
    {
        auto it = _dispatches.find(kernel.dispatchId);
        if(it == _dispatches.end())
        {
            throw std::runtime_error("kernel '" + toString(kernel.kernelId)
                                     + "' names unknown dispatch descriptor '"
                                     + toString(kernel.dispatchId) + "'");
        }
        return it->second.handler;
    }

    /// Resolves how to size and launch @p kernel, copying the definition for a plan to
    /// own.
    /// @throws std::runtime_error if the kernel's dispatch descriptor is unknown.
    KernelDispatcher<THandle> getDispatchDetails(const KernelDefinition& kernel) const
    {
        const auto* handler = dispatchHandlerFor(kernel);
        return {kernel, handler};
    }

    /// The distinct values @p field takes across @p kernels, in ranked-first order.
    static std::vector<MetadataValue>
        knobValues(const std::vector<const KernelDefinition*>& kernels, const std::string& field)
    {
        std::vector<MetadataValue> values;
        for(const KernelDefinition* kernel : kernels)
        {
            const auto value = kernel->tryGetMetadata(field);
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
    /// An `INT` field addresses itself; other types use their index in the field's ordinal
    /// domain (see buildOrdinalDomains). No fallback index: it would address a wrong kernel.
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

    /// Does @p kernel's value for @p field carry the pinned ordinal? A kernel lacking the field
    /// does not match.
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
    /// non-integer field some kernel carries.
    bool isOrdinalKnob(const std::string& field) const
    {
        return _ordinalDomains.find(field) != _ordinalDomains.end();
    }

private:
    /// Orders @p catalog by the record for @p context's graph and device and attaches that
    /// record, when the record fully covers @p catalog; false, leaving @p catalog untouched,
    /// otherwise.
    ///
    /// @p catalogKey is the catalog key built from @p context. Its graph content is the
    /// winner key's too, so reusing it spares a second verified copy and hash of the graph.
    /// A graph with no key has no bytes to key a record on, and such graphs never match
    /// each other either.
    bool adoptCoveringRecord(Catalog& catalog,
                             const MatchContext& context,
                             const std::optional<CatalogKey>& catalogKey) const
    {
        const auto& entries = catalog.entries;
        // Cheap rejection first: mightHaveWinnerFor() accounts for an on-disk shard this
        // process has not read yet, unlike a bare winnerCacheSize() check.
        if(entries.empty() || !catalogKey.has_value()
           || !mightHaveWinnerFor(context.deviceProperties.gcnArchName))
        {
            return false;
        }

        const WinnerKey key{catalogKey->graph, DeviceKey{context.deviceProperties}};

        auto record = winnerFor(key);
        if(!record.has_value())
        {
            return false;
        }

        auto ordered = orderIfFullyCovered(*record, entries);
        if(!ordered.has_value())
        {
            return false;
        }
        HIPDNN_PLUGIN_LOG_INFO("ingestor: ordered " << ordered->size()
                                                    << " catalog entries from a benchmarked "
                                                       "record; heuristic ranking skipped");
        catalog.entries = std::move(*ordered);
        catalog.measuredRecord = std::make_shared<const WinnerRecord>(std::move(*record));
        catalog.isSorted = true;
        return true;
    }

    /// measuredCatalog() for a caller that already built @p key from @p context.
    Catalog measuredCatalogFor(const MatchContext& context,
                               const std::optional<CatalogKey>& key) const
    {
        Catalog catalog = catalogFor(context, key);
        // A heuristic order is provisional (a sweep can postdate it), so look up a record
        // even when already sorted. An unresolved device's empty catalog is never covered.
        if(catalog.measuredRecord == nullptr && adoptCoveringRecord(catalog, context, key))
        {
            if(key.has_value())
            {
                // put, not putIfAbsent: a measured order replaces a provisional one.
                _catalogCache.put(*key, catalog);
            }
        }
        return catalog;
    }

    /// One memoized full-catalog calibrated ranking, with the id of the model that produced it.
    struct CalibratedRanking
    {
        std::vector<ScoredKernel> ranking;
        std::string modelId;
    };

    /// @p ranking in its own order, keeping only entries naming a kernel in @p candidates.
    /// Dropping rows without reordering is what makes filtering and ranking commute
    /// (RFC 0019 §5 step 8).
    static std::vector<ScoredKernel>
        restrictRanking(const std::vector<ScoredKernel>& ranking,
                        const std::vector<const KernelDefinition*>& candidates)
    {
        std::vector<ScoredKernel> restricted;
        restricted.reserve(std::min(ranking.size(), candidates.size()));
        for(const auto& scored : ranking)
        {
            const bool survived = std::any_of(
                candidates.begin(), candidates.end(), [&scored](const KernelDefinition* kernel) {
                    return kernel->kernelId == scored.kernelId;
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
    /// only place they are ever computed: buildCatalog references them per query rather
    /// than completing each kernel's metadata again on every graph.
    void validateAndIndexPacks()
    {
        // Two kernels may share a tuple when no single device sees both at the same arch
        // tier (per-arch shards, or a more specific kernel that buildCatalog shadows
        // per device). Uniqueness is per competing-arch group, keyed by the tuple's values
        // in a hash map rather than scanned, which would be quadratic.
        size_t kernelCount = 0;
        for(const auto& pack : _packs)
        {
            kernelCount += pack.kernels.size();
        }
        std::unordered_map<MetadataTupleValues,
                           std::vector<std::vector<std::string>>,
                           MetadataTupleValuesHash>
            archesClaimingTuple;
        archesClaimingTuple.reserve(kernelCount);
        // Rankings and enumeration name a candidate by kernel id, so an id is unique per
        // overlapping-arch group too.
        std::unordered_map<DescriptorId, std::vector<std::vector<std::string>>, DescriptorIdHash>
            archesClaimingId;
        archesClaimingId.reserve(kernelCount);
        // Records @p arch unless @p conflicts finds a claimant it collides with: any shared
        // device for an id, the same arch tier on a shared device for a tuple.
        const auto claim = [](std::vector<std::vector<std::string>>& claimants,
                              const std::vector<std::string>& arch,
                              const auto& conflicts) {
            const bool taken = std::any_of(
                claimants.begin(), claimants.end(), [&arch, &conflicts](const auto& claimed) {
                    return conflicts(claimed, arch);
                });
            if(!taken)
            {
                claimants.push_back(arch);
            }
            return !taken;
        };

        _definitions.reserve(_packs.size());
        _kernelsByArch.reserve(_packs.size());
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
            // What this pack's native handler launches, read once per pack. The empty set is
            // a single-launch handler, which reads `kernel_source`.
            const std::set<std::string> handlerLaunches
                = _dispatches.find(pack.dispatchId)->second.handler->launchNames();
            for(const auto& kernel : pack.kernels)
            {
                const auto isAdaptable = [](const KernelSource& source) {
                    return source.kind == KernelSourceKind::EMBEDDED_SOURCE
                           || source.kind == KernelSourceKind::KPACK;
                };
                // A multi-launch kernel keeps no `source`: every launch entry answers for
                // itself, and one this build cannot adapt costs the whole kernel.
                if(kernel.sources.empty() ? !isAdaptable(kernel.source)
                                          : !std::all_of(kernel.sources.begin(),
                                                         kernel.sources.end(),
                                                         [&](const auto& launch) {
                                                             return isAdaptable(launch.second);
                                                         }))
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

                // RFC 0017 §8.6: an accepted kernel is always launchable. A handler launches
                // exactly the launches it declares, so a kernel carrying any other set --
                // fewer, more, or the same number under other names -- is dropped here,
                // before applicability, rather than failing at plan build.
                std::set<std::string> kernelLaunches;
                for(const auto& launch : kernel.sources)
                {
                    kernelLaunches.insert(launch.first);
                }
                if(kernelLaunches != handlerLaunches)
                {
                    const auto spell = [](const std::set<std::string>& names) {
                        if(names.empty())
                        {
                            return std::string("{} (one launch, 'kernel_source')");
                        }
                        std::string text = "{";
                        for(const auto& name : names)
                        {
                            text += (text.size() > 1 ? ", " : "") + name;
                        }
                        return text + "}";
                    };
                    HIPDNN_PLUGIN_LOG_ERROR(
                        "ingestor: " << describeDescriptor("kernel", kernel.name, kernel.id)
                                     << " carries the launches " << spell(kernelLaunches)
                                     << ", but the dispatch handler of "
                                     << describeDescriptor("pack", pack.name, pack.id)
                                     << " launches " << spell(handlerLaunches)
                                     << "; a variant with a different launch set belongs in its"
                                        " own pack, dropping it");
                    continue;
                }

                auto key = completeMetadata(kernel);
                // A kernel that declared no arch of its own runs wherever its pack does;
                // one that declared a narrower list claims only that. Claiming by the
                // kernel rather than the pack is what lets two kernels of ONE pack share a
                // tuple under disjoint arch -- one implementation per capability -- while
                // still catching two that compete at the same arch tier on a device.
                // Kernels at different tiers do not collide: the better tier wins.
                std::vector<std::string> kernelArch = kernel.arch.empty() ? pack.arch : kernel.arch;
                MetadataTupleValues tuple;
                tuple.reserve(key.size());
                for(const auto& entry : key)
                {
                    tuple.push_back(entry.second);
                }
                // try_emplace, not operator[], only because misc-const-correctness
                // misreads the operator[] form here and demands a const map.
                if(!claim(archesClaimingId.try_emplace(kernel.id).first->second,
                          kernelArch,
                          archOverlaps))
                {
                    throw std::invalid_argument(
                        "kernel '" + toString(kernel.id)
                        + "' is declared twice on an arch both declarations reach; a kernel id "
                        + "names one candidate and must be unique per device");
                }
                if(!claim(archesClaimingTuple.try_emplace(std::move(tuple)).first->second,
                          kernelArch,
                          archesCompete))
                {
                    throw std::invalid_argument(
                        "kernel '" + toString(kernel.id)
                        + "' duplicates the metadata tuple of another kernel under schema '"
                        + _schema.name
                        + "' at the same arch tier on a device both reach; a better tier "
                        + "shadows a worse one, so equal tiers cannot be told apart");
                }

                packDefinitions.push_back(KernelDefinition{kernel.id,
                                                           pack.id,
                                                           pack.dispatchId,
                                                           kernel.source,
                                                           std::move(key),
                                                           kernel.priority,
                                                           std::move(kernelArch),
                                                           kernel.originDirectory,
                                                           kernel.name,
                                                           kernel.treeRoot,
                                                           kernel.sources});
            }
            // Pushed even when every kernel was dropped: _definitions is indexed by pack
            // position, so skipping one entry would bind every later pack's definitions to
            // the wrong pack and run the last index out of range.
            _definitions.push_back(std::move(packDefinitions));

            std::vector<ArchKernelCount> counts;
            for(const auto& definition : _definitions.back())
            {
                const auto group
                    = std::find_if(counts.begin(), counts.end(), [&definition](const auto& count) {
                          return count.arch == definition.arch;
                      });
                if(group == counts.end())
                {
                    counts.push_back({definition.arch, 1});
                }
                else
                {
                    ++group->kernels;
                }
            }
            _kernelsByArch.push_back(std::move(counts));
        }

        buildOrdinalDomains();
    }

    /// The value set of every non-integer metadata field, in the order that numbers it
    /// (RFC 0019 §13.2: a knob is an int64, so non-`INT` fields are addressed by index).
    ///
    /// Spans every pack, not one device's catalog, so an index means the same kernel on every
    /// graph. Ordered by `std::variant`'s `<`; `uhd_gen/addressing.py` must derive the same
    /// indices.
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

        return catalogFor(context, cacheKey(context));
    }

    /// The lookup-or-build half of catalogFor, for a caller that already passed its
    /// device guard and built @p key from the same @p context.
    Catalog catalogFor(const MatchContext& context, const std::optional<CatalogKey>& key) const
    {
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
                "ingestor: graph supplies no content key, so its catalog cannot be cached");
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

    /// Runs @p context's graph_match once (absent means the engine binds nothing and
    /// always proceeds with an empty map) and every pack's UMDs: graph-scoped criteria
    /// (memoized across packs), then kernel-scoped ones. The two stages it composes are
    /// one pass over the packs, split where graph_match has run.
    Catalog buildCatalog(const MatchContext& context) const
    {
        CatalogGraphStage stage = matchGraph(context);
        if(stage.firstPack < _packs.size() && !stage.bound.has_value())
        {
            return Catalog{};
        }
        return matchVariants(context, std::move(stage));
    }

    /// The pack-level arch gate up to the first pack that clears it, then graph_match,
    /// run lazily there so an engine no pack of which supports the device never runs it.
    CatalogGraphStage matchGraph(const MatchContext& context) const
    {
        CatalogGraphStage stage;
        while(stage.firstPack < _packs.size() && !packSupportsDevice(stage.firstPack, context))
        {
            ++stage.firstPack;
        }
        if(stage.firstPack == _packs.size())
        {
            return stage;
        }

        stage.bound = _graphMatchFn == nullptr ? std::optional<BoundTokens>(BoundTokens{})
                                               : _graphMatchFn(context);
        if(!stage.bound.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: engine declined graph_match for device "
                                   << context.deviceId);
            return stage;
        }

        const auto& deviceArch = context.deviceProperties.gcnArchName;
        for(size_t packIndex = stage.firstPack; packIndex < _packs.size(); ++packIndex)
        {
            if(packIndex != stage.firstPack && !archSupports(_packs[packIndex].arch, deviceArch))
            {
                continue;
            }
            for(const auto& group : _kernelsByArch[packIndex])
            {
                if(archSupports(group.arch, deviceArch))
                {
                    stage.predictedVariants += group.kernels;
                }
            }
        }
        return stage;
    }

    /// The rest of the pass matchGraph began: from @p stage's first pack on, every pack's
    /// graph-scoped criteria and its kernels' arch and kernel-scoped matchers. With no
    /// pack supporting the device this only reports the empty catalog.
    Catalog matchVariants(const MatchContext& context, CatalogGraphStage stage) const
    {
        Catalog catalog;
        if(stage.bound.has_value())
        {
            catalog.bound = std::move(*stage.bound);
        }
        GraphMatcherMemo graphVerdicts;

        for(size_t packIndex = stage.firstPack; packIndex < _packs.size(); ++packIndex)
        {
            const auto& pack = _packs[packIndex];

            // matchGraph already cleared the first pack.
            if(packIndex != stage.firstPack && !packSupportsDevice(packIndex, context))
            {
                continue;
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

                // Referenced, not rebuilt or copied: every field was settled at
                // construction, and `_definitions` never changes after it.
                if(kernelLevelMatchersPass(pack, context, catalog.bound, precomputed))
                {
                    catalog.entries.push_back(&precomputed);
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

        shadowByArchTier(catalog, context);

        HIPDNN_PLUGIN_LOG_INFO("ingestor: catalog for device "
                               << context.deviceId << " holds " << catalog.entries.size()
                               << " kernel(s) from " << _packs.size() << " pack(s)");
        return catalog;
    }

    /// The pack-level arch gate, logging a decline.
    bool packSupportsDevice(size_t packIndex, const MatchContext& context) const
    {
        const auto& pack = _packs[packIndex];
        if(archSupports(pack.arch, context.deviceProperties.gcnArchName))
        {
            return true;
        }
        HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                 << " does not support device arch '"
                                                 << context.deviceProperties.gcnArchName << "'");
        return false;
    }

    /// Per metadata tuple, keeps only the entries with the best arch tier for the device.
    /// Runs after matcher admission, so a declined kernel never hides a lower-tier one.
    static void shadowByArchTier(Catalog& catalog, const MatchContext& context)
    {
        auto& entries = catalog.entries;
        if(entries.size() < 2)
        {
            return;
        }

        struct ByMetadata
        {
            bool operator()(const MetadataValues* lhs, const MetadataValues* rhs) const
            {
                return *lhs < *rhs;
            }
        };
        constexpr int SHADOWED = -1;
        std::vector<int> tiers;
        tiers.reserve(entries.size());
        std::map<const MetadataValues*, size_t, ByMetadata> winners;
        for(size_t i = 0; i < entries.size(); ++i)
        {
            tiers.push_back(
                static_cast<int>(archTier(entries[i]->arch, context.deviceProperties.gcnArchName)
                                     .value_or(ArchTier::UNRESTRICTED)));
            const auto [it, inserted] = winners.try_emplace(&entries[i]->metadata, i);
            if(!inserted && tiers[i] < tiers[it->second])
            {
                it->second = i;
            }
        }

        bool anyShadowed = false;
        for(size_t i = 0; i < entries.size(); ++i)
        {
            const size_t winner = winners.at(&entries[i]->metadata);
            if(tiers[i] != tiers[winner])
            {
                tiers[i] = SHADOWED;
                anyShadowed = true;
                HIPDNN_PLUGIN_LOG_INFO("ingestor: kernel " << toString(entries[i]->kernelId)
                                                           << " is shadowed by kernel "
                                                           << toString(entries[winner]->kernelId)
                                                           << " at a better arch tier on device "
                                                           << context.deviceProperties.gcnArchName);
            }
        }
        if(!anyShadowed)
        {
            return;
        }

        // Compacted in place, in order: the survivors keep their relative position.
        size_t kept = 0;
        for(size_t i = 0; i < entries.size(); ++i)
        {
            if(tiers[i] != SHADOWED)
            {
                entries[kept++] = entries[i];
            }
        }
        entries.resize(kept);
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

        if(_engine.name.empty() || !_engine.contentIdentified)
        {
            // Without a name there is no shard path, and without content identity no shard
            // tied to the model; latch so later lookups stay in-memory only.
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

        // Latch and merge under one hold, so a concurrent lookup never sees the shard as
        // loaded before its records are merged. `_winnerCache` never takes this mutex.
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

        // mergeAbsent() applies last-line-wins and keeps in-memory keys (newer than the file)
        // as one atomic step, so eviction during the merge cannot resurrect a stale ranking.
        std::vector<std::pair<WinnerKey, WinnerRecord>> newestFirst;
        newestFirst.reserve(decoded.size());
        for(auto it = decoded.rbegin(); it != decoded.rend(); ++it)
        {
            if(!it->first.graph.isUsable())
            {
                // A key whose graph carries no content is not even equal to itself
                // (GraphContentKey::operator==), so it can never be looked up -- and as
                // a non-reflexive key it would violate the cache's KeyEqual requirements.
                continue;
            }
            newestFirst.push_back(std::move(*it));
        }
        _winnerCache.mergeAbsent(std::move(newestFirst));
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
        if(_engine.name.empty() || !_engine.contentIdentified)
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
    /// definitions, completed once at construction and never mutated after, so the
    /// pointers every Catalog holds into it stay valid for this manager's lifetime.
    std::vector<std::vector<KernelDefinition>> _definitions;
    /// How many kernels a pack holds under one resolved arch list.
    struct ArchKernelCount
    {
        std::vector<std::string> arch;
        size_t kernels = 0;
    };
    /// Parallel to _packs: each pack's kernels counted by distinct arch list, so
    /// predicting a catalog build's work tests each list once rather than each kernel.
    std::vector<std::vector<ArchKernelCount>> _kernelsByArch;
    /// Per non-integer metadata field, its value set in index order, so the field is
    /// addressable by an int64 knob. Fixed at construction.
    std::map<std::string, std::vector<MetadataValue>> _ordinalDomains;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    GraphMatchFn _graphMatchFn = nullptr;
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
    /// The full catalog's calibrated ranking; a knob-pinned request restricts it instead of
    /// ranking its own subset (RFC 0019 §9.2).
    mutable LruCache<CatalogKey, CalibratedRanking, CatalogKeyHash> _calibratedCache;

    /// The engine's identity. The name locates the on-disk winner-cache shard (empty disables
    /// it); the revision is the version half of every CatalogKey.
    EngineIdentity _engine;

    /// Separate from _catalogCache so benchmarked rankings and catalogs cannot evict each other.
    mutable LruCache<WinnerKey, WinnerRecord, WinnerKeyHash> _winnerCache;
    /// Guards the two sets below, not _winnerCache, which locks internally.
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
