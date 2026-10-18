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
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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

/// One slot's selection state as a slotted state manager is constructed with it (RFC 0021 §3):
/// the per-slot counterpart of the schema, packs and ranker an unslotted engine passes directly.
struct KernelSlotInput
{
    /// The slot's name, unique within the engine.
    std::string name;
    MetadataSchema schema;
    /// The packs whose `slot` names this one; may be empty (the engine then declines).
    std::vector<KernelDescriptorPack> packs;
    std::shared_ptr<IKernelHeuristic> heuristic;
};

/// Whether an engine accepts one graph on one device, and why not (RFC 0021 §3 rule 2).
struct SlotProgramVerdict
{
    bool applicable = false;
    /// Why the engine declined; empty when it is applicable. For an invalid shared-workspace
    /// token or offset it names the token.
    std::string declineReason;
    /// The validated `shared_workspace` token value in bytes; 0 without a token.
    size_t sharedWorkspaceBytes = 0;
};

/// The alignment, in bytes, of the shared region of a slotted engine's workspace (RFC 0021 §6).
inline constexpr size_t SHARED_WORKSPACE_ALIGNMENT = 256;

/// The bytes the shared region occupies: @p sharedBytes rounded up to
/// SHARED_WORKSPACE_ALIGNMENT, which is the offset at which scratch begins.
constexpr size_t alignSharedWorkspace(size_t sharedBytes)
{
    return (sharedBytes + SHARED_WORKSPACE_ALIGNMENT - 1) / SHARED_WORKSPACE_ALIGNMENT
           * SHARED_WORKSPACE_ALIGNMENT;
}

/// The bound token a slotted engine's graph match publishes with the scratch offset: the
/// shared size rounded up to SHARED_WORKSPACE_ALIGNMENT.
inline constexpr const char* SCRATCH_OFFSET_TOKEN = "ws.scratch_offset";

/// The engine's view of its own kernels: which apply to a graph, in what order, and
/// how to launch one. Answers isApplicable (unsortedDefinitions non-empty), getDetails
/// (sortedDefinitions), getMaxWorkspaceSize (dispatchHandlerFor per survivor, max), and
/// initializeExecutionContext (sortedDefinitions().front(), getDispatchDetails).
///
/// Slots (RFC 0021). The manager holds one SlotState per slot: its schema, definitions,
/// ordinal domains, ranker and catalogs. An unslotted engine is exactly one implicit slot with
/// an empty name, and the accessors above and below that take no slot answer for it only. A
/// slotted engine answers through the overloads that take a slot index; the slot-less ones
/// throw std::logic_error, because no single catalog answers for several slots. One graph match
/// serves a catalog build for every slot, and the engine applies only if every slot's catalog is
/// non-empty and its `shared_workspace` token is valid (slotProgramVerdict).
///
/// Thread safety. `_catalogCache`, `_calibratedCache`, `_verdictCache`, `_winnerCache`,
/// `_programCache` and `_slotCatalogIds` are `LruCache`s that synchronize internally;
/// `_winnerCacheMutex` guards the per-shard "loaded" set and the one-shot failure-log set.
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
    /// One slot's selection state; an unslotted engine has exactly one, unnamed.
    struct SlotState
    {
        std::string name;
        MetadataSchema schema;
        std::vector<KernelDescriptorPack> packs;
        /// One entry per pack, parallel to `packs`: its kernels' context-independent
        /// definitions, completed once at construction.
        std::vector<std::vector<KernelDefinition>> definitions;
        /// Per non-integer metadata field, its value set in index order, so the field is
        /// addressable by an int64 knob. Fixed at construction.
        std::map<std::string, std::vector<MetadataValue>> ordinalDomains;
        std::shared_ptr<IKernelHeuristic> heuristic;
    };

public:
    /// How many (graph, device) catalogs to retain; eviction costs a rematch, never a
    /// wrong answer.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /// How many benchmarked rankings to retain in memory (RFC 0019 §9.2). An evicted record is
    /// re-read from the on-disk shard, so eviction costs at most one redundant sweep.
    static constexpr size_t DEFAULT_WINNER_CACHE_CAPACITY = 4096;

    /// An unslotted engine: one implicit slot.
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
        : KernelIngestorStateManager(
              false,
              implicitSlot(std::move(schema), std::move(packs), std::move(heuristic)),
              {},
              std::move(matchers),
              std::move(dispatches),
              graphMatchSymbol,
              describedBy,
              catalogCacheCapacity,
              std::move(engine),
              winnerCacheCapacity)
    {
    }

    // Pinned in place: catalog pointers into `_definitions` rely on a stable address.
    KernelIngestorStateManager(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager& operator=(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager(KernelIngestorStateManager&&) = delete;
    KernelIngestorStateManager& operator=(KernelIngestorStateManager&&) = delete;

    // PORT(S): C1-07 slotted constructor and slot accessors after develop's deleted copy/move
    /// A slotted engine: @p slots in launch order. As the unslotted constructor, except that a
    /// pack breaking a per-slot uniqueness rule (a metadata tuple repeated within a slot, a
    /// kernel id repeated on overlapping arches within a slot) is dropped with an ERROR instead
    /// of thrown (RFC 0021 §8). A kernel id shared by two slots is the loader's to refuse.
    /// @param matchers, dispatches Shared by every slot's packs.
    /// @param sharedWorkspaceToken The bound token holding the shared region's size (RFC 0021
    ///        §6); empty when the engine declares none.
    /// @throws std::invalid_argument as the unslotted constructor, and a slot list that is
    ///         empty or repeats a name.
    KernelIngestorStateManager(std::vector<KernelSlotInput> slots,
                               std::vector<MatchDescriptor> matchers,
                               std::vector<DispatchDescriptor> dispatches,
                               const std::string& graphMatchSymbol,
                               std::string sharedWorkspaceToken = {},
                               const std::string& describedBy = {},
                               size_t catalogCacheCapacity = DEFAULT_CATALOG_CACHE_CAPACITY,
                               EngineIdentity engine = {},
                               size_t winnerCacheCapacity = DEFAULT_WINNER_CACHE_CAPACITY)
        : KernelIngestorStateManager(true,
                                     slotStates(std::move(slots)),
                                     std::move(sharedWorkspaceToken),
                                     std::move(matchers),
                                     std::move(dispatches),
                                     graphMatchSymbol,
                                     describedBy,
                                     catalogCacheCapacity,
                                     std::move(engine),
                                     winnerCacheCapacity)
    {
    }

    /// Whether the engine declares slots (RFC 0021); an unslotted engine has one implicit slot.
    bool isSlotted() const
    {
        return _slotted;
    }

    /// How many slots select kernels: one for an unslotted engine.
    size_t slotCount() const
    {
        return _slots.size();
    }

    /// The slot's name; empty for an unslotted engine's implicit slot.
    const std::string& slotName(size_t slot) const
    {
        return slotAt(slot).name;
    }

    /// The metadata schema slot @p slot's kernels declare.
    const MetadataSchema& slotSchema(size_t slot) const
    {
        return slotAt(slot).schema;
    }

    /// The `shared_workspace` token; empty when the engine declares none.
    const std::string& sharedWorkspaceToken() const
    {
        return _sharedWorkspaceToken;
    }

    /// The unslotted engine's schema.
    /// @throws std::logic_error on a slotted engine; use slotSchema().
    const MetadataSchema& metadataSchema() const
    {
        requireUnslotted("metadataSchema");
        return _slots.front().schema;
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
        requireUnslotted("calibratedRanking");
        const auto key = cacheKey(context);
        if(key.has_value())
        {
            if(auto cached = _calibratedCache.get(*key); cached.has_value())
            {
                modelId = std::move(cached->modelId);
                return restrictRanking(cached->ranking, filtered);
            }
        }

        auto ranking = _slots.front().heuristic->calibratedRanking(full, context, modelId);

        if(key.has_value() && !ranking.empty())
        {
            // Not memoized when empty: empty means "cannot calibrate", and caching it would
            // make a miss and a hit indistinguishable.
            _calibratedCache.put(*key, CalibratedRanking{ranking, modelId});
        }
        return restrictRanking(ranking, filtered);
    }

    /// The model calibratedRanking() would use for @p metric on @p arch, from bindings alone.
    /// Empty for a slotted engine: a slot ranker predicts no time (RFC 0021 §7).
    std::string calibratedModelId(const std::string& metric, const std::string& arch) const
    {
        return _slotted ? std::string{} : _slots.front().heuristic->calibratedModelId(metric, arch);
    }

    /// Whether the engine accepts the graph and device @p context names, and why not. For an
    /// unslotted engine that is whether its catalog is non-empty; for a slotted engine, whether
    /// every slot's is and the shared-workspace token is valid (RFC 0021 §3 rule 2, §6). A
    /// slotted engine's graph match runs once per catalog build, however many slots ask.
    SlotProgramVerdict slotProgramVerdict(const MatchContext& context) const
    {
        if(!_slotted)
        {
            SlotProgramVerdict verdict;
            verdict.applicable = !catalogFor(_slots.front(), 0, context).entries.empty();
            if(!verdict.applicable)
            {
                verdict.declineReason = "no kernel applies to the graph";
            }
            return verdict;
        }
        if(!deviceResolved(context))
        {
            return {false, "no device resolved or the device reports no gcnArchName", 0};
        }
        const auto key = cacheKey(context, {});
        if(key.has_value())
        {
            if(auto cached = _verdictCache.get(*key); cached.has_value())
            {
                return std::move(*cached);
            }
        }
        auto built = buildSlotted(context);
        auto verdict = built.verdict;
        storeSlotted(context, std::move(built));
        return verdict;
    }

    /// Whether the engine accepts the graph and device @p context names; slotProgramVerdict()
    /// carries the reason when it does not.
    bool isApplicable(const MatchContext& context) const
    {
        return slotProgramVerdict(context).applicable;
    }

    /// Every kernel that applies to the graph and device @p context names, unordered.
    std::vector<const KernelDefinition*> unsortedDefinitions(const MatchContext& context) const
    {
        requireUnslotted("unsortedDefinitions");
        return catalogFor(_slots.front(), 0, context).entries;
    }

    /// The unranked catalog and the state matching bound, from one lookup.
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        requireUnslotted("unsortedCatalog");
        return catalogFor(_slots.front(), 0, context);
    }

    /// Slot @p slot's unranked catalog. Every slot's catalog is empty when graph_match declines
    /// the graph or the `shared_workspace` token is invalid. A slot with no applicable kernel
    /// leaves the other slots' catalogs populated, so an empty or non-empty catalog does not say
    /// whether the engine applies: slotProgramVerdict() is the authority.
    Catalog unsortedCatalog(size_t slot, const MatchContext& context) const
    {
        return catalogFor(slotAt(slot), slot, context);
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
        // PORT(S): C1-08 enumerableCatalog: unslotted guard and per-slot overload (body moved to enumerableCatalogOf)
        requireUnslotted("enumerableCatalog");
        return enumerableCatalogOf(0, context);
    }

    /// Slot @p slot's catalog in descriptor-ID order, as enumerableCatalog() for one slot.
    Catalog enumerableCatalog(size_t slot, const MatchContext& context) const
    {
        static_cast<void>(slotAt(slot));
        return enumerableCatalogOf(slot, context);
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
        // PORT(S): C1-09 sortedCatalog: unslotted guard
        requireUnslotted("sortedCatalog");
        return sortedCatalogOf(0, context);
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
    // PORT(S): C1-10 sortedCatalog(slot) overload beside develop's key-reusing body
    /// Slot @p slot's catalog ordered by that slot's ranker, or by declared order without one.
    /// A benchmarked slot record, when one covers the catalog, reorders it (see sortedCatalogOf).
    Catalog sortedCatalog(size_t slot, const MatchContext& context) const
    {
        static_cast<void>(slotAt(slot));
        return sortedCatalogOf(slot, context);
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
        // PORT(S): C1-11 measuredCatalog: unslotted guard over catalogFor(slot, 0, context)
        requireUnslotted("measuredCatalog");
        Catalog catalog = catalogFor(_slots.front(), 0, context);
        // A heuristic order is provisional (a sweep can postdate it), so look up a record
        // even when already sorted. An unresolved device's empty catalog is never covered.
        if(catalog.measuredRecord == nullptr && adoptCoveringRecord(catalog, context))
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
        // A slot record belongs to a slotted engine, and only slotted engines write v3, so an
        // unslotted engine's shard bytes cannot change.
        if(_slotted == key.slot.empty())
        {
            throw std::logic_error(
                _slotted ? "KernelIngestorStateManager::recordWinner needs a slot on a slotted "
                           "engine; use recordSlotted for a program"
                         : "KernelIngestorStateManager::recordWinner takes no slot on an "
                           "unslotted engine");
        }
        if(_slotted && !hasSlot(key.slot))
        {
            throw std::invalid_argument("recordWinner: unknown slot '" + key.slot + "'");
        }

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

    /// The ranking recorded for @p key (a slot record when `key.slot` is set), or nullopt, as a
    /// copy. A lookup refreshes @p key's recency.
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

    /// The program record for @p baseKey (whose `slot` MUST be empty), or nullopt.
    ///
    /// Validated at read time, never trusted as stored. A record is returned only if, for every
    /// slot in launch order, it names a kernel that is in that slot's catalog as this manager
    /// last saw it for the graph and device, that slot's record covers every kernel of that
    /// catalog, and the named kernel is that record's winner (first entry). A rewritten slot
    /// record with a different winner, a catalog that gained a kernel the record does not
    /// cover, or a catalog this manager has not seen all yield nullopt. Always nullopt on an
    /// unslotted engine.
    std::optional<ProgramRecord> programFor(const WinnerKey& baseKey) const
    {
        if(!_slotted || !baseKey.slot.empty() || !baseKey.graph.isUsable())
        {
            return std::nullopt;
        }

        loadShardIfAbsent(baseKey.device.properties().gcnArchName);
        auto program = _programCache.get(baseKey);
        if(!program.has_value() || program->picks.size() != _slots.size())
        {
            return std::nullopt;
        }

        for(size_t i = 0; i < _slots.size(); ++i)
        {
            const auto& pick = program->picks[i];
            if(pick.slot != _slots[i].name)
            {
                return std::nullopt;
            }
            WinnerKey slotKey = baseKey;
            slotKey.slot = pick.slot;
            const auto kernels = _slotCatalogIds.get(slotKey);
            const auto record = winnerFor(slotKey);
            if(!kernels.has_value() || !record.has_value() || record->empty()
               || record->front().kernelId != pick.kernelId
               || std::find(kernels->begin(), kernels->end(), pick.kernelId) == kernels->end())
            {
                return std::nullopt;
            }
            const bool covered = std::all_of(
                kernels->begin(), kernels->end(), [&record](const DescriptorId& kernel) {
                    return std::any_of(
                        record->begin(), record->end(), [&kernel](const RankedEntry& entry) {
                            return entry.kernelId == kernel;
                        });
                });
            if(!covered)
            {
                return std::nullopt;
            }
        }
        return program;
    }

    /// Records slotted benchmark results as ONE locked append: each @p slotRecords entry
    /// (slot name, per-slot ranking) is written under `baseKey` with its slot set, and
    /// @p program, if present, is written as the program line for @p baseKey (whose
    /// `slot` MUST be empty). Either part may be absent: a program-line-only call is legal,
    /// and an empty ranking is skipped as in `recordWinner`. @p slotCause applies to the per-slot
    /// records and @p programCause to the program line, with the same semantics as
    /// `recordWinner`'s cause.
    ///
    /// A program line is written only if, once this call's slot records are in, every slot's
    /// record has the program's pick as its winner; otherwise it is dropped (logged) and the
    /// slot records are still written. Slot lines precede the program line, so a failed append
    /// never leaves a program line without its slot records.
    /// @throws std::logic_error on an unslotted engine.
    /// @throws std::invalid_argument @p baseKey names a slot, a record names an unknown slot,
    ///         or @p program does not name every slot once, in launch order.
    void recordSlotted(const WinnerKey& baseKey,
                       const std::vector<std::pair<std::string, WinnerRecord>>& slotRecords,
                       const std::optional<ProgramRecord>& program,
                       WinnerWriteCause slotCause,
                       WinnerWriteCause programCause) const
    {
        requireSlotted("recordSlotted");
        if(!baseKey.slot.empty())
        {
            throw std::invalid_argument("recordSlotted: the base key must not name a slot");
        }
        for(const auto& entry : slotRecords)
        {
            if(!hasSlot(entry.first))
            {
                throw std::invalid_argument("recordSlotted: unknown slot '" + entry.first + "'");
            }
        }
        if(program.has_value())
        {
            const bool inLaunchOrder
                = program->picks.size() == _slots.size()
                  && std::equal(program->picks.begin(),
                                program->picks.end(),
                                _slots.begin(),
                                [](const ProgramPick& pick, const SlotState& slot) {
                                    return pick.slot == slot.name;
                                });
            if(!inLaunchOrder)
            {
                throw std::invalid_argument(
                    "recordSlotted: the program must name every slot once, in launch order");
            }
        }

        if(!baseKey.graph.isUsable())
        {
            // As recordWinner: an unkeyable graph's record could never be looked up.
            HIPDNN_PLUGIN_LOG_INFO("ingestor: a benchmarked slotted ranking could not be cached "
                                   "because its graph yields no key");
            return;
        }

        auto plan = writeSlottedToShard(baseKey, slotRecords, program, slotCause, programCause);
        for(auto& [slot, record] : plan.slotRecords)
        {
            WinnerKey key = baseKey;
            key.slot = slot;
            _winnerCache.put(key, std::move(record));
        }
        if(plan.program.has_value())
        {
            _programCache.put(baseKey, std::move(*plan.program));
        }
    }

    /// How many rankings are held: slot and whole-graph records, each capped at the winner-cache
    /// capacity, plus program records, capped at the same capacity separately.
    size_t winnerCacheSize() const
    {
        return _winnerCache.size() + _programCache.size();
    }

    /// Is it worth building a WinnerKey for @p gcnArchName? True if the in-memory caches
    /// hold anything, or this arch's shard has not been attempted yet. Probes the
    /// stripped arch, matching how `loadShardIfAbsent()` latches.
    bool mightHaveWinnerFor(const std::string& gcnArchName) const
    {
        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        // size(), not empty(): LruCache exposes no empty().
        return _winnerCache.size() != 0 || _programCache.size() != 0
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
        requireUnslotted("knobOrdinal");
        return knobOrdinal(0, field, value);
    }

    /// As knobOrdinal(field, value), over slot @p slot's ordinal domains.
    std::optional<int64_t>
        knobOrdinal(size_t slot, const std::string& field, const MetadataValue& value) const
    {
        if(const auto* intValue = std::get_if<int64_t>(&value))
        {
            return *intValue;
        }

        const auto& ordinalDomains = slotAt(slot).ordinalDomains;
        const auto domain = ordinalDomains.find(field);
        if(domain == ordinalDomains.end())
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
        requireUnslotted("knobMatches");
        return knobMatches(0, kernel, field, pinned);
    }

    /// As knobMatches(kernel, field, pinned), over slot @p slot's ordinal domains.
    bool knobMatches(size_t slot,
                     const KernelDefinition& kernel,
                     const std::string& field,
                     int64_t pinned) const
    {
        const auto value = kernel.tryGetMetadata(field);
        if(!value.has_value())
        {
            return false;
        }
        const auto ordinal = knobOrdinal(slot, field, *value);
        return ordinal.has_value() && *ordinal == pinned;
    }

    /// Is @p field addressed by an ordinal rather than by its own value? True exactly for a
    /// non-integer field some kernel carries.
    bool isOrdinalKnob(const std::string& field) const
    {
        requireUnslotted("isOrdinalKnob");
        return isOrdinalKnob(0, field);
    }

    /// As isOrdinalKnob(field), over slot @p slot's ordinal domains.
    bool isOrdinalKnob(size_t slot, const std::string& field) const
    {
        const auto& ordinalDomains = slotAt(slot).ordinalDomains;
        return ordinalDomains.find(field) != ordinalDomains.end();
    }

private:
    /// @p matchers and @p dispatches resolve here, eagerly, for the unslotted and slotted
    /// constructors alike.
    KernelIngestorStateManager(bool slotted,
                               std::vector<SlotState> slots,
                               std::string sharedWorkspaceToken,
                               std::vector<MatchDescriptor> matchers,
                               std::vector<DispatchDescriptor> dispatches,
                               const std::string& graphMatchSymbol,
                               const std::string& describedBy,
                               size_t catalogCacheCapacity,
                               EngineIdentity engine,
                               size_t winnerCacheCapacity)
        : _slotted(slotted)
        , _slots(std::move(slots))
        , _sharedWorkspaceToken(std::move(sharedWorkspaceToken))
        , _graphMatchFn(graphMatchSymbol.empty()
                            ? nullptr
                            : GraphMatchRegistry::resolve(graphMatchSymbol, describedBy))
        , _catalogCache(catalogCacheCapacity)
        // Same key and capacity as _catalogCache: both answer for (graph, device, engine
        // version). Kept outside `Catalog` because no matcher, scorer or formula reads it.
        , _calibratedCache(catalogCacheCapacity)
        , _verdictCache(catalogCacheCapacity)
        , _engine(std::move(engine))
        , _winnerCache(winnerCacheCapacity)
        , _programCache(winnerCacheCapacity)
        , _slotCatalogIds(catalogCacheCapacity)
    {
        if(_slots.empty())
        {
            throw std::invalid_argument("kernel ingestor requires at least one slot");
        }
        std::unordered_set<std::string> slotNames;
        for(const auto& slot : _slots)
        {
            if(slot.heuristic == nullptr)
            {
                throw std::invalid_argument("kernel ingestor requires a heuristic");
            }
            if(_slotted && (slot.name.empty() || !slotNames.insert(slot.name).second))
            {
                throw std::invalid_argument("slot name '" + slot.name
                                            + "' is empty or declared twice");
            }
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

        for(auto& slot : _slots)
        {
            indexSlot(slot);
        }
    }

    static std::vector<SlotState> implicitSlot(MetadataSchema schema,
                                               std::vector<KernelDescriptorPack> packs,
                                               std::shared_ptr<IKernelHeuristic> heuristic)
    {
        std::vector<SlotState> slots;
        slots.push_back(
            SlotState{{}, std::move(schema), std::move(packs), {}, {}, std::move(heuristic)});
        return slots;
    }

    static std::vector<SlotState> slotStates(std::vector<KernelSlotInput> inputs)
    {
        std::vector<SlotState> slots;
        slots.reserve(inputs.size());
        for(auto& input : inputs)
        {
            slots.push_back(SlotState{std::move(input.name),
                                      std::move(input.schema),
                                      std::move(input.packs),
                                      {},
                                      {},
                                      std::move(input.heuristic)});
        }
        return slots;
    }

    const SlotState& slotAt(size_t slot) const
    {
        if(slot >= _slots.size())
        {
            throw std::out_of_range("slot index " + std::to_string(slot) + " is out of range for "
                                    + std::to_string(_slots.size()) + " slot(s)");
        }
        return _slots[slot];
    }

    /// The slot-less accessors answer for one catalog, which only an unslotted engine has.
    void requireUnslotted(const char* accessor) const
    {
        if(_slotted)
        {
            throw std::logic_error(std::string("KernelIngestorStateManager::") + accessor
                                   + " needs a slot: this engine is slotted");
        }
    }

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
    // PORT(S): C1-12 adoptCoveringRecord signature: slot parameter beside develop's catalogKey parameter
    /// otherwise. @p slot names the slot record to adopt; empty for the whole-graph record of
    /// an unslotted engine.
    bool adoptCoveringRecord(Catalog& catalog,
                             const MatchContext& context,
                             std::string_view slot = {}) const
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
        // PORT(S): C1-13 adoptCoveringRecord winner key: slot-keyed WinnerKey beside develop's catalogKey->graph
        const WinnerKey key{
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{context.graph},
            DeviceKey{context.deviceProperties},
            std::string(slot)};
        if(!key.graph.isUsable())
        {
            // No bytes to key on; such graphs never match each other either.
            return false;
        }

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
    // PORT(S): C1-14 noteSlotCatalog, requireSlotted, hasSlot beside develop's measuredCatalogFor
    /// Remembers which kernels @p slot's @p catalog holds for @p context's graph and device,
    /// so programFor() can compare a program record with the catalogs it is asked about without
    /// a MatchContext. A catalog is a function of (graph, device) for the life of the manager.
    void noteSlotCatalog(const SlotState& slot,
                         const MatchContext& context,
                         const Catalog& catalog) const
    {
        if(!deviceResolved(context))
        {
            return;
        }
        const WinnerKey key{
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{context.graph},
            DeviceKey{context.deviceProperties},
            slot.name};
        if(!key.graph.isUsable())
        {
            return;
        }
        std::vector<DescriptorId> ids;
        ids.reserve(catalog.entries.size());
        for(const auto& entry : catalog.entries)
        {
            ids.push_back(entry.kernelId);
        }
        _slotCatalogIds.put(key, std::move(ids));
    }

    void requireSlotted(const char* accessor) const
    {
        if(!_slotted)
        {
            throw std::logic_error(std::string("KernelIngestorStateManager::") + accessor
                                   + " needs a slotted engine");
        }
    }

    bool hasSlot(const std::string& name) const
    {
        return std::any_of(_slots.begin(), _slots.end(), [&name](const SlotState& slot) {
            return slot.name == name;
        });
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
    // PORT(S): C1-15 indexSlot doc comment, per slot
    /// Indexes every pack of @p slot: validates its references and builds the KernelDefinition
    /// for each of its kernels. Every field of a definition is context-independent, so this is
    /// the only place they are ever computed: buildCatalog copies them per query rather
    /// than completing each kernel's metadata again on every graph.
    ///
    /// Uniqueness is per slot. A pack that breaks it is dropped whole, with an ERROR, in a
    /// slotted engine (RFC 0021 §8) and throws in an unslotted one.
    void indexSlot(SlotState& slot) const
    {
        // Two kernels may share a tuple when no single device sees both at the same arch
        // tier (per-arch shards, or a more specific kernel that buildCatalog shadows
        // per device). Uniqueness is per competing-arch group, keyed by the tuple's values
        // in a hash map rather than scanned, which would be quadratic.
        size_t kernelCount = 0;
        for(const auto& pack : slot.packs)
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
        // PORT(S): C1-16 indexSlot: loop over slot.packs/definitions beside develop's _definitions/_kernelsByArch reserve
        slot.definitions.reserve(slot.packs.size());
        for(const auto& pack : slot.packs)
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
            // PORT(S): C1-17 indexSlot: claim/refusal bookkeeping beside develop's handlerLaunches
            // What this pack has claimed so far, so a slotted engine can release it when a
            // later kernel of the pack is refused. Left empty for an unslotted engine, which
            // throws instead.
            std::vector<DescriptorId> claimedIds;
            std::vector<MetadataTupleValues> claimedTuples;
            std::string refusal;
            // A violation throws for an unslotted engine; a slotted engine records it and
            // drops the pack.
            const auto refuse = [this, &refusal](std::string message) {
                if(!_slotted)
                {
                    throw std::invalid_argument(message);
                }
                refusal = std::move(message);
            };

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
                // PORT(S): C1-18 indexSlot: completeMetadata(slot.schema, kernel) beside develop's launch-set check
                auto key = completeMetadata(slot.schema, kernel);
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
                    refuse("kernel '" + toString(kernel.id)
                           + "' is declared twice on an arch both declarations reach; a kernel id "
                           + "names one candidate and must be unique per device");
                    break;
                }
                if(_slotted)
                {
                    claimedIds.push_back(kernel.id);
                }
                MetadataTupleValues claimedTuple;
                if(_slotted)
                {
                    claimedTuple = tuple;
                }
                if(!claim(archesClaimingTuple.try_emplace(std::move(tuple)).first->second,
                          kernelArch,
                          archesCompete))
                {
                    refuse("kernel '" + toString(kernel.id)
                           + "' duplicates the metadata tuple of another kernel under schema '"
                           + slot.schema.name
                           + "' at the same arch tier on a device both reach; a better tier "
                           + "shadows a worse one, so equal tiers cannot be told apart");
                    break;
                }
                if(_slotted)
                {
                    claimedTuples.push_back(std::move(claimedTuple));
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

            if(!refusal.empty())
            {
                // Newest claim first: a key claimed twice by this pack holds both entries, and
                // the later one is last.
                for(auto it = claimedIds.rbegin(); it != claimedIds.rend(); ++it)
                {
                    archesClaimingId.at(*it).pop_back();
                }
                for(auto it = claimedTuples.rbegin(); it != claimedTuples.rend(); ++it)
                {
                    archesClaimingTuple.at(*it).pop_back();
                }
                packDefinitions.clear();
                HIPDNN_PLUGIN_LOG_ERROR("ingestor: slot '" << slot.name << "' drops pack "
                                                           << toString(pack.id) << ": " << refusal);
            }
            // Pushed even when every kernel was dropped: `definitions` is indexed by pack
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
            // PORT(S): C1-19 indexSlot: slot.definitions.push_back beside develop's _definitions push and _kernelsByArch counts
            slot.definitions.push_back(std::move(packDefinitions));
        }

        buildOrdinalDomains(slot);
    }

    /// The value set of every non-integer metadata field of @p slot, in the order that numbers
    /// it (RFC 0019 §13.2: a knob is an int64, so non-`INT` fields are addressed by index).
    ///
    /// Spans every pack of the slot, not one device's catalog, so an index means the same
    /// kernel on every graph. Ordered by `std::variant`'s `<`; `uhd_gen/addressing.py` must
    /// derive the same indices.
    static void buildOrdinalDomains(SlotState& slot)
    {
        for(const auto& field : slot.schema.fields)
        {
            if(field.type == MetadataType::INT)
            {
                continue;
            }

            std::set<MetadataValue> observed;
            for(const auto& packDefinitions : slot.definitions)
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
                slot.ordinalDomains.emplace(
                    field.name, std::vector<MetadataValue>(observed.begin(), observed.end()));
            }
        }
    }

    /// A kernel's metadata values with @p schema's defaults filled in; the completed
    /// tuple, not the descriptor id, is the catalog key.
    static MetadataValues completeMetadata(const MetadataSchema& schema,
                                           const KernelDescriptor& kernel)
    {
        MetadataValues complete;

        for(const auto& field : schema.fields)
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
    // PORT(S): C1-20 cacheKey with slot parameter and deviceResolved beside develop's catalogFor(context) signature
    /// Device comes from the context, not a separate argument, so one device's catalog
    /// never caches under another's key. The engine version comes from this manager. The
    /// ranking metric is part of the key because both orders depend on it (RFC 0019 §11.4);
    /// the registry's own name is stored because the key outlives the request. @p slot names
    /// the slot a catalog belongs to; empty for an unslotted engine and for a slotted engine's
    /// engine-wide entries.
    std::optional<CatalogKey> cacheKey(const MatchContext& context,
                                       std::string_view slot = {}) const
    {
        // PORT(S): C1-28 semantic: tryGetGraphId no longer exists; develop keys the catalog on graph content, not Graph.id
        const auto graphId = tryGetGraphId(context.graph);
        const auto* metric = hipdnn_data_sdk::utilities::findRankingMetric(context.rankingMetric);
        if(!graphId.has_value() || metric == nullptr)
        {
            return std::nullopt;
        }
        return CatalogKey{
            *graphId, context.deviceId, _engine.version, metric->name, std::string(slot)};
    }

    /// Pack pruning and matchers both read the device, so nothing can be answered without one.
    /// Checked once here rather than in every provider's matchers, where an omission is
    /// invisible.
    static bool deviceResolved(const MatchContext& context)
    {
        return context.deviceId != NO_DEVICE && !context.deviceProperties.gcnArchName.empty();
    }

    /// Slot @p slotIndex's catalog for @p context, cached. A slotted engine builds every slot's
    /// catalog on a miss, with one graph match, and caches them together.
    Catalog catalogFor(const SlotState& slot, size_t slotIndex, const MatchContext& context) const
    {
        if(!deviceResolved(context))
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
        // PORT(S): C1-21 catalogFor(slot...): slot-keyed lookup beside develop's key-taking catalogFor split
        const auto key = cacheKey(context, slot.name);
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

        if(_slotted)
        {
            auto built = buildSlotted(context);
            auto catalog = built.catalogs[slotIndex];
            storeSlotted(context, std::move(built));
            return catalog;
        }

        GraphMatchState match;
        Catalog catalog = buildCatalog(slot, context, match);
        if(match.bound.has_value())
        {
            catalog.bound = std::move(*match.bound);
        }

        if(key.has_value())
        {
            // putIfAbsent: another thread may already have installed a sorted catalog
            // here; overwriting with this unsorted one would discard that ranking.
            _catalogCache.putIfAbsent(*key, catalog);
        }

        return catalog;
    }

    Catalog enumerableCatalogOf(size_t slot, const MatchContext& context) const
    {
        auto catalog = catalogFor(_slots[slot], slot, context);
        if(_slotted)
        {
            noteSlotCatalog(_slots[slot], context, catalog);
        }
        std::sort(catalog.entries.begin(),
                  catalog.entries.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.kernelId < rhs.kernelId; });
        return catalog;
    }

    /// Slot @p slotIndex's catalog with its slot record adopted when one covers it. A
    /// benchmarked record outranks the slot's ranker, and may postdate a cached ranker order,
    /// so it is looked up even when the cached catalog is already sorted.
    Catalog measuredSlotCatalog(size_t slotIndex, const MatchContext& context) const
    {
        const auto& slot = _slots[slotIndex];
        Catalog catalog = catalogFor(slot, slotIndex, context);
        noteSlotCatalog(slot, context, catalog);
        if(catalog.measuredRecord == nullptr && adoptCoveringRecord(catalog, context, slot.name))
        {
            if(const auto key = cacheKey(context, slot.name); key.has_value())
            {
                // put, not putIfAbsent: a measured order replaces a provisional one.
                _catalogCache.put(*key, catalog);
            }
        }
        return catalog;
    }

    Catalog sortedCatalogOf(size_t slotIndex, const MatchContext& context) const
    {
        const auto& slot = _slots[slotIndex];
        // Mirrors catalogFor's own reject guard: cacheKey() below only reads graph and
        // device ordinal, not arch, so without this an unresolved-arch context would
        // cache an empty catalog under the SAME key a later, resolved call for this
        // device reuses -- permanently hiding that device's real catalog.
        if(!deviceResolved(context))
        {
            return catalogFor(slot, slotIndex, context);
        }

        // A benchmarked record (whole-graph, or the slot's own) outranks the ranker.
        Catalog catalog
            = _slotted ? measuredSlotCatalog(slotIndex, context) : measuredCatalog(context);
        if(catalog.isSorted)
        {
            return catalog;
        }

        catalog.entries = slot.heuristic->rank(catalog, context);
        catalog.isSorted = true;
        if(const auto key = cacheKey(context, slot.name); key.has_value())
        {
            // put, not putIfAbsent: sorted is strictly better than whatever is cached.
            _catalogCache.put(*key, catalog);
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
    // PORT(S): C1-22 buildCatalog(slot, context, GraphMatchState) beside develop's buildCatalog
    /// What one catalog build shares across its slots: the engine's graph match, run lazily
    /// and at most once, and the graph-scoped criteria's verdicts.
    struct GraphMatchState
    {
        /// Whether the graph match has run.
        bool ran = false;
        /// What it bound; nullopt until it runs, and after it declines the graph.
        std::optional<BoundTokens> bound;
        GraphMatcherMemo verdicts;
    };

    /// Runs the engine's graph match, once per @p match. Absent means the engine binds nothing
    /// and always proceeds with an empty map.
    void runGraphMatch(const MatchContext& context, GraphMatchState& match) const
    {
        if(match.ran)
        {
            return;
        }
        match.ran = true;
        match.bound = _graphMatchFn == nullptr ? std::optional<BoundTokens>(BoundTokens{})
                                               : _graphMatchFn(context);
    }

    /// The catalogs of every slot and whether the engine accepts the graph.
    struct SlottedBuild
    {
        std::vector<Catalog> catalogs;
        SlotProgramVerdict verdict;
    };

    /// Builds every slot's catalog from one graph match, then decides the engine's verdict. A
    /// graph_match decline or an invalid shared-workspace token empties every catalog. A slot
    /// with no applicable kernel only sets the decline reason and leaves the other slots'
    /// catalogs populated, so the verdict, not a catalog, says whether the engine applies.
    SlottedBuild buildSlotted(const MatchContext& context) const
    {
        SlottedBuild built;
        built.catalogs.reserve(_slots.size());
        GraphMatchState match;
        for(const auto& slot : _slots)
        {
            built.catalogs.push_back(buildCatalog(slot, context, match));
        }

        auto& verdict = built.verdict;
        if(match.ran && !match.bound.has_value())
        {
            verdict.declineReason = "graph_match declined the graph";
        }
        else if(match.ran && !_sharedWorkspaceToken.empty())
        {
            verdict.declineReason
                = checkSharedWorkspace(*match.bound, verdict.sharedWorkspaceBytes);
            if(!verdict.declineReason.empty())
            {
                HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine declined the graph on device "
                                        << context.deviceId << ": " << verdict.declineReason);
                for(auto& catalog : built.catalogs)
                {
                    catalog.entries.clear();
                }
            }
        }
        if(verdict.declineReason.empty())
        {
            for(size_t i = 0; i < _slots.size(); ++i)
            {
                if(built.catalogs[i].entries.empty())
                {
                    verdict.declineReason
                        = "slot '" + _slots[i].name + "' has no applicable kernel";
                    HIPDNN_PLUGIN_LOG_INFO("ingestor: engine declined the graph on device "
                                           << context.deviceId << ": " << verdict.declineReason);
                    break;
                }
            }
        }
        verdict.applicable = verdict.declineReason.empty();
        if(!verdict.applicable)
        {
            verdict.sharedWorkspaceBytes = 0;
        }

        if(match.bound.has_value())
        {
            // The slots share one binding; the last takes it, the others copy.
            for(size_t i = 0; i + 1 < built.catalogs.size(); ++i)
            {
                built.catalogs[i].bound = *match.bound;
            }
            built.catalogs.back().bound = std::move(*match.bound);
        }
        return built;
    }

    /// Caches @p built under @p context's key: each slot's catalog and the engine's verdict.
    void storeSlotted(const MatchContext& context, SlottedBuild built) const
    {
        for(size_t i = 0; i < _slots.size(); ++i)
        {
            noteSlotCatalog(_slots[i], context, built.catalogs[i]);
        }
        const auto engineKey = cacheKey(context, {});
        if(!engineKey.has_value())
        {
            return;
        }
        for(size_t i = 0; i < _slots.size(); ++i)
        {
            auto slotKey = *engineKey;
            slotKey.slot = _slots[i].name;
            // putIfAbsent, as for an unslotted catalog: a ranked catalog may already be there.
            _catalogCache.putIfAbsent(slotKey, std::move(built.catalogs[i]));
        }
        _verdictCache.putIfAbsent(*engineKey, std::move(built.verdict));
    }

    /// Why the shared region @p bound describes is unusable, or empty when it is usable: the
    /// `shared_workspace` token must be a non-negative integer, `ws.scratch_offset` must be
    /// published as that size rounded up to the engine contract's alignment, and every
    /// `ws.<name>.offset` token must be a non-negative integer. @p sharedBytes receives the
    /// token's value when it is. The messages name the token and are stable.
    std::string checkSharedWorkspace(const BoundTokens& bound, size_t& sharedBytes) const
    {
        const auto declared = "shared_workspace token '" + _sharedWorkspaceToken + "'";
        const auto sharedIt = bound.find(_sharedWorkspaceToken);
        if(sharedIt == bound.end())
        {
            return declared + " was not published by the graph match";
        }
        const auto* size = std::get_if<int64_t>(&sharedIt->second);
        if(size == nullptr)
        {
            return declared + " is not an integer";
        }
        if(*size < 0)
        {
            return declared + " is negative (" + std::to_string(*size) + ")";
        }
        const auto aligned = alignSharedWorkspace(static_cast<size_t>(*size));

        const auto scratchIt = bound.find(SCRATCH_OFFSET_TOKEN);
        if(scratchIt == bound.end())
        {
            return std::string(SCRATCH_OFFSET_TOKEN) + " was not published by the graph match";
        }
        const auto* scratchOffset = std::get_if<int64_t>(&scratchIt->second);
        if(scratchOffset == nullptr)
        {
            return std::string(SCRATCH_OFFSET_TOKEN) + " is not an integer";
        }
        if(*scratchOffset < 0 || static_cast<size_t>(*scratchOffset) != aligned)
        {
            return std::string(SCRATCH_OFFSET_TOKEN) + " (" + std::to_string(*scratchOffset)
                   + ") does not equal " + declared + " value " + std::to_string(*size)
                   + " rounded up to " + std::to_string(SHARED_WORKSPACE_ALIGNMENT) + " bytes ("
                   + std::to_string(aligned) + ")";
        }

        // The first offender by name, so the message does not depend on hash order.
        const std::string* badOffset = nullptr;
        for(const auto& [name, value] : bound)
        {
            constexpr std::string_view ROOT = "ws.";
            constexpr std::string_view SUFFIX = ".offset";
            const bool isOffsetToken
                = name.size() > ROOT.size() + SUFFIX.size()
                  && name.compare(0, ROOT.size(), ROOT) == 0
                  && name.compare(name.size() - SUFFIX.size(), SUFFIX.size(), SUFFIX) == 0;
            if(!isOffsetToken)
            {
                continue;
            }
            const auto* offset = std::get_if<int64_t>(&value);
            if((offset == nullptr || *offset < 0) && (badOffset == nullptr || name < *badOffset))
            {
                badOffset = &name;
            }
        }
        if(badOffset != nullptr)
        {
            return "offset token '" + *badOffset + "' is not a non-negative integer";
        }

        sharedBytes = static_cast<size_t>(*size);
        return {};
    }

    /// Runs @p match's graph match (lazily, on the first pack that clears the arch gate) and
    /// every pack's UMDs of @p slot: graph-scoped criteria (memoized across packs and slots),
    /// then kernel-scoped ones. The catalog's `bound` is left to the caller; it is
    /// `match.bound`, shared by every slot of a build.
    Catalog buildCatalog(const SlotState& slot,
                         const MatchContext& context,
                         GraphMatchState& match) const
    {
        CatalogGraphStage stage = matchGraph(context);
        if(stage.firstPack < _packs.size() && !stage.bound.has_value())
        {
            return Catalog{};
        }
        return matchVariants(context, std::move(stage));
    }
        // PORT(S): C1-23 buildCatalog body start beside develop's matchGraph call
        Catalog catalog;

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
        // PORT(S): C1-24 slot.packs loop beside develop's matchGraph and matchVariants
        for(size_t packIndex = 0; packIndex < slot.packs.size(); ++packIndex)
        {
            const auto& pack = slot.packs[packIndex];

            // matchGraph already cleared the first pack.
            if(packIndex != stage.firstPack && !packSupportsDevice(packIndex, context))
            {
                continue;
            }

            if(!graphLevelMatchersPass(pack, context, graphVerdicts, catalog.bound))
            // PORT(S): C1-25 runGraphMatch call and match.verdicts beside develop's graphLevelMatchersPass call
            runGraphMatch(context, match);

            if(!match.bound.has_value())
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: engine declined graph_match for device "
                                       << context.deviceId);
                return catalog;
            }

            if(!graphLevelMatchersPass(pack, context, match.verdicts, *match.bound))
            {
                HIPDNN_PLUGIN_LOG_INFO("ingestor: pack " << toString(pack.id)
                                                         << " declined at a graph-scoped matcher");
                continue;
            }

            size_t admitted = 0;
            for(const auto& precomputed : slot.definitions[packIndex])
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
                // PORT(S): C1-26 kernelLevelMatchersPass with match.bound beside develop's referenced definitions
                // Copied only when admitted: matchers take it by const reference, and most
                // kernels are rejected.
                if(kernelLevelMatchersPass(pack, context, *match.bound, precomputed))
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

        HIPDNN_PLUGIN_LOG_INFO("ingestor: "
                               << (_slotted ? "slot '" + slot.name + "' catalog" : "catalog")
                               << " for device " << context.deviceId << " holds "
                               << catalog.entries.size() << " kernel(s) from " << slot.packs.size()
                               << " pack(s)");
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

        std::vector<DecodedWinnerLine> decoded;
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
                = hipdnn_data_sdk::utilities::readAllLines(*shard, &decodeWinnerLine);
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
        // A line of the other engine kind is skipped: an unslotted engine reads whole-graph
        // records only, a slotted one slot records and program lines.
        std::vector<std::pair<WinnerKey, WinnerRecord>> newestFirst;
        std::vector<std::pair<WinnerKey, ProgramRecord>> newestFirstPrograms;
        newestFirst.reserve(decoded.size());
        for(auto it = decoded.rbegin(); it != decoded.rend(); ++it)
        {
            if(!it->key.graph.isUsable())
            {
                // A key whose graph carries no content is not even equal to itself
                // (GraphContentKey::operator==), so it can never be looked up -- and as
                // a non-reflexive key it would violate the cache's KeyEqual requirements.
                continue;
            }
            if(it->program.has_value())
            {
                if(_slotted)
                {
                    newestFirstPrograms.emplace_back(std::move(it->key), std::move(*it->program));
                }
            }
            else if(_slotted == !it->key.slot.empty())
            {
                newestFirst.emplace_back(std::move(it->key), std::move(it->record));
            }
        }
        _winnerCache.mergeAbsent(std::move(newestFirst));
        _programCache.mergeAbsent(std::move(newestFirstPrograms));
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

    /// What one `recordSlotted` call settles while holding the shard lock.
    struct SlottedWritePlan
    {
        /// The slot rankings to keep in memory: each one appended, or the on-disk record a fresh
        /// miss adopted.
        std::vector<std::pair<std::string, WinnerRecord>> slotRecords;
        /// The program record to keep in memory; empty when none was given or it was invalid.
        std::optional<ProgramRecord> program;
        /// Encoded lines to append, in order: every slot record, then the program line.
        std::vector<std::string> lines;
    };

    /// Decides what @p slotRecords and @p program contribute, from the lines @p onDisk holds
    /// (nullptr when no shard is usable, in which case the in-memory slot rankings stand in).
    /// Each slot record follows @p slotCause as `writeBackToShard` does, the program line
    /// @p programCause. A program line is planned only if it names, for every slot, the winner
    /// (first entry) of that slot's record as it will stand after this write; a mismatch drops
    /// the program line and keeps the slot records, so a program line never lands without
    /// valid slot records. Last-line-wins over @p onDisk, as in the reader.
    SlottedWritePlan
        planSlottedWrite(const WinnerKey& baseKey,
                         const std::vector<std::pair<std::string, WinnerRecord>>& slotRecords,
                         const std::optional<ProgramRecord>& program,
                         WinnerWriteCause slotCause,
                         WinnerWriteCause programCause,
                         const std::vector<DecodedWinnerLine>* onDisk) const
    {
        std::unordered_map<std::string, const WinnerRecord*> diskSlots;
        const ProgramRecord* diskProgram = nullptr;
        if(onDisk != nullptr)
        {
            for(const auto& line : *onDisk)
            {
                if(line.key.device != baseKey.device || line.key.graph != baseKey.graph)
                {
                    continue;
                }
                if(line.program.has_value())
                {
                    diskProgram = &*line.program;
                }
                else if(!line.key.slot.empty())
                {
                    diskSlots[line.key.slot] = &line.record;
                }
            }
        }

        SlottedWritePlan plan;
        plan.slotRecords.reserve(slotRecords.size());
        plan.lines.reserve(slotRecords.size() + 1);
        for(const auto& [slot, record] : slotRecords)
        {
            if(record.empty())
            {
                // As recordWinner: an all-unusable sweep has nothing to record.
                continue;
            }
            const auto existing = diskSlots.find(slot);
            if(existing != diskSlots.end() && slotCause == WinnerWriteCause::FRESH_MISS)
            {
                plan.slotRecords.emplace_back(slot, *existing->second);
                continue;
            }
            WinnerKey key = baseKey;
            key.slot = slot;
            plan.lines.push_back(encodeWinnerRecordLine(key, record));
            plan.slotRecords.emplace_back(slot, record);
        }

        if(!program.has_value())
        {
            return plan;
        }

        for(const auto& pick : program->picks)
        {
            WinnerRecord inMemory;
            const WinnerRecord* winner = nullptr;
            const auto planned
                = std::find_if(plan.slotRecords.begin(),
                               plan.slotRecords.end(),
                               [&pick](const auto& entry) { return entry.first == pick.slot; });
            if(planned != plan.slotRecords.end())
            {
                winner = &planned->second;
            }
            else if(onDisk != nullptr)
            {
                const auto existing = diskSlots.find(pick.slot);
                winner = existing != diskSlots.end() ? existing->second : nullptr;
            }
            else
            {
                WinnerKey key = baseKey;
                key.slot = pick.slot;
                if(auto cached = _winnerCache.get(key); cached.has_value())
                {
                    inMemory = std::move(*cached);
                    winner = &inMemory;
                }
            }
            if(winner == nullptr || winner->empty() || winner->front().kernelId != pick.kernelId)
            {
                HIPDNN_PLUGIN_LOG_WARN("ingestor: a program record was not written because slot '"
                                       << pick.slot << "' has no record whose winner is kernel '"
                                       << toString(pick.kernelId) << "'");
                return plan;
            }
        }

        if(diskProgram != nullptr && programCause == WinnerWriteCause::FRESH_MISS)
        {
            plan.program = *diskProgram;
            return plan;
        }
        plan.lines.push_back(encodeProgramLine(baseKey, *program));
        plan.program = *program;
        return plan;
    }

    /// `recordSlotted`'s disk half: reads the shard, plans and appends every line under one hold
    /// of the shard's lock. A disk failure degrades to the in-memory plan, as `writeBackToShard`
    /// does. A failed append stops the sequence, so the program line, last, never lands without
    /// its slot records.
    SlottedWritePlan
        writeSlottedToShard(const WinnerKey& baseKey,
                            const std::vector<std::pair<std::string, WinnerRecord>>& slotRecords,
                            const std::optional<ProgramRecord>& program,
                            WinnerWriteCause slotCause,
                            WinnerWriteCause programCause) const
    {
        if(_engine.name.empty() || !_engine.contentIdentified)
        {
            return planSlottedWrite(
                baseKey, slotRecords, program, slotCause, programCause, nullptr);
        }

        const auto& gcnArchName = baseKey.device.properties().gcnArchName;
        auto [shard, openStatus] = openWinnerCacheShard(_engine, gcnArchName);
        if(openStatus == hipdnn_data_sdk::utilities::LineStoreStatus::VERSION_MISMATCH)
        {
            logShardFailureOnce(WinnerShardFailureKind::VERSION_MISMATCH, gcnArchName);
            return planSlottedWrite(
                baseKey, slotRecords, program, slotCause, programCause, nullptr);
        }
        if(openStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK || !shard.has_value())
        {
            logShardFailureOnce(WinnerShardFailureKind::OPEN, gcnArchName);
            return planSlottedWrite(
                baseKey, slotRecords, program, slotCause, programCause, nullptr);
        }
        if(hipdnn_data_sdk::utilities::lockLineStore(*shard)
           != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            logShardFailureOnce(WinnerShardFailureKind::LOCK, gcnArchName);
            return planSlottedWrite(
                baseKey, slotRecords, program, slotCause, programCause, nullptr);
        }

        const auto [existing, readStatus]
            = hipdnn_data_sdk::utilities::readAllLines(*shard, &decodeWinnerLine);
        if(readStatus != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            hipdnn_data_sdk::utilities::unlockLineStore(*shard);
            logShardFailureOnce(WinnerShardFailureKind::READ, gcnArchName);
            return planSlottedWrite(
                baseKey, slotRecords, program, slotCause, programCause, nullptr);
        }

        auto plan
            = planSlottedWrite(baseKey, slotRecords, program, slotCause, programCause, &existing);
        for(const auto& line : plan.lines)
        {
            if(hipdnn_data_sdk::utilities::appendLine(*shard, line)
               != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
            {
                logShardFailureOnce(WinnerShardFailureKind::APPEND, gcnArchName);
                break;
            }
        }
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
        return plan;
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

    bool _slotted;
    /// In launch order; exactly one, unnamed, for an unslotted engine.
    std::vector<SlotState> _slots;
    /// The bound token holding the shared workspace size; empty when none is declared.
    std::string _sharedWorkspaceToken;
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
    // PORT(S): C1-27 develop's per-engine member list kept; slotted model keeps per-slot state instead (nothing re-inserted)
    GraphMatchFn _graphMatchFn = nullptr;
    /// Keyed by slot: an unslotted engine's catalogs under the empty slot, a slotted engine's
    /// one entry per slot name.
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;
    /// The full catalog's calibrated ranking; a knob-pinned request restricts it instead of
    /// ranking its own subset (RFC 0019 §9.2).
    mutable LruCache<CatalogKey, CalibratedRanking, CatalogKeyHash> _calibratedCache;
    /// A slotted engine's verdict per (graph, device, engine version, metric), under the empty
    /// slot; built with the slot catalogs by one graph match.
    mutable LruCache<CatalogKey, SlotProgramVerdict, CatalogKeyHash> _verdictCache;

    /// The engine's identity. The name locates the on-disk winner-cache shard (empty disables
    /// it); the revision is the version half of every CatalogKey.
    EngineIdentity _engine;

    /// Separate from _catalogCache so benchmarked rankings and catalogs cannot evict each other.
    mutable LruCache<WinnerKey, WinnerRecord, WinnerKeyHash> _winnerCache;
    /// A slotted engine's program records, keyed by the whole-graph key (empty slot). Its own
    /// cache at the same capacity, so program records and slot records are each LRU-accounted
    /// and neither evicts the other.
    mutable LruCache<WinnerKey, ProgramRecord, WinnerKeyHash> _programCache;
    /// Which kernels each slot's catalog held for a (graph, device), keyed like a slot record.
    /// Lets programFor() check a program record against the catalogs without a MatchContext;
    /// refreshed by every slotted catalog access, so a missing entry means "not seen".
    mutable LruCache<WinnerKey, std::vector<DescriptorId>, WinnerKeyHash> _slotCatalogIds;
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
