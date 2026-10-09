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
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
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
    /// A kernel-scoped matcher's optional KernelPrepareFn, registered under its symbol.
    KernelPrepareFn prepareFn = nullptr;
};

/// A kernel-scoped matcher's CandidateFilterFn, resolved once with the matcher, and the
/// packs whose admitted kernels it refines as one set: every pack listing the matcher.
struct ResolvedCandidateFilter
{
    DescriptorId matcherId;
    std::string description;
    CandidateFilterFn filter = nullptr;
    std::unordered_set<DescriptorId, DescriptorIdHash> listingPacks;
};

/// A UDD plus the handler its dispatchSymbol resolved to at construction.
template <typename THandle>
struct ResolvedDispatch
{
    DispatchDescriptor descriptor;
    const IKernelDispatchHandler<THandle>* handler = nullptr;
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

/// The part of an engine's kernel-ingestor state that never changes once built: its
/// schema, resolved matchers and dispatches, packs, every kernel's precomputed
/// definition, heuristic, and graph_match. Building it is O(kernels) -- every kernel's
/// metadata is completed and its tuple checked for uniqueness -- so it is built once
/// per descriptor set and shared, through `std::shared_ptr<const ...>`, by every
/// KernelIngestorStateManager serving that set. Each manager keeps its own mutable
/// caches beside it.
///
/// Thread safety: immutable after construction and read only through const members,
/// so any number of managers on any number of threads may share one instance.
template <typename THandle>
class KernelIngestorSharedState
{
public:
    /// @throws std::invalid_argument bad pack reference, or duplicate metadata tuple.
    /// @throws std::runtime_error a UMD or the engine's graph_match names a symbol this
    /// build does not ship.
    /// @param describedBy Names the engine in the graph_match resolution failure.
    KernelIngestorSharedState(MetadataSchema schema,
                              std::vector<MatchDescriptor> matchers,
                              std::vector<DispatchDescriptor> dispatches,
                              std::vector<KernelDescriptorPack> packs,
                              std::shared_ptr<IKernelHeuristic> heuristic,
                              const std::string& graphMatchSymbol,
                              const std::string& describedBy = {})
        : _schema(std::move(schema))
        , _packs(std::move(packs))
        , _heuristic(std::move(heuristic))
        , _graphMatchFn(graphMatchSymbol.empty()
                            ? nullptr
                            : GraphMatchRegistry::resolve(graphMatchSymbol, describedBy))
    {
        if(_heuristic == nullptr)
        {
            throw std::invalid_argument("kernel ingestor requires a heuristic");
        }

        for(auto& matcher : matchers)
        {
            const auto id = matcher.id;
            const auto description = describeDescriptor("matcher", matcher.name, matcher.id);
            ResolvedMatcher resolved{std::move(matcher), nullptr, nullptr, nullptr};
            if(resolved.descriptor.scope == MatchScope::GRAPH)
            {
                resolved.graphFn
                    = GraphCriterionRegistry::resolve(resolved.descriptor.matchSymbol, description);
            }
            else
            {
                resolved.kernelFn
                    = KernelMatcherRegistry::resolve(resolved.descriptor.matchSymbol, description);
                resolved.prepareFn
                    = KernelPrepareRegistry::tryResolve(resolved.descriptor.matchSymbol);
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
        prepareDefinitions();
        resolveCandidateFilters();
    }

    KernelIngestorSharedState(const KernelIngestorSharedState&) = delete;
    KernelIngestorSharedState& operator=(const KernelIngestorSharedState&) = delete;
    KernelIngestorSharedState(KernelIngestorSharedState&&) = delete;
    KernelIngestorSharedState& operator=(KernelIngestorSharedState&&) = delete;
    ~KernelIngestorSharedState() = default;

    const MetadataSchema& schema() const
    {
        return _schema;
    }

    const std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash>& matchers() const
    {
        return _matchers;
    }

    const std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash>&
        dispatches() const
    {
        return _dispatches;
    }

    const std::vector<KernelDescriptorPack>& packs() const
    {
        return _packs;
    }

    /// One entry per pack, parallel to packs(): its kernels' context-independent
    /// definitions, completed once at construction.
    const std::vector<std::vector<KernelDefinition>>& definitions() const
    {
        return _definitions;
    }

    const IKernelHeuristic& heuristic() const
    {
        return *_heuristic;
    }

    GraphMatchFn graphMatchFn() const
    {
        return _graphMatchFn;
    }

    /// Every kernel-scoped matcher whose symbol also names a CandidateFilterFn, in the
    /// order the packs first list them. Resolved at construction, like the matchers: the
    /// registry is immutable once the provider loaded, and absence means no refinement.
    const std::vector<ResolvedCandidateFilter>& candidateFilters() const
    {
        return _candidateFilters;
    }

private:
    /// Runs each pack's KernelPrepareFn, if one of its kernel-scoped matchers registers
    /// one, over every definition of the pack, once. A pack may have at most one: its
    /// kernels carry a single payload.
    /// @throws std::invalid_argument a pack lists two matchers with different preparers.
    void prepareDefinitions()
    {
        for(size_t packIndex = 0; packIndex < _packs.size(); ++packIndex)
        {
            const auto& pack = _packs[packIndex];
            KernelPrepareFn prepare = nullptr;
            for(const auto& matcherId : pack.matcherIds)
            {
                const auto candidate = _matchers.at(matcherId).prepareFn;
                if(candidate == nullptr || candidate == prepare)
                {
                    continue;
                }
                if(prepare != nullptr)
                {
                    throw std::invalid_argument("pack '" + toString(pack.id)
                                                + "' lists two kernel-scoped matchers with "
                                                  "different kernel preparers; a kernel "
                                                  "carries one prepared payload");
                }
                prepare = candidate;
            }
            if(prepare == nullptr)
            {
                continue;
            }
            for(auto& definition : _definitions[packIndex])
            {
                definition.prepared = prepare(definition);
                definition.preparedBy = prepare;
            }
        }
    }

    void resolveCandidateFilters()
    {
        std::unordered_set<DescriptorId, DescriptorIdHash> visited;
        for(const auto& pack : _packs)
        {
            for(const auto& matcherId : pack.matcherIds)
            {
                const auto& matcher = _matchers.at(matcherId);
                if(matcher.descriptor.scope != MatchScope::KERNEL
                   || !visited.insert(matcherId).second)
                {
                    continue;
                }
                const auto filter
                    = CandidateFilterRegistry::tryResolve(matcher.descriptor.matchSymbol);
                if(filter == nullptr)
                {
                    continue;
                }
                ResolvedCandidateFilter resolved{
                    matcherId,
                    describeDescriptor("matcher", matcher.descriptor.name, matcher.descriptor.id),
                    filter,
                    {}};
                for(const auto& other : _packs)
                {
                    if(std::find(other.matcherIds.begin(), other.matcherIds.end(), matcherId)
                       != other.matcherIds.end())
                    {
                        resolved.listingPacks.insert(other.id);
                    }
                }
                _candidateFilters.push_back(std::move(resolved));
            }
        }
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
        // one device. Keyed by the tuple's values in a hash map rather than scanned, which
        // would be quadratic.
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
                MetadataTupleValues tuple;
                tuple.reserve(key.size());
                for(const auto& entry : key)
                {
                    tuple.push_back(entry.second);
                }
                // try_emplace, not operator[], only because misc-const-correctness
                // misreads the operator[] form here and demands a const map.
                std::vector<std::vector<std::string>>& claimants
                    = archesClaimingTuple.try_emplace(std::move(tuple)).first->second;
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
                                                           kernel.treeRoot,
                                                           nullptr,
                                                           nullptr});
            }
            // Pushed even when every kernel was dropped: _definitions is indexed by pack
            // position, so skipping one entry would bind every later pack's definitions to
            // the wrong pack and run the last index out of range.
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

    MetadataSchema _schema;
    std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash> _matchers;
    std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash> _dispatches;
    std::vector<KernelDescriptorPack> _packs;
    std::vector<std::vector<KernelDefinition>> _definitions;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    GraphMatchFn _graphMatchFn = nullptr;
    std::vector<ResolvedCandidateFilter> _candidateFilters;
};

/// The engine's view of its own kernels: which apply to a graph, in what order, and
/// how to launch one. Answers isApplicable (unsortedDefinitions non-empty), getDetails
/// (sortedDefinitions), getMaxWorkspaceSize (getDispatchDetails per survivor, max), and
/// initializeExecutionContext (sortedDefinitions().front(), getDispatchDetails).
///
/// Thread safety. Two independent caches, each guarding itself:
///
/// - `_catalogCache` (`LruCache`) synchronizes internally.
/// - `_winnerCache` is guarded by `_winnerCacheMutex`, as is the one-shot growth warning.
///
/// Neither lock is ever held across a call that takes the other, so the two cannot
/// deadlock. Matchers, the heuristic, and the accessors below all run outside both.
///
/// A cached catalog is a `CatalogView`: pointers into the shared state's immutable
/// definitions, published through `std::shared_ptr<const CatalogView>` and never mutated
/// afterwards. Ordering one builds a new view and stores it, so a caller still holding
/// the old one is never raced. Holding a view costs 8 bytes per kernel rather than a
/// copy of each definition, which is what bounds the cache's memory: each entry is a
/// pointer list, not the catalog's definitions. `winnerFor` returns a copy, since its
/// cache is mutable.
///
/// Concurrent callers can therefore duplicate work -- two threads may rank the same
/// catalog, or record a ranking for the same key -- and the last store wins. Both
/// stores are equally valid, so the cost is the redundant work, never a wrong answer.
template <typename THandle>
class KernelIngestorStateManager
{
public:
    /// How many (graph, device) catalogs to retain; eviction costs a rematch, never a
    /// wrong answer. Each holds a pointer per applicable kernel, so a full cache costs
    /// at most capacity x 8 bytes x the engine's kernel count.
    static constexpr size_t DEFAULT_CATALOG_CACHE_CAPACITY = 256;

    /// A tripwire, not a cap: the winner cache never evicts, because discarding a
    /// measured ranking costs a GPU sweep or a silent quality regression. Crossing this
    /// logs once and changes nothing.
    static constexpr size_t WINNER_CACHE_WARNING_THRESHOLD = 4096;

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
    /// @param engineName The engine's own scoped name (`EngineDescriptor::name`), used to
    ///        compose the on-disk winner-cache shard path -- not `describedBy`, which is
    ///        a diagnostic string, not a path component. Defaults to empty so existing
    ///        direct-construction call sites keep compiling; an empty name disables the
    ///        disk cache rather than sharing one shard across every unnamed instance.
    KernelIngestorStateManager(MetadataSchema schema,
                               std::vector<MatchDescriptor> matchers,
                               std::vector<DispatchDescriptor> dispatches,
                               std::vector<KernelDescriptorPack> packs,
                               std::shared_ptr<IKernelHeuristic> heuristic,
                               const std::string& graphMatchSymbol,
                               const std::string& describedBy = {},
                               size_t catalogCacheCapacity = DEFAULT_CATALOG_CACHE_CAPACITY,
                               std::string engineName = {})
        : KernelIngestorStateManager(
              std::make_shared<const KernelIngestorSharedState<THandle>>(std::move(schema),
                                                                         std::move(matchers),
                                                                         std::move(dispatches),
                                                                         std::move(packs),
                                                                         std::move(heuristic),
                                                                         graphMatchSymbol,
                                                                         describedBy),
              catalogCacheCapacity,
              std::move(engineName))
    {
    }

    /// A manager over @p state, which another manager may already be serving: nothing is
    /// resolved, completed or validated again, so this is O(1) in the number of kernels.
    /// The caches (catalog, winner records, shard latches) are this manager's own.
    /// @param engineName As above; empty disables the on-disk winner cache.
    /// @throws std::invalid_argument @p state is null.
    explicit KernelIngestorStateManager(
        std::shared_ptr<const KernelIngestorSharedState<THandle>> state,
        size_t catalogCacheCapacity = DEFAULT_CATALOG_CACHE_CAPACITY,
        std::string engineName = {})
        : _state(requireState(std::move(state)))
        , _schema(_state->schema())
        , _matchers(_state->matchers())
        , _dispatches(_state->dispatches())
        , _packs(_state->packs())
        , _definitions(_state->definitions())
        , _heuristic(_state->heuristic())
        , _graphMatchFn(_state->graphMatchFn())
        , _catalogCache(catalogCacheCapacity)
        , _engineName(std::move(engineName))
    {
    }

    /// Not relocatable: the members above alias into *_state, and the engine's plan
    /// builder holds a reference to this manager.
    KernelIngestorStateManager(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager& operator=(const KernelIngestorStateManager&) = delete;
    KernelIngestorStateManager(KernelIngestorStateManager&&) = delete;
    KernelIngestorStateManager& operator=(KernelIngestorStateManager&&) = delete;
    ~KernelIngestorStateManager() = default;

    /// The immutable state this manager serves, for building another manager over it.
    const std::shared_ptr<const KernelIngestorSharedState<THandle>>& sharedState() const
    {
        return _state;
    }

    const MetadataSchema& metadataSchema() const
    {
        return _schema;
    }

    /// The kernels that apply to the graph and device @p context names, unordered, as a
    /// shared snapshot that points into the shared state rather than copying it. The
    /// first call for a (graph, device) matches every kernel once; later calls return
    /// the cached snapshot.
    std::shared_ptr<const CatalogView> catalogView(const MatchContext& context) const
    {
        return catalogFor(context);
    }

    /// catalogView(), best first: a benchmarked record covering the whole catalog
    /// supplies a measured order and `rank()` is never called; otherwise the heuristic
    /// orders it. The ordered snapshot replaces the unordered one in the cache.
    std::shared_ptr<const CatalogView> sortedCatalogView(const MatchContext& context) const
    {
        // Mirrors catalogFor's own reject guard: cacheKey() below only reads graph and
        // device ordinal, not arch, so without this an unresolved-arch context would
        // cache an empty catalog under the SAME key a later, resolved call for this
        // device reuses -- permanently hiding that device's real catalog.
        if(context.deviceId == NO_DEVICE || context.deviceProperties.gcnArchName.empty())
        {
            return catalogFor(context);
        }

        auto view = catalogFor(context);

        // A measured order is final; a heuristic one is provisional, so this lookup runs
        // again even when the catalog is already sorted -- a sweep can postdate the
        // memoized sort.
        if(view->isSorted && view->orderedFromRecord)
        {
            return view;
        }

        auto sorted = std::make_shared<CatalogView>();
        sorted->bound = view->bound;
        sorted->owner = view->owner;
        if(auto ordered = orderFromWinnerRecord(view->entries, context); ordered.has_value())
        {
            sorted->entries = std::move(*ordered);
            sorted->orderedFromRecord = true;
        }
        else
        {
            if(view->isSorted)
            {
                return view;
            }
            sorted->entries = _heuristic.rankView(view->entries, view->bound, context);
        }
        sorted->isSorted = true;

        std::shared_ptr<const CatalogView> published = std::move(sorted);
        if(const auto key = cacheKey(context); key.has_value())
        {
            // put, not putIfAbsent: sorted is strictly better than whatever is cached.
            _catalogCache.put(*key, published);
        }
        return published;
    }

    /// Every kernel that applies to the graph and device @p context names, unordered;
    /// copied out of catalogView().
    std::vector<KernelDefinition> unsortedDefinitions(const MatchContext& context) const
    {
        return catalogView(context)->toCatalog().entries;
    }

    /// The unranked catalog and the state matching bound, from one lookup; a copy of
    /// catalogView().
    Catalog unsortedCatalog(const MatchContext& context) const
    {
        return catalogView(context)->toCatalog();
    }

    /// Every kernel that applies to the graph and device @p context names, best first;
    /// copied out of sortedCatalogView().
    std::vector<KernelDefinition> sortedDefinitions(const MatchContext& context) const
    {
        return sortedCatalogView(context)->toCatalog().entries;
    }

    /// The ordered catalog and the state matching bound, from one lookup; a copy of
    /// sortedCatalogView().
    Catalog sortedCatalog(const MatchContext& context) const
    {
        return sortedCatalogView(context)->toCatalog();
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

        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        _winnerCache[key] = std::move(adopted);

        if(_winnerCache.size() > WINNER_CACHE_WARNING_THRESHOLD && !_winnerCacheGrowthWarned)
        {
            _winnerCacheGrowthWarned = true;
            HIPDNN_PLUGIN_LOG_WARN(
                "ingestor: winner cache holds "
                << _winnerCache.size() << " entries, past the soft threshold of "
                << WINNER_CACHE_WARNING_THRESHOLD
                << "; it does not evict, so this is reported once rather than acted on");
        }
    }

    /// The ranking recorded for @p key, or nullopt. Returns a copy: the caller walks it
    /// outside the lock, and a reference would outlive the guard.
    ///
    /// Read-through: an in-memory miss loads @p key's on-disk shard once per shard (a
    /// per-shard "already loaded" flag), taking only `_winnerCacheMutex` to merge
    /// decoded entries into `_winnerCache` -- unlike write-back, this never holds a file
    /// lock across the read.
    std::optional<WinnerRecord> winnerFor(const WinnerKey& key) const
    {
        {
            const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
            const auto found = _winnerCache.find(key);
            if(found != _winnerCache.end())
            {
                return found->second;
            }
        }

        loadShardIfAbsent(key.device.properties().gcnArchName);

        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        const auto found = _winnerCache.find(key);
        if(found == _winnerCache.end())
        {
            return std::nullopt;
        }
        return found->second;
    }

    /// How many rankings are held. For tests and diagnostics; the cache never evicts, so
    /// this only grows.
    size_t winnerCacheSize() const
    {
        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        return _winnerCache.size();
    }

    /// Is it worth building a WinnerKey for @p gcnArchName? True if the in-memory cache
    /// holds anything, or this arch's shard has not been attempted yet. Probes the
    /// stripped arch, matching how `loadShardIfAbsent()` latches.
    bool mightHaveWinnerFor(const std::string& gcnArchName) const
    {
        const std::lock_guard<std::mutex> guard(_winnerCacheMutex);
        return !_winnerCache.empty()
               || _loadedWinnerShards.find(std::string(stripArchFeatures(gcnArchName)))
                      == _loadedWinnerShards.end();
    }

    /// Resolves how to size and launch @p kernel.
    /// @throws std::runtime_error if the kernel's dispatch descriptor is unknown.
    KernelDispatcher<THandle> getDispatchDetails(const KernelDefinition& kernel) const
    {
        return {kernel, dispatchHandler(kernel)};
    }

    /// The handler that sizes and launches @p kernel, without copying the kernel.
    /// @throws std::runtime_error if the kernel's dispatch descriptor is unknown.
    const IKernelDispatchHandler<THandle>* dispatchHandler(const KernelDefinition& kernel) const
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

    /// The distinct values @p field takes across @p kernels, in ranked-first order.
    static std::vector<MetadataValue> knobValues(const std::vector<KernelDefinition>& kernels,
                                                 const std::string& field)
    {
        std::vector<const KernelDefinition*> view;
        view.reserve(kernels.size());
        for(const auto& kernel : kernels)
        {
            view.push_back(&kernel);
        }
        return knobValues(view, field);
    }

    /// knobValues() over a catalog view.
    static std::vector<MetadataValue>
        knobValues(const std::vector<const KernelDefinition*>& kernels, const std::string& field)
    {
        std::vector<MetadataValue> values;
        for(const auto* kernel : kernels)
        {
            const auto it = kernel->metadata.find(field);
            if(it == kernel->metadata.end())
            {
                continue;
            }
            if(std::find(values.begin(), values.end(), it->second) == values.end())
            {
                values.push_back(it->second);
            }
        }
        return values;
    }

private:
    static std::shared_ptr<const KernelIngestorSharedState<THandle>>
        requireState(std::shared_ptr<const KernelIngestorSharedState<THandle>> state)
    {
        if(state == nullptr)
        {
            throw std::invalid_argument("kernel ingestor state manager requires a shared state");
        }
        return state;
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

    std::shared_ptr<const CatalogView> catalogFor(const MatchContext& context) const
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
            return std::make_shared<const CatalogView>();
        }

        const auto key = cacheKey(context);
        if(key.has_value())
        {
            if(auto cached = _catalogCache.get(*key); cached.has_value())
            {
                HIPDNN_PLUGIN_LOG_TRACE("ingestor: catalog cache hit for device "
                                        << context.deviceId);
                return std::move(*cached);
            }
        }
        else
        {
            HIPDNN_PLUGIN_LOG_TRACE(
                "ingestor: graph carries no identity, so its catalog cannot be cached");
        }

        std::shared_ptr<const CatalogView> catalog
            = std::make_shared<const CatalogView>(buildCatalog(context));

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
    CatalogView buildCatalog(const MatchContext& context) const
    {
        CatalogView catalog;
        catalog.owner = _state;
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

            // The pack's kernel-scoped matchers, resolved once per pack rather than looked
            // up per kernel.
            const auto kernelFns = kernelMatcherFns(pack);
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

                // Pointed at, not copied: every field was settled when the shared state was
                // built, and nothing downstream mutates a definition.
                if(std::all_of(kernelFns.begin(), kernelFns.end(), [&](KernelMatcherFn kernelFn) {
                       return kernelFn(context, catalog.bound, precomputed);
                   }))
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

        applyCandidateFilters(context, catalog);

        HIPDNN_PLUGIN_LOG_INFO("ingestor: catalog for device "
                               << context.deviceId << " holds " << catalog.entries.size()
                               << " kernel(s) from " << _packs.size() << " pack(s)");
        return catalog;
    }

    /// Each kernel-scoped matcher whose symbol also names a CandidateFilterFn narrows, as
    /// one set, every kernel admitted from the packs listing it. Runs once every pack has
    /// been matched, so a set sharded across packs is refined whole; what stays keeps its
    /// order. The filters and their packs were resolved with the shared state.
    void applyCandidateFilters(const MatchContext& context, CatalogView& catalog) const
    {
        if(catalog.entries.empty())
        {
            return;
        }
        std::vector<bool> keep(catalog.entries.size(), true);
        for(const auto& resolved : _state->candidateFilters())
        {
            std::vector<size_t> indices;
            std::vector<const KernelDefinition*> candidates;
            for(size_t i = 0; i < catalog.entries.size(); ++i)
            {
                if(keep[i] && resolved.listingPacks.count(catalog.entries[i]->packId) != 0)
                {
                    indices.push_back(i);
                    candidates.push_back(catalog.entries[i]);
                }
            }
            if(candidates.empty())
            {
                continue;
            }

            const auto kept = resolved.filter(context, catalog.bound, candidates);
            if(kept.size() != candidates.size())
            {
                throw std::runtime_error("the candidate filter of " + resolved.description
                                         + " returned " + std::to_string(kept.size())
                                         + " flag(s) for " + std::to_string(candidates.size())
                                         + " kernel(s)");
            }
            size_t survivors = 0;
            for(size_t i = 0; i < kept.size(); ++i)
            {
                if(kept[i])
                {
                    ++survivors;
                }
                else
                {
                    keep[indices[i]] = false;
                }
            }
            HIPDNN_PLUGIN_LOG_INFO("ingestor: the candidate filter of "
                                   << resolved.description << " kept " << survivors << " of "
                                   << candidates.size() << " kernel(s)");
        }

        size_t next = 0;
        for(size_t i = 0; i < catalog.entries.size(); ++i)
        {
            if(keep[i])
            {
                catalog.entries[next++] = catalog.entries[i];
            }
        }
        catalog.entries.resize(next);
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

    std::vector<KernelMatcherFn> kernelMatcherFns(const KernelDescriptorPack& pack) const
    {
        std::vector<KernelMatcherFn> kernelFns;
        for(const auto& matcherId : pack.matcherIds)
        {
            const auto& matcher = _matchers.at(matcherId);
            if(matcher.descriptor.scope == MatchScope::KERNEL)
            {
                kernelFns.push_back(matcher.kernelFn);
            }
        }
        return kernelFns;
    }

    /// The measured order for @p context's graph and device, or nullopt when no record
    /// covers @p entries and the caller must rank as it always has.
    ///
    /// Full coverage is required: a partial record cannot order the rest, and
    /// interleaving measured with unmeasured entries would not be a valid order.
    std::optional<std::vector<const KernelDefinition*>>
        orderFromWinnerRecord(const std::vector<const KernelDefinition*>& entries,
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

        auto ordered = orderIfFullyCoveredView(*record, entries);
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

        if(_engineName.empty())
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
        auto [shard, openStatus] = openWinnerCacheShard(_engineName, gcnArchName);
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
        // Walk in reverse and never overwrite: within the file the last line for a key
        // wins, and a key already in memory was put there by this process's own
        // measurement, which is newer than anything the file can offer.
        for(auto it = decoded.rbegin(); it != decoded.rend(); ++it)
        {
            if(!it->first.graph.isUsable())
            {
                // A key whose graph carries no content is not even equal to itself
                // (GraphContentKey::operator==), so it can never be looked up -- and as
                // a non-reflexive key it would violate unordered_map's requirements on
                // KeyEqual. recordWinner() rejects these on the write side too.
                continue;
            }
            _winnerCache.try_emplace(it->first, std::move(it->second));
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
        if(_engineName.empty())
        {
            return record;
        }

        const auto& gcnArchName = key.device.properties().gcnArchName;
        auto [shard, openStatus] = openWinnerCacheShard(_engineName, gcnArchName);
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
                                   << _engineName << "' arch '" << gcnArchName
                                   << "'; on-disk miss, in-memory behavior continues");
            break;
        case WinnerShardFailureKind::LOCK:
            HIPDNN_PLUGIN_LOG_INFO(
                "ingestor: could not lock the on-disk winner cache shard for engine '"
                << _engineName << "' arch '" << gcnArchName
                << "'; on-disk write-back skipped for this call");
            break;
        case WinnerShardFailureKind::READ:
            HIPDNN_PLUGIN_LOG_INFO("ingestor: on-disk winner cache for engine '"
                                   << _engineName << "' arch '" << gcnArchName
                                   << "' could not be read; on-disk miss, in-memory behavior "
                                      "continues");
            break;
        case WinnerShardFailureKind::APPEND:
            HIPDNN_PLUGIN_LOG_INFO(
                "ingestor: could not append a benchmarked ranking to the on-disk winner "
                "cache for engine '"
                << _engineName << "' arch '" << gcnArchName
                << "'; the ranking is kept in-memory only for this process");
            break;
        case WinnerShardFailureKind::VERSION_MISMATCH:
            HIPDNN_PLUGIN_LOG_WARN("ingestor: on-disk winner cache for engine '"
                                   << _engineName << "' arch '" << gcnArchName
                                   << "' declined: version mismatch; on-disk miss, in-memory "
                                      "behavior continues");
            break;
        default:
            // Unreachable; present because clang-tidy requires an explicit default.
            break;
        }
    }

    /// Shared, immutable, and declared first: every alias below binds into it.
    std::shared_ptr<const KernelIngestorSharedState<THandle>> _state;
    const MetadataSchema& _schema;
    const std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash>& _matchers;
    const std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash>&
        _dispatches;
    const std::vector<KernelDescriptorPack>& _packs;
    /// One entry per pack, parallel to _packs: its kernels' context-independent
    /// definitions, completed once when the shared state was built.
    const std::vector<std::vector<KernelDefinition>>& _definitions;
    const IKernelHeuristic& _heuristic;
    GraphMatchFn _graphMatchFn = nullptr;
    mutable LruCache<CatalogKey, std::shared_ptr<const CatalogView>, CatalogKeyHash> _catalogCache;

    /// The engine's own scoped name, used to locate its on-disk winner-cache shard.
    /// Empty disables the disk cache for this manager entirely.
    std::string _engineName;

    /// Unbounded, never evicted, own mutex (see WINNER_CACHE_WARNING_THRESHOLD). Separate
    /// from _catalogCache's lock: a shared lock would serialize benchmarking write-back
    /// against ordinary catalog lookups.
    mutable std::unordered_map<WinnerKey, WinnerRecord, WinnerKeyHash> _winnerCache;
    mutable std::mutex _winnerCacheMutex;
    mutable bool _winnerCacheGrowthWarned = false;
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
