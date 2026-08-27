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
/// Thread safety. Two independent caches, each guarding itself:
///
/// - `_catalogCache` (`LruCache`) synchronizes internally.
/// - `_winnerCache` is guarded by `_winnerCacheMutex`, as is the one-shot growth warning.
///
/// Neither lock is ever held across a call that takes the other, so the two cannot
/// deadlock. Matchers, the heuristic, and the accessors below all run outside both.
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
        : _schema(std::move(schema))
        , _packs(std::move(packs))
        , _heuristic(std::move(heuristic))
        , _graphMatchFn(graphMatchSymbol.empty()
                            ? nullptr
                            : GraphMatchRegistry::resolve(graphMatchSymbol, describedBy))
        , _catalogCache(catalogCacheCapacity)
        , _engineName(std::move(engineName))
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

    /// The ordered catalog and the state matching bound, from one lookup.
    ///
    /// A benchmarked record covering the whole catalog supplies a measured order and
    /// `rank()` is never called; otherwise the heuristic orders it.
    Catalog sortedCatalog(const MatchContext& context) const
    {
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

    /// Records @p record under @p key, replacing any earlier ranking for it: a later
    /// sweep measured the current candidate set, and the coverage gate only widens.
    ///
    /// Write-back: if this manager has an engine name, the record is also appended to
    /// the on-disk shard. Unlike the in-memory-only read-through path below, this holds
    /// the shard's own file lock across the whole read-then-append sequence as one
    /// critical section -- `_winnerCacheMutex` alone cannot serialize against a second
    /// process sharing the file. If the shard's current ranking for this key already
    /// matches, nothing is written and the on-disk record is adopted; if it differs, the
    /// new ranking is appended and supersedes the old one under the reader's
    /// last-line-wins merge. Any disk failure degrades to in-memory-only, keeping the
    /// measurement just taken, logged once per manager.
    void recordWinner(const WinnerKey& key, const WinnerRecord& record) const
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

        WinnerRecord adopted = writeBackToShard(key, record);

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
        // Pack pruning and matchers both read the device, so nothing below can be
        // answered without one. Checked here rather than in every provider's matchers,
        // where an omission is invisible.
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

                // Copied, not rebuilt: every field was settled at construction, and the
                // kernel matcher below reads the definition without mutating it.
                KernelDefinition definition = precomputed;

                if(kernelLevelMatchersPass(pack, context, catalog.bound, definition))
                {
                    catalog.entries.push_back(std::move(definition));
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

    /// Write-back: re-reads @p key's shard under its LineStore lock and appends
    /// @p record unless the shard's current ranking for @p key already matches it, as
    /// one critical section under the shard's own lock (mirrors
    /// `LineStoreLockHelper.cpp`). The file lock rather than `_winnerCacheMutex`,
    /// because the racing writer may be another process.
    ///
    /// A record is never immutable: a shard may hold several lines for one key, and the
    /// reader resolves them last-line-wins (see `loadShardIfAbsent()`), so appending a
    /// newer ranking is how a record is replaced. Without that, a catalog that gains a
    /// kernel could never satisfy the coverage gate again, and every later run would
    /// re-benchmark and discard the result forever.
    ///
    /// @return The on-disk record, if its ranked ids already match @p record's -- an
    ///     unchanged catalog, so nothing is written; otherwise @p record itself, both
    ///     when it was appended and on every disk failure, since write-back is
    ///     best-effort and the caller's own measurement is the right value to keep in
    ///     memory.
    WinnerRecord writeBackToShard(const WinnerKey& key, WinnerRecord record) const
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

        if(onDiskWinner != nullptr && rankedIdsEqual(*onDiskWinner, record))
        {
            // Same ranking already on disk: adopt it rather than appending a duplicate
            // line for an unchanged catalog.
            hipdnn_data_sdk::utilities::unlockLineStore(*shard);
            return *onDiskWinner;
        }

        if(hipdnn_data_sdk::utilities::appendLine(*shard, encodeWinnerRecordLine(key, record))
           != hipdnn_data_sdk::utilities::LineStoreStatus::OK)
        {
            logShardFailureOnce(WinnerShardFailureKind::APPEND, gcnArchName);
        }
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
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

    MetadataSchema _schema;
    std::unordered_map<DescriptorId, ResolvedMatcher, DescriptorIdHash> _matchers;
    std::unordered_map<DescriptorId, ResolvedDispatch<THandle>, DescriptorIdHash> _dispatches;
    std::vector<KernelDescriptorPack> _packs;
    /// One entry per pack, parallel to _packs: its kernels' context-independent
    /// definitions, completed once at construction.
    std::vector<std::vector<KernelDefinition>> _definitions;
    std::shared_ptr<IKernelHeuristic> _heuristic;
    GraphMatchFn _graphMatchFn = nullptr;
    mutable LruCache<CatalogKey, Catalog, CatalogKeyHash> _catalogCache;

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
