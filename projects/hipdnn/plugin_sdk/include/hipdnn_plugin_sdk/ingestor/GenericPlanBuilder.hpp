// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/RankingMetric.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/SlotProgramBenchmark.hpp>
#include <hipdnn_plugin_sdk/ingestor/StagedApplicability.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// A caller's requested value for each knob it explicitly set, keyed by KMD field
/// name.
using KnobFilter = std::map<std::string, int64_t>;

namespace detail
{

/// One MetadataValue as the JSON value it already is, kept verbatim: categorical
/// encoding belongs to the feature extractor (RFC 0019 §7).
inline nlohmann::json metadataValueToJson(const MetadataValue& value)
{
    return std::visit([](const auto& held) { return nlohmann::json(held); }, value);
}

inline void addMetadataFeature(nlohmann::json& features,
                               const std::string& name,
                               const MetadataValue& value)
{
    features[name] = metadataValueToJson(value);
    if(const auto* values = std::get_if<std::vector<int64_t>>(&value))
    {
        for(size_t i = 0; i < values->size(); ++i)
        {
            // emplace: a directly published scalar wins over a synthesized indexed name.
            features.emplace(name + "[" + std::to_string(i) + "]", (*values)[i]);
        }
    }
}

/// @p value as lowercase hex without leading zeros. std::to_chars ignores the global locale,
/// which a stream would consult and could group the digits under.
inline std::string toHex(uint64_t value)
{
    std::array<char, 16> digits{};
    const auto end = std::to_chars(digits.data(), digits.data() + digits.size(), value, 16).ptr;
    return {digits.data(), end};
}

/// An integer knob setting, as an engine config and an enumerated candidate carry it.
inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>
    makeIntKnobSetting(const std::string& name, int64_t value)
{
    auto setting = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>();
    setting->knob_id = name;
    hipdnn_flatbuffers_sdk::data_objects::IntValueT integer;
    integer.value = value;
    setting->value.Set(integer);
    return setting;
}

} // namespace detail

/// What a `TSettings` used with GenericPlanBuilder must carry, grouped so a second
/// provider embeds one member rather than replicating loose fields by name.
struct IngestorSettings
{
    KnobFilter knobFilter;
    bool benchmarkingEnabled = false;
    std::optional<int64_t> workspaceLimit;
};

/// The one plan builder a descriptor-backed engine has: a catalog entry is a
/// candidate, and this builds a plan for whichever one selection chose.
/// @tparam THandle Must expose `hipStream_t getStream() const`, the stream benchmarking
///         times kernels on. Required of ingestor users only -- validateHandleType()
///         does not ask for it.
/// @tparam TSettings Must carry an `IngestorSettings ingestorSettings` member.
/// @tparam TContext Must expose `const TSettings& executionSettings() const`, holding
///         the settings initializeExecutionSettings() populated.
template <typename THandle, typename TSettings, typename TContext>
class GenericPlanBuilder : public IPlanBuilder<THandle, TSettings, TContext>
{
    static_assert(HasGetStream<THandle>::value,
                  "A handle used with the kernel ingestor must have a "
                  "'hipStream_t getStream() const' method: benchmarking times candidate "
                  "kernels with HIP events on that stream");

public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /// References (@p engine, @p deviceResolver) are owned by the engine, which
    /// outlives its builder. @p timer overrides BenchmarkPlan's default HIP-event
    /// timer; tests inject a deterministic one so the real write-back factory below is
    /// exercised without a device.
    GenericPlanBuilder(const EngineDescriptor& engine,
                       const KernelIngestorStateManager<THandle>& stateManager,
                       const IDeviceResolver<THandle>& deviceResolver,
                       typename BenchmarkPlan<THandle>::Timer timer = {})
        : _engine(engine)
        , _stateManager(stateManager)
        , _deviceResolver(deviceResolver)
        , _timer(std::move(timer))
        , _knobs(buildKnobs(_engine, _stateManager))
    {
    }

    /// Declines on any error rather than propagating: engine enumeration walks every
    /// engine in one loop, so a throw here would deny the caller the engines that would
    /// have answered. Device resolution, matching, and the schema read can all throw.
    bool isApplicable(const THandle& handle, const IGraph& opGraph) const override
    {
        try
        {
            if(!understandsGraph(opGraph))
            {
                return false;
            }
            return _stateManager.isApplicable(contextFor(handle, opGraph));
        }
        catch(const std::exception& error)
        {
            logApplicabilityFailure(error);
            return false;
        }
    }

    /// isApplicable() split for IStagedApplicability: the same verdict, cache contents,
    /// logging and error handling, with the catalog's per-variant loop left pending when
    /// the caller thread cannot settle it cheaply. The device and catalog key come from
    /// @p scope when an earlier engine already resolved them through the same resolver,
    /// and are recorded there when this engine is the first.
    /// The returned pending check borrows @p opGraph and @p scope, and what @p scope holds:
    /// keep them alive until its finishApplicability() has returned (see PendingApplicability).
    StagedApplicability beginApplicability(const THandle& handle,
                                           const IGraph& opGraph,
                                           ApplicabilityScope& scope) const
    {
        try
        {
            if(!understandsGraph(opGraph))
            {
                return decided(false);
            }

            auto& shared = scope.device();
            const void* resolver = &_deviceResolver;
            DeviceId deviceId = NO_DEVICE;
            // Borrowed, never owned here: from the scope, or from this engine's own resolver
            // when it is not the scope's. A pending check copies the latter before it outlives
            // this call, so it never depends on the resolver keeping the reference.
            const DeviceProperties* deviceProperties = nullptr;
            bool fromResolver = false;
            if(shared.has_value() && shared->resolver == resolver)
            {
                deviceId = shared->deviceId;
                deviceProperties = &shared->properties;
            }
            else
            {
                const auto resolved = contextFor(handle, opGraph);
                deviceId = resolved.deviceId;
                if(!shared.has_value())
                {
                    shared = ApplicabilityScope::Device{
                        resolver, deviceId, resolved.deviceProperties, std::nullopt, false};
                    deviceProperties = &shared->properties;
                }
                else
                {
                    deviceProperties = &resolved.deviceProperties;
                    fromResolver = true;
                }
            }
            const MatchContext context{opGraph, deviceId, *deviceProperties};

            // The key carries no resolver identity, only the device id and this engine's
            // version, so the scope's key serves only engines of the version that built it.
            std::optional<CatalogKey> ownedKey;
            const CatalogKey* key = nullptr;
            if(shared->deviceId == deviceId
               && (!shared->keyBuilt || shared->keyVersion == _engine.revision))
            {
                if(!shared->keyBuilt)
                {
                    shared->key = _stateManager.cacheKey(context);
                    shared->keyVersion = _engine.revision;
                    shared->keyBuilt = true;
                }
                key = shared->key.has_value() ? &*shared->key : nullptr;
            }
            else
            {
                ownedKey = _stateManager.cacheKey(context);
                key = ownedKey.has_value() ? &*ownedKey : nullptr;
            }

            if(key == nullptr)
            {
                return decided(_stateManager.applicableUncached(context));
            }
            if(const auto hit = _stateManager.cachedVerdict(context, *key); hit.has_value())
            {
                return decided(*hit);
            }

            auto stage = _stateManager.beginCatalog(context, *key);
            if(!stage.has_value())
            {
                return decided(false);
            }

            auto pending = std::make_unique<PendingCatalog>();
            pending->predictedVariants = stage->predictedVariants;
            pending->owner = &_stateManager;
            pending->graph = &opGraph;
            pending->deviceId = deviceId;
            if(fromResolver)
            {
                // Set only once the copy is in the heap-allocated state, which does not move.
                pending->ownedProperties = *deviceProperties;
                pending->deviceProperties = &*pending->ownedProperties;
            }
            else
            {
                pending->deviceProperties = deviceProperties;
            }
            pending->ownedKey = std::move(ownedKey);
            pending->key = pending->ownedKey.has_value() ? &*pending->ownedKey : key;
            pending->stage = std::move(*stage);
            return {false, false, PendingApplicability(std::move(pending))};
        }
        catch(const std::exception& error)
        {
            logApplicabilityFailure(error);
            return decided(false);
        }
    }

    /// The per-variant loop @p pending carries; any thread, no HIP calls. A pending this
    /// builder's state manager did not produce is declined with an error.
    bool finishApplicability(PendingApplicability&& pending) const
    {
        const auto state = pending.release();
        try
        {
            // `owner` is set only by this builder's begin, which always derives
            // PendingCatalog, so a match makes the downcast safe without RTTI.
            auto* catalog = state != nullptr && state->owner == &_stateManager
                                ? static_cast<PendingCatalog*>(state.get())
                                : nullptr;
            if(catalog == nullptr)
            {
                throw std::invalid_argument("the pending applicability check belongs to "
                                            "another engine");
            }
            return _stateManager.finishCatalog(
                catalog->context(), *catalog->key, std::move(catalog->stage));
        }
        catch(const std::exception& error)
        {
            logApplicabilityFailure(error);
            return false;
        }
    }

    bool understandsGraph(const IGraph& opGraph) const
    {
        const auto schemaFloor = graphSchemaFloor(opGraph);
        if(_engine.sdkVersion < schemaFloor)
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                   << _engine.name << "' declined the graph: it understands graph "
                                   << "schema " << _engine.sdkVersion.str()
                                   << " but this graph requires " << schemaFloor.str());
            return false;
        }
        return true;
    }

    /// The scratch a call needs: the largest any surviving kernel needs, reused, not
    /// partitioned, since kernels launch one at a time on one stream. A slotted engine adds
    /// the aligned shared region to the largest per-slot scratch.
    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const TSettings& executionSettings) const override
    {
        const auto context = contextFor(handle, opGraph);
        if(_stateManager.isSlotted())
        {
            const auto selection
                = slotCandidates(context, executionSettings.ingestorSettings, false);
            size_t maxScratch = 0;
            for(const auto& kernels : selection.perSlot)
            {
                for(const KernelDefinition* kernel : kernels)
                {
                    const auto* handler = _stateManager.dispatchHandlerFor(*kernel);
                    maxScratch = std::max(
                        maxScratch, handler->workspaceBytes(context, selection.bound, *kernel));
                }
            }
            return selection.alignedSharedBytes + maxScratch;
        }
        const auto catalog = _stateManager.unsortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        const auto filtered
            = applyConstraints(catalog, executionSettings.ingestorSettings, context);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(
                0, executionSettings.ingestorSettings.knobFilter, catalog.entries.size());
        }

        size_t maxBytes = 0;
        for(const KernelDefinition* kernel : filtered)
        {
            const auto* handler = _stateManager.dispatchHandlerFor(*kernel);
            maxBytes = std::max(maxBytes, handler->workspaceBytes(context, catalog.bound, *kernel));
        }
        return maxBytes;
    }

    /// The override is consulted unconditionally: it must change the outcome even when
    /// engineConfig is invalid or carries no knob, which is what makes a plain
    /// hipdnnExecute benchmark. readBenchmarkingEnabled() always runs so the knob's own
    /// answer is available to value_or().
    void initializeExecutionSettings(const THandle& handle,
                                     const IGraph& opGraph,
                                     const IEngineConfig& engineConfig,
                                     TSettings& executionSettings) const override
    {
        auto& settings = executionSettings.ingestorSettings;
        settings.knobFilter = readKnobFilter(engineConfig);
        if(_stateManager.isSlotted() && !settings.knobFilter.empty())
        {
            rejectEmptiedSlots(contextFor(handle, opGraph), settings.knobFilter);
        }
        settings.workspaceLimit = heuristics::workspaceLimit(engineConfig);
        settings.benchmarkingEnabled
            = benchmarkingOverrideFromEnv().value_or(readBenchmarkingEnabled(engineConfig));
    }

    /// Kernel choice follows the engine configuration's ranking metric (RFC 0019 §11.4):
    /// that metric's `sort_kernel_catalog` UHD ranks the catalog when it has one, the
    /// engine's default ranker otherwise, and the sorted order is cached per metric.
    void buildPlan(const THandle& handle,
                   const IGraph& opGraph,
                   const IEngineConfig& engineConfig,
                   TContext& executionContext) const override
    {
        const auto context
            = contextFor(handle, opGraph, heuristics::rankingMetric(engineConfig).name);
        const auto& settings = executionContext.executionSettings().ingestorSettings;
        if(_stateManager.isSlotted())
        {
            buildSlottedPlan(opGraph, context, settings, executionContext);
            return;
        }
        const auto catalog = _stateManager.sortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        const auto filtered = applyConstraints(catalog, settings, context);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(0, settings.knobFilter, catalog.entries.size());
        }

        // Orderability is decided once, on the full catalog, in sortedCatalog(); `filtered` is
        // that measured order restricted. Re-asking against `filtered` would let a pin change
        // the order source (RFC 0019 §5 step 8).
        if(catalog.measuredRecord != nullptr)
        {
            // Walk the ranked list rather than take its front: a GenericPlan can fail to
            // build, and a cache hit must not be stricter than an empty cache.
            for(size_t rank = 0; rank < filtered.size(); ++rank)
            {
                std::string failure;
                try
                {
                    auto plan = std::make_unique<GenericPlan<THandle>>(
                        _stateManager.getDispatchDetails(*filtered[rank]), context, catalog.bound);

                    // The record is the catalog's own snapshot, so this holds even after the
                    // winner cache evicts its copy.
                    HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                           << _engine.name << "' served kernel "
                                           << toString(filtered[rank]->kernelId) << " at rank "
                                           << rank << " from a benchmarked record of "
                                           << catalog.measuredRecord->size() << " entry(s) for "
                                           << filtered.size() << " candidate(s)");

                    executionContext.setPlan(std::move(plan));
                    return;
                }
                catch(const HipdnnPluginException& error)
                {
                    // A malformed descriptor is the author's mistake: rethrow rather than
                    // silently serve a different kernel.
                    if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
                    {
                        throw;
                    }
                    failure = error.what();
                }
                catch(const std::exception& error)
                {
                    failure = error.what();
                }

                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' could not build a plan for "
                                       << toString(filtered[rank]->kernelId) << " at rank " << rank
                                       << ": " << failure << "; trying the next ranked entry");
            }

            HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                   << _engine.name
                                   << "' found a benchmarked record whose entries no longer "
                                      "resolve; falling back to normal selection");
        }

        // The lookup stays lazy: a WinnerKey hashes the whole graph, so it is not worth
        // building when neither a benchmark write nor a possible hit needs one.
        std::optional<WinnerKey> winnerKey;
        std::optional<WinnerRecord> record;
        if(settings.benchmarkingEnabled
           || _stateManager.mightHaveWinnerFor(context.deviceProperties.gcnArchName))
        {
            winnerKey
                = WinnerKey{hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{opGraph},
                            DeviceKey{context.deviceProperties}};
            record = _stateManager.winnerFor(*winnerKey);
        }

        if(catalog.measuredRecord == nullptr && record.has_value() && settings.benchmarkingEnabled)
        {
            // A record only reorders candidates measured together; one that does not cover
            // the whole catalog is ignored rather than partially trusted.
            HIPDNN_PLUGIN_LOG_INFO(
                "ingestor: engine '"
                << _engine.name << "' has a benchmarked record that does not fully cover its "
                << catalog.entries.size() << " applicable kernel(s); re-benchmarking "
                << filtered.size() << " candidate(s)");
        }

        if(!settings.benchmarkingEnabled)
        {
            // Constructing a GenericPlan runs prepare()/workspaceBytes(), so a kernel whose
            // code object cannot be loaded must cost only itself while a sibling that loads
            // still serves the graph. Same reason the ranked walk above walks.
            std::vector<std::string> failures;
            for(size_t rank = 0; rank < filtered.size(); ++rank)
            {
                try
                {
                    auto plan = std::make_unique<GenericPlan<THandle>>(
                        _stateManager.getDispatchDetails(*filtered[rank]), context, catalog.bound);

                    HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                           << _engine.name << "' selected kernel "
                                           << toString(filtered[rank]->kernelId) << " at rank "
                                           << rank << " from " << filtered.size()
                                           << " candidate(s) (" << catalog.entries.size()
                                           << " before knob filtering) ranked by metric '"
                                           << context.rankingMetric << "'");

                    executionContext.setPlan(std::move(plan));
                    return;
                }
                catch(const HipdnnPluginException& error)
                {
                    // A malformed descriptor is the author's mistake, not a kernel that
                    // happens not to fit this graph: falling past it would hide the fault
                    // and silently serve a different kernel than the one authored.
                    if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
                    {
                        throw;
                    }
                    failures.emplace_back(toString(filtered[rank]->kernelId) + ": " + error.what());
                }
                catch(const std::exception& error)
                {
                    failures.emplace_back(toString(filtered[rank]->kernelId) + ": " + error.what());
                }

                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' could not build a plan for "
                                       << toString(filtered[rank]->kernelId) << " at rank " << rank
                                       << ": " << failures.back() << "; trying the next candidate");
            }

            throwNoBuildableKernel(filtered.size(), failures);
        }

        HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '" << _engine.name << "' will benchmark "
                                                    << filtered.size() << " candidate(s) ("
                                                    << catalog.entries.size()
                                                    << " before knob filtering), ranked front "
                                                    << toString(filtered.front()->kernelId));

        // Every candidate walk applies the same policy: an unbuildable candidate is a reason
        // to carry, a malformed descriptor stops the build. Absorbing here what the others
        // rethrow would make the diagnosis a consequence of a tuning setting.
        std::vector<std::string> benchmarkFailures;
        std::vector<typename BenchmarkPlan<THandle>::Candidate> candidates;
        candidates.reserve(filtered.size());
        const auto problem = problemFeaturesJson(context, catalog.bound);
        for(const KernelDefinition* kernel : filtered)
        {
            try
            {
                candidates.push_back(
                    {kernel->kernelId,
                     std::make_unique<GenericPlan<THandle>>(
                         _stateManager.getDispatchDetails(*kernel), context, catalog.bound),
                     kernel->packId,
                     kernel->dispatchId,
                     candidateFeatures(problem, *kernel)});
                continue;
            }
            catch(const HipdnnPluginException& error)
            {
                if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
                {
                    throw;
                }
                benchmarkFailures.emplace_back(toString(kernel->kernelId) + ": " + error.what());
            }
            catch(const std::exception& error)
            {
                benchmarkFailures.emplace_back(toString(kernel->kernelId) + ": " + error.what());
            }

            HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '" << _engine.name
                                                        << "' dropped benchmarking candidate '"
                                                        << toString(kernel->kernelId)
                                                        << "': " << benchmarkFailures.back());
        }

        // Every candidate dropped. Reported here, with the reasons gathered above, rather
        // than left to BenchmarkPlan's constructor, whose INTERNAL_ERROR names neither the
        // engine nor a single kernel that failed or why.
        if(candidates.empty())
        {
            throwNoBuildableKernel(filtered.size(), benchmarkFailures);
        }

        // The callback is the write-back channel, already bound to the key: it captures
        // the state manager by reference, which the engine owns and which strictly
        // outlives every plan it hands out.
        // The graph and device halves of the winner key, in hex, so an exporter can group
        // rows per (graph, device) problem exactly as the cache keys it. `winnerKey` is
        // engaged here: benchmarking always builds it above.
        auto benchmarkId = detail::toHex(winnerKey->graph.hash());
        auto deviceId = detail::toHex(winnerKey->device.hash());

        // A record that exists but did not serve this graph -- either it failed the coverage gate
        // or none of its ranked entries still resolved -- is being superseded, so its write must
        // append rather than adopt.
        // `catalog.measuredRecord` counts too: the bounded winner cache may have evicted the
        // record that ordered this catalog.
        const auto cause = record.has_value() || catalog.measuredRecord != nullptr
                               ? WinnerWriteCause::COVERAGE_REBENCHMARK
                               : WinnerWriteCause::FRESH_MISS;

        executionContext.setPlan(makeBenchmarkPlan(
            std::move(candidates),
            handle,
            [&stateManager = _stateManager, winnerKey = std::move(*winnerKey), cause](
                const std::vector<RankedEntry>& ranking) {
                stateManager.recordWinner(winnerKey, ranking, cause);
            },
            std::move(benchmarkId),
            std::move(deviceId)));
    }

    /// One knob per KMD field the engine exposes; default is the top-ranked value. A slotted
    /// engine exposes `<slot>.<field>`, with values and default computed over that slot's
    /// catalog (RFC 0021 §5).
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        getCustomKnobs(const THandle& handle, const IGraph& opGraph) const override
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        const auto context = contextFor(handle, opGraph);
        std::vector<KnobT> knobs;
        for(size_t slot = 0; slot < _stateManager.slotCount(); ++slot)
        {
            const auto ranked = _stateManager.sortedCatalog(slot, context).entries;
            if(ranked.empty())
            {
                continue;
            }

            for(const auto& knobRef : _knobs)
            {
                if(knobRef.slot != slot)
                {
                    continue;
                }
                const auto values
                    = KernelIngestorStateManager<THandle>::knobValues(ranked, knobRef.field);

                // A non-integer value is advertised as its ordinal, which is what a caller pins
                // to select that kernel (RFC 0019 §13.2).
                std::vector<int64_t> choices;
                choices.reserve(values.size());
                for(const auto& value : values)
                {
                    if(const auto ordinal = _stateManager.knobOrdinal(slot, knobRef.field, value);
                       ordinal.has_value())
                    {
                        choices.push_back(*ordinal);
                    }
                }
                if(choices.empty())
                {
                    continue;
                }

                KnobT knob;
                knob.knob_id = knobRef.id;
                knob.description = "Kernel metadata field '" + knobRef.field + "' of "
                                   + (_stateManager.isSlotted()
                                          ? "slot '" + _stateManager.slotName(slot) + "' of "
                                          : std::string())
                                   + "engine '" + _engine.name + "'"
                                   + (_stateManager.isOrdinalKnob(slot, knobRef.field)
                                          ? " (ordinal: an index into the field's value set)"
                                          : "");

                IntValueT defaultValue;
                defaultValue.value = choices.front();
                knob.default_value.Set(defaultValue);

                IntConstraintT constraint;
                constraint.min_value = *std::min_element(choices.begin(), choices.end());
                constraint.max_value = *std::max_element(choices.begin(), choices.end());
                constraint.step = 1;
                constraint.valid_values = std::move(choices);
                knob.constraint.Set(constraint);

                knobs.push_back(std::move(knob));
            }
        }

        return knobs;
    }

    /// Enumerates the actual matched catalog, never the Cartesian product of knob
    /// domains. Defaults do not narrow the catalog; only explicitly pinned knobs do.
    hipdnn_flatbuffers_sdk::data_objects::EngineCandidatePageT
        enumerateCandidates(const THandle& handle,
                            const IGraph& opGraph,
                            const IEngineConfig& config,
                            uint64_t offset,
                            uint64_t limit) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        if(limit == 0 || limit > 10000 || !understandsGraph(opGraph))
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "Invalid candidate page limit or unsupported graph schema");
        }
        // Unknown/duplicate knobs must not silently broaden the requested scope.
        std::unordered_set<std::string> seen;
        for(const auto& setting : config.knobSettingWrappers())
        {
            const auto name = setting->knobId();
            if(!seen.insert(name).second || findKnob(name) == nullptr)
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "Unknown or duplicate enumeration knob '" + name + "'");
            }
        }
        const auto context = contextFor(handle, opGraph);
        if(_stateManager.isSlotted())
        {
            return enumerateSlotted(opGraph, context, config, offset, limit);
        }
        const auto catalog = _stateManager.enumerableCatalog(context);
        const auto filtered = filterSlot(catalog.entries, 0, readKnobFilter(config));
        if(offset > filtered.size())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "Candidate offset exceeds scoped catalog size");
        }

        // Check the entire scoped snapshot, not just this page: a collision across a
        // page boundary is still unaddressable by EngineVariant enrollment.
        std::map<KnobFilter, DescriptorId> identities;
        for(const KernelDefinition* kernel : filtered)
        {
            const auto tuple = candidateKnobs(*kernel);
            const auto inserted = identities.emplace(tuple, kernel->kernelId);
            if(!inserted.second)
            {
                throw HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "Ambiguous enrolled knob tuple for candidates '"
                        + toString(inserted.first->second) + "' and '" + toString(kernel->kernelId)
                        + "'; expose enough KMD integer fields in the collection UED knobs");
            }
        }

        EngineCandidatePageT page;
        fillPageContext(page, opGraph, context, catalog.bound);
        page.total_count = filtered.size();
        page.offset = offset;
        const auto end = offset + std::min<uint64_t>(limit, filtered.size() - offset);
        page.candidates.reserve(static_cast<size_t>(end - offset));
        for(auto i = offset; i < end; ++i)
        {
            const KernelDefinition& kernel = *filtered[static_cast<size_t>(i)];
            auto candidate = std::make_unique<EngineCandidateT>();
            candidate->id = toString(kernel.kernelId);
            for(const auto& [name, value] : candidateKnobs(kernel))
            {
                candidate->knob_settings.push_back(detail::makeIntKnobSetting(name, value));
            }
            nlohmann::json features = nlohmann::json::object();
            for(const auto& [name, value] : kernel.metadata)
            {
                detail::addMetadataFeature(features, "kernel." + name, value);
            }
            candidate->kernel_features = features.dump();
            page.candidates.push_back(std::move(candidate));
        }
        return page;
    }

    /// @brief Canonical features plus existing graph-match bindings, without touching a catalog.
    uhd::FeatureExtractionContext predictionFeatures(const THandle& handle,
                                                     const IGraph& graph,
                                                     const IEngineConfig& config,
                                                     std::string& arch) const
    {
        const auto context = contextFor(handle, graph);
        arch = context.deviceProperties.gcnArchName;
        auto features = heuristics::engineFeatures(graph, config, context.deviceProperties);
        try
        {
            if(const auto bound = _stateManager.graphBindings(context))
            {
                detail::bindGraphMatchBindings(features, *bound, _stateManager.isSlotted());
            }
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_WARN(
                "ingestor: optional prediction bindings unavailable: " << error.what());
        }
        return features;
    }

    /// @brief Predicts the configuration plan build would serve, in the ranking metric
    ///        @p config carries (RFC 0019 §5 step 9).
    ///
    /// Under a covering benchmark record the value is measured, or the calibrated model's
    /// estimate when the record lacks the metric; otherwise only that metric's calibrated
    /// ranker answers (RFC 0019 §11.4).
    void predictConfiguration(const THandle& handle,
                              const IGraph& graph,
                              const IEngineConfig& config,
                              hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT& result) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        const auto& metric = heuristics::rankingMetric(config);
        result.metric = std::string(metric.name);
        validateKnobConstraints(config);
        if(readBenchmarkingEnabled(config))
        {
            throw HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "An exact configuration prediction cannot preserve a benchmarking request");
        }
        TSettings executionSettings;
        initializeExecutionSettings(handle, graph, config, executionSettings);
        const auto context = contextFor(handle, graph, metric.name);
        if(_stateManager.isSlotted())
        {
            predictSlotted(graph, config, executionSettings.ingestorSettings, context, result);
            return;
        }
        // One snapshot: the catalog carries the record that ordered it, so a winner-cache
        // eviction cannot split plan build and this prediction.
        auto catalog = _stateManager.measuredCatalog(context);
        const auto filtered
            = applyConstraints(catalog, executionSettings.ingestorSettings, context);
        std::string modelId;
        // The full catalog decides the order source and ranking; `filtered` only limits what
        // the answer may name, so a pin changes neither (RFC 0019 §9.2, §5 steps 8-9).
        const bool measured = catalog.measuredRecord != nullptr;
        std::vector<ScoredKernel> ranking;
        if(measured)
        {
            bool measuresMetric = false;
            ranking = measuredRanking(
                *catalog.measuredRecord, filtered, metric, context, measuresMetric);
            if(!measuresMetric)
            {
                // A metric the record does not measure is the model's to answer -- for the
                // record's configurations, never by choosing among them.
                const auto estimates
                    = _stateManager.calibratedRanking(catalog, filtered, context, modelId);
                for(auto& entry : ranking)
                {
                    const auto estimate
                        = std::find_if(estimates.begin(), estimates.end(), [&](const auto& scored) {
                              return scored.kernelId == entry.kernelId;
                          });
                    entry.score = estimate == estimates.end() ? 0.0 : estimate->score;
                }
            }
        }
        else
        {
            ranking = _stateManager.calibratedRanking(catalog, filtered, context, modelId);
        }
        catalog.entries = filtered;
        result.reason
            = "No calibrated '" + result.metric + "' configuration prediction is available";
        for(const auto& scored : ranking)
        {
            const bool valued
                = scored.score != 0.0
                  && hipdnn_data_sdk::utilities::isValidMetricValue(metric, scored.score);
            // Under a record the order is decided: a candidate without a value is still the one
            // plan build serves, so it is answered UNAVAILABLE below rather than skipped.
            if(!valued && !measured)
            {
                continue;
            }
            const auto selected = std::find_if(catalog.entries.begin(),
                                               catalog.entries.end(),
                                               [&](const KernelDefinition* kernel) {
                                                   return kernel->kernelId == scored.kernelId;
                                               });
            if(selected == catalog.entries.end())
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                            "Ranker returned an unknown candidate");
            }
            // Plan build walks past a candidate it cannot build, so the prediction does too.
            try
            {
                const GenericPlan<THandle> prepared(
                    _stateManager.getDispatchDetails(**selected), context, catalog.bound);
            }
            catch(const HipdnnPluginException& error)
            {
                if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
                {
                    throw;
                }
                continue;
            }
            catch(const std::exception&)
            {
                continue;
            }

            // This is the configuration plan build serves. Naming a later candidate when this
            // one cannot be named would advertise a kernel the request does not build.
            const auto knobs = candidateKnobs(**selected);
            // Same comparison as filterSlot(): the tuple holds ordinals for non-integer knobs.
            const auto matching = std::count_if(
                catalog.entries.begin(),
                catalog.entries.end(),
                [this, &knobs](const KernelDefinition* kernel) {
                    return std::all_of(
                        knobs.begin(), knobs.end(), [this, kernel](const auto& setting) {
                            return _stateManager.knobMatches(
                                *kernel, setting.first, setting.second);
                        });
                });
            if(matching != 1)
            {
                result.reason
                    = "The served configuration cannot be identified by its exposed knobs";
                return;
            }
            if(!valued)
            {
                // Only reachable under a record: nothing can say what this configuration is
                // worth in this metric.
                result.reason = "The benchmarked configuration has no '" + result.metric
                                + "' value: the record does not measure it and no "
                                  "calibrated model estimates it";
                return;
            }
            result.engine_config = pinnedConfig(config, result.engine_id, knobs);
            result.value = scored.score;
            // Empty when the record supplied the value: no model produced the number.
            result.uhd_id = modelId;
            result.status = PredictionStatus::AVAILABLE;
            result.reason.clear();
            return;
        }
    }
    void validateKnobConstraints(const IEngineConfig& config) const
    {
        if(!config.isValid())
        {
            return;
        }
        std::unordered_set<std::string> seen;
        for(const auto& setting : config.knobSettingWrappers())
        {
            const auto name = setting->knobId();
            if(!seen.insert(name).second
               || (name != BENCHMARKING_KNOB_NAME && name != WORKSPACE_SIZE_LIMIT_KNOB_NAME
                   && findKnob(name) == nullptr))
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "Unknown or duplicate selection constraint '" + name
                                                + "'");
            }
        }
    }

private:
    /// What finishApplicability needs: the context begin resolved and the catalog's
    /// graph-level stage. The graph is borrowed from the caller, which keeps it alive until
    /// every finish has returned. The device properties and the key are borrowed from the
    /// scope, or owned here when this engine's resolver or device is not the scope's.
    struct PendingCatalog : PendingApplicability::State
    {
        const IGraph* graph = nullptr;
        DeviceId deviceId = NO_DEVICE;
        const DeviceProperties* deviceProperties = nullptr;
        /// Set when this engine's resolver is not the scope's: its own copy of the properties.
        std::optional<DeviceProperties> ownedProperties;
        /// Set when this engine's device differs from the scope's, so the key is its own.
        std::optional<CatalogKey> ownedKey;
        const CatalogKey* key = nullptr;
        CatalogGraphStage stage;

        MatchContext context() const
        {
            return MatchContext{*graph, deviceId, *deviceProperties};
        }
    };

    static StagedApplicability decided(bool verdict)
    {
        return {true, verdict, PendingApplicability{}};
    }

    void logApplicabilityFailure(const std::exception& error) const
    {
        HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine '"
                                << _engine.name
                                << "' declined the graph: deciding applicability failed: "
                                << error.what());
    }

    /// @p record's order over @p filtered, valued in @p metric: the time, or for `tflops`
    /// `graph.flops / (ms * 1e9)`, the label uhd_gen trains on (RFC 0019 §5 step 9).
    /// @param measuresMetric Set false when the record cannot supply @p metric; every value
    ///        is then 0 and only the order is the record's.
    static std::vector<ScoredKernel>
        measuredRanking(const WinnerRecord& record,
                        const std::vector<const KernelDefinition*>& filtered,
                        const hipdnn_data_sdk::utilities::RankingMetric& metric,
                        const MatchContext& context,
                        bool& measuresMetric)
    {
        std::optional<double> flops;
        if(metric.name == "tflops")
        {
            flops = graphFlops(context);
        }
        measuresMetric = metric.name == "time" || flops.has_value();

        std::vector<ScoredKernel> ranking;
        ranking.reserve(filtered.size());
        for(const auto& entry : record)
        {
            const bool admitted = std::any_of(
                filtered.begin(), filtered.end(), [&entry](const KernelDefinition* kernel) {
                    return kernel->kernelId == entry.kernelId && kernel->packId == entry.packId
                           && kernel->dispatchId == entry.dispatchId;
                });
            if(!admitted)
            {
                continue;
            }
            const double value = metricValue(metric, flops, entry.timeMs);
            ranking.push_back({entry.kernelId, value});
        }
        return ranking;
    }

    /// @p kernels whose workspace for @p context, added to @p offset, stays within @p limit.
    /// @param offset Bytes already taken before the kernel's own scratch: a slotted engine's
    ///        aligned shared region.
    std::vector<const KernelDefinition*>
        withinWorkspace(std::vector<const KernelDefinition*> kernels,
                        const MatchContext& context,
                        const BoundTokens& bound,
                        int64_t limit,
                        size_t offset = 0) const
    {
        kernels.erase(
            std::remove_if(kernels.begin(),
                           kernels.end(),
                           [&](const KernelDefinition* kernel) {
                               const auto* handler = _stateManager.dispatchHandlerFor(*kernel);
                               return offset + handler->workspaceBytes(context, bound, *kernel)
                                      > static_cast<uint64_t>(limit);
                           }),
            kernels.end());
        return kernels;
    }

    std::vector<const KernelDefinition*> applyConstraints(const Catalog& catalog,
                                                          const IngestorSettings& settings,
                                                          const MatchContext& context) const
    {
        auto filtered = filterSlot(catalog.entries, 0, settings.knobFilter);
        if(settings.workspaceLimit)
        {
            return withinWorkspace(
                std::move(filtered), context, catalog.bound, *settings.workspaceLimit);
        }
        return filtered;
    }

    /// One engine knob: a KMD field of slot @p slot, advertised under @p id. An unslotted
    /// engine's id is the bare field name; a slotted engine's is `<slot>.<field>`.
    struct KnobRef
    {
        size_t slot;
        std::string field;
        std::string id;
    };

    /// What a slotted engine's pruning leaves of every slot, ordered for plan build or
    /// enumeration, with what they share.
    struct SlotCandidates
    {
        std::vector<std::vector<const KernelDefinition*>> perSlot;
        /// Per slot, the benchmarked record whose order `perSlot` follows; null when none covers
        /// the slot's catalog.
        std::vector<std::shared_ptr<const WinnerRecord>> measured;
        /// What the engine's one graph match bound, which every slot's kernels read.
        BoundTokens bound;
        /// The shared region, rounded to its alignment: the offset at which scratch begins.
        size_t alignedSharedBytes = 0;
    };

    static std::vector<KnobRef> buildKnobs(const EngineDescriptor& engine,
                                           const KernelIngestorStateManager<THandle>& stateManager)
    {
        std::vector<KnobRef> knobs;
        if(!stateManager.isSlotted())
        {
            knobs.reserve(engine.knobs.size());
            for(const auto& field : engine.knobs)
            {
                knobs.push_back({0, field, field});
            }
            return knobs;
        }
        size_t count = 0;
        for(const auto& slot : engine.slots)
        {
            count += slot.knobs.size();
        }
        knobs.reserve(count);
        for(size_t index = 0; index < stateManager.slotCount(); ++index)
        {
            for(const auto& slot : engine.slots)
            {
                if(slot.name != stateManager.slotName(index))
                {
                    continue;
                }
                for(const auto& field : slot.knobs)
                {
                    knobs.push_back({index, field, slot.name + "." + field});
                }
            }
        }
        return knobs;
    }

    const KnobRef* findKnob(const std::string& id) const
    {
        const auto found = std::find_if(
            _knobs.begin(), _knobs.end(), [&id](const KnobRef& knob) { return knob.id == id; });
        return found == _knobs.end() ? nullptr : &*found;
    }

    /// @p kernel's value of every knob of slot @p slot, as the knob settings that select it.
    KnobFilter candidateKnobs(const KernelDefinition& kernel, size_t slot = 0) const
    {
        KnobFilter tuple;
        for(const auto& knob : _knobs)
        {
            if(knob.slot != slot)
            {
                continue;
            }
            const auto it = kernel.metadata.find(knob.field);
            // The tuple must address this kernel, since replay pins exactly these values.
            // Non-integer values enter as their ordinal (RFC 0019 §13.2).
            if(it != kernel.metadata.end())
            {
                if(const auto ordinal = _stateManager.knobOrdinal(slot, knob.field, it->second);
                   ordinal.has_value())
                {
                    tuple.emplace(knob.id, *ordinal);
                }
            }
        }
        return tuple;
    }

    /// The settings of @p filter that name a knob of slot @p slot.
    std::vector<std::pair<const KnobRef*, int64_t>> slotPins(size_t slot,
                                                             const KnobFilter& filter) const
    {
        std::vector<std::pair<const KnobRef*, int64_t>> pins;
        pins.reserve(filter.size());
        for(const auto& knob : _knobs)
        {
            const auto setting = filter.find(knob.id);
            if(knob.slot == slot && setting != filter.end())
            {
                pins.emplace_back(&knob, setting->second);
            }
        }
        return pins;
    }

    /// @p entries of slot @p slot that every setting of @p filter naming that slot's knobs admits.
    std::vector<const KernelDefinition*>
        filterSlot(const std::vector<const KernelDefinition*>& entries,
                   size_t slot,
                   const KnobFilter& filter) const
    {
        const auto pins = slotPins(slot, filter);
        if(pins.empty())
        {
            return entries;
        }
        std::vector<const KernelDefinition*> filtered;
        filtered.reserve(entries.size());
        for(const KernelDefinition* kernel : entries)
        {
            const bool admitted = std::all_of(pins.begin(), pins.end(), [&](const auto& pin) {
                return _stateManager.knobMatches(slot, *kernel, pin.first->field, pin.second);
            });
            if(admitted)
            {
                filtered.push_back(kernel);
            }
        }
        return filtered;
    }

    /// Set-time check of a slotted engine's knob settings (RFC 0021 §5): a setting that leaves
    /// its slot with no kernel is rejected here, naming the knob. A declined graph is left to
    /// plan build, which reports the decline.
    void rejectEmptiedSlots(const MatchContext& context, const KnobFilter& filter) const
    {
        if(!_stateManager.isApplicable(context))
        {
            return;
        }
        for(size_t slot = 0; slot < _stateManager.slotCount(); ++slot)
        {
            if(slotPins(slot, filter).empty())
            {
                continue;
            }
            const auto catalog = _stateManager.unsortedCatalog(slot, context);
            if(filterSlot(catalog.entries, slot, filter).empty())
            {
                throwUnsatisfiableKnobFilter(slot, filter, catalog.entries.size());
            }
        }
    }

    /// Every slot's kernels that survive @p settings: the knob filter, then the workspace limit
    /// applied to the total of aligned shared region and the kernel's scratch (RFC 0021 §6).
    /// @param ranked Order each slot by its ranker rather than as declared.
    /// @throws HipdnnPluginException INTERNAL_ERROR when the engine declines the graph, and
    ///         INVALID_VALUE when the settings leave the engine with no program: a slot with no
    ///         kernel, or a shared region alone over the limit.
    SlotCandidates slotCandidates(const MatchContext& context,
                                  const IngestorSettings& settings,
                                  bool ranked) const
    {
        const auto verdict = _stateManager.slotProgramVerdict(context);
        if(!verdict.applicable)
        {
            throwNoApplicableKernel(verdict.declineReason);
        }

        SlotCandidates selection;
        selection.alignedSharedBytes = alignSharedWorkspace(verdict.sharedWorkspaceBytes);
        if(settings.workspaceLimit
           && selection.alignedSharedBytes > static_cast<uint64_t>(*settings.workspaceLimit))
        {
            throw HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "engine '" + _engine.name + "' declines the graph: its shared workspace region of "
                    + std::to_string(verdict.sharedWorkspaceBytes) + " bytes ("
                    + std::to_string(selection.alignedSharedBytes)
                    + " aligned) exceeds the workspace limit of "
                    + std::to_string(*settings.workspaceLimit) + " bytes");
        }

        selection.perSlot.reserve(_stateManager.slotCount());
        selection.measured.reserve(_stateManager.slotCount());
        for(size_t slot = 0; slot < _stateManager.slotCount(); ++slot)
        {
            auto catalog = ranked ? _stateManager.sortedCatalog(slot, context)
                                  : _stateManager.unsortedCatalog(slot, context);
            auto kept = filterSlot(catalog.entries, slot, settings.knobFilter);
            if(kept.empty())
            {
                throwUnsatisfiableKnobFilter(slot, settings.knobFilter, catalog.entries.size());
            }
            if(settings.workspaceLimit)
            {
                const auto before = kept.size();
                kept = withinWorkspace(std::move(kept),
                                       context,
                                       catalog.bound,
                                       *settings.workspaceLimit,
                                       selection.alignedSharedBytes);
                if(kept.empty())
                {
                    throw HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "engine '" + _engine.name + "' has no kernel in slot '"
                            + _stateManager.slotName(slot) + "' within the workspace limit of "
                            + std::to_string(*settings.workspaceLimit) + " bytes ("
                            + std::to_string(before) + " kernel(s) remained before the limit; "
                            + "the shared region takes "
                            + std::to_string(selection.alignedSharedBytes) + " bytes)");
                }
            }
            if(slot == 0)
            {
                selection.bound = std::move(catalog.bound);
            }
            selection.measured.push_back(catalog.measuredRecord);
            selection.perSlot.push_back(std::move(kept));
        }
        return selection;
    }

    /// The plan of @p kernel for @p context, or null when it cannot be built, with the reason
    /// appended to @p failures. A malformed descriptor is the author's mistake, not a kernel that
    /// happens not to fit this graph: it stops the build instead of costing only its kernel.
    /// @param onBuilt Called with the built plan inside the same guard, so what it throws costs
    ///        only @p kernel as the build itself does.
    /// @throws HipdnnPluginException INVALID_VALUE from a malformed descriptor.
    std::shared_ptr<GenericPlan<THandle>> tryBuildPlan(
        const KernelDefinition& kernel,
        const MatchContext& context,
        const BoundTokens& bound,
        std::vector<std::string>& failures,
        const std::function<void(const std::shared_ptr<GenericPlan<THandle>>&)>& onBuilt = {}) const
    {
        try
        {
            auto plan = std::make_shared<GenericPlan<THandle>>(
                _stateManager.getDispatchDetails(kernel), context, bound);
            if(onBuilt)
            {
                onBuilt(plan);
            }
            return plan;
        }
        catch(const HipdnnPluginException& error)
        {
            if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
            {
                throw;
            }
            failures.emplace_back(toString(kernel.kernelId) + ": " + error.what());
        }
        catch(const std::exception& error)
        {
            failures.emplace_back(toString(kernel.kernelId) + ": " + error.what());
        }
        return nullptr;
    }

    /// Each slot's first kernel of @p selection that builds, in slot order. A kernel that fails
    /// to build costs only its slot's pick, as for an unslotted engine; a malformed descriptor
    /// stops the build.
    /// @throws HipdnnPluginException INTERNAL_ERROR when a slot has no buildable kernel.
    std::vector<SlotStage<THandle>> buildSlotStages(const MatchContext& context,
                                                    const SlotCandidates& selection) const
    {
        std::vector<SlotStage<THandle>> stages;
        stages.reserve(selection.perSlot.size());
        for(size_t slot = 0; slot < selection.perSlot.size(); ++slot)
        {
            const auto& slotName = _stateManager.slotName(slot);
            const auto& kernels = selection.perSlot[slot];
            std::vector<std::string> failures;
            std::shared_ptr<GenericPlan<THandle>> plan;
            for(size_t rank = 0; rank < kernels.size() && plan == nullptr; ++rank)
            {
                plan = tryBuildPlan(*kernels[rank], context, selection.bound, failures);
                if(plan != nullptr)
                {
                    HIPDNN_PLUGIN_LOG_INFO(
                        "ingestor: engine '"
                        << _engine.name << "' slot '" << slotName << "' selected kernel "
                        << toString(kernels[rank]->kernelId) << " at rank " << rank << " from "
                        << kernels.size() << " candidate(s)");
                    continue;
                }

                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' slot '" << slotName
                                       << "' could not build a plan for "
                                       << toString(kernels[rank]->kernelId) << " at rank " << rank
                                       << ": " << failures.back() << "; trying the next candidate");
            }
            if(plan == nullptr)
            {
                throwNoBuildableKernel(kernels.size(), failures, slotName);
            }
            stages.push_back({slotName, std::move(plan)});
        }
        return stages;
    }

    void buildSlottedPlan(const IGraph& opGraph,
                          const MatchContext& context,
                          const IngestorSettings& settings,
                          TContext& executionContext) const
    {
        const auto selection = slotCandidates(context, settings, true);
        if(settings.benchmarkingEnabled
           && setSlotProgramBenchmark(opGraph, context, settings, selection, executionContext))
        {
            return;
        }
        executionContext.setPlan(std::make_unique<SlotProgramPlan<THandle>>(
            buildSlotStages(context, selection), selection.alignedSharedBytes));
    }

    /// Plans the benchmark of a slotted engine's program (RFC 0021 §7) when one is due: some
    /// slot's catalog is not covered by a winner record, or the program record is not valid.
    /// While a knob filter or workspace limit is active only an uncovered slot makes it due, and
    /// nothing is recorded, since the records would describe a narrowed catalog.
    /// @return false, leaving @p executionContext untouched, when nothing needs measuring.
    bool setSlotProgramBenchmark(const IGraph& opGraph,
                                 const MatchContext& context,
                                 const IngestorSettings& settings,
                                 const SlotCandidates& selection,
                                 TContext& executionContext) const
    {
        const bool narrowed = !settings.knobFilter.empty() || settings.workspaceLimit.has_value();
        const WinnerKey baseKey{
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{opGraph},
            DeviceKey{context.deviceProperties}};
        const bool anyUncovered = std::any_of(selection.measured.begin(),
                                              selection.measured.end(),
                                              [](const auto& record) { return record == nullptr; });
        if(!anyUncovered && (narrowed || _stateManager.programFor(baseKey).has_value()))
        {
            return false;
        }

        // Every candidate walk applies the same policy: an unbuildable kernel is a reason to
        // carry, a malformed descriptor stops the build. A covered slot needs only its first
        // buildable kernel, its measured winner; when that winner fails to build the slot is
        // uncovered after all and is re-timed over the kernels that do build.
        const auto problem = problemFeaturesJson(context, selection.bound, true);
        std::vector<typename SlotProgramBenchmark<THandle>::Slot> slots;
        slots.reserve(selection.perSlot.size());
        size_t timedKernels = 0;
        bool timedSlotHasRecord = false;
        for(size_t slot = 0; slot < selection.perSlot.size(); ++slot)
        {
            const auto& slotName = _stateManager.slotName(slot);
            bool covered = selection.measured[slot] != nullptr;
            typename SlotProgramBenchmark<THandle>::Slot built{slotName, {}, covered};
            built.kernels.reserve(selection.perSlot[slot].size());
            std::vector<std::string> failures;
            for(const KernelDefinition* kernel : selection.perSlot[slot])
            {
                if(tryBuildPlan(*kernel,
                                context,
                                selection.bound,
                                failures,
                                [&](const auto& plan) {
                                    built.kernels.push_back({kernel->kernelId,
                                                             kernel->packId,
                                                             kernel->dispatchId,
                                                             plan,
                                                             candidateFeatures(problem, *kernel)});
                                })
                   != nullptr)
                {
                    if(covered)
                    {
                        break;
                    }
                    continue;
                }
                covered = false;
                built.covered = false;

                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' slot '" << slotName
                                       << "' dropped benchmarking candidate '"
                                       << toString(kernel->kernelId) << "': " << failures.back());
            }
            if(built.kernels.empty())
            {
                throwNoBuildableKernel(selection.perSlot[slot].size(), failures, slotName);
            }
            if(!covered)
            {
                timedKernels += built.kernels.size();
                WinnerKey slotKey = baseKey;
                slotKey.slot = slotName;
                timedSlotHasRecord
                    = timedSlotHasRecord || _stateManager.winnerFor(slotKey).has_value();
            }
            slots.push_back(std::move(built));
        }

        HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                               << _engine.name << "' will benchmark its slot program: "
                               << timedKernels << " kernel(s) in uncovered slot(s)"
                               << (narrowed ? "; a knob filter or workspace limit is active, "
                                              "so no records will be written"
                                            : ""));

        typename SlotProgramBenchmark<THandle>::RecordFn record;
        if(!narrowed)
        {
            // The write-back channel, already bound to the key: it captures the state manager by
            // reference, which the engine owns and which strictly outlives every plan it hands
            // out. A slot record that exists but did not cover its catalog is being superseded,
            // so its write must append rather than adopt; the program is only ever timed because
            // its record was missing or invalid, so its line supersedes too.
            const auto slotCause = timedSlotHasRecord ? WinnerWriteCause::COVERAGE_REBENCHMARK
                                                      : WinnerWriteCause::FRESH_MISS;
            record = [&stateManager = _stateManager, baseKey, slotCause](
                         const typename SlotProgramBenchmark<THandle>::SlotRecords& slotRecords,
                         const std::optional<ProgramRecord>& program) {
                stateManager.recordSlotted(baseKey,
                                           slotRecords,
                                           program,
                                           slotCause,
                                           WinnerWriteCause::COVERAGE_REBENCHMARK);
            };
        }

        executionContext.setPlan(
            std::make_unique<SlotProgramBenchmark<THandle>>(std::move(slots),
                                                            selection.alignedSharedBytes,
                                                            _timer,
                                                            std::move(record),
                                                            detail::toHex(baseKey.graph.hash()),
                                                            detail::toHex(baseKey.device.hash())));
        return true;
    }

    /// The graph's work in FLOPs, when the canonical problem features know it.
    static std::optional<double> graphFlops(const MatchContext& context)
    {
        const auto problem = heuristics::problemFeatures(context.graph, context.deviceProperties);
        if(const auto* value = problem.getContext().find("graph.flops"))
        {
            if(const auto* known = std::get_if<double>(value))
            {
                return *known;
            }
        }
        return std::nullopt;
    }

    /// @p timeMs in @p metric: the time itself, or for `tflops` `flops / (timeMs * 1e9)`, the
    /// label uhd_gen trains on; 0 when the metric cannot be valued. @p flops is the graph's work,
    /// when known.
    static double metricValue(const hipdnn_data_sdk::utilities::RankingMetric& metric,
                              const std::optional<double>& flops,
                              double timeMs)
    {
        if(metric.name == "time")
        {
            return timeMs;
        }
        if(flops.has_value() && timeMs > 0.0)
        {
            return *flops / (timeMs * 1e9);
        }
        return 0.0;
    }

    /// @p config with the knob settings @p knobs added for every knob it does not already set.
    static std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>
        pinnedConfig(const IEngineConfig& config, int64_t engineId, const KnobFilter& knobs)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        auto exact = config.isValid()
                         ? std::unique_ptr<EngineConfigT>(config.getEngineConfig().UnPack())
                         : std::make_unique<EngineConfigT>();
        exact->engine_id = engineId;
        for(const auto& field : knobs)
        {
            const auto& name = field.first;
            const auto value = field.second;
            auto existing
                = std::find_if(exact->knobs.begin(), exact->knobs.end(), [&](const auto& setting) {
                      return setting->knob_id == name;
                  });
            if(existing == exact->knobs.end())
            {
                exact->knobs.push_back(detail::makeIntKnobSetting(name, value));
            }
        }
        return exact;
    }

    /// The page fields every enumeration states about its graph and device; the live ranker's
    /// problem features, with `device.*` split into their own field.
    void fillPageContext(hipdnn_flatbuffers_sdk::data_objects::EngineCandidatePageT& page,
                         const IGraph& opGraph,
                         const MatchContext& context,
                         const BoundTokens& bound) const
    {
        page.engine_name = _engine.name;
        page.engine_descriptor_id = toString(_engine.id);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graphKey{opGraph};
        if(!graphKey.isUsable())
        {
            throw HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "Cannot enumerate a graph without stable serialized content");
        }
        page.graph_id = std::to_string(graphKey.hash());
        page.device_id = std::to_string(DeviceKey{context.deviceProperties}.hash());
        page.device_arch = context.deviceProperties.gcnArchName;
        auto problem = problemFeaturesJson(context, bound, _stateManager.isSlotted());
        nlohmann::json device = nlohmann::json::object();
        for(auto it = problem.begin(); it != problem.end();)
        {
            if(it.key().rfind("device.", 0) == 0)
            {
                device.emplace(it.key(), std::move(it.value()));
                it = problem.erase(it);
            }
            else
            {
                ++it;
            }
        }
        page.problem_features = problem.dump();
        page.device_features = device.dump();
    }

    /// A slotted engine's enumeration (RFC 0021 §5): every combination of per-slot kernels, in
    /// kernel-id order with the first slot varying slowest. Entries are computed from the page's
    /// indices, so the product is never materialized and a page does not depend on rankers or
    /// records.
    hipdnn_flatbuffers_sdk::data_objects::EngineCandidatePageT
        enumerateSlotted(const IGraph& opGraph,
                         const MatchContext& context,
                         const IEngineConfig& config,
                         uint64_t offset,
                         uint64_t limit) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        const auto filter = readKnobFilter(config);
        const auto slotCount = _stateManager.slotCount();
        std::vector<std::vector<const KernelDefinition*>> perSlot;
        perSlot.reserve(slotCount);
        BoundTokens bound;
        for(size_t slot = 0; slot < slotCount; ++slot)
        {
            auto catalog = _stateManager.enumerableCatalog(slot, context);
            auto filtered = filterSlot(catalog.entries, slot, filter);
            // Per slot, not over the product: a combination is unique exactly when each slot's
            // tuple is, and a collision is unaddressable by EngineVariant enrollment.
            std::map<KnobFilter, DescriptorId> identities;
            for(const KernelDefinition* kernel : filtered)
            {
                const auto inserted
                    = identities.emplace(candidateKnobs(*kernel, slot), kernel->kernelId);
                if(!inserted.second)
                {
                    throw HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "Ambiguous enrolled knob tuple in slot '" + _stateManager.slotName(slot)
                            + "' for candidates '" + toString(inserted.first->second) + "' and '"
                            + toString(kernel->kernelId)
                            + "'; expose enough KMD integer fields in the slot's UED knobs");
                }
            }
            if(slot == 0)
            {
                bound = std::move(catalog.bound);
            }
            perSlot.push_back(std::move(filtered));
        }

        uint64_t total = 1;
        if(std::any_of(
               perSlot.begin(), perSlot.end(), [](const auto& kernels) { return kernels.empty(); }))
        {
            total = 0;
        }
        else
        {
            for(const auto& kernels : perSlot)
            {
                if(total > std::numeric_limits<uint64_t>::max() / kernels.size())
                {
                    throw HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                        "Candidate count of the slot product exceeds 64 bits");
                }
                total *= kernels.size();
            }
        }
        if(offset > total)
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "Candidate offset exceeds scoped catalog size");
        }

        EngineCandidatePageT page;
        fillPageContext(page, opGraph, context, bound);
        page.total_count = total;
        page.offset = offset;
        const auto end = offset + std::min<uint64_t>(limit, total - offset);
        page.candidates.reserve(static_cast<size_t>(end - offset));
        std::vector<const KernelDefinition*> picks(slotCount, nullptr);
        for(auto index = offset; index < end; ++index)
        {
            auto rest = index;
            for(size_t slot = slotCount; slot-- > 0;)
            {
                const auto& kernels = perSlot[slot];
                picks[slot] = kernels[static_cast<size_t>(rest % kernels.size())];
                rest /= kernels.size();
            }

            auto candidate = std::make_unique<EngineCandidateT>();
            candidate->id = detail::compositeCandidateId(slotCount, [this, &picks](size_t slot) {
                return std::forward_as_tuple(_stateManager.slotName(slot), picks[slot]->kernelId);
            });
            nlohmann::json features = nlohmann::json::object();
            for(size_t slot = 0; slot < slotCount; ++slot)
            {
                const auto& slotName = _stateManager.slotName(slot);
                for(const auto& [name, value] : candidateKnobs(*picks[slot], slot))
                {
                    candidate->knob_settings.push_back(detail::makeIntKnobSetting(name, value));
                }
                for(const auto& [name, value] : picks[slot]->metadata)
                {
                    detail::addMetadataFeature(
                        features,
                        std::string("kernel.").append(slotName).append(".").append(name),
                        value);
                }
            }
            candidate->kernel_features = features.dump();
            page.candidates.push_back(std::move(candidate));
        }
        return page;
    }

    /// A slotted engine's configuration prediction (RFC 0021 §7): it comes only from
    /// measurement, so it exists while a valid program record does and the settings in @p config
    /// leave that program unchanged. Every other case is answered UNAVAILABLE in @p result.
    void predictSlotted(const IGraph& graph,
                        const IEngineConfig& config,
                        const IngestorSettings& settings,
                        const MatchContext& context,
                        hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT& result) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        result.reason = "No valid program record exists: a slotted engine's configuration "
                        "prediction comes only from measurement";
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graphKey{graph};
        if(!graphKey.isUsable())
        {
            return;
        }
        // slotCandidates notes every slot catalog, which programFor needs to validate the record.
        const auto candidates = slotCandidates(context, settings, true);
        const auto program
            = _stateManager.programFor(WinnerKey{graphKey, DeviceKey{context.deviceProperties}});
        if(!program.has_value())
        {
            return;
        }

        const auto stages = buildSlotStages(context, candidates);
        const bool unchanged
            = stages.size() == program->picks.size()
              && std::equal(stages.begin(),
                            stages.end(),
                            program->picks.begin(),
                            [](const auto& stage, const auto& pick) {
                                return stage.slot == pick.slot
                                       && stage.plan->kernel().kernelId == pick.kernelId;
                            });
        if(!unchanged)
        {
            result.reason = "The knob settings select a program other than the measured one";
            return;
        }

        KnobFilter knobs;
        for(size_t slot = 0; slot < stages.size(); ++slot)
        {
            const auto tuple = candidateKnobs(stages[slot].plan->kernel(), slot);
            const auto catalog = _stateManager.unsortedCatalog(slot, context);
            const auto matching = std::count_if(catalog.entries.begin(),
                                                catalog.entries.end(),
                                                [&](const KernelDefinition* kernel) {
                                                    return candidateKnobs(*kernel, slot) == tuple;
                                                });
            if(matching != 1)
            {
                result.reason
                    = "The served configuration cannot be identified by its exposed knobs";
                return;
            }
            knobs.insert(tuple.begin(), tuple.end());
        }

        const auto& metric = heuristics::rankingMetric(config);
        const auto flops = metric.name == "tflops" ? graphFlops(context) : std::nullopt;
        const double value = metricValue(metric, flops, program->timeMs);
        if(value == 0.0 || !hipdnn_data_sdk::utilities::isValidMetricValue(metric, value))
        {
            result.reason = "The program record has no '" + result.metric + "' value";
            return;
        }

        result.engine_config = pinnedConfig(config, result.engine_id, knobs);
        result.value = value;
        // Empty: the measurement supplied the value, no model produced it.
        result.uhd_id.clear();
        result.status = PredictionStatus::AVAILABLE;
        result.reason.clear();
    }

    /// The problem half of a `sort_kernel_catalog` row as JSON: the scalars the live ranker
    /// binds (detail::catalogProblemFeatures) plus each int-list token whole. `device.*`
    /// facts distinguish boards of one arch. Keys drop the leading '$'.
    /// @param reserveWorkspace Leave `ws.*` tokens out, as a slotted engine's problem features
    ///        do (RFC 0021 §6).
    static nlohmann::json problemFeaturesJson(const MatchContext& context,
                                              const BoundTokens& bound,
                                              bool reserveWorkspace = false)
    {
        auto features = detail::catalogProblemFeatures(context, bound, reserveWorkspace).toJson();
        for(const auto& [token, value] : bound)
        {
            // A list has no scalar binding, so a model reads it only through the indexed
            // names already present; the whole list rides along for readers of the row.
            const auto* values = std::get_if<std::vector<int64_t>>(&value);
            if(values != nullptr && !detail::isReservedFeatureName(token, reserveWorkspace))
            {
                features[!token.empty() && token.front() == '$' ? token.substr(1) : token]
                    = *values;
            }
        }
        return features;
    }

    /// The features of one benchmarked (problem, kernel) pair: @p problem plus the kernel's
    /// KMD metadata under `kernel.`. Built for every sweep, whether or not a UHD ships.
    static nlohmann::json candidateFeatures(const nlohmann::json& problem,
                                            const KernelDefinition& kernel)
    {
        auto features = problem;
        for(const auto& [field, value] : kernel.metadata)
        {
            detail::addMetadataFeature(features, "kernel." + field, value);
        }
        return features;
    }

    /// The seam for a deterministic test timer is the constructor's `timer` parameter,
    /// not this factory: tests exercise this exact code path rather than overriding it.
    std::unique_ptr<IPlan<THandle>>
        makeBenchmarkPlan(std::vector<typename BenchmarkPlan<THandle>::Candidate> candidates,
                          const THandle& handle,
                          typename BenchmarkPlan<THandle>::RecordRankingFn recordRanking,
                          std::string benchmarkId,
                          std::string deviceId) const
    {
        return std::make_unique<BenchmarkPlan<THandle>>(std::move(candidates),
                                                        handle,
                                                        _timer,
                                                        std::move(recordRanking),
                                                        std::move(benchmarkId),
                                                        std::move(deviceId));
    }

    /// An arch-independent pack (empty `arch` list, itself legal) passes `archSupports`
    /// regardless of device identity, so the catalog can be non-empty with no device
    /// resolved.
    MatchContext contextFor(const THandle& handle, const IGraph& opGraph) const
    {
        return contextFor(handle, opGraph, hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC);
    }

    /// @param rankingMetric A registered metric's name, viewing the registry's storage (see
    ///        MatchContext::rankingMetric).
    MatchContext contextFor(const THandle& handle,
                            const IGraph& opGraph,
                            std::string_view rankingMetric) const
    {
        const auto deviceId = _deviceResolver.deviceId(handle);
        const auto& deviceProperties = _deviceResolver.deviceProperties(deviceId);
        if(deviceId == NO_DEVICE || deviceProperties.gcnArchName.empty())
        {
            const auto* reason = deviceId == NO_DEVICE
                                     ? "the device could not be resolved from the handle"
                                     : "the resolved device reports no gcnArchName";
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine '" << _engine.name
                                                         << "' cannot build a plan: " << reason);
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "engine '" + _engine.name
                                            + "' cannot build a plan: " + reason);
        }
        return MatchContext{opGraph, deviceId, deviceProperties, rankingMetric};
    }

    KnobFilter readKnobFilter(const IEngineConfig& engineConfig) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        KnobFilter filter;
        if(!engineConfig.isValid())
        {
            return filter;
        }

        for(const auto& knob : _knobs)
        {
            if(!engineConfig.hasKnobSetting(knob.id))
            {
                continue;
            }

            const auto& setting = engineConfig.getKnobSettingByName(knob.id);
            if(setting.valueType() != KnobValue::IntValue)
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "engine '" + _engine.name + "' knob '" + knob.id
                                                + "' must be set to an integer value");
            }
            filter[knob.id] = setting.template valueAs<IntValue>().value();
        }
        return filter;
    }

    /// Separate from readKnobFilter(): this knob is a plain on/off, never a metadata
    /// filter entry. Absent knob or invalid config both read as false; a non-int setting
    /// throws, matching every other knob's type contract.
    bool readBenchmarkingEnabled(const IEngineConfig& engineConfig) const
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        if(!engineConfig.isValid() || !engineConfig.hasKnobSetting(BENCHMARKING_KNOB_NAME))
        {
            return false;
        }

        const auto& setting = engineConfig.getKnobSettingByName(BENCHMARKING_KNOB_NAME);
        if(setting.valueType() != KnobValue::IntValue)
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "engine '" + _engine.name + "' knob '"
                                            + BENCHMARKING_KNOB_NAME
                                            + "' must be set to an integer value");
        }
        return setting.valueAs<IntValue>().value() != 0;
    }

    [[noreturn]] void throwNoApplicableKernel(const std::string& reason = {}) const
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "engine '" + _engine.name
                                        + "' accepted this graph but has no applicable kernel"
                                        + (reason.empty() ? "" : " (" + reason + ")"));
    }

    /// @param reasons Why each candidate was rejected, in the order they were tried.
    ///                Carried in the message because the per-candidate warnings are
    ///                logged at WARN, which the default log level does not emit: without
    ///                this the caller sees only that everything failed, not why.
    /// @param slot The slot whose kernels failed; empty for an unslotted engine.
    [[noreturn]] void throwNoBuildableKernel(size_t candidates,
                                             const std::vector<std::string>& reasons,
                                             const std::string& slot = {}) const
    {
        std::string detail;
        for(const auto& reason : reasons)
        {
            detail += (detail.empty() ? "" : "; ") + reason;
        }

        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "engine '" + _engine.name + "' could not build a plan for any "
                                        + "of its " + std::to_string(candidates)
                                        + " applicable kernel(s)"
                                        + (slot.empty() ? "" : " in slot '" + slot + "'")
                                        + (detail.empty() ? "" : " (" + detail + ")"));
    }

    /// @param slot The slot whose knob settings no kernel satisfies; 0 for an unslotted engine.
    ///        Names the settings alphabetically, only those addressing @p slot's knobs.
    [[noreturn]] void throwUnsatisfiableKnobFilter(size_t slot,
                                                   const KnobFilter& filter,
                                                   size_t survivorsBeforeFilter) const
    {
        std::string settingsText;
        for(const auto& [knobName, value] : filter)
        {
            const auto* knob = findKnob(knobName);
            if(knob == nullptr || knob->slot != slot)
            {
                continue;
            }
            if(!settingsText.empty())
            {
                settingsText += ", ";
            }
            settingsText += knobName + "=" + std::to_string(value);
        }

        throw HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "engine '" + _engine.name + "' has no kernel"
                + (_stateManager.isSlotted() ? " in slot '" + _stateManager.slotName(slot) + "'"
                                             : std::string())
                + " satisfying the requested knob setting(s) " + settingsText + " ("
                + std::to_string(survivorsBeforeFilter)
                + " kernel(s) matched the graph before knob filtering)");
    }

    const EngineDescriptor& _engine;
    const KernelIngestorStateManager<THandle>& _stateManager;
    const IDeviceResolver<THandle>& _deviceResolver;
    typename BenchmarkPlan<THandle>::Timer _timer;
    /// Every knob the engine exposes, in slot then declaration order.
    std::vector<KnobRef> _knobs;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
