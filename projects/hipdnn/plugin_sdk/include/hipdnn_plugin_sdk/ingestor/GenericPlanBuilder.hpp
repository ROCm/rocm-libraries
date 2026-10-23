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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
            return !_stateManager.unsortedDefinitions(contextFor(handle, opGraph)).empty();
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine '"
                                    << _engine.name
                                    << "' declined the graph: deciding applicability failed: "
                                    << error.what());
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

    /// The largest scratch any surviving kernel needs; reused, not partitioned, since
    /// kernels launch one at a time on one stream.
    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const TSettings& executionSettings) const override
    {
        const auto context = contextFor(handle, opGraph);
        const auto catalog = _stateManager.unsortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        const auto filtered
            = applyConstraints(catalog, executionSettings.ingestorSettings, context);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(executionSettings.ingestorSettings.knobFilter,
                                         catalog.entries.size());
        }

        size_t maxBytes = 0;
        for(const auto& kernel : filtered)
        {
            const auto dispatcher = _stateManager.getDispatchDetails(kernel);
            maxBytes = std::max(maxBytes,
                                dispatcher.handler->workspaceBytes(context, catalog.bound, kernel));
        }
        return maxBytes;
    }

    /// The override is consulted unconditionally: it must change the outcome even when
    /// engineConfig is invalid or carries no knob, which is what makes a plain
    /// hipdnnExecute benchmark. readBenchmarkingEnabled() always runs so the knob's own
    /// answer is available to value_or().
    void initializeExecutionSettings(const THandle& /*handle*/,
                                     const IGraph& /*opGraph*/,
                                     const IEngineConfig& engineConfig,
                                     TSettings& executionSettings) const override
    {
        auto& settings = executionSettings.ingestorSettings;
        settings.knobFilter = readKnobFilter(engineConfig);
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
        const auto catalog = _stateManager.sortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        const auto filtered = applyConstraints(catalog, settings, context);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(settings.knobFilter, catalog.entries.size());
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
                        _stateManager.getDispatchDetails(filtered[rank]), context, catalog.bound);

                    // The record is the catalog's own snapshot, so this holds even after the
                    // winner cache evicts its copy.
                    HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                           << _engine.name << "' served kernel "
                                           << toString(filtered[rank].kernelId) << " at rank "
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
                                       << toString(filtered[rank].kernelId) << " at rank " << rank
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
                        _stateManager.getDispatchDetails(filtered[rank]), context, catalog.bound);

                    HIPDNN_PLUGIN_LOG_INFO(
                        "ingestor: engine '"
                        << _engine.name << "' selected kernel " << toString(filtered[rank].kernelId)
                        << " at rank " << rank << " from " << filtered.size() << " candidate(s) ("
                        << catalog.entries.size() << " before knob filtering) ranked by metric '"
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
                    failures.emplace_back(toString(filtered[rank].kernelId) + ": " + error.what());
                }
                catch(const std::exception& error)
                {
                    failures.emplace_back(toString(filtered[rank].kernelId) + ": " + error.what());
                }

                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' could not build a plan for "
                                       << toString(filtered[rank].kernelId) << " at rank " << rank
                                       << ": " << failures.back() << "; trying the next candidate");
            }

            throwNoBuildableKernel(filtered.size(), failures);
        }

        HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '" << _engine.name << "' will benchmark "
                                                    << filtered.size() << " candidate(s) ("
                                                    << catalog.entries.size()
                                                    << " before knob filtering), ranked front "
                                                    << toString(filtered.front().kernelId));

        // Every candidate walk applies the same policy: an unbuildable candidate is a reason
        // to carry, a malformed descriptor stops the build. Absorbing here what the others
        // rethrow would make the diagnosis a consequence of a tuning setting.
        std::vector<std::string> benchmarkFailures;
        std::vector<typename BenchmarkPlan<THandle>::Candidate> candidates;
        candidates.reserve(filtered.size());
        const auto problem = problemFeaturesJson(context, catalog.bound);
        for(const auto& kernel : filtered)
        {
            try
            {
                candidates.push_back(
                    {kernel.kernelId,
                     std::make_unique<GenericPlan<THandle>>(
                         _stateManager.getDispatchDetails(kernel), context, catalog.bound),
                     kernel.packId,
                     kernel.dispatchId,
                     candidateFeatures(problem, kernel)});
                continue;
            }
            catch(const HipdnnPluginException& error)
            {
                if(error.getStatus() == HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
                {
                    throw;
                }
                benchmarkFailures.emplace_back(toString(kernel.kernelId) + ": " + error.what());
            }
            catch(const std::exception& error)
            {
                benchmarkFailures.emplace_back(toString(kernel.kernelId) + ": " + error.what());
            }

            HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '" << _engine.name
                                                        << "' dropped benchmarking candidate '"
                                                        << toString(kernel.kernelId)
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
    /// One knob per KMD field the engine exposes; default is the top-ranked value.
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        getCustomKnobs(const THandle& handle, const IGraph& opGraph) const override
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        const auto ranked = _stateManager.sortedDefinitions(contextFor(handle, opGraph));
        std::vector<KnobT> knobs;
        if(ranked.empty())
        {
            return knobs;
        }

        for(const auto& knobName : _engine.knobs)
        {
            const auto values = KernelIngestorStateManager<THandle>::knobValues(ranked, knobName);

            // A non-integer value is advertised as its ordinal, which is what a caller pins
            // to select that kernel (RFC 0019 §13.2).
            std::vector<int64_t> choices;
            choices.reserve(values.size());
            for(const auto& value : values)
            {
                if(const auto ordinal = _stateManager.knobOrdinal(knobName, value);
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
            knob.knob_id = knobName;
            knob.description = "Kernel metadata field '" + knobName + "' of engine '" + _engine.name
                               + "'"
                               + (_stateManager.isOrdinalKnob(knobName)
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
            if(!seen.insert(name).second
               || std::find(_engine.knobs.begin(), _engine.knobs.end(), name)
                      == _engine.knobs.end())
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "Unknown or duplicate enumeration knob '" + name + "'");
            }
        }
        const auto context = contextFor(handle, opGraph);
        const auto catalog = _stateManager.enumerableCatalog(context);
        const auto filtered = applyKnobFilter(catalog.entries, readKnobFilter(config));
        if(offset > filtered.size())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "Candidate offset exceeds scoped catalog size");
        }

        // Check the entire scoped snapshot, not just this page: a collision across a
        // page boundary is still unaddressable by EngineVariant enrollment.
        std::map<KnobFilter, DescriptorId> identities;
        for(const auto& kernel : filtered)
        {
            const auto tuple = candidateKnobs(kernel);
            const auto inserted = identities.emplace(tuple, kernel.kernelId);
            if(!inserted.second)
            {
                throw HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "Ambiguous enrolled knob tuple for candidates '"
                        + toString(inserted.first->second) + "' and '" + toString(kernel.kernelId)
                        + "'; expose enough KMD integer fields in the collection UED knobs");
            }
        }

        EngineCandidatePageT page;
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
        page.total_count = filtered.size();
        page.offset = offset;
        // The live ranker's problem features, with `device.*` split into their own field.
        auto problem = problemFeaturesJson(context, catalog.bound);
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
        const auto end = offset + std::min<uint64_t>(limit, filtered.size() - offset);
        page.candidates.reserve(static_cast<size_t>(end - offset));
        for(auto i = offset; i < end; ++i)
        {
            const auto& kernel = filtered[static_cast<size_t>(i)];
            auto candidate = std::make_unique<EngineCandidateT>();
            candidate->id = toString(kernel.kernelId);
            for(const auto& [name, value] : candidateKnobs(kernel))
            {
                auto setting = std::make_unique<KnobSettingT>();
                setting->knob_id = name;
                IntValueT integer;
                integer.value = value;
                setting->value.Set(integer);
                candidate->knob_settings.push_back(std::move(setting));
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
                detail::bindGraphMatchBindings(features, *bound);
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
            const auto selected = std::find_if(
                catalog.entries.begin(), catalog.entries.end(), [&](const auto& kernel) {
                    return kernel.kernelId == scored.kernelId;
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
                    _stateManager.getDispatchDetails(*selected), context, catalog.bound);
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
            const auto knobs = candidateKnobs(*selected);
            // Same comparison as applyKnobFilter(): the tuple holds ordinals for non-integer knobs.
            const auto matching = std::count_if(
                catalog.entries.begin(), catalog.entries.end(), [this, &knobs](const auto& kernel) {
                    return std::all_of(
                        knobs.begin(), knobs.end(), [this, &kernel](const auto& setting) {
                            return _stateManager.knobMatches(kernel, setting.first, setting.second);
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
            auto exact = config.isValid()
                             ? std::unique_ptr<EngineConfigT>(config.getEngineConfig().UnPack())
                             : std::make_unique<EngineConfigT>();
            exact->engine_id = result.engine_id;
            for(const auto& field : knobs)
            {
                const auto& name = field.first;
                const auto value = field.second;
                auto existing
                    = std::find_if(exact->knobs.begin(),
                                   exact->knobs.end(),
                                   [&](const auto& setting) { return setting->knob_id == name; });
                if(existing == exact->knobs.end())
                {
                    auto setting = std::make_unique<KnobSettingT>();
                    setting->knob_id = name;
                    IntValueT integer;
                    integer.value = value;
                    setting->value.Set(integer);
                    exact->knobs.push_back(std::move(setting));
                }
            }
            result.engine_config = std::move(exact);
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
                   && std::find(_engine.knobs.begin(), _engine.knobs.end(), name)
                          == _engine.knobs.end()))
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "Unknown or duplicate selection constraint '" + name
                                                + "'");
            }
        }
    }

private:
    /// @p record's order over @p filtered, valued in @p metric: the time, or for `tflops`
    /// `graph.flops / (ms * 1e9)`, the label uhd_gen trains on (RFC 0019 §5 step 9).
    /// @param measuresMetric Set false when the record cannot supply @p metric; every value
    ///        is then 0 and only the order is the record's.
    static std::vector<ScoredKernel>
        measuredRanking(const WinnerRecord& record,
                        const std::vector<KernelDefinition>& filtered,
                        const hipdnn_data_sdk::utilities::RankingMetric& metric,
                        const MatchContext& context,
                        bool& measuresMetric)
    {
        std::optional<double> flops;
        if(metric.name == "tflops")
        {
            const auto problem
                = heuristics::problemFeatures(context.graph, context.deviceProperties);
            if(const auto* value = problem.getContext().find("graph.flops"))
            {
                if(const auto* known = std::get_if<double>(value))
                {
                    flops = *known;
                }
            }
        }
        measuresMetric = metric.name == "time" || flops.has_value();

        std::vector<ScoredKernel> ranking;
        ranking.reserve(filtered.size());
        for(const auto& entry : record)
        {
            const bool admitted
                = std::any_of(filtered.begin(), filtered.end(), [&entry](const auto& kernel) {
                      return kernel.kernelId == entry.kernelId && kernel.packId == entry.packId
                             && kernel.dispatchId == entry.dispatchId;
                  });
            if(!admitted)
            {
                continue;
            }
            double value = 0.0;
            if(metric.name == "time")
            {
                value = entry.timeMs;
            }
            else if(flops.has_value() && entry.timeMs > 0.0)
            {
                value = *flops / (entry.timeMs * 1e9);
            }
            ranking.push_back({entry.kernelId, value});
        }
        return ranking;
    }

    std::vector<KernelDefinition> applyConstraints(const Catalog& catalog,
                                                   const IngestorSettings& settings,
                                                   const MatchContext& context) const
    {
        auto filtered = applyKnobFilter(catalog.entries, settings.knobFilter);
        if(settings.workspaceLimit)
        {
            filtered.erase(
                std::remove_if(filtered.begin(),
                               filtered.end(),
                               [&](const auto& kernel) {
                                   const auto dispatch = _stateManager.getDispatchDetails(kernel);
                                   return dispatch.handler->workspaceBytes(
                                              context, catalog.bound, kernel)
                                          > static_cast<uint64_t>(*settings.workspaceLimit);
                               }),
                filtered.end());
        }
        return filtered;
    }

    KnobFilter candidateKnobs(const KernelDefinition& kernel) const
    {
        KnobFilter tuple;
        for(const auto& name : _engine.knobs)
        {
            const auto it = kernel.metadata.find(name);
            // The tuple must address this kernel, since replay pins exactly these values.
            // Non-integer values enter as their ordinal (RFC 0019 §13.2).
            if(it != kernel.metadata.end())
            {
                if(const auto ordinal = _stateManager.knobOrdinal(name, it->second);
                   ordinal.has_value())
                {
                    tuple.emplace(name, *ordinal);
                }
            }
        }
        return tuple;
    }

    /// The problem half of a `sort_kernel_catalog` row as JSON: the scalars the live ranker
    /// binds (detail::catalogProblemFeatures) plus each int-list token whole. `device.*`
    /// facts distinguish boards of one arch. Keys drop the leading '$'.
    static nlohmann::json problemFeaturesJson(const MatchContext& context, const BoundTokens& bound)
    {
        auto features = detail::catalogProblemFeatures(context, bound).toJson();
        for(const auto& [token, value] : bound)
        {
            // A list has no scalar binding, so a model reads it only through the indexed
            // names already present; the whole list rides along for readers of the row.
            const auto* values = std::get_if<std::vector<int64_t>>(&value);
            if(values != nullptr && !detail::isReservedFeatureName(token))
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

        for(const auto& knobName : _engine.knobs)
        {
            if(!engineConfig.hasKnobSetting(knobName))
            {
                continue;
            }

            const auto& setting = engineConfig.getKnobSettingByName(knobName);
            if(setting.valueType() != KnobValue::IntValue)
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "engine '" + _engine.name + "' knob '" + knobName
                                                + "' must be set to an integer value");
            }
            filter[knobName] = setting.valueAs<IntValue>().value();
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

    std::vector<KernelDefinition> applyKnobFilter(const std::vector<KernelDefinition>& catalog,
                                                  const KnobFilter& filter) const
    {
        if(filter.empty())
        {
            return catalog;
        }

        std::vector<KernelDefinition> filtered;
        filtered.reserve(catalog.size());
        for(const auto& kernel : catalog)
        {
            // Pins match through the engine's ordinal domain, so non-integer fields are pinnable.
            const bool matchesEverySetKnob
                = std::all_of(filter.begin(), filter.end(), [this, &kernel](const auto& setting) {
                      return _stateManager.knobMatches(kernel, setting.first, setting.second);
                  });
            if(matchesEverySetKnob)
            {
                filtered.push_back(kernel);
            }
        }
        return filtered;
    }

    [[noreturn]] void throwNoApplicableKernel() const
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "engine '" + _engine.name
                                        + "' accepted this graph but has no applicable kernel");
    }

    /// @param reasons Why each candidate was rejected, in the order they were tried.
    ///                Carried in the message because the per-candidate warnings are
    ///                logged at WARN, which the default log level does not emit: without
    ///                this the caller sees only that everything failed, not why.
    [[noreturn]] void throwNoBuildableKernel(size_t candidates,
                                             const std::vector<std::string>& reasons) const
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
                                        + (detail.empty() ? "" : " (" + detail + ")"));
    }

    [[noreturn]] void throwUnsatisfiableKnobFilter(const KnobFilter& filter,
                                                   size_t survivorsBeforeFilter) const
    {
        std::string settingsText;
        for(const auto& [knobName, value] : filter)
        {
            if(!settingsText.empty())
            {
                settingsText += ", ";
            }
            settingsText += knobName + "=" + std::to_string(value);
        }

        throw HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "engine '" + _engine.name + "' has no kernel satisfying the requested knob setting(s) "
                + settingsText + " (" + std::to_string(survivorsBeforeFilter)
                + " kernel(s) matched the graph before knob filtering)");
    }

    const EngineDescriptor& _engine;
    const KernelIngestorStateManager<THandle>& _stateManager;
    const IDeviceResolver<THandle>& _deviceResolver;
    typename BenchmarkPlan<THandle>::Timer _timer;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
