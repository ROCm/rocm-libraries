// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// A caller's requested value for each knob it explicitly set, keyed by KMD field
/// name.
using KnobFilter = std::map<std::string, int64_t>;

/// What a `TSettings` used with GenericPlanBuilder must carry, grouped so a second
/// provider embeds one member rather than replicating loose fields by name.
struct IngestorSettings
{
    KnobFilter knobFilter;
    bool benchmarkingEnabled = false;
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
    /// outlives its builder.
    GenericPlanBuilder(const EngineDescriptor& engine,
                       const KernelIngestorStateManager<THandle>& stateManager,
                       const IDeviceResolver<THandle>& deviceResolver)
        : _engine(engine)
        , _stateManager(stateManager)
        , _deviceResolver(deviceResolver)
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
            = applyKnobFilter(catalog.entries, executionSettings.ingestorSettings.knobFilter);
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
        executionSettings.ingestorSettings.knobFilter = readKnobFilter(engineConfig);
        executionSettings.ingestorSettings.benchmarkingEnabled
            = benchmarkingOverrideFromEnv().value_or(readBenchmarkingEnabled(engineConfig));
    }

    void buildPlan(const THandle& handle,
                   const IGraph& opGraph,
                   const IEngineConfig& /*engineConfig*/,
                   TContext& executionContext) const override
    {
        const auto context = contextFor(handle, opGraph);
        const auto catalog = _stateManager.sortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        // The settings this context already carries, not a second parse of engineConfig:
        // initializeExecutionSettings() ran against this same config immediately before
        // and the engine stored the result.
        const auto& settings = executionContext.executionSettings().ingestorSettings;
        const auto filtered = applyKnobFilter(catalog.entries, settings.knobFilter);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(settings.knobFilter, catalog.entries.size());
        }

        if(!settings.benchmarkingEnabled)
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '"
                                   << _engine.name << "' selected kernel "
                                   << toString(filtered.front().kernelId) << " from "
                                   << filtered.size() << " candidate(s) (" << catalog.entries.size()
                                   << " before knob filtering)");

            executionContext.setPlan(std::make_unique<GenericPlan<THandle>>(
                _stateManager.getDispatchDetails(filtered.front()), context, catalog.bound));
            return;
        }

        HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '" << _engine.name << "' will benchmark "
                                                    << filtered.size() << " candidate(s) ("
                                                    << catalog.entries.size()
                                                    << " before knob filtering), ranked front "
                                                    << toString(filtered.front().kernelId));

        std::vector<typename BenchmarkPlan<THandle>::Candidate> candidates;
        candidates.reserve(filtered.size());
        for(const auto& kernel : filtered)
        {
            try
            {
                candidates.push_back(
                    {kernel.kernelId,
                     std::make_unique<GenericPlan<THandle>>(
                         _stateManager.getDispatchDetails(kernel), context, catalog.bound)});
            }
            catch(const std::exception& error)
            {
                HIPDNN_PLUGIN_LOG_WARN("ingestor: engine '"
                                       << _engine.name << "' dropped benchmarking candidate '"
                                       << toString(kernel.kernelId) << "': " << error.what());
            }
        }

        // An empty vector here (every candidate's GenericPlan threw) throws
        // INTERNAL_ERROR out of BenchmarkPlan's constructor, propagating unhandled.
        executionContext.setPlan(
            std::make_unique<BenchmarkPlan<THandle>>(std::move(candidates), handle));
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

            std::vector<int64_t> choices;
            choices.reserve(values.size());
            for(const auto& value : values)
            {
                if(const auto* intValue = std::get_if<int64_t>(&value))
                {
                    choices.push_back(*intValue);
                }
            }
            if(choices.empty())
            {
                continue;
            }

            KnobT knob;
            knob.knob_id = knobName;
            knob.description
                = "Kernel metadata field '" + knobName + "' of engine '" + _engine.name + "'";

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

private:
    /// A plan built for a device that was never identified cannot be correct: an
    /// arch-independent pack (empty `arch` list, itself legal) is admitted by
    /// `archSupports` regardless of device identity, so the catalog can come back
    /// non-empty even though no device was actually resolved. Rejecting here, at the
    /// one `MatchContext` construction site every entry point shares, closes that gap
    /// once instead of at each caller.
    MatchContext contextFor(const THandle& handle, const IGraph& opGraph) const
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
        return MatchContext{opGraph, deviceId, deviceProperties};
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
            const bool matchesEverySetKnob
                = std::all_of(filter.begin(), filter.end(), [&kernel](const auto& setting) {
                      const auto value = kernel.tryGetMetadata(setting.first);
                      const auto* intValue
                          = value.has_value() ? std::get_if<int64_t>(&*value) : nullptr;
                      return intValue != nullptr && *intValue == setting.second;
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
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
