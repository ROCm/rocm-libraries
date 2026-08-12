// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// A caller's requested value for each knob it explicitly set, keyed by knob (= KMD
/// field) name. Only integer-valued fields are expressible as knobs today.
///
/// A `TSettings` used with GenericPlanBuilder must carry one of these named
/// `ingestorKnobFilter`, populated by initializeExecutionSettings() and read back by
/// getMaxWorkspaceSize() and buildPlan().
using KnobFilter = std::map<std::string, int64_t>;

/**
 * @brief The one plan builder a descriptor-backed engine has.
 *
 * One builder regardless of catalog size: a catalog entry is a candidate, and this
 * builds a plan for whichever candidate selection chose.
 *
 * Every method here reads state the state manager already computed and cached;
 * nothing rebuilds.
 *
 * @tparam TSettings Must carry a `KnobFilter ingestorKnobFilter` member, populated by
 *         initializeExecutionSettings() so getMaxWorkspaceSize() can apply the same
 *         filter buildPlan() reads directly from IEngineConfig.
 */
template <typename THandle, typename TSettings, typename TContext>
class GenericPlanBuilder : public IPlanBuilder<THandle, TSettings, TContext>
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /// @param engine The UED whose knobs this builder reports and whose name its
    ///        diagnostics carry. Held by reference; owned by the engine, which
    ///        outlives its builder.
    /// @param stateManager The descriptor state to select over. Shared, not owned, so
    ///        the engine and its builder read one catalog cache.
    /// @param deviceResolver Answers which device each call is for. Held by reference;
    ///        owned by the engine, which outlives its builder.
    GenericPlanBuilder(const EngineDescriptor& engine,
                       const KernelIngestorStateManager<THandle>& stateManager,
                       const IDeviceResolver<THandle>& deviceResolver)
        : _engine(engine)
        , _stateManager(stateManager)
        , _deviceResolver(deviceResolver)
    {
    }

    /// Applicable exactly when some kernel survived matching. Deliberately does not
    /// rank: a membership test needs no order.
    bool isApplicable(const THandle& handle, const IGraph& opGraph) const override
    {
        return !_stateManager.unsortedDefinitions(contextFor(handle, opGraph)).empty();
    }

    /**
     * @brief The largest scratch any surviving kernel needs.
     *
     * The buffer is reused rather than partitioned: kernels launch one at a time on
     * one stream, so a candidate's scratch is live only while it runs.
     *
     * @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) if the catalog
     *         is empty (see throwNoApplicableKernel()).
     * @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) if knob
     *         filtering empties a non-empty catalog (see throwUnsatisfiableKnobFilter()).
     */
    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const TSettings& executionSettings) const override
    {
        const auto context = contextFor(handle, opGraph);
        // Unsorted: one lookup for the entries and the bound state together.
        const auto catalog = _stateManager.unsortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        // Filtered before the max is taken: knob settings request a catalog subset,
        // same contract as buildPlan().
        const auto filtered
            = applyKnobFilter(catalog.entries, executionSettings.ingestorKnobFilter);
        if(filtered.empty())
        {
            // Same failure buildPlan() reports, for the same reason.
            throwUnsatisfiableKnobFilter(executionSettings.ingestorKnobFilter,
                                         catalog.entries.size());
        }

        size_t maxBytes = 0;
        // Unsorted deliberately: the maximum is order-independent, and ranking here
        // would load every candidate engine's heuristic for a number that doesn't need it.
        for(const auto& kernel : filtered)
        {
            const auto dispatcher = _stateManager.getDispatchDetails(kernel);
            maxBytes = std::max(maxBytes,
                                dispatcher.handler->workspaceBytes(context, catalog.bound, kernel));
        }
        return maxBytes;
    }

    /**
     * @brief Reads the caller's knob settings from @p engineConfig into TSettings.
     *
     * Populates `executionSettings.ingestorKnobFilter` so getMaxWorkspaceSize() --
     * which receives only TSettings, not an IEngineConfig -- reads the same filter
     * buildPlan() reads directly from @p engineConfig.
     */
    void initializeExecutionSettings(const THandle& /*handle*/,
                                     const IGraph& /*opGraph*/,
                                     const IEngineConfig& engineConfig,
                                     TSettings& executionSettings) const override
    {
        executionSettings.ingestorKnobFilter = readKnobFilter(engineConfig);
    }

    /**
     * @brief Builds a plan for the top-ranked kernel satisfying every knob the caller
     *        set.
     *
     * @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR) if the catalog
     *         is empty before knob filtering (see throwNoApplicableKernel()).
     * @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) if knob
     *         filtering empties a non-empty catalog (see throwUnsatisfiableKnobFilter()).
     */
    void buildPlan(const THandle& handle,
                   const IGraph& opGraph,
                   const IEngineConfig& engineConfig,
                   TContext& executionContext) const override
    {
        const auto context = contextFor(handle, opGraph);
        // One lookup for both the order and the bound state.
        const auto catalog = _stateManager.sortedCatalog(context);
        if(catalog.entries.empty())
        {
            throwNoApplicableKernel();
        }

        // Filtered after ranking: filtering and ranking commute (see this class's
        // tparam doc), yielding the same kernel filter-then-rank would without a
        // second heuristic pass.
        const auto filter = readKnobFilter(engineConfig);
        const auto filtered = applyKnobFilter(catalog.entries, filter);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(filter, catalog.entries.size());
        }

        // Logs the selected kernel (RFC 0017 §10); the only place the choice is
        // observable without a debugger.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: engine '" << _engine.name << "' selected kernel "
                                                    << toString(filtered.front().kernelId)
                                                    << " from " << filtered.size()
                                                    << " candidate(s) (" << catalog.entries.size()
                                                    << " before knob filtering)");

        executionContext.setPlan(std::make_unique<GenericPlan<THandle>>(
            _stateManager.getDispatchDetails(filtered.front()), context, catalog.bound));
    }

    /**
     * @brief One knob per KMD field the engine exposes.
     *
     * A knob's legal values are what the catalog implements for this graph. Its
     * default is whatever the heuristic ranked first, so leaving every knob alone
     * reproduces the out-of-the-box selection.
     */
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
                // Only integer-valued fields are expressible as knobs today; a
                // string-valued field is skipped rather than stringified.
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
            // First entry is the top-ranked kernel's value, so leaving the knob alone
            // reproduces the out-of-the-box selection.
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
    /// Binds the graph and the device this call is for. Resolving the device here, once
    /// per call, is what keeps the cache key and the matchers reading the same one.
    MatchContext contextFor(const THandle& handle, const IGraph& opGraph) const
    {
        const auto deviceId = _deviceResolver.deviceId(handle);
        return MatchContext{opGraph, deviceId, _deviceResolver.deviceProperties(deviceId)};
    }

    /// Reads @p engineConfig's settings for this engine's own knobs into a KnobFilter.
    /// Only engine-exposed knobs are read: a setting for a knob this engine never
    /// advertised cannot mean anything to it.
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
                // Only integer-valued fields are expressible as knobs today (see
                // getCustomKnobs()); a setting of another type is a caller error.
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "engine '" + _engine.name + "' knob '" + knobName
                                                + "' must be set to an integer value");
            }
            filter[knobName] = setting.valueAs<IntValue>().value();
        }
        return filter;
    }

    /// The catalog entries matching every SET knob in @p filter. An entry with no
    /// value for a filtered field never matches: no value means no choice offered.
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

    /**
     * @brief Reports an empty catalog: shared by getMaxWorkspaceSize() and buildPlan(),
     *        since both run only after isApplicable() already accepted this graph.
     */
    [[noreturn]] void throwNoApplicableKernel() const
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "engine '" + _engine.name
                                        + "' accepted this graph but has no applicable kernel");
    }

    /**
     * @brief Reports a knob combination no kernel in the catalog implements.
     *
     * HIPDNN_PLUGIN_STATUS_INVALID_VALUE, not INTERNAL_ERROR: a bad caller input, not
     * a matcher/applicability disagreement like buildPlan()'s empty-catalog case.
     *
     * Names every set knob and value, and how many kernels survived matching before
     * knob filtering.
     */
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
    /// Held by reference; owned by the engine, which outlives this builder.
    const KernelIngestorStateManager<THandle>& _stateManager;
    const IDeviceResolver<THandle>& _deviceResolver;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
