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
/// field) name. Only integer-valued fields are expressible as knobs today (see
/// GenericPlanBuilder::getCustomKnobs), so this is the same restriction knobValues()
/// already has, not a new one.
///
/// A `TSettings` used with GenericPlanBuilder must carry one of these named
/// `ingestorKnobFilter`, populated by initializeExecutionSettings() and read back by
/// getMaxWorkspaceSize(): that method's IPlanBuilder signature carries no
/// IEngineConfig, so TSettings is the only channel by which a knob setting can reach
/// it. buildPlan() gets IEngineConfig directly and reads the filter from there instead.
using KnobFilter = std::map<std::string, int64_t>;

/**
 * @brief The one plan builder a descriptor-backed engine has.
 *
 * Not one per kernel: a catalog entry is a candidate, and this builder's job is to
 * produce a plan that can launch whichever candidate selection chose. An engine with 150
 * kernels still has one builder and pays only for the kernel a plan needs.
 *
 * Every method here is a read of state the state manager already computed and cached.
 * Applicability builds the catalog; the knob query and the plan build read the ranked
 * order; the workspace query reads the dispatch descriptors. Nothing rebuilds.
 *
 * @tparam TSettings The provider's settings type. Must carry a `KnobFilter
 *         ingestorKnobFilter` member (see KnobFilter's doc for why): knob settings
 *         reach the catalog as a filter applied before ranking, RFC 0017 §4's
 *         catalog -> filter -> rank order. Filtering after GenericEngine's already-
 *         ranked read and taking the filtered list's front is equivalent to filtering
 *         first: IKernelHeuristic's score() takes one kernel at a time and is never
 *         handed the catalog, so filtering and ranking commute (see its doc), and a
 *         stable filter over an already-ranked list preserves that ranking.
 */
template <typename THandle, typename TSettings, typename TContext>
class GenericPlanBuilder : public IPlanBuilder<THandle, TSettings, TContext>
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /// @param engine The UED whose knobs this builder reports and whose name its
    ///        diagnostics carry. Held by reference; owned by the engine, which outlives
    ///        its builder.
    /// @param stateManager The descriptor state to select over. Shared rather than owned
    ///        so the engine and its builder read one catalog cache.
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

    /// Applicable exactly when some kernel survived matching. Deliberately does not rank:
    /// a membership test needs no order, so the heuristic is never run here.
    bool isApplicable(const THandle& handle, const IGraph& opGraph) const override
    {
        return !_stateManager.unsortedDefinitions(contextFor(handle, opGraph)).empty();
    }

    /**
     * @brief The largest scratch any surviving kernel needs.
     *
     * The maximum suffices because the buffer is reused rather than partitioned: kernels
     * launch one at a time on one stream, so a candidate's scratch is live only while it
     * runs. A kernel needing less over-allocates, which is accepted.
     */
    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const TSettings& executionSettings) const override
    {
        const auto context = contextFor(handle, opGraph);
        // Unsorted, but still one lookup for the entries and the bound state together.
        const auto catalog = _stateManager.unsortedCatalog(context);
        // Filtered before the max is taken, same contract as buildPlan: the caller's
        // knob settings are a request for a subset of the catalog, not a value the
        // survivors are merely ranked by.
        const auto filtered
            = applyKnobFilter(catalog.entries, executionSettings.ingestorKnobFilter);
        if(!catalog.entries.empty() && filtered.empty())
        {
            // Same failure buildPlan reports for the same reason: a workspace query and
            // a plan build must not disagree about whether a knob combination is legal.
            throwUnsatisfiableKnobFilter(executionSettings.ingestorKnobFilter,
                                         catalog.entries.size());
        }

        size_t maxBytes = 0;
        // Unsorted deliberately: a maximum over the survivors is order-independent, and
        // this call arrives for every candidate engine. Ranking here would load and run
        // each engine's heuristic to compute a number that does not depend on the order,
        // which is the cost the lazy-ranking split exists to avoid.
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
     * Populates `executionSettings.ingestorKnobFilter` so getMaxWorkspaceSize() -- which
     * receives only TSettings, not an IEngineConfig -- can apply the same filter
     * buildPlan() reads directly from @p engineConfig. This is the one channel by which
     * a knob setting reaches that call.
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
     *         is empty before knob filtering. Reaching here with nothing to build means
     *         applicability accepted a graph this engine cannot serve at all — RFC 0017
     *         §8.6 makes that a bug, not a legal outcome, so it surfaces as a failed
     *         plan build rather than a silent decline. Distinct from the case below:
     *         this is a matcher/applicability disagreement, not a bad caller request.
     * @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) if the catalog
     *         is non-empty but knob filtering empties it. See
     *         throwUnsatisfiableKnobFilter() for why this is the caller's fault rather
     *         than the engine's: a graph this engine can serve exists, but no kernel
     *         implements the specific combination of values the caller asked for.
     */
    void buildPlan(const THandle& handle,
                   const IGraph& opGraph,
                   const IEngineConfig& engineConfig,
                   TContext& executionContext) const override
    {
        const auto context = contextFor(handle, opGraph);
        // One lookup for both the order and the bound state: asking separately would
        // match twice for a graph that carries no identity and so cannot be cached.
        const auto catalog = _stateManager.sortedCatalog(context);
        if(catalog.entries.empty())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "engine '" + _engine.name
                                            + "' accepted this graph but has no applicable kernel");
        }

        // Filtered after ranking rather than before: filtering and ranking commute (see
        // this class's own tparam doc), so filtering the already-ranked list and taking
        // its front is the same kernel a filter-then-rank pipeline would choose, without
        // a second heuristic pass. RFC 0017 §4's catalog -> filter -> rank ordering is
        // preserved in effect, not in the order these two calls execute.
        const auto filter = readKnobFilter(engineConfig);
        const auto filtered = applyKnobFilter(catalog.entries, filter);
        if(filtered.empty())
        {
            throwUnsatisfiableKnobFilter(filter, catalog.entries.size());
        }

        // The selection an operator would otherwise have to infer. RFC 0017 §10 asks
        // that a resolved plan say which kernel it resolved to; with the catalog built
        // from descriptors rather than a switch statement, this line is the only place
        // the choice is observable without a debugger.
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
     * A knob's legal values are what the catalog implements for this graph, not the
     * field's theoretical range — offering a value no surviving kernel carries would
     * produce a request nothing can serve. Its default is whatever the heuristic ranked
     * first, so leaving every knob alone reproduces the out-of-the-box selection.
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
                // Only integer-valued fields are expressible as knobs today, since
                // hipDNN's integer knob constraint carries int64 choices. A
                // string-valued KMD field is skipped rather than stringified into a
                // number that would mean nothing to a caller.
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
            // knobValues() preserves ranked order, so the first entry is the top-ranked
            // kernel's value: leaving the knob alone reproduces the out-of-the-box
            // selection.
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

    /**
     * @brief Reads @p engineConfig's settings for this engine's own knobs into a
     *        KnobFilter.
     *
     * Only engine-exposed knobs are read: `_engine.knobs` names the KMD fields this
     * engine advertises through getCustomKnobs(), and a filter is built only from those
     * -- a setting for a knob this engine never exposed cannot mean anything to it.
     */
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
                // getCustomKnobs()), so a setting of another type names a knob this
                // engine never advertised as anything but an integer -- a caller error,
                // reported the same way an unsatisfiable value is below.
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "engine '" + _engine.name + "' knob '" + knobName
                                                + "' must be set to an integer value");
            }
            filter[knobName] = setting.valueAs<IntValue>().value();
        }
        return filter;
    }

    /**
     * @brief The catalog entries matching every SET knob in @p filter.
     *
     * An entry with no value for a filtered field never matches: knobValues() already
     * treats "no value for this field" as "does not offer a choice here" (see its doc),
     * and a filter is a request for a specific choice, so a kernel with nothing to say
     * about that field cannot be what the caller asked for.
     */
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
     * @brief Reports a knob combination no kernel in the catalog implements.
     *
     * HIPDNN_PLUGIN_STATUS_INVALID_VALUE, not HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR: the
     * caller supplied a value combination nothing implements, which is a bad input, not
     * an internal bug -- unlike buildPlan()'s empty-catalog-before-filtering case, which
     * really is a matcher/applicability disagreement. The two must stay distinguishable
     * by status and by message, since a caller diagnosing which case they hit has only
     * this exception to read.
     *
     * Names every SET knob and its requested value, not just the first mismatch: a
     * caller who set three knobs needs to see all three to know which to back off. Also
     * states how many kernels survived matching before knob filtering, which
     * distinguishes "the graph matched nothing" (irrelevant here; this path is only
     * reached when that count is nonzero) from "the graph matched, but your knobs
     * excluded everything" (always the case here).
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
    /// Held by reference; owned by the engine, which owns this builder and so outlives
    /// it. Sharing ownership here would reintroduce exactly the cross-engine aliasing the
    /// engine's ownership rules out.
    const KernelIngestorStateManager<THandle>& _stateManager;
    const IDeviceResolver<THandle>& _deviceResolver;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
