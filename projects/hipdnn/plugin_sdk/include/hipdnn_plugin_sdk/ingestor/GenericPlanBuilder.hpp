// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

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
 * @tparam TSettings The provider's settings type. Carried through unchanged: knob
 *         settings reach the catalog as a filter (future work, see initializeExecutionSettings),
 *         not as provider-specific state this builder interprets.
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
                               const TSettings& /*executionSettings*/) const override
    {
        const auto context = contextFor(handle, opGraph);
        // Unsorted, but still one lookup for the entries and the bound state together.
        const auto catalog = _stateManager.unsortedCatalog(context);
        size_t maxBytes = 0;
        // Unsorted deliberately: a maximum over the survivors is order-independent, and
        // this call arrives for every candidate engine. Ranking here would load and run
        // each engine's heuristic to compute a number that does not depend on the order,
        // which is the cost the lazy-ranking split exists to avoid.
        for(const auto& kernel : catalog.entries)
        {
            const auto dispatcher = _stateManager.getDispatchDetails(kernel);
            maxBytes = std::max(maxBytes,
                                dispatcher.handler->workspaceBytes(context, catalog.bound, kernel));
        }
        return maxBytes;
    }

    /// Knob settings are read into provider settings here in the full design, where they
    /// filter the catalog before ranking. The first engine exposes its knob for
    /// reporting only, so there is nothing to carry across.
    void initializeExecutionSettings(const THandle& /*handle*/,
                                     const IGraph& /*opGraph*/,
                                     const IEngineConfig& /*engineConfig*/,
                                     TSettings& /*executionSettings*/) const override
    {
    }

    /**
     * @brief Builds a plan for the top-ranked kernel.
     *
     * @throws HipdnnPluginException if the catalog is empty. Reaching here with nothing
     *         to build means applicability accepted a graph this engine cannot serve —
     *         RFC 0017 §8.6 makes that a bug, not a legal outcome, so it surfaces as a
     *         failed plan build rather than a silent decline.
     */
    void buildPlan(const THandle& handle,
                   const IGraph& opGraph,
                   const IEngineConfig& /*engineConfig*/,
                   TContext& executionContext) const override
    {
        const auto context = contextFor(handle, opGraph);
        // One lookup for both the order and the bound state: asking separately would
        // match twice for a graph that carries no identity and so cannot be cached.
        const auto catalog = _stateManager.sortedCatalog(context);
        const auto& ranked = catalog.entries;
        if(ranked.empty())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "engine '" + _engine.name
                                            + "' accepted this graph but has no applicable kernel");
        }

        executionContext.setPlan(std::make_unique<GenericPlan<THandle>>(
            _stateManager.getDispatchDetails(ranked.front()), context, catalog.bound));
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

    const EngineDescriptor& _engine;
    /// Held by reference; owned by the engine, which owns this builder and so outlives
    /// it. Sharing ownership here would reintroduce exactly the cross-engine aliasing the
    /// engine's ownership rules out.
    const KernelIngestorStateManager<THandle>& _stateManager;
    const IDeviceResolver<THandle>& _deviceResolver;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
