// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief One hipDNN engine, defined entirely by a UED and the packs naming it.
 *
 * Satisfies hipDNN's existing IEngine contract using descriptor data. No new engine or
 * plugin-ABI interface is introduced: from the host's side this is an ordinary engine,
 * and everything data-driven about it lives behind that contract.
 *
 * The engine's id is its UED name hashed into hipDNN's engine-id space, the same
 * derivation a hand-written engine's registered name goes through. Because a
 * descriptor-backed engine is defined by data rather than by a compile-time
 * registration, its name is registered when the engine is constructed rather than at
 * build time — which is why an id collision surfaces here, as a failure to create the
 * engine, rather than as a link-time or startup error.
 *
 * Holds exactly one plan builder. A catalog entry is a candidate, not a builder, so an
 * engine with 150 kernels still has one.
 */
template <typename THandle, typename TSettings, typename TContext>
class GenericEngine : public IEngine<THandle, TSettings, TContext>
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /**
     * @param stateManager    The engine's descriptor state, already validated.
     * @param deviceProperties Supplies `$device.*`. Held by value: providers hand these
     *        out by value, so a reference would bind to a temporary.
     * @param deviceId        The device this engine's catalogs are keyed on.
     */
    GenericEngine(std::shared_ptr<KernelIngestorStateManager<THandle>> stateManager,
                  const hipDeviceProp_t& deviceProperties,
                  DeviceId deviceId)
        : _stateManager(std::move(stateManager))
        , _id(hipdnn_data_sdk::utilities::engineNameToId(_stateManager->engine().name))
        , _planBuilder(_stateManager, deviceProperties, deviceId)
    {
    }

    int64_t id() const override
    {
        return _id;
    }

    bool isApplicable(THandle& handle, const IGraph& opGraph) const override
    {
        return _planBuilder.isApplicable(handle, opGraph);
    }

    /**
     * @brief Reports this engine's knobs for @p opGraph.
     *
     * The returned buffer is detached and handed to the caller's handle to own, matching
     * how every other engine in this provider answers the query: the host reads the
     * buffer through the returned pointer and releases it by handle later.
     */
    void getDetails(THandle& handle,
                    const IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override
    {
        flatbuffers::FlatBufferBuilder builder;

        std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobOffsets;
        for(const auto& knob : _planBuilder.getCustomKnobs(handle, opGraph))
        {
            knobOffsets.push_back(hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knob));
        }

        auto knobs = builder.CreateVector(knobOffsets);
        auto engineDetails
            = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
        builder.Finish(engineDetails);

        auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
        detailsOut.ptr = detachedBuffer->data();
        detailsOut.size = detachedBuffer->size();
        handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
    }

    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const IEngineConfig& engineConfig) const override
    {
        TSettings executionSettings;
        _planBuilder.initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
        return _planBuilder.getMaxWorkspaceSize(handle, opGraph, executionSettings);
    }

    void initializeExecutionContext(const THandle& handle,
                                    const IGraph& opGraph,
                                    const IEngineConfig& engineConfig,
                                    TContext& executionContext) const override
    {
        TSettings executionSettings;
        _planBuilder.initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
        executionContext.setExecutionSettings(executionSettings);
        _planBuilder.buildPlan(handle, opGraph, engineConfig, executionContext);
    }

private:
    std::shared_ptr<KernelIngestorStateManager<THandle>> _stateManager;
    int64_t _id;
    GenericPlanBuilder<THandle, TSettings, TContext> _planBuilder;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
