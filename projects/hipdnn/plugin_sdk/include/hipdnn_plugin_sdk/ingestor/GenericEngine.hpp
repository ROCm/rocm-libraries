// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief One hipDNN engine, defined entirely by a UED and the packs naming it.
 *
 * Satisfies hipDNN's existing IEngine contract using descriptor data; from the host's
 * side this is an ordinary engine.
 *
 * The engine's id is its UED name hashed into hipDNN's engine-id space, registered when
 * the engine is constructed rather than at build time, so an id collision surfaces here
 * as a failure to create the engine.
 *
 * Holds exactly one plan builder: a catalog entry is a candidate, not a builder, so an
 * engine with 150 kernels still has one.
 */
template <typename THandle, typename TSettings, typename TContext>
class GenericEngine : public IEngine<THandle, TSettings, TContext>
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /**
     * @param engine         The UED this engine is; 1:1 with a hipDNN engine, so its
     *        name is this engine's identity and its knob list is what it exposes.
     * @param stateManager   The descriptor state this engine selects over, already
     *        validated.
     * @param deviceResolver Answers which device each call is for. Held by reference;
     *        owned by the provider, which must keep it alive for the engine's lifetime.
     *
     * @throws std::invalid_argument if a knob names no field in the engine's metadata
     *         schema (RFC 0017 §4): a knob is only a name, with the field supplying its
     *         type, default and legal values.
     */
    GenericEngine(EngineDescriptor engine,
                  std::unique_ptr<KernelIngestorStateManager<THandle>> stateManager,
                  const IDeviceResolver<THandle>& deviceResolver)
        : _engine(std::move(engine))
        , _stateManager(std::move(stateManager))
        , _id(hipdnn_data_sdk::utilities::engineNameToId(_engine.name))
        , _planBuilder(_engine, *_stateManager, deviceResolver)
    {
        const auto& fields = _stateManager->metadataSchema().fields;
        for(const auto& knob : _engine.knobs)
        {
            const auto declared
                = std::any_of(fields.begin(), fields.end(), [&knob](const MetadataField& field) {
                      return field.name == knob;
                  });
            if(!declared)
            {
                throw std::invalid_argument("engine '" + _engine.name + "' exposes knob '" + knob
                                            + "', which its metadata schema does not declare");
            }
        }
    }

    /// Not relocatable: _planBuilder holds references into _engine and *_stateManager.
    GenericEngine(const GenericEngine&) = delete;
    GenericEngine& operator=(const GenericEngine&) = delete;
    GenericEngine(GenericEngine&&) = delete;
    GenericEngine& operator=(GenericEngine&&) = delete;

    /// The descriptor this engine was built from, for diagnostics.
    const EngineDescriptor& descriptor() const
    {
        return _engine;
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
     * how every other engine in this provider answers the query.
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
        // The UED's notes reach the caller the same way every other engine's do.
        auto behaviorNotes = builder.CreateVector(_engine.behaviorNotes);
        auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
            builder, _id, knobs, behaviorNotes);
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
    EngineDescriptor _engine;
    /// Owned outright: see the constructor for why this must not be shared. Held by
    /// pointer so the plan builder can bind a reference to it in the member
    /// initializer list, which needs a stable address.
    std::unique_ptr<KernelIngestorStateManager<THandle>> _stateManager;
    int64_t _id;
    GenericPlanBuilder<THandle, TSettings, TContext> _planBuilder;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
