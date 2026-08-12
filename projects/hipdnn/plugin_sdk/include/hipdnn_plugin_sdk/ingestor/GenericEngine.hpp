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
     * @param engine         The UED this engine is. A UED is 1:1 with a hipDNN engine, so
     *        the engine owns it: its name is this engine's identity, and its knob list is
     *        what this engine exposes to a caller.
     * @param stateManager   The descriptor state this engine selects over, already
     *        validated.
     * @param deviceResolver Answers which device each call is for, from the handle that
     *        carries it. Held by reference; owned by the provider, which must keep it
     *        alive for the engine's lifetime.
     *
     * @throws std::invalid_argument if a knob names no field in the engine's metadata
     *         schema. RFC 0017 §4 makes that a load error: a knob is only a name, with
     *         the field supplying its type, its default and its legal values, so a knob
     *         matching no field can never be reported or honoured. Caught here because
     *         this is the one place holding both the UED and its KMD.
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

    /// Not relocatable: _planBuilder holds references into _engine and *_stateManager,
    /// so a move or copy would leave it referring into the source object.
    GenericEngine(const GenericEngine&) = delete;
    GenericEngine& operator=(const GenericEngine&) = delete;
    GenericEngine(GenericEngine&&) = delete;
    GenericEngine& operator=(GenericEngine&&) = delete;

    /// The descriptor this engine was built from, for diagnostics and for a caller that
    /// needs the engine's declared knobs or notes.
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
        // The UED's notes reach the caller the same way every other engine's do. Passing
        // nothing here would report a descriptor-backed engine as having no behavior
        // notes at all, which is a claim rather than an omission.
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
    /// pointer rather than by value so the plan builder can bind a reference to it in the
    /// member initializer list, which needs a stable address.
    std::unique_ptr<KernelIngestorStateManager<THandle>> _stateManager;
    int64_t _id;
    GenericPlanBuilder<THandle, TSettings, TContext> _planBuilder;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
