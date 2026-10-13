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
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The first knob @p engine exposes that @p fields does not declare, or nullptr. RFC 0017
/// §4 treats an undeclared knob as a load error, since the field supplies its type,
/// default, and legal values. Shared with the descriptor loader so a bad engine is
/// rejected while reading it, before its id is ever advertised.
inline const std::string* findUndeclaredKnob(const EngineDescriptor& engine,
                                             const std::vector<MetadataField>& fields)
{
    for(const auto& knob : engine.knobs)
    {
        const auto declared
            = std::any_of(fields.begin(), fields.end(), [&knob](const MetadataField& field) {
                  return field.name == knob;
              });
        if(!declared)
        {
            return &knob;
        }
    }
    return nullptr;
}

/// One hipDNN engine, defined entirely by a UED and the packs naming it. The
/// engine's id is its UED name hashed into hipDNN's engine-id space.
template <typename THandle, typename TSettings, typename TContext>
class GenericEngine : public IEngine<THandle, TSettings, TContext>
{
public:
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    /// @throws std::invalid_argument if a knob names no field in the metadata schema.
    GenericEngine(EngineDescriptor engine,
                  std::unique_ptr<KernelIngestorStateManager<THandle>> stateManager,
                  const IDeviceResolver<THandle>& deviceResolver)
        : _engine(std::move(engine))
        , _stateManager(std::move(stateManager))
        , _id(hipdnn_data_sdk::utilities::engineNameToId(_engine.name))
        , _planBuilder(_engine, *_stateManager, deviceResolver)
    {
        if(const auto* undeclared
           = findUndeclaredKnob(_engine, _stateManager->metadataSchema().fields))
        {
            throw std::invalid_argument("engine '" + _engine.name + "' exposes knob '" + *undeclared
                                        + "', which its metadata schema does not declare");
        }
    }

    /// Not relocatable: _planBuilder holds references into _engine and *_stateManager.
    GenericEngine(const GenericEngine&) = delete;
    GenericEngine& operator=(const GenericEngine&) = delete;
    GenericEngine(GenericEngine&&) = delete;
    GenericEngine& operator=(GenericEngine&&) = delete;

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

    void getDetails(THandle& handle,
                    const IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override
    {
        flatbuffers::FlatBufferBuilder builder;

        std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobOffsets;
        // Advertised out-of-band: staying out of _engine.knobs keeps findUndeclaredKnob
        // from seeing it and readKnobFilter from filtering on it. Unconditional, since
        // the class-level static_assert already requires THandle to supply a stream.
        knobOffsets.push_back(KnobFactory::createIntKnob(
            builder, BENCHMARKING_KNOB_NAME, "Enable benchmarking", 0, 0, 1, 1, {}));
        for(const auto& knob : _planBuilder.getCustomKnobs(handle, opGraph))
        {
            knobOffsets.push_back(hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knob));
        }

        auto knobs = builder.CreateVector(knobOffsets);
        auto behaviorNotes = builder.CreateVector(_engine.behaviorNotes);
        auto name = builder.CreateString(_engine.name);
        auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
            builder, _id, knobs, behaviorNotes, name);
        builder.Finish(engineDetails);

        // Detached buffer outlives this call; the handle takes ownership.
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
    std::unique_ptr<KernelIngestorStateManager<THandle>> _stateManager;
    int64_t _id;
    GenericPlanBuilder<THandle, TSettings, TContext> _planBuilder;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
