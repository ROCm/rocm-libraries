// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
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
                  const IDeviceResolver<THandle>& deviceResolver,
                  std::map<std::string, HeuristicDescriptor> predictions = {},
                  std::set<std::string> unavailablePredictionArches = {},
                  std::string selectorRevision = {},
                  nlohmann::json provenance = nlohmann::json::object())
        : _engine(std::move(engine))
        , _stateManager(std::move(stateManager))
        , _id(hipdnn_data_sdk::utilities::engineNameToId(_engine.name))
        , _planBuilder(_engine, *_stateManager, deviceResolver)
        , _unavailablePredictionArches(std::move(unavailablePredictionArches))
        , _selectorRevision(selectorRevision.empty() ? "generic-untuned-v1/" + toString(_engine.id)
                                                           + "/" + _engine.revision.str()
                                                     : std::move(selectorRevision))
        , _provenance(std::move(provenance))
    {
        if(const auto* undeclared
           = findUndeclaredKnob(_engine, _stateManager->metadataSchema().fields))
        {
            throw std::invalid_argument("engine '" + _engine.name + "' exposes knob '" + *undeclared
                                        + "', which its metadata schema does not declare");
        }
        for(const auto& [arch, descriptor] : predictions)
        {
            _predictions.emplace(arch, UhdKernelHeuristic::configFrom(descriptor));
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

    void enumerateCandidates(THandle& handle,
                             const IGraph& opGraph,
                             const IEngineConfig& engineConfig,
                             uint64_t offset,
                             uint64_t limit,
                             hipdnnPluginConstData_t& detailsOut) const override
    {
        hipdnn_flatbuffers_sdk::data_objects::EngineDetailsT details;
        details.engine_id = _id;
        details.candidate_page
            = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineCandidatePageT>(
                _planBuilder.enumerateCandidates(handle, opGraph, engineConfig, offset, limit));
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EngineDetails::Pack(builder, &details));
        auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
        detailsOut = {buffer->data(), buffer->size()};
        handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(buffer));
    }

    /// @brief L1 uses only graph bindings; L2 returns the calibrated ranker's exact candidate.
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        getPrediction(THandle& handle,
                      const IGraph& graph,
                      const IEngineConfig& config,
                      hipdnnEnginePredictionKind_t kind,
                      bool evaluate) const override
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        if(evaluate && kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
        {
            EnginePredictionT result;
            result.engine_id = _id;
            result.kind = PredictionKind::CONFIGURATION;
            result.status = PredictionStatus::UNAVAILABLE;
            try
            {
                _planBuilder.predictConfiguration(handle, graph, config, result);
            }
            catch(const std::exception& error)
            {
                result.status = PredictionStatus::UNAVAILABLE;
                result.reason = error.what();
            }
            return result;
        }
        std::string arch;
        const auto features = _planBuilder.predictionFeatures(handle, graph, config, arch);
        std::string selectedArch;
        const uhd::UhdConfig* selected = nullptr;
        bool invalid = false;
        for(const auto& [target, model] : _predictions)
        {
            if((target == "default" && selectedArch.empty())
               || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                   && (selectedArch == "default" || target.size() > selectedArch.size())))
            {
                selected = &model;
                selectedArch = target;
            }
        }
        for(const auto& target : _unavailablePredictionArches)
        {
            if((target == "default" && selectedArch.empty())
               || (target != "default" && archMatches(arch, target, ArchMatchMode::PREFIX)
                   && (selectedArch == "default" || target.size() >= selectedArch.size())))
            {
                invalid = true;
            }
        }
        // Nothing outside the UED role map can bind a model to this engine, so an
        // engine with no resolved `predict_engine_tflops` role simply has none.
        static const uhd::UhdConfig UNBOUND;
        const auto& uhdConfig = selected != nullptr ? *selected : UNBOUND;
        const bool evaluateEngine
            = evaluate && !invalid && kind == HIPDNN_ENGINE_PREDICTION_ENGINE;
        std::shared_ptr<const uhd::prediction_detail::Model> compiled;
        if(evaluateEngine && selected != nullptr)
        {
            compiled = compiledModel(selectedArch, *selected);
        }
        auto result = uhd::predictEngine(_id,
                                         _engine.name,
                                         _selectorRevision,
                                         arch,
                                         features,
                                         evaluateEngine,
                                         uhdConfig,
                                         compiled);
        if(!result.binding_json.empty())
        {
            auto binding = nlohmann::json::parse(result.binding_json);
            for(const auto& dependency : _provenance.items())
            {
                binding["trained_against"][dependency.key()] = dependency.value();
            }
            result.binding_json = binding.dump();
        }
        if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
        {
            result.kind = PredictionKind::CONFIGURATION;
            result.status = PredictionStatus::UNAVAILABLE;
            result.uhd_id.clear();
            result.reason = "Exact configuration prediction was not evaluated";
            if(!result.binding_json.empty())
            {
                auto binding = nlohmann::json::parse(result.binding_json);
                binding["role"] = "sort_kernel_catalog";
                binding.erase("uhd_id");
                result.binding_json = binding.dump();
            }
        }
        else if(invalid)
        {
            result.status = PredictionStatus::INVALID;
            result.reason = "The selected engine-prediction UHD is missing or incompatible";
        }
        return result;
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
    /// Compiling a UHD rebuilds its feature contract and reads its artifact off disk.
    /// The descriptor tree is immutable for a provider lifetime, so each architecture's
    /// model is compiled once and shared by every later query on this engine.
    std::shared_ptr<const uhd::prediction_detail::Model>
        compiledModel(const std::string& arch, const uhd::UhdConfig& config) const
    {
        const std::lock_guard<std::mutex> lock(_modelMutex);
        if(const auto cached = _modelCache.find(arch); cached != _modelCache.end())
        {
            return cached->second;
        }
        return _modelCache.emplace(arch, uhd::prediction_detail::model(config)).first->second;
    }

    EngineDescriptor _engine;
    std::unique_ptr<KernelIngestorStateManager<THandle>> _stateManager;
    int64_t _id;
    GenericPlanBuilder<THandle, TSettings, TContext> _planBuilder;
    std::map<std::string, uhd::UhdConfig> _predictions;
    std::set<std::string> _unavailablePredictionArches;
    std::string _selectorRevision;
    nlohmann::json _provenance;
    mutable std::mutex _modelMutex;
    mutable std::map<std::string, std::shared_ptr<const uhd::prediction_detail::Model>>
        _modelCache;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
