// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_graph_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// A GET-only enum array value served from shim state.
template <typename Enum>
StoredAttribute enumArrayValue(cudnnBackendAttributeType_t type, const std::vector<Enum>& values)
{
    static_assert(sizeof(Enum) == sizeof(std::int32_t));
    StoredAttribute value;
    value.type = type;
    value.count = static_cast<std::int64_t>(values.size());
    value.bytes.resize(values.size() * sizeof(Enum));
    if(!values.empty())
    {
        std::memcpy(value.bytes.data(), values.data(), value.bytes.size());
    }
    return value;
}

class KnobInfoDescriptor final : public Descriptor
{
public:
    KnobInfoDescriptor()
        : Descriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR)
    {
    }

    static std::shared_ptr<KnobInfoDescriptor> describe(const v8::EngineKnob& knob)
    {
        auto info = std::make_shared<KnobInfoDescriptor>();
        info->storeComputed(CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &knob.type);
        info->storeComputed(
            CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, CUDNN_TYPE_INT64, 1, &knob.range.minValue);
        info->storeComputed(
            CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, CUDNN_TYPE_INT64, 1, &knob.range.maxValue);
        info->storeComputed(CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &knob.range.stride);
        info->_described = true;
        info->finalize();
        return info;
    }

protected:
    void finalizeImpl() override
    {
        if(!_described)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR is filled by "
                            "CUDNN_ATTR_ENGINE_KNOB_INFO and cannot be finalized by the caller");
        }
    }

private:
    bool _described = false;
};

class KnobChoiceDescriptor final : public Descriptor
{
public:
    KnobChoiceDescriptor()
        : Descriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR)
    {
    }

    static std::shared_ptr<KnobChoiceDescriptor> choose(cudnnBackendKnobType_t type,
                                                        std::int64_t value)
    {
        auto choice = std::make_shared<KnobChoiceDescriptor>();
        choice->storeComputed(CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &type);
        choice->storeComputed(CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &value);
        choice->finalize();
        return choice;
    }

    // Valid once finalized.
    cudnnBackendKnobType_t knobType() const noexcept
    {
        return _knobType;
    }

    std::int64_t value() const noexcept
    {
        return _value;
    }

protected:
    void finalizeImpl() override
    {
        _knobType = requireAttribute(
            scalarAttribute<cudnnBackendKnobType_t>(CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE),
            CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
        _value = requireAttribute(scalarAttribute<std::int64_t>(CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE),
                                  CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
    }

private:
    cudnnBackendKnobType_t _knobType = CUDNN_KNOB_TYPE_COUNTS;
    std::int64_t _value = 0;
};

class EngineDescriptor final : public Descriptor
{
public:
    EngineDescriptor()
        : Descriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR)
    {
    }

    // An engine of a finalized graph, as ENGINEHEUR RESULTS yields it.
    static std::shared_ptr<EngineDescriptor>
        forRecord(std::shared_ptr<Descriptor> graph, std::shared_ptr<const v8::EngineRecord> record)
    {
        auto engine = std::make_shared<EngineDescriptor>();
        const std::int64_t index = record->index();
        engine->storeComputedDescriptors(CUDNN_ATTR_ENGINE_OPERATION_GRAPH, {graph});
        engine->storeComputed(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
        engine->_graphView = &descriptorCast<OperationGraphDescriptor>(
            *graph, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        engine->_graph = std::move(graph);
        engine->_record = std::move(record);
        engine->finalize();
        return engine;
    }

    // An engine reconstructed from a serialized plan; it has no graph.
    static std::shared_ptr<EngineDescriptor> loaded(std::shared_ptr<const v8::EngineRecord> record)
    {
        auto engine = std::make_shared<EngineDescriptor>();
        const std::int64_t index = record->index();
        engine->storeComputed(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
        engine->_record = std::move(record);
        engine->finalize();
        return engine;
    }

    // Valid once finalized.
    const std::shared_ptr<const v8::EngineRecord>& record() const noexcept
    {
        return _record;
    }

    std::int64_t engineId() const noexcept
    {
        return _record->id();
    }

    std::int64_t engineIndex() const noexcept
    {
        return _record->index();
    }

    bool isLoadedPlan() const noexcept
    {
        return _record->isLoaded();
    }

    const OperationGraphDescriptor& graph() const
    {
        requireGraph();
        return *_graphView;
    }

    std::shared_ptr<Descriptor> graphRef() const
    {
        requireGraph();
        return _graph;
    }

    hipdnnBackendDescriptor_t hipdnnEngine() const
    {
        return _record->hipdnnEngine();
    }

    std::vector<cudnnBackendBehaviorNote_t> behaviorNotes() const
    {
        return _record->behaviorNotes();
    }

    std::vector<cudnnBackendNumericalNote_t> numericalNotes() const
    {
        return _record->numericalNotes();
    }

    std::optional<std::string> engineName() const
    {
        return _record->name();
    }

    const std::vector<v8::EngineKnob>& visibleKnobs() const
    {
        return _record->knobs();
    }

    const v8::EngineKnob* visibleKnob(cudnnBackendKnobType_t type) const
    {
        return _record->knob(type);
    }

protected:
    void finalizeImpl() override
    {
        if(_record)
        {
            return;
        }
        std::shared_ptr<Descriptor> graphRef
            = descriptorAttribute(CUDNN_ATTR_ENGINE_OPERATION_GRAPH);
        if(!graphRef)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_ENGINE_OPERATION_GRAPH is required");
        }
        const std::int64_t index
            = requireAttribute(scalarAttribute<std::int64_t>(CUDNN_ATTR_ENGINE_GLOBAL_INDEX),
                               CUDNN_ATTR_ENGINE_GLOBAL_INDEX);
        const auto& graph = descriptorCast<OperationGraphDescriptor>(
            *graphRef, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        _record
            = v8::makeEngineRecord(graph.requireCore(), index, "CUDNN_ATTR_ENGINE_GLOBAL_INDEX");
        _graphView = &graph;
        _graph = std::move(graphRef);
    }

    std::optional<StoredAttribute> lazyAttribute(cudnnBackendAttributeName_t name,
                                                 bool /*countOnly*/) const override
    {
        switch(name)
        {
        case CUDNN_ATTR_ENGINE_OPERATION_GRAPH:
            requireGraph();
            return std::nullopt;
        case CUDNN_ATTR_ENGINE_KNOB_INFO:
        {
            requireGraph();
            std::call_once(_knobInfoOnce, [this] {
                const std::vector<v8::EngineKnob>& knobs = _record->knobs();
                std::vector<std::shared_ptr<Descriptor>> knobInfo;
                knobInfo.reserve(knobs.size());
                for(const v8::EngineKnob& knob : knobs)
                {
                    knobInfo.push_back(KnobInfoDescriptor::describe(knob));
                }
                _knobInfo = std::move(knobInfo);
            });
            StoredAttribute value;
            value.type = CUDNN_TYPE_BACKEND_DESCRIPTOR;
            value.count = static_cast<std::int64_t>(_knobInfo.size());
            value.descriptors = _knobInfo;
            return value;
        }
        case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
            return enumArrayValue(CUDNN_TYPE_BEHAVIOR_NOTE, _record->behaviorNotes());
        case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
            return enumArrayValue(CUDNN_TYPE_NUMERICAL_NOTE, _record->numericalNotes());
        default:
            return std::nullopt;
        }
    }

private:
    void requireGraph() const
    {
        static_cast<void>(_record->graph());
    }

    std::shared_ptr<Descriptor> _graph;
    const OperationGraphDescriptor* _graphView = nullptr;
    std::shared_ptr<const v8::EngineRecord> _record;

    mutable std::once_flag _knobInfoOnce;
    mutable std::vector<std::shared_ptr<Descriptor>> _knobInfo;
};

class EngineConfigDescriptor final : public Descriptor
{
public:
    EngineConfigDescriptor()
        : Descriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)
    {
    }

    // A knob-free config of `engine`, as ENGINEHEUR RESULTS yields it.
    static std::shared_ptr<EngineConfigDescriptor>
        forEngine(std::shared_ptr<EngineDescriptor> engine)
    {
        auto config = std::make_shared<EngineConfigDescriptor>();
        config->storeComputedDescriptors(CUDNN_ATTR_ENGINECFG_ENGINE, {std::move(engine)});
        config->finalize();
        return config;
    }

    // The config of a deserialized plan: ENGINE and KNOB_CHOICES are gettable,
    // nothing reaches hipDNN.
    static std::shared_ptr<EngineConfigDescriptor>
        forLoadedPlan(std::shared_ptr<const v8::ConfigRecord> record)
    {
        auto config = std::make_shared<EngineConfigDescriptor>();
        std::shared_ptr<EngineDescriptor> engine = EngineDescriptor::loaded(record->engineRecord());
        std::vector<std::shared_ptr<Descriptor>> choices;
        choices.reserve(record->knobChoices().size());
        for(const auto& [type, value] : record->knobChoices())
        {
            choices.push_back(KnobChoiceDescriptor::choose(type, value));
        }
        config->storeComputedDescriptors(CUDNN_ATTR_ENGINECFG_ENGINE, {engine});
        config->storeComputedDescriptors(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, std::move(choices));
        config->_engine = std::move(engine);
        config->_record = std::move(record);
        config->finalize();
        return config;
    }

    static std::shared_ptr<EngineConfigDescriptor>
        forLoadedPlan(std::int64_t id,
                      std::int64_t index,
                      std::vector<cudnnBackendBehaviorNote_t> behaviorNotes,
                      v8::KnobChoices knobChoices,
                      std::optional<std::string> name)
    {
        return forLoadedPlan(v8::makeLoadedConfigRecord(
            id, index, std::move(behaviorNotes), std::move(knobChoices), std::move(name)));
    }

    // Valid once finalized.
    const std::shared_ptr<const v8::ConfigRecord>& record() const noexcept
    {
        return _record;
    }

    const EngineDescriptor& engine() const noexcept
    {
        return *_engine;
    }

    std::int64_t engineId() const noexcept
    {
        return _record->engineId();
    }

    std::int64_t engineIndex() const noexcept
    {
        return _record->engineIndex();
    }

    bool isLoadedPlan() const noexcept
    {
        return _record->isLoaded();
    }

    const OperationGraphDescriptor& graph() const
    {
        return _engine->graph();
    }

    std::shared_ptr<Descriptor> graphRef() const
    {
        return _engine->graphRef();
    }

    const v8::KnobChoices& knobChoices() const noexcept
    {
        return _record->knobChoices();
    }

    hipdnnBackendDescriptor_t hipdnnEngineConfig() const
    {
        return _record->hipdnnEngineConfig();
    }

protected:
    void finalizeImpl() override
    {
        if(_record)
        {
            return;
        }
        const std::shared_ptr<Descriptor> engineRef
            = descriptorAttribute(CUDNN_ATTR_ENGINECFG_ENGINE);
        if(!engineRef)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_ENGINECFG_ENGINE is required");
        }
        descriptorCast<EngineDescriptor>(*engineRef, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        auto engine = std::static_pointer_cast<EngineDescriptor>(engineRef);

        v8::KnobChoices requested;
        for(const std::shared_ptr<Descriptor>& descriptor :
            descriptorArrayAttribute(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES))
        {
            const auto& choice = descriptorCast<KnobChoiceDescriptor>(
                *descriptor, CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            requested.emplace_back(choice.knobType(), choice.value());
        }
        _record = v8::makeConfigRecord(
            engine->record(), requested, "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES");
        _engine = std::move(engine);
    }

private:
    std::shared_ptr<EngineDescriptor> _engine;
    std::shared_ptr<const v8::ConfigRecord> _record;
};

class EngineHeuristicDescriptor final : public Descriptor
{
public:
    EngineHeuristicDescriptor()
        : Descriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)
    {
    }

protected:
    void validateSet(cudnnBackendAttributeName_t name, const StoredAttribute& /*value*/) override
    {
        if(name == CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET
           || name == CUDNN_ATTR_ENGINEHEUR_DEVICEPROP)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            attributeName(name) + " is not supported by the hipDNN cuDNN shim");
        }
    }

    // Every mode runs hipDNN's fallback selection; the results are the graph's
    // rank snapshot, so finalize makes no hipDNN call once the graph has
    // enumerated its engines.
    void finalizeImpl() override
    {
        const std::shared_ptr<Descriptor> graphRef
            = descriptorAttribute(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH);
        if(!graphRef)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH is required");
        }
        const cudnnBackendHeurMode_t mode
            = requireAttribute(scalarAttribute<cudnnBackendHeurMode_t>(CUDNN_ATTR_ENGINEHEUR_MODE),
                               CUDNN_ATTR_ENGINEHEUR_MODE);
        v8::warnUnhonoredHeuristicMode(mode);

        const auto& graph = descriptorCast<OperationGraphDescriptor>(
            *graphRef, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        std::vector<std::shared_ptr<const v8::EngineRecord>> engines
            = v8::rankedEngineRecords(graph.requireCore());
        std::vector<std::shared_ptr<Descriptor>> results;
        results.reserve(engines.size());
        for(std::shared_ptr<const v8::EngineRecord>& engine : engines)
        {
            results.push_back(EngineConfigDescriptor::forEngine(
                EngineDescriptor::forRecord(graphRef, std::move(engine))));
        }
        storeComputedDescriptors(CUDNN_ATTR_ENGINEHEUR_RESULTS, std::move(results));
    }
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
