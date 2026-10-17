// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
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
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/knob_wrapper.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/detail/KnobUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

using KnobChoices = std::vector<std::pair<cudnnBackendKnobType_t, std::int64_t>>;

// A hipDNN knob the shim shows through CUDNN_ATTR_ENGINE_KNOB_INFO and accepts
// in a KNOB_CHOICE: it projects onto cuDNN min/max/stride, and no other knob of
// the same engine maps onto the same cuDNN type.
struct EngineKnob
{
    cudnnBackendKnobType_t type;
    cudnn_frontend::Knob range;
    ::hipdnn_frontend::Knob native;
};

inline std::string knobTypeLabel(cudnnBackendKnobType_t type)
{
    if(const std::optional<KnobType_t> mapped = toFe(type))
    {
        return std::string("CUDNN_KNOB_TYPE_") + knobTypeName(*mapped);
    }
    return "cuDNN knob type " + std::to_string(static_cast<int>(type));
}

inline std::vector<EngineKnob>
    visibleEngineKnobs(const std::vector<::hipdnn_frontend::Knob>& nativeKnobs)
{
    std::vector<::hipdnn_frontend::Knob> projectable;
    std::vector<cudnn_frontend::Knob> ranges;
    for(const ::hipdnn_frontend::Knob& nativeKnob : nativeKnobs)
    {
        if(const std::optional<cudnn_frontend::Knob> range = projectNativeKnob(nativeKnob))
        {
            projectable.push_back(nativeKnob);
            ranges.push_back(*range);
        }
    }
    // A cuDNN type claimed by two projectable knobs maps to a null entry.
    const auto lookup = buildKnobTypeLookup(projectable);
    std::vector<EngineKnob> visible;
    visible.reserve(projectable.size());
    for(std::size_t i = 0; i < projectable.size(); ++i)
    {
        const std::optional<cudnnBackendKnobType_t> type = toCudnn(ranges[i].type);
        if(!type || lookup.at(ranges[i].type) == nullptr)
        {
            HIPDNN_FE_LOG_WARN("[cudnn_frontend] Omitting hipDNN knob '"
                               << projectable[i].knobId()
                               << "'; another knob of this engine maps onto the same cuDNN knob "
                                  "type.");
            continue;
        }
        visible.push_back(EngineKnob{*type, ranges[i], projectable[i]});
    }
    return visible;
}

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

inline void warnUnhonoredHeuristicMode(cudnnBackendHeurMode_t mode)
{
    switch(mode)
    {
    case CUDNN_HEUR_MODE_INSTANT:
        warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::INSTANT);
        break;
    case CUDNN_HEUR_MODE_A:
        warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::A);
        break;
    case CUDNN_HEUR_MODE_B:
        warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::B);
        break;
    case CUDNN_HEUR_MODE_FALLBACK:
    default:
        break;
    }
}

class KnobInfoDescriptor final : public Descriptor
{
public:
    KnobInfoDescriptor()
        : Descriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR)
    {
    }

    static std::shared_ptr<KnobInfoDescriptor> describe(const EngineKnob& knob)
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

    // The engine at `index` of a finalized graph, as ENGINEHEUR RESULTS yields it.
    static std::shared_ptr<EngineDescriptor> atIndex(std::shared_ptr<Descriptor> graph,
                                                     std::int64_t index)
    {
        auto engine = std::make_shared<EngineDescriptor>();
        engine->storeComputedDescriptors(CUDNN_ATTR_ENGINE_OPERATION_GRAPH, {std::move(graph)});
        engine->storeComputed(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
        engine->finalize();
        return engine;
    }

    // An engine reconstructed from a serialized plan; it has no graph.
    static std::shared_ptr<EngineDescriptor>
        loaded(std::int64_t id,
               std::int64_t index,
               std::vector<cudnnBackendBehaviorNote_t> behaviorNotes,
               std::optional<std::string> name)
    {
        auto engine = std::make_shared<EngineDescriptor>();
        engine->storeComputed(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
        engine->_loaded = true;
        engine->_id = id;
        engine->_index = index;
        engine->_behaviorNotes = std::move(behaviorNotes);
        engine->_name = std::move(name);
        engine->finalize();
        return engine;
    }

    // Valid once finalized.
    std::int64_t engineId() const noexcept
    {
        return _id;
    }

    std::int64_t engineIndex() const noexcept
    {
        return _index;
    }

    bool isLoadedPlan() const noexcept
    {
        return _loaded;
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

    // The graph's cached hipDNN ENGINE (owned by the graph).
    hipdnnBackendDescriptor_t hipdnnEngine() const
    {
        requireGraph();
        return _hipdnnEngine;
    }

    // Reported only where the hipDNN note has a cuDNN counterpart.
    std::vector<cudnnBackendBehaviorNote_t> behaviorNotes() const
    {
        loadNotes();
        return _behaviorNotes;
    }

    // hipDNN engines declare no numerical notes; only a determinism claim
    // lifts NONDETERMINISTIC.
    std::vector<cudnnBackendNumericalNote_t> numericalNotes() const
    {
        if(isDeterminismClaimed(_id))
        {
            return {};
        }
        return {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC};
    }

    std::optional<std::string> engineName() const
    {
        loadNotes();
        return _name;
    }

    const std::vector<EngineKnob>& visibleKnobs() const
    {
        loadKnobs();
        return _knobs;
    }

    // The single visibility rule shared by KNOB_INFO and KNOB_CHOICE.
    const EngineKnob* visibleKnob(cudnnBackendKnobType_t type) const
    {
        for(const EngineKnob& knob : visibleKnobs())
        {
            if(knob.type == type)
            {
                return &knob;
            }
        }
        return nullptr;
    }

protected:
    void finalizeImpl() override
    {
        if(_loaded)
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
        const std::optional<std::int64_t> id = graph.engineIdAt(index);
        if(!id)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_ENGINE_GLOBAL_INDEX = " + std::to_string(index)
                                + " is outside [0, " + std::to_string(graph.engineCount()) + ")");
        }
        _hipdnnEngine = graph.defaultEngine(*id);
        _graphView = &graph;
        _graph = std::move(graphRef);
        _id = *id;
        _index = index;
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
            loadKnobs();
            StoredAttribute value;
            value.type = CUDNN_TYPE_BACKEND_DESCRIPTOR;
            value.count = static_cast<std::int64_t>(_knobInfo.size());
            value.descriptors = _knobInfo;
            return value;
        }
        case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
            return enumArrayValue(CUDNN_TYPE_BEHAVIOR_NOTE, behaviorNotes());
        case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
            return enumArrayValue(CUDNN_TYPE_NUMERICAL_NOTE, numericalNotes());
        default:
            return std::nullopt;
        }
    }

private:
    void requireGraph() const
    {
        if(_loaded)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "an engine restored from a serialized execution plan has no "
                            "operation graph or knob information");
        }
    }

    void loadKnobs() const
    {
        if(_loaded)
        {
            return;
        }
        std::call_once(_knobsOnce, [this] {
            std::vector<::hipdnn_frontend::Knob> nativeKnobs;
            checkFrontend(
                ::hipdnn_frontend::detail::unpackKnobsFromDescriptors(_hipdnnEngine, nativeKnobs),
                "reading the knobs of engine " + std::to_string(_id));
            std::vector<EngineKnob> knobs = visibleEngineKnobs(nativeKnobs);
            std::vector<std::shared_ptr<Descriptor>> knobInfo;
            knobInfo.reserve(knobs.size());
            for(const EngineKnob& knob : knobs)
            {
                knobInfo.push_back(KnobInfoDescriptor::describe(knob));
            }
            _knobs = std::move(knobs);
            _knobInfo = std::move(knobInfo);
        });
    }

    void loadNotes() const
    {
        if(_loaded)
        {
            return;
        }
        std::call_once(_notesOnce, [this] {
            std::vector<hipdnnBackendBehaviorNote_t> hipdnnNotes;
            checkFrontend(
                ::hipdnn_frontend::detail::getDescriptorAttrVec(_hipdnnEngine,
                                                                HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                                                HIPDNN_TYPE_BEHAVIOR_NOTE,
                                                                hipdnnNotes,
                                                                "engine behavior notes"),
                "reading the behavior notes of engine " + std::to_string(_id));
            std::vector<cudnnBackendBehaviorNote_t> notes;
            for(const hipdnnBackendBehaviorNote_t note : hipdnnNotes)
            {
                const std::optional<BehaviorNote_t> mapped
                    = ::hipdnn_frontend::fromHipdnnBehaviorNote(note);
                const std::optional<cudnnBackendBehaviorNote_t> cudnnNote
                    = mapped ? toCudnn(*mapped) : std::nullopt;
                if(cudnnNote)
                {
                    notes.push_back(*cudnnNote);
                }
            }
            // The name is informational, so a backend without it is not an error.
            std::string name;
            std::optional<std::string> engineName;
            if(::hipdnn_frontend::detail::getDescriptorAttrString(
                   _hipdnnEngine, HIPDNN_ATTR_ENGINE_NAME_EXT, name, "engine name")
                   .is_good()
               && !name.empty())
            {
                engineName = std::move(name);
            }
            _behaviorNotes = std::move(notes);
            _name = std::move(engineName);
        });
    }

    std::shared_ptr<Descriptor> _graph;
    const OperationGraphDescriptor* _graphView = nullptr;
    hipdnnBackendDescriptor_t _hipdnnEngine = nullptr;
    std::int64_t _id = 0;
    std::int64_t _index = 0;
    bool _loaded = false;

    mutable std::once_flag _knobsOnce;
    mutable std::vector<EngineKnob> _knobs;
    mutable std::vector<std::shared_ptr<Descriptor>> _knobInfo;
    mutable std::once_flag _notesOnce;
    mutable std::vector<cudnnBackendBehaviorNote_t> _behaviorNotes;
    mutable std::optional<std::string> _name;
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
        forLoadedPlan(std::int64_t id,
                      std::int64_t index,
                      std::vector<cudnnBackendBehaviorNote_t> behaviorNotes,
                      KnobChoices knobChoices,
                      std::optional<std::string> name)
    {
        auto config = std::make_shared<EngineConfigDescriptor>();
        std::shared_ptr<EngineDescriptor> engine
            = EngineDescriptor::loaded(id, index, std::move(behaviorNotes), std::move(name));
        std::vector<std::shared_ptr<Descriptor>> choices;
        choices.reserve(knobChoices.size());
        for(const auto& [type, value] : knobChoices)
        {
            choices.push_back(KnobChoiceDescriptor::choose(type, value));
        }
        config->storeComputedDescriptors(CUDNN_ATTR_ENGINECFG_ENGINE, {engine});
        config->storeComputedDescriptors(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, std::move(choices));
        config->_engine = std::move(engine);
        config->_choices = std::move(knobChoices);
        config->_loaded = true;
        config->finalize();
        return config;
    }

    // Valid once finalized.
    const EngineDescriptor& engine() const noexcept
    {
        return *_engine;
    }

    std::int64_t engineId() const noexcept
    {
        return _engine->engineId();
    }

    std::int64_t engineIndex() const noexcept
    {
        return _engine->engineIndex();
    }

    bool isLoadedPlan() const noexcept
    {
        return _loaded;
    }

    const OperationGraphDescriptor& graph() const
    {
        return _engine->graph();
    }

    std::shared_ptr<Descriptor> graphRef() const
    {
        return _engine->graphRef();
    }

    // In the order the caller set them.
    const KnobChoices& knobChoices() const noexcept
    {
        return _choices;
    }

    // The finalized hipDNN ENGINECFG. Without knob choices this is the graph's
    // shared default config; with them it is built on first use, because the
    // backend accepts knobs only before its config is finalized.
    hipdnnBackendDescriptor_t hipdnnEngineConfig() const
    {
        if(_loaded)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "the engine config of a deserialized execution plan has no hipDNN "
                            "engine config");
        }
        if(_settings.empty())
        {
            return graph().defaultEngineConfig(engineId());
        }
        std::call_once(_configOnce, [this] { buildKnobConfig(); });
        return _knobConfig.get();
    }

protected:
    void finalizeImpl() override
    {
        if(_loaded)
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
        if(engine->isLoadedPlan())
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "an engine restored from a serialized execution plan cannot be "
                            "configured");
        }

        constexpr std::int64_t UNSET_KNOB_VALUE = -1;
        KnobChoices choices;
        std::vector<::hipdnn_frontend::KnobSetting> settings;
        for(const std::shared_ptr<Descriptor>& descriptor :
            descriptorArrayAttribute(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES))
        {
            const auto& choice = descriptorCast<KnobChoiceDescriptor>(
                *descriptor, CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            const cudnnBackendKnobType_t type = choice.knobType();
            const EngineKnob* knob = engine->visibleKnob(type);
            // Once any knob is set, upstream's builder sends -1 for every other
            // knob of the engine; that means "unset" unless the knob accepts -1.
            if(choice.value() == UNSET_KNOB_VALUE
               && (knob == nullptr
                   || knob->native
                          .validate(::hipdnn_frontend::KnobSetting(knob->native.knobId(),
                                                                   UNSET_KNOB_VALUE))
                          .is_bad()))
            {
                continue;
            }
            const std::string label = knobTypeLabel(type);
            if(knob == nullptr)
            {
                throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                                "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: " + label
                                    + " is not exposed by engine "
                                    + std::to_string(engine->engineId()));
            }
            for(const auto& previous : choices)
            {
                if(previous.first == type)
                {
                    throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                    "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: " + label
                                        + " is chosen more than once");
                }
            }
            const auto converted = appendNativeKnobSetting(knob->native, choice.value(), settings);
            if(converted.is_bad())
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: " + converted.get_message());
            }
            const auto valid = knob->native.validate(settings.back());
            if(valid.is_bad())
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES: " + label + " = "
                                    + std::to_string(choice.value()) + ": " + valid.get_message());
            }
            choices.emplace_back(type, choice.value());
        }
        _choices = std::move(choices);
        _settings = std::move(settings);
        _engine = std::move(engine);
    }

private:
    void buildKnobConfig() const
    {
        using hipdnn_frontend::detail::hipdnnBackend;
        const std::string context = "engine " + std::to_string(engineId());
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor config = createHipdnnDescriptor(
            HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, "creating the hipDNN engine config of " + context);
        hipdnnBackendDescriptor_t engine = _engine->hipdnnEngine();
        checkHipdnn(hipdnnBackend()->backendSetAttribute(config.get(),
                                                         HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         static_cast<const void*>(&engine)),
                    "setting the hipDNN engine of " + context);
        checkFrontend(
            ::hipdnn_frontend::detail::applyKnobSettingsViaDescriptors(config.get(), _settings),
            "applying knob choices to " + context);
        checkHipdnn(hipdnnBackend()->backendFinalize(config.get()),
                    "finalizing the hipDNN engine config of " + context);
        _knobConfig = std::move(config);
    }

    // Declared first so the hipDNN config below is released before the engine
    // (and through it the graph) whose ENGINE it references.
    std::shared_ptr<EngineDescriptor> _engine;
    KnobChoices _choices;
    std::vector<::hipdnn_frontend::KnobSetting> _settings;
    bool _loaded = false;
    mutable std::once_flag _configOnce;
    mutable hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _knobConfig;
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
        warnUnhonoredHeuristicMode(mode);

        const auto& graph = descriptorCast<OperationGraphDescriptor>(
            *graphRef, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        const std::vector<std::int64_t>& rankedIds = graph.rankedEngineIds();
        std::vector<std::shared_ptr<Descriptor>> results;
        results.reserve(rankedIds.size());
        for(const std::int64_t id : rankedIds)
        {
            const std::optional<std::int64_t> index = graph.engineIndexOf(id);
            if(!index)
            {
                throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                                "ranked engine " + std::to_string(id) + " has no index");
            }
            results.push_back(
                EngineConfigDescriptor::forEngine(EngineDescriptor::atIndex(graphRef, *index)));
        }
        storeComputedDescriptors(CUDNN_ATTR_ENGINEHEUR_RESULTS, std::move(results));
    }
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
