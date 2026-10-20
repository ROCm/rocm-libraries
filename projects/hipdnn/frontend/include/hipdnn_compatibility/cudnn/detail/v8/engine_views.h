// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/knob_wrapper.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/detail/KnobUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

using KnobChoices = std::vector<std::pair<cudnnBackendKnobType_t, std::int64_t>>;

// A hipDNN knob the shim shows and accepts: it projects onto cuDNN
// min/max/stride, and no other knob of the same engine maps onto the same cuDNN
// type.
struct EngineKnob
{
    cudnnBackendKnobType_t type;
    cudnn_frontend::Knob range;
    ::hipdnn_frontend::Knob native;
};

inline std::string knobTypeLabel(cudnnBackendKnobType_t type)
{
    if(const std::optional<KnobType_t> mapped = backend::toFe(type))
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
        const std::optional<cudnnBackendKnobType_t> type = backend::toCudnn(ranges[i].type);
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

inline void warnUnhonoredHeuristicMode(cudnnBackendHeurMode_t mode)
{
    switch(mode)
    {
    case CUDNN_HEUR_MODE_INSTANT:
        detail::warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::INSTANT);
        break;
    case CUDNN_HEUR_MODE_A:
        detail::warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::A);
        break;
    case CUDNN_HEUR_MODE_B:
        detail::warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::B);
        break;
    case CUDNN_HEUR_MODE_FALLBACK:
    default:
        break;
    }
}

// One engine of a graph's index space, or an engine restored from a serialized
// plan (no graph; id, index, notes and name come from the plan JSON). Knobs and
// notes are read from hipDNN on first use, so building a record makes no
// hipDNN call.
class EngineRecord final : public RecordOf<CUDNN_BACKEND_ENGINE_DESCRIPTOR>
{
public:
    EngineRecord(std::shared_ptr<const GraphCore> graph, std::int64_t index, std::int64_t id)
        : _graph(std::move(graph))
        , _hipdnnEngine(_graph->defaultEngine(id))
        , _id(id)
        , _index(index)
    {
    }

    EngineRecord(std::int64_t id,
                 std::int64_t index,
                 std::vector<cudnnBackendBehaviorNote_t> behaviorNotes,
                 std::optional<std::string> name)
        : _id(id)
        , _index(index)
        , _behaviorNotes(std::move(behaviorNotes))
        , _name(std::move(name))
    {
    }

    std::int64_t id() const noexcept
    {
        return _id;
    }

    std::int64_t index() const noexcept
    {
        return _index;
    }

    bool isLoaded() const noexcept
    {
        return _graph == nullptr;
    }

    const GraphCore& graph() const
    {
        requireGraph();
        return *_graph;
    }

    const std::shared_ptr<const GraphCore>& graphCore() const
    {
        requireGraph();
        return _graph;
    }

    // The graph's default hipDNN ENGINE for this id (owned by the graph).
    hipdnnBackendDescriptor_t hipdnnEngine() const
    {
        requireGraph();
        return _hipdnnEngine;
    }

    // Empty for a restored engine.
    const std::vector<EngineKnob>& knobs() const
    {
        loadKnobs();
        return _knobs;
    }

    // The single visibility rule shared by knob info and knob choices.
    const EngineKnob* knob(cudnnBackendKnobType_t type) const
    {
        for(const EngineKnob& candidate : knobs())
        {
            if(candidate.type == type)
            {
                return &candidate;
            }
        }
        return nullptr;
    }

    // Only notes with a cuDNN counterpart.
    const std::vector<cudnnBackendBehaviorNote_t>& behaviorNotes() const
    {
        loadNotes();
        return _behaviorNotes;
    }

    const std::optional<std::string>& name() const
    {
        loadNotes();
        return _name;
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

private:
    void requireGraph() const
    {
        if(isLoaded())
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "an engine restored from a serialized execution plan has no "
                            "operation graph or knob information");
        }
    }

    void loadKnobs() const
    {
        if(isLoaded())
        {
            return;
        }
        std::call_once(_knobsOnce, [this] {
            std::vector<::hipdnn_frontend::Knob> nativeKnobs;
            checkFrontend(
                ::hipdnn_frontend::detail::unpackKnobsFromDescriptors(_hipdnnEngine, nativeKnobs),
                "reading the knobs of engine " + std::to_string(_id));
            _knobs = visibleEngineKnobs(nativeKnobs);
        });
    }

    void loadNotes() const
    {
        if(isLoaded())
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
                    = mapped ? backend::toCudnn(*mapped) : std::nullopt;
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

    std::shared_ptr<const GraphCore> _graph;
    hipdnnBackendDescriptor_t _hipdnnEngine = nullptr;
    std::int64_t _id;
    std::int64_t _index;

    mutable std::once_flag _knobsOnce;
    mutable std::vector<EngineKnob> _knobs;
    mutable std::once_flag _notesOnce;
    mutable std::vector<cudnnBackendBehaviorNote_t> _behaviorNotes;
    mutable std::optional<std::string> _name;
};

// The engine at dense `index` of `graph`; `indexLabel` names the index in the
// out-of-range message.
inline std::shared_ptr<const EngineRecord> makeEngineRecord(std::shared_ptr<const GraphCore> graph,
                                                            std::int64_t index,
                                                            std::string_view indexLabel)
{
    const std::optional<std::int64_t> id = graph->engineIdAt(index);
    if(!id)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(indexLabel) + " = " + std::to_string(index) + " is outside [0, "
                            + std::to_string(graph->engineCount()) + ")");
    }
    return std::make_shared<const EngineRecord>(std::move(graph), index, *id);
}

// Every engine of `graph` in heuristic rank order. Makes no hipDNN call once
// the graph has enumerated its engines.
inline std::vector<std::shared_ptr<const EngineRecord>>
    rankedEngineRecords(const std::shared_ptr<const GraphCore>& graph)
{
    const std::vector<std::int64_t>& rankedIds = graph->rankedEngineIds();
    std::vector<std::shared_ptr<const EngineRecord>> engines;
    engines.reserve(rankedIds.size());
    for(const std::int64_t id : rankedIds)
    {
        const std::optional<std::int64_t> index = graph->engineIndexOf(id);
        if(!index)
        {
            throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                            "ranked engine " + std::to_string(id) + " has no index");
        }
        engines.push_back(std::make_shared<const EngineRecord>(graph, *index, id));
    }
    return engines;
}

// A hipDNN ENGINECFG for `engine` carrying `settings`, finalized.
inline ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
    buildKnobConfig(hipdnnBackendDescriptor_t engine,
                    std::int64_t engineId,
                    const std::vector<::hipdnn_frontend::KnobSetting>& settings)
{
    using ::hipdnn_frontend::detail::hipdnnBackend;
    const std::string context = "engine " + std::to_string(engineId);
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor config = createHipdnnDescriptor(
        HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, "creating the hipDNN engine config of " + context);
    checkHipdnn(hipdnnBackend()->backendSetAttribute(config.get(),
                                                     HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     static_cast<const void*>(&engine)),
                "setting the hipDNN engine of " + context);
    checkFrontend(
        ::hipdnn_frontend::detail::applyKnobSettingsViaDescriptors(config.get(), settings),
        "applying knob choices to " + context);
    checkHipdnn(hipdnnBackend()->backendFinalize(config.get()),
                "finalizing the hipDNN engine config of " + context);
    return config;
}

// An engine plus validated knob choices, or the config of a restored plan
// (whose engine is restored too; nothing reaches hipDNN).
class ConfigRecord final : public RecordOf<CUDNN_BACKEND_ENGINECFG_DESCRIPTOR>
{
public:
    ConfigRecord(std::shared_ptr<const EngineRecord> engine,
                 KnobChoices choices,
                 std::vector<::hipdnn_frontend::KnobSetting> settings)
        : _engine(std::move(engine))
        , _choices(std::move(choices))
        , _settings(std::move(settings))
    {
    }

    const EngineRecord& engine() const noexcept
    {
        return *_engine;
    }

    const std::shared_ptr<const EngineRecord>& engineRecord() const noexcept
    {
        return _engine;
    }

    std::int64_t engineId() const noexcept
    {
        return _engine->id();
    }

    std::int64_t engineIndex() const noexcept
    {
        return _engine->index();
    }

    bool isLoaded() const noexcept
    {
        return _engine->isLoaded();
    }

    // In the order the caller chose them.
    const KnobChoices& knobChoices() const noexcept
    {
        return _choices;
    }

    // The finalized hipDNN ENGINECFG. Without knob settings this is the graph's
    // shared default config; with them it is built on first use, because the
    // backend accepts knobs only before its config is finalized.
    hipdnnBackendDescriptor_t hipdnnEngineConfig() const
    {
        if(isLoaded())
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "the engine config of a deserialized execution plan has no hipDNN "
                            "engine config");
        }
        if(_settings.empty())
        {
            return _engine->graph().defaultEngineConfig(engineId());
        }
        std::call_once(_configOnce, [this] {
            _knobConfig = buildKnobConfig(_engine->hipdnnEngine(), engineId(), _settings);
        });
        return _knobConfig.get();
    }

private:
    // Declared first so the hipDNN config below is released before the engine
    // (and through it the graph) whose ENGINE it references.
    std::shared_ptr<const EngineRecord> _engine;
    KnobChoices _choices;
    std::vector<::hipdnn_frontend::KnobSetting> _settings;
    mutable std::once_flag _configOnce;
    mutable ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _knobConfig;
};

// Validates `requested` against the engine's visible knobs, in order. A -1
// means "unset" unless the knob accepts -1: once any knob is set, upstream's
// builder sends -1 for every other knob of the engine. `choicesLabel` prefixes
// every knob message.
inline std::shared_ptr<const ConfigRecord>
    makeConfigRecord(std::shared_ptr<const EngineRecord> engine,
                     const KnobChoices& requested,
                     std::string_view choicesLabel)
{
    if(engine->isLoaded())
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        "an engine restored from a serialized execution plan cannot be "
                        "configured");
    }

    constexpr std::int64_t UNSET_KNOB_VALUE = -1;
    const std::string prefix = std::string(choicesLabel) + ": ";
    KnobChoices choices;
    std::vector<::hipdnn_frontend::KnobSetting> settings;
    for(const auto& [type, value] : requested)
    {
        const EngineKnob* knob = engine->knob(type);
        if(value == UNSET_KNOB_VALUE
           && (knob == nullptr
               || knob->native
                      .validate(
                          ::hipdnn_frontend::KnobSetting(knob->native.knobId(), UNSET_KNOB_VALUE))
                      .is_bad()))
        {
            continue;
        }
        const std::string label = knobTypeLabel(type);
        if(knob == nullptr)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            prefix + label + " is not exposed by engine "
                                + std::to_string(engine->id()));
        }
        for(const auto& previous : choices)
        {
            if(previous.first == type)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                prefix + label + " is chosen more than once");
            }
        }
        const auto converted = appendNativeKnobSetting(knob->native, value, settings);
        if(converted.is_bad())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, prefix + converted.get_message());
        }
        const auto valid = knob->native.validate(settings.back());
        if(valid.is_bad())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            prefix + label + " = " + std::to_string(value) + ": "
                                + valid.get_message());
        }
        choices.emplace_back(type, value);
    }
    return std::make_shared<const ConfigRecord>(
        std::move(engine), std::move(choices), std::move(settings));
}

inline std::shared_ptr<const ConfigRecord>
    makeLoadedConfigRecord(std::int64_t id,
                           std::int64_t index,
                           std::vector<cudnnBackendBehaviorNote_t> behaviorNotes,
                           KnobChoices knobChoices,
                           std::optional<std::string> name)
{
    return std::make_shared<const ConfigRecord>(
        std::make_shared<const EngineRecord>(id, index, std::move(behaviorNotes), std::move(name)),
        std::move(knobChoices),
        std::vector<::hipdnn_frontend::KnobSetting>{});
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
