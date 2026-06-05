// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobWrapper.hpp>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class IEngineDetails
{
public:
    virtual ~IEngineDetails() = default;

    virtual const hipdnn_flatbuffers_sdk::data_objects::EngineDetails& getEngineDetails() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;

    virtual uint32_t knobCount() const = 0;
    virtual std::vector<int32_t> behaviorNotes() const = 0;
    virtual const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnob>>&
        knobWrappers() const
        = 0;
    virtual const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnob&
        getKnobByName(const std::string& knobName) const
        = 0;
};

class EngineDetailsWrapper : public IEngineDetails
{
public:
    explicit EngineDetailsWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::EngineDetails>())
            {
                _shallowEngineDetails
                    = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::EngineDetails>(
                        buffer);
            }
        }
    }

    explicit EngineDetailsWrapper(
        const hipdnn_flatbuffers_sdk::data_objects::EngineDetails* engineDetails)
        : _shallowEngineDetails(engineDetails)
    {
    }

    const hipdnn_flatbuffers_sdk::data_objects::EngineDetails& getEngineDetails() const override
    {
        throwIfNotValid();
        return *_shallowEngineDetails;
    }

    bool isValid() const override
    {
        return _shallowEngineDetails != nullptr;
    }

    int64_t engineId() const override
    {
        throwIfNotValid();

        return _shallowEngineDetails->engine_id();
    }

    uint32_t knobCount() const override
    {
        throwIfNotValid();

        auto knobs = _shallowEngineDetails->knobs();
        if(knobs == nullptr)
        {
            return 0;
        }
        return knobs->size();
    }

    std::vector<int32_t> behaviorNotes() const override
    {
        throwIfNotValid();

        auto rawNotes = _shallowEngineDetails->behavior_notes();
        if(rawNotes == nullptr)
        {
            return {};
        }

        std::vector<int32_t> notes;
        notes.reserve(rawNotes->size());
        for(uint32_t i = 0; i < rawNotes->size(); ++i)
        {
            notes.push_back(rawNotes->Get(i));
        }
        return notes;
    }

    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnob>>&
        knobWrappers() const override
    {
        throwIfNotValid();
        populateKnobWrappers();
        return _knobWrappers;
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnob&
        getKnobByName(const std::string& knobName) const override
    {
        throwIfNotValid();
        populateKnobWrappers();

        auto it = _knobNameToIndex.find(knobName);
        if(it == _knobNameToIndex.end())
        {
            throw std::out_of_range("Knob with name '" + knobName + "' not found");
        }

        return *_knobWrappers[it->second];
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("Engine details is not valid");
        }
    }

    void populateKnobWrappers() const
    {
        if(_knobsPopulated)
        {
            return;
        }

        auto knobs = _shallowEngineDetails->knobs();
        if(knobs != nullptr)
        {
            _knobWrappers.reserve(knobs->size());
            for(uint32_t i = 0; i < knobs->size(); ++i)
            {
                auto knob = knobs->Get(i);
                auto wrapper
                    = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::KnobWrapper>(
                        knob);
                auto knobName = wrapper->knobId();
                _knobNameToIndex[knobName] = i;
                _knobWrappers.push_back(std::move(wrapper));
            }
        }
        _knobsPopulated = true;
    }

    // Pointer to the flatbuffer representation of the engine details. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_flatbuffers_sdk::data_objects::EngineDetails* _shallowEngineDetails = nullptr;

    // Lazily populated cache of knob wrappers
    mutable std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnob>>
        _knobWrappers;
    mutable std::unordered_map<std::string, size_t> _knobNameToIndex;
    mutable bool _knobsPopulated = false;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
