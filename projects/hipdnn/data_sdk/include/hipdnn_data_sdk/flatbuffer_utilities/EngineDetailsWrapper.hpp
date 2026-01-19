// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <unordered_map>

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_plugin_sdk
{

class IEngineDetails
{
public:
    virtual ~IEngineDetails() = default;

    virtual const hipdnn_data_sdk::data_objects::EngineDetails& getEngineDetails() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;

    virtual uint32_t knobCount() const = 0;
    virtual const std::vector<std::unique_ptr<hipdnn_data_sdk::flatbuffer_utilities::IKnob>>&
        knobWrappers() const
        = 0;
    virtual const hipdnn_data_sdk::flatbuffer_utilities::IKnob& getKnobById(int64_t knobId) const
        = 0;
    virtual const hipdnn_data_sdk::flatbuffer_utilities::IKnob&
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
            if(verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::EngineDetails>())
            {
                _shallowEngineDetails
                    = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::EngineDetails>(buffer);
            }
        }
    }

    const hipdnn_data_sdk::data_objects::EngineDetails& getEngineDetails() const override
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

    const std::vector<std::unique_ptr<hipdnn_data_sdk::flatbuffer_utilities::IKnob>>&
        knobWrappers() const override
    {
        throwIfNotValid();
        populateKnobWrappers();
        return _knobWrappers;
    }

    const hipdnn_data_sdk::flatbuffer_utilities::IKnob& getKnobById(int64_t knobId) const override
    {
        throwIfNotValid();
        populateKnobWrappers();

        auto it = _knobIdToIndex.find(knobId);
        if(it == _knobIdToIndex.end())
        {
            throw std::out_of_range("Knob with id " + std::to_string(knobId) + " not found");
        }

        return *_knobWrappers[it->second];
    }

    const hipdnn_data_sdk::flatbuffer_utilities::IKnob&
        getKnobByName(const std::string& knobName) const override
    {
        auto knobId = static_cast<int64_t>(hipdnn_data_sdk::utilities::fnv1aHash(knobName));
        return getKnobById(knobId);
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
                    = std::make_unique<hipdnn_data_sdk::flatbuffer_utilities::KnobWrapper>(knob);
                auto knobId = wrapper->knobId();
                _knobIdToIndex[knobId] = i;
                _knobWrappers.push_back(std::move(wrapper));
            }
        }
        _knobsPopulated = true;
    }

    // Pointer to the flatbuffer representation of the engine details. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_data_sdk::data_objects::EngineDetails* _shallowEngineDetails = nullptr;

    // Lazily populated cache of knob wrappers
    mutable std::vector<std::unique_ptr<hipdnn_data_sdk::flatbuffer_utilities::IKnob>>
        _knobWrappers;
    mutable std::unordered_map<int64_t, size_t> _knobIdToIndex;
    mutable bool _knobsPopulated = false;
};

}
