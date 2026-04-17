// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobSettingWrapper.hpp>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class IEngineConfig
{
public:
    virtual ~IEngineConfig() = default;

    virtual const hipdnn_flatbuffers_sdk::data_objects::EngineConfig& getEngineConfig() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t engineId() const = 0;

    virtual uint32_t knobSettingCount() const = 0;
    virtual const std::vector<
        std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting>>&
        knobSettingWrappers() const
        = 0;
    virtual const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting&
        getKnobSettingByName(const std::string& knobName) const
        = 0;

    virtual bool hasKnobSetting(const std::string& knobName) const = 0;
};

class EngineConfigWrapper : public IEngineConfig
{
public:
    explicit EngineConfigWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::EngineConfig>())
            {
                _shallowEngineConfig
                    = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::EngineConfig>(
                        buffer);
            }
        }
    }

    const hipdnn_flatbuffers_sdk::data_objects::EngineConfig& getEngineConfig() const override
    {
        throwIfNotValid();

        return *_shallowEngineConfig;
    }

    bool isValid() const override
    {
        return _shallowEngineConfig != nullptr;
    }

    int64_t engineId() const override
    {
        throwIfNotValid();

        return _shallowEngineConfig->engine_id();
    }

    uint32_t knobSettingCount() const override
    {
        throwIfNotValid();

        auto knobs = _shallowEngineConfig->knobs();
        if(knobs == nullptr)
        {
            return 0;
        }
        return knobs->size();
    }

    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting>>&
        knobSettingWrappers() const override
    {
        throwIfNotValid();
        populateKnobSettingWrappers();
        return _knobSettingWrappers;
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting&
        getKnobSettingByName(const std::string& knobName) const override
    {
        throwIfNotValid();
        populateKnobSettingWrappers();

        auto it = _knobSettingNameToIndex.find(knobName);
        if(it == _knobSettingNameToIndex.end())
        {
            throw std::out_of_range("KnobSetting with name '" + knobName + "' not found");
        }

        return *_knobSettingWrappers[it->second];
    }

    bool hasKnobSetting(const std::string& knobName) const override
    {
        throwIfNotValid();
        populateKnobSettingWrappers();
        return _knobSettingNameToIndex.find(knobName) != _knobSettingNameToIndex.end();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("Engine config is not valid");
        }
    }

    void populateKnobSettingWrappers() const
    {
        if(_knobSettingsPopulated)
        {
            return;
        }

        auto knobs = _shallowEngineConfig->knobs();
        if(knobs != nullptr)
        {
            _knobSettingWrappers.reserve(knobs->size());
            for(uint32_t i = 0; i < knobs->size(); ++i)
            {
                auto knob = knobs->Get(i);
                auto wrapper = std::make_unique<
                    hipdnn_flatbuffers_sdk::flatbuffer_utilities::KnobSettingWrapper>(knob);
                auto knobName = wrapper->knobId();
                _knobSettingNameToIndex[knobName] = i;
                _knobSettingWrappers.push_back(std::move(wrapper));
            }
        }
        _knobSettingsPopulated = true;
    }

    // Pointer to the flatbuffer representation of the engine config. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_flatbuffers_sdk::data_objects::EngineConfig* _shallowEngineConfig = nullptr;

    // Lazily populated cache of knob setting wrappers
    mutable std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting>>
        _knobSettingWrappers;
    mutable std::unordered_map<std::string, size_t> _knobSettingNameToIndex;
    mutable bool _knobSettingsPopulated = false;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
