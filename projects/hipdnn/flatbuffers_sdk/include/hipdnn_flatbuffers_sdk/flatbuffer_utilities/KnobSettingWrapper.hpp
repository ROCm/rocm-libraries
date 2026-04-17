// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class IKnobSetting
{
public:
    virtual ~IKnobSetting() = default;

    virtual const hipdnn_flatbuffers_sdk::data_objects::KnobSetting& getKnobSetting() const = 0;
    virtual bool isValid() const = 0;
    virtual std::string knobId() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::KnobValue valueType() const = 0;

    virtual std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>
        toKnobSettingT() const = 0;

    template <typename T>
    const T& valueAs() const
    {
        if(valueType() != hipdnn_flatbuffers_sdk::data_objects::KnobValueTraits<T>::enum_value)
        {
            throw std::invalid_argument("Value is not of the expected type");
        }

        auto* val = value();
        if(val == nullptr)
        {
            throw std::invalid_argument("Value is null");
        }

        return *static_cast<const T*>(val);
    }

private:
    virtual const void* value() const = 0;
};

class KnobSettingWrapper : public IKnobSetting
{
public:
    explicit KnobSettingWrapper(
        const hipdnn_flatbuffers_sdk::data_objects::KnobSetting* knobSetting)
        : _shallowKnobSetting(knobSetting)
    {
    }

    explicit KnobSettingWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>())
            {
                _shallowKnobSetting
                    = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>(
                        buffer);
            }
        }
    }

    const hipdnn_flatbuffers_sdk::data_objects::KnobSetting& getKnobSetting() const override
    {
        throwIfNotValid();
        return *_shallowKnobSetting;
    }

    bool isValid() const override
    {
        return _shallowKnobSetting != nullptr;
    }

    std::string knobId() const override
    {
        throwIfNotValid();
        auto knobIdPtr = _shallowKnobSetting->knob_id();
        return knobIdPtr != nullptr ? knobIdPtr->str() : "";
    }

    hipdnn_flatbuffers_sdk::data_objects::KnobValue valueType() const override
    {
        throwIfNotValid();
        return _shallowKnobSetting->value_type();
    }

    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>
        toKnobSettingT() const override
    {
        throwIfNotValid();

        auto knobSettingT = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>();
        auto knobIdPtr = _shallowKnobSetting->knob_id();
        knobSettingT->knob_id = knobIdPtr != nullptr ? knobIdPtr->str() : "";

        auto knobValueType = _shallowKnobSetting->value_type();
        auto knobValuePtr = _shallowKnobSetting->value();

        if(knobValuePtr != nullptr)
        {
            knobSettingT->value.type = knobValueType;
            knobSettingT->value.value
                = hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion::UnPack(
                    knobValuePtr, knobValueType, nullptr);
        }

        return knobSettingT;
    }

private:
    const void* value() const override
    {
        throwIfNotValid();
        return _shallowKnobSetting->value();
    }

    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("KnobSetting is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the knob setting. We do not own this memory
    // as we're just reading from the buffer passed during construction.
    const hipdnn_flatbuffers_sdk::data_objects::KnobSetting* _shallowKnobSetting = nullptr;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
