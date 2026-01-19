// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>

namespace hipdnn_data_sdk::flatbuffer_utilities
{

class IKnobSetting
{
public:
    virtual ~IKnobSetting() = default;

    virtual const hipdnn_data_sdk::data_objects::KnobSetting& getKnobSetting() const = 0;
    virtual bool isValid() const = 0;
    virtual int64_t knobId() const = 0;
    virtual hipdnn_data_sdk::data_objects::KnobValue valueType() const = 0;
    virtual const std::type_info& valueClassType() const = 0;

    virtual std::unique_ptr<hipdnn_data_sdk::data_objects::KnobSettingT> toKnobSettingT() const = 0;

    template <typename T>
    const T& valueAs() const
    {
        if(valueClassType() != typeid(T))
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
    explicit KnobSettingWrapper(const hipdnn_data_sdk::data_objects::KnobSetting* knobSetting)
        : _shallowKnobSetting(knobSetting)
    {
    }

    explicit KnobSettingWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::KnobSetting>())
            {
                _shallowKnobSetting
                    = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::KnobSetting>(buffer);
            }
        }
    }

    const hipdnn_data_sdk::data_objects::KnobSetting& getKnobSetting() const override
    {
        throwIfNotValid();
        return *_shallowKnobSetting;
    }

    bool isValid() const override
    {
        return _shallowKnobSetting != nullptr;
    }

    int64_t knobId() const override
    {
        throwIfNotValid();
        return _shallowKnobSetting->knob_id();
    }

    hipdnn_data_sdk::data_objects::KnobValue valueType() const override
    {
        throwIfNotValid();
        return _shallowKnobSetting->value_type();
    }

    const std::type_info& valueClassType() const override
    {
        throwIfNotValid();
        switch(valueType())
        {
        case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
            return typeid(hipdnn_data_sdk::data_objects::IntValue);
        case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
            return typeid(hipdnn_data_sdk::data_objects::FloatValue);
        case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
            return typeid(hipdnn_data_sdk::data_objects::StringValue);
        case hipdnn_data_sdk::data_objects::KnobValue::NONE:
        default:
            throw std::invalid_argument("Value type is not recognized");
        }
    }

    std::unique_ptr<hipdnn_data_sdk::data_objects::KnobSettingT> toKnobSettingT() const override
    {
        throwIfNotValid();

        auto knobSettingT = std::make_unique<hipdnn_data_sdk::data_objects::KnobSettingT>();
        knobSettingT->knob_id = _shallowKnobSetting->knob_id();

        auto knobValueType = _shallowKnobSetting->value_type();
        auto knobValuePtr = _shallowKnobSetting->value();

        if(knobValuePtr != nullptr)
        {
            knobSettingT->value.type = knobValueType;
            knobSettingT->value.value = hipdnn_data_sdk::data_objects::KnobValueUnion::UnPack(
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
    const hipdnn_data_sdk::data_objects::KnobSetting* _shallowKnobSetting = nullptr;
};

} // namespace hipdnn_data_sdk::flatbuffer_utilities
