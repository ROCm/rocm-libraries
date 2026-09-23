// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class IKnob
{
public:
    virtual ~IKnob() = default;

    virtual const hipdnn_flatbuffers_sdk::data_objects::Knob& getKnob() const = 0;
    virtual bool isValid() const = 0;
    virtual std::string knobId() const = 0;
    virtual std::string description() const = 0;
    virtual bool isDeprecated() const = 0;

    // Default value accessors
    virtual bool hasDefaultValue() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::KnobValue defaultValueType() const = 0;

    // Constraint accessors
    virtual bool hasConstraint() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::KnobConstraint constraintType() const = 0;

    template <typename T>
    const T& defaultValueAs() const
    {
        if(defaultValueType()
           != hipdnn_flatbuffers_sdk::data_objects::KnobValueTraits<T>::enum_value)
        {
            throw std::invalid_argument("Default value is not of the expected type");
        }

        auto* val = defaultValue();
        if(val == nullptr)
        {
            throw std::invalid_argument("Default value is null");
        }

        return *static_cast<const T*>(val);
    }

    template <typename T>
    const T& constraintAs() const
    {
        if(constraintType()
           != hipdnn_flatbuffers_sdk::data_objects::KnobConstraintTraits<T>::enum_value)
        {
            throw std::invalid_argument("Constraint is not of the expected type");
        }

        auto* c = constraint();
        if(c == nullptr)
        {
            throw std::invalid_argument("Constraint is null");
        }

        return *static_cast<const T*>(c);
    }

private:
    // Raw pointer accessors (for serialization use cases)
    virtual const void* defaultValue() const = 0;
    virtual const void* constraint() const = 0;
};

class KnobWrapper : public IKnob
{
public:
    explicit KnobWrapper(const hipdnn_flatbuffers_sdk::data_objects::Knob* knob)
        : _shallowKnob(knob)
    {
    }

    explicit KnobWrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Knob>())
            {
                _shallowKnob
                    = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Knob>(buffer);
            }
        }
    }

    const hipdnn_flatbuffers_sdk::data_objects::Knob& getKnob() const override
    {
        throwIfNotValid();
        return *_shallowKnob;
    }

    bool isValid() const override
    {
        return _shallowKnob != nullptr;
    }

    std::string knobId() const override
    {
        throwIfNotValid();
        auto str = _shallowKnob->knob_id();
        return str != nullptr ? str->str() : "";
    }

    std::string description() const override
    {
        throwIfNotValid();
        auto desc = _shallowKnob->description();
        return desc != nullptr ? desc->str() : "";
    }

    bool isDeprecated() const override
    {
        throwIfNotValid();
        return _shallowKnob->deprecated();
    }

    bool hasDefaultValue() const override
    {
        throwIfNotValid();
        return _shallowKnob->default_value() != nullptr;
    }

    hipdnn_flatbuffers_sdk::data_objects::KnobValue defaultValueType() const override
    {
        throwIfNotValid();
        return _shallowKnob->default_value_type();
    }

    bool hasConstraint() const override
    {
        throwIfNotValid();
        return _shallowKnob->constraint() != nullptr;
    }

    hipdnn_flatbuffers_sdk::data_objects::KnobConstraint constraintType() const override
    {
        throwIfNotValid();
        return _shallowKnob->constraint_type();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("Knob is not valid");
        }
    }

    const void* defaultValue() const override
    {
        throwIfNotValid();
        return _shallowKnob->default_value();
    }

    const void* constraint() const override
    {
        throwIfNotValid();
        return _shallowKnob->constraint();
    }

    // Pointer to the flatbuffer representation of the knob. We do not own this memory
    // as we're just reading from the buffer passed during construction.
    const hipdnn_flatbuffers_sdk::data_objects::Knob* _shallowKnob = nullptr;
};

} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
