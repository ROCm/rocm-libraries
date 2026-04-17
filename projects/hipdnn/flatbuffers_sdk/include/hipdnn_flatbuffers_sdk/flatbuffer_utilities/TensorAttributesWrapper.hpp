// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{

class ITensorAttributesWrapper
{
public:
    virtual ~ITensorAttributesWrapper() = default;
    virtual bool isValid() const = 0;
    virtual const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensorAttributes() const
        = 0;

    virtual int64_t uid() const = 0;
    virtual std::string name() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::DataType dataType() const = 0;
    virtual std::vector<int64_t> strides() const = 0;
    virtual std::vector<int64_t> dims() const = 0;
    virtual bool isVirtual() const = 0;
    virtual hipdnn_flatbuffers_sdk::data_objects::TensorValue valueType() const = 0;
    virtual const void* value() const = 0;

    template <typename T>
    const T& valueAs() const
    {
        if(valueType() != hipdnn_flatbuffers_sdk::data_objects::TensorValueTraits<T>::enum_value)
        {
            throw std::invalid_argument("Value of tensor attributes is not of the expected type");
        }

        auto* v = value();
        if(v == nullptr)
        {
            throw std::invalid_argument("Value of tensor attributes is null");
        }

        return *static_cast<const T*>(v);
    }
};

class TensorAttributesWrapper : public ITensorAttributesWrapper
{
public:
    explicit TensorAttributesWrapper(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* t)
        : _shallowTensor(t)
    {
        throwIfNotValid();
    }

    bool isValid() const override
    {
        return _shallowTensor != nullptr;
    }

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensorAttributes() const override
    {
        throwIfNotValid();
        return *_shallowTensor;
    }

    int64_t uid() const override
    {
        throwIfNotValid();
        return _shallowTensor->uid();
    }

    std::string name() const override
    {
        throwIfNotValid();
        const auto& t = tensorAttributes();
        return t.name() != nullptr ? t.name()->str() : "";
    }

    hipdnn_flatbuffers_sdk::data_objects::DataType dataType() const override
    {
        throwIfNotValid();
        return _shallowTensor->data_type();
    }

    std::vector<int64_t> strides() const override
    {
        throwIfNotValid();
        const auto* strides = _shallowTensor->strides();
        if(strides == nullptr)
        {
            throw std::invalid_argument("TensorAttribute has null strides");
        }

        return {strides->cbegin(), strides->cend()};
    }

    std::vector<int64_t> dims() const override
    {
        throwIfNotValid();
        const auto* dims = _shallowTensor->dims();
        if(dims == nullptr)
        {
            throw std::invalid_argument("TensorAttribute has null dims");
        }

        return {dims->cbegin(), dims->cend()};
    }

    bool isVirtual() const override
    {
        throwIfNotValid();
        return _shallowTensor->virtual_();
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorValue valueType() const override
    {
        throwIfNotValid();
        return _shallowTensor->value_type();
    }

    const void* value() const override
    {
        throwIfNotValid();
        return _shallowTensor->value();
    }

private:
    void throwIfNotValid() const
    {
        if(!isValid())
        {
            throw std::invalid_argument("TensorAttribute is null");
        }
    }

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _shallowTensor = nullptr;
};
} // namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
