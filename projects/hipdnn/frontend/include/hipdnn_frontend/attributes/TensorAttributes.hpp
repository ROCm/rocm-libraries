// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "GraphAttributes.hpp"
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace hipdnn_frontend::graph
{

class TensorAttributes
{
public:
    using ValueVariant
        = std::variant<std::monostate, double, float, half, hip_bfloat16, uint8_t, int32_t>;

    TensorAttributes() = default;

    template <typename T>
    TensorAttributes(T const& scalar)
    {
        set_value(scalar);
    }

    bool get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        return !std::holds_alternative<std::monostate>(_value);
    }

    template <typename T>
    std::optional<T> get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        if(auto p = std::get_if<T>(&_value))
        {
            return *p;
        }
        return std::nullopt;
    }

    template <typename T>
    TensorAttributes& set_value(T v) // NOLINT(readability-identifier-naming)
    {

        static_assert(std::disjunction_v<std::is_same<T, float>,
                                         std::is_same<T, double>,
                                         std::is_same<T, half>,
                                         std::is_same<T, hip_bfloat16>,
                                         std::is_same<T, uint8_t>,
                                         std::is_same<T, int32_t>>,
                      "Unsupported type for Tensor_attributes::set_value");
        _value = v;
        _dataType = getDataTypeEnumFromType<T>();
        _dim = _stride = {1};
        return *this;
    }

    TensorAttributes& clear_value() // NOLINT(readability-identifier-naming)
    {
        _value = {};
        return *this;
    }

    int64_t get_uid() const // NOLINT(readability-identifier-naming)
    {
        return _uid;
    }

    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return _name;
    }

    DataType get_data_type() const // NOLINT(readability-identifier-naming)
    {
        return _dataType;
    }

    const std::vector<int64_t>& get_stride() const // NOLINT(readability-identifier-naming)
    {
        return _stride;
    }

    const std::vector<int64_t>& get_dim() const // NOLINT(readability-identifier-naming)
    {
        return _dim;
    }

    int64_t get_volume() const // NOLINT(readability-identifier-naming)
    {
        int64_t volume = 1;
        for(const auto& d : _dim)
        {
            volume *= d;
        }
        return volume;
    }

    bool get_is_virtual() const // NOLINT(readability-identifier-naming)
    {
        return _isVirtual;
    }

    bool has_uid() const // NOLINT(readability-identifier-naming)
    {
        return _uidSet;
    }

    TensorAttributes& set_uid(int64_t uid) // NOLINT(readability-identifier-naming)
    {
        _uid = uid;
        _uidSet = true;
        return *this;
    }

    TensorAttributes& set_name(const std::string& name) // NOLINT(readability-identifier-naming)
    {
        _name = name;
        return *this;
    }

    TensorAttributes& set_data_type(DataType dataType) // NOLINT(readability-identifier-naming)
    {
        _dataType = dataType;
        return *this;
    }

    TensorAttributes&
        set_stride(const std::vector<int64_t>& stride) // NOLINT(readability-identifier-naming)
    {
        _stride = stride;
        return *this;
    }

    TensorAttributes&
        set_dim(const std::vector<int64_t>& dim) // NOLINT(readability-identifier-naming)
    {
        _dim = dim;
        return *this;
    }

    TensorAttributes& set_is_virtual(bool isVirtual) // NOLINT(readability-identifier-naming)
    {
        _isVirtual = isVirtual;
        return *this;
    }

    TensorAttributes& set_output(bool output) // NOLINT(readability-identifier-naming)
    {
        return set_is_virtual(!output);
    }

    TensorAttributes& clear_uid() // NOLINT(readability-identifier-naming)
    {
        _uid = 0;
        _uidSet = false;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    TensorAttributes& fill_from_context(const GraphAttributes& graphAttributes)
    {
        if(_dataType == DataType::NOT_SET)
        {
            if(_isVirtual)
            {
                _dataType = graphAttributes.get_intermediate_data_type();
            }
            else
            {
                _dataType = graphAttributes.get_io_data_type();
            }
        }

        return *this;
    }

    Error validate() const
    {
        if(_dataType == DataType::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Tensor " + _name + " does not have a data type set"};
        }

        HIPDNN_RETURN_IF_TRUE(_isVirtual && get_pass_by_value(),
                              ErrorCode::INVALID_VALUE,
                              "Tensor " + _name + " cannot be virtual and pass by value");
        HIPDNN_RETURN_IF_NE(_dim.size(),
                            _stride.size(),
                            ErrorCode::INVALID_VALUE,
                            "Tensor " + _name + " dims and strides have different sizes");

        HIPDNN_RETURN_IF_TRUE(_dim.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "Tensor " + _name + " dims must be non-empty");

        auto isPositive = [](int64_t value) constexpr { return value > 0; };
        HIPDNN_RETURN_IF_FALSE(std::all_of(_dim.begin(), _dim.end(), isPositive),
                               ErrorCode::INVALID_VALUE,
                               "Tensor " + _name + " must have only positive dimensions");

        return {ErrorCode::OK, ""};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto result = std::visit(
            [&](auto&& arg) -> std::pair<hipdnn_data_sdk::data_objects::TensorValue,
                                         flatbuffers::Offset<void>> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, float>)
                {
                    hipdnn_data_sdk::data_objects::Float32Value floatVal(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
                            builder.CreateStruct(floatVal).Union()};
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    hipdnn_data_sdk::data_objects::Float64Value doubleVal(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                            builder.CreateStruct(doubleVal).Union()};
                }
                else if constexpr(std::is_same_v<T, half>)
                {
                    hipdnn_data_sdk::data_objects::Float16Value halfVal(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::Float16Value,
                            builder.CreateStruct(halfVal).Union()};
                }
                else if constexpr(std::is_same_v<T, hip_bfloat16>)
                {
                    hipdnn_data_sdk::data_objects::BFloat16Value bfVal(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::BFloat16Value,
                            builder.CreateStruct(bfVal).Union()};
                }
                else if constexpr(std::is_same_v<T, uint8_t>)
                {
                    hipdnn_data_sdk::data_objects::Float8Value uint8Val(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::Float8Value,
                            builder.CreateStruct(uint8Val).Union()};
                }
                else if constexpr(std::is_same_v<T, int32_t>)
                {
                    hipdnn_data_sdk::data_objects::Int32Value int32Val(arg);
                    return {hipdnn_data_sdk::data_objects::TensorValue::Int32Value,
                            builder.CreateStruct(int32Val).Union()};
                }
                else
                {
                    // For std::monostate case
                    return {hipdnn_data_sdk::data_objects::TensorValue::NONE, 0};
                }
            },
            _value);

        return hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                           _uid,
                                                                           _name.c_str(),
                                                                           toSdkType(_dataType),
                                                                           &_stride,
                                                                           &_dim,
                                                                           _isVirtual,
                                                                           result.first,
                                                                           result.second);
    }

    static std::shared_ptr<TensorAttributes>
        fromFlatBuffer(const hipdnn_data_sdk::data_objects::TensorAttributes* fb)
    {
        if(fb == nullptr)
        {
            return nullptr;
        }

        auto tensor = std::make_shared<TensorAttributes>();

        tensor->set_uid(fb->uid());

        if(fb->name() != nullptr)
        {
            tensor->set_name(fb->name()->c_str());
        }

        tensor->set_data_type(fromSdkType(fb->data_type()));

        if(fb->dims() != nullptr)
        {
            std::vector<int64_t> dims(fb->dims()->begin(), fb->dims()->end());
            tensor->set_dim(dims);
        }

        if(fb->strides() != nullptr)
        {
            std::vector<int64_t> strides(fb->strides()->begin(), fb->strides()->end());
            tensor->set_stride(strides);
        }

        tensor->set_is_virtual(fb->virtual_());

        auto valueType = fb->value_type();
        if(valueType != hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            switch(valueType)
            {
            case hipdnn_data_sdk::data_objects::TensorValue::Float32Value:
                tensor->set_value(fb->value_as_Float32Value()->value());
                break;
            case hipdnn_data_sdk::data_objects::TensorValue::Float64Value:
                tensor->set_value(fb->value_as_Float64Value()->value());
                break;
            case hipdnn_data_sdk::data_objects::TensorValue::Float16Value:
                tensor->set_value(static_cast<half>(fb->value_as_Float16Value()->value()));
                break;
            case hipdnn_data_sdk::data_objects::TensorValue::BFloat16Value:
                tensor->set_value(static_cast<hip_bfloat16>(fb->value_as_BFloat16Value()->value()));
                break;
            case hipdnn_data_sdk::data_objects::TensorValue::Float8Value:
                tensor->set_value(fb->value_as_Float8Value()->value());
                break;
            case hipdnn_data_sdk::data_objects::TensorValue::Int32Value:
                tensor->set_value(fb->value_as_Int32Value()->value());
                break;
            default:
                break;
            }
        }

        return tensor;
    }

private:
    int64_t _uid = 0;
    bool _uidSet = false;
    std::string _name;
    DataType _dataType = DataType::NOT_SET;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _dim;
    bool _isVirtual = false;
    ValueVariant _value;
};
typedef TensorAttributes Tensor_attributes;
} // namespace hipdnn_frontend::graph
