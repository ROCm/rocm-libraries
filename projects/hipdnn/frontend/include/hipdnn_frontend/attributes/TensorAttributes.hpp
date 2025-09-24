// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "GraphAttributes.hpp"
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
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

    bool validate_dims_set_and_positive() const // NOLINT(readability-identifier-naming
    {
        auto isPositive = [](int64_t value) constexpr { return value > 0; };
        return !_dim.empty() && std::all_of(_dim.begin(), _dim.end(), isPositive);
    }

    bool validate_dims_and_strides_set_and_positive() const // NOLINT(readability-identifier-naming
    {
        auto isPositive = [](int64_t value) constexpr { return value > 0; };
        return validate_dims_set_and_positive() && _stride.size() == _dim.size()
               && std::all_of(_stride.begin(), _stride.end(), isPositive);
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto result = std::visit(
            [&](auto&& arg)
                -> std::pair<hipdnn_sdk::data_objects::TensorValue, flatbuffers::Offset<void>> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr(std::is_same_v<T, float>)
                {
                    hipdnn_sdk::data_objects::Float32Value floatVal(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::Float32Value,
                            builder.CreateStruct(floatVal).Union()};
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    hipdnn_sdk::data_objects::Float64Value doubleVal(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::Float64Value,
                            builder.CreateStruct(doubleVal).Union()};
                }
                else if constexpr(std::is_same_v<T, half>)
                {
                    hipdnn_sdk::data_objects::Float16Value halfVal(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::Float16Value,
                            builder.CreateStruct(halfVal).Union()};
                }
                else if constexpr(std::is_same_v<T, hip_bfloat16>)
                {
                    hipdnn_sdk::data_objects::BFloat16Value bfVal(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::BFloat16Value,
                            builder.CreateStruct(bfVal).Union()};
                }
                else if constexpr(std::is_same_v<T, uint8_t>)
                {
                    hipdnn_sdk::data_objects::Float8Value uint8Val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::Float8Value,
                            builder.CreateStruct(uint8Val).Union()};
                }
                else if constexpr(std::is_same_v<T, int32_t>)
                {
                    hipdnn_sdk::data_objects::Int32Value int32Val(arg);
                    return {hipdnn_sdk::data_objects::TensorValue::Int32Value,
                            builder.CreateStruct(int32Val).Union()};
                }
                else
                {
                    // For std::monostate case
                    return {hipdnn_sdk::data_objects::TensorValue::NONE, 0};
                }
            },
            _value);

        return hipdnn_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                      _uid,
                                                                      _name.c_str(),
                                                                      toSdkType(_dataType),
                                                                      &_stride,
                                                                      &_dim,
                                                                      _isVirtual,
                                                                      result.first,
                                                                      result.second);
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
}
}
