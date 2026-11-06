// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/// @file IngestorScalarBindings.hpp
/// Saves and restores the pass-by-value scalar operands of a handler's launch (RFC 0016
/// §4.6, §4.9), as launch values.
///
/// A role names one scalar of a launch, such as `seed`. A binding writes the values
/// `<role>.uid`, `<role>.data_type`, `<role>.user_supplied` and, only when the scalar is
/// baked at plan-build time, `<role>.baked`. A user-supplied scalar never saves a value: the
/// restored plan reads it from the device buffers at execute.
namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace detail
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;

inline std::string scalarBindingName(std::string_view role, std::string_view field)
{
    std::string name(role);
    name += '.';
    name += field;
    return name;
}

// True when a ScalarValue has an arm for this data type.
inline bool isScalarValueDataType(DataType dataType)
{
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::INT32:
    case DataType::INT64:
    case DataType::BOOLEAN:
        return true;
    default:
        return false;
    }
}

// The ScalarValue arm that holds a scalar of this data type, as its index in the variant.
inline size_t scalarValueIndexOf(DataType dataType)
{
    switch(dataType)
    {
    case DataType::DOUBLE:
        return 0;
    case DataType::FLOAT:
        return 1;
    case DataType::HALF:
        return 2;
    case DataType::BFLOAT16:
        return 3;
    case DataType::INT32:
        return 4;
    case DataType::INT64:
        return 5;
    case DataType::BOOLEAN:
        return 6;
    default:
        break;
    }
    return std::variant_npos;
}

inline bool requireLaunchBool(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                              std::string_view name)
{
    const std::string expected
        = hipdnn_plugin_sdk::ingestor::toString(hipdnn_plugin_sdk::ingestor::MetadataType::BOOL);
    const auto& value = requireLaunchValue(inputs, name, expected);
    const auto* flag = std::get_if<bool>(&value);
    if(flag == nullptr)
    {
        refuseLaunchValue(inputs,
                          name,
                          expected,
                          hipdnn_plugin_sdk::ingestor::toString(
                              hipdnn_plugin_sdk::ingestor::metadataTypeOf(value)));
    }
    return *flag;
}

} // namespace detail

/// Writes the binding of @p operand under @p role into @p values. Throws
/// `HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR` for an operand whose data type has no `ScalarValue`
/// arm, or whose baked value is held in another arm than its data type names.
inline void putScalarBinding(hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                             std::string_view role,
                             const hipdnn_plugin_sdk::ScalarOperand& operand)
{
    if(!detail::isScalarValueDataType(operand.dataType)
       || (!operand.isRuntimeUserSupplied
           && operand.bakedDefault.index() != detail::scalarValueIndexOf(operand.dataType)))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot save scalar '" + std::string(role) + "': its data type "
                + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(operand.dataType)
                + " does not match a pass-by-value scalar value");
    }

    values[detail::scalarBindingName(role, "uid")] = operand.uid;
    values[detail::scalarBindingName(role, "data_type")]
        = std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(operand.dataType));
    values[detail::scalarBindingName(role, "user_supplied")] = operand.isRuntimeUserSupplied;
    if(operand.isRuntimeUserSupplied)
    {
        return;
    }

    hipdnn_plugin_sdk::ingestor::MetadataValue baked;
    std::visit(
        [&baked](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr(std::is_same_v<T, bool>)
            {
                baked = value;
            }
            else if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
            {
                baked = static_cast<int64_t>(value);
            }
            else
            {
                // double, float, half and bfloat16 are all exactly representable as double.
                baked = static_cast<double>(value);
            }
        },
        operand.bakedDefault);
    values[detail::scalarBindingName(role, "baked")] = std::move(baked);
}

/// True when @p inputs holds a binding for @p role: the `<role>.uid` value is present.
inline bool hasScalarBinding(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                             std::string_view role)
{
    return inputs.values.find(detail::scalarBindingName(role, "uid")) != inputs.values.end();
}

/// Rebuilds the operand `putScalarBinding` saved under @p role. Refuses the plan
/// (`INCOMPATIBLE`) when a value is missing or has another type, when `<role>.baked` sits
/// next to a user-supplied scalar or is missing next to a baked one, or when the data type
/// has no `ScalarValue` arm. A baked value is rebuilt in the arm of its data type.
inline hipdnn_plugin_sdk::ScalarOperand
    requireScalarBinding(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                         std::string_view role)
{
    using hipdnn_plugin_sdk::ScalarOperand;
    using hipdnn_plugin_sdk::ScalarValue;
    using hipdnn_plugin_sdk::ingestor::MetadataValue;

    const std::string uidName = detail::scalarBindingName(role, "uid");
    const std::string typeName = detail::scalarBindingName(role, "data_type");
    const std::string userName = detail::scalarBindingName(role, "user_supplied");
    const std::string bakedName = detail::scalarBindingName(role, "baked");

    ScalarOperand operand;
    operand.uid = requireLaunchInt(inputs, uidName);

    const std::string typeText = requireLaunchString(inputs, typeName);
    bool found = false;
    for(const auto candidate : hipdnn_flatbuffers_sdk::data_objects::EnumValuesDataType())
    {
        if(typeText == hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(candidate))
        {
            operand.dataType = candidate;
            found = true;
            break;
        }
    }
    if(!found || !detail::isScalarValueDataType(operand.dataType))
    {
        detail::refuseLaunchValue(inputs,
                                  typeName,
                                  "the name of a data type a pass-by-value scalar can hold",
                                  "'" + typeText + "'");
    }

    operand.isRuntimeUserSupplied = detail::requireLaunchBool(inputs, userName);
    const bool hasBaked = inputs.values.find(bakedName) != inputs.values.end();
    if(operand.isRuntimeUserSupplied)
    {
        if(hasBaked)
        {
            serialization::refuseIngestorPlan(
                serialization::IngestorPlanRefusal::INCOMPATIBLE,
                "launch value '" + bakedName + "' of dispatch '" + inputs.dispatchSymbol
                    + "' must be absent: '" + userName
                    + "' is true, and a user-supplied scalar saves no value");
        }
        return operand;
    }

    if(!hasBaked)
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "dispatch '" + inputs.dispatchSymbol
                                              + "' needs launch value '" + bakedName + "', as '"
                                              + userName + "' is false, and the plan has none");
    }

    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(operand.dataType)
    {
    case DataType::DOUBLE:
        operand.bakedDefault = ScalarValue{requireLaunchDouble(inputs, bakedName)};
        break;
    case DataType::FLOAT:
        operand.bakedDefault
            = ScalarValue{static_cast<float>(requireLaunchDouble(inputs, bakedName))};
        break;
    case DataType::HALF:
        operand.bakedDefault
            = ScalarValue{hipdnn_data_sdk::types::half(requireLaunchDouble(inputs, bakedName))};
        break;
    case DataType::BFLOAT16:
        operand.bakedDefault
            = ScalarValue{hipdnn_data_sdk::types::bfloat16(requireLaunchDouble(inputs, bakedName))};
        break;
    case DataType::INT32:
        operand.bakedDefault = ScalarValue{
            static_cast<int32_t>(requireLaunchInt(inputs,
                                                  bakedName,
                                                  std::numeric_limits<int32_t>::min(),
                                                  std::numeric_limits<int32_t>::max()))};
        break;
    case DataType::INT64:
        operand.bakedDefault = ScalarValue{requireLaunchInt(inputs, bakedName)};
        break;
    case DataType::BOOLEAN:
        operand.bakedDefault = ScalarValue{detail::requireLaunchBool(inputs, bakedName)};
        break;
    default:
        break;
    }
    return operand;
}

/// Appends the launch value names of one role to @p names: the allow-list entries a handler
/// passes to `requireOnlyLaunchValues`.
inline void appendScalarBindingNames(std::vector<std::string>& names, std::string_view role)
{
    for(const std::string_view field : {"uid", "data_type", "user_supplied", "baked"})
    {
        names.push_back(detail::scalarBindingName(role, field));
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
