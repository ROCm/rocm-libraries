// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests
{

// An input that has no allocation of its own. The view (`uid`) reads the memory of
// another input, the owner (`ofUid`), from `byteOffset` bytes into the owner's
// allocation. The view's own dims, strides and data type give the layout.
struct TensorAlias
{
    int64_t uid = 0;
    int64_t ofUid = 0;
    int64_t byteOffset = 0;

    bool operator==(const TensorAlias& other) const
    {
        return uid == other.uid && ofUid == other.ofUid && byteOffset == other.byteOffset;
    }
};

namespace tensor_alias_detail
{

inline std::string describe(const TensorAlias& alias)
{
    return "alias view " + std::to_string(alias.uid) + " of owner " + std::to_string(alias.ofUid);
}

// The value as an int64_t, or nullopt when it is not an integer in the int64_t range.
inline std::optional<int64_t> integerValue(const nlohmann::json& value)
{
    if(!value.is_number_integer())
    {
        return std::nullopt;
    }
    if(value.is_number_unsigned()
       && value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        return std::nullopt;
    }
    return value.get<int64_t>();
}

// Throws if `entry` has a key outside `allowedKeys`, naming `where` and the key.
inline void rejectUnknownKeys(const nlohmann::json& entry,
                              const std::string& where,
                              std::initializer_list<std::string_view> allowedKeys)
{
    for(const auto& item : entry.items())
    {
        const auto& key = item.key();
        bool known = false;
        for(const auto allowed : allowedKeys)
        {
            if(key == allowed)
            {
                known = true;
                break;
            }
        }
        if(!known)
        {
            std::string message = where;
            message.append(" has unknown key \"").append(key).append("\"");
            throw std::invalid_argument(message);
        }
    }
}

// Reads `key` as a 64-bit signed integer, throwing (naming `where`) when it is absent
// or not one.
inline int64_t
    readRequiredInteger(const nlohmann::json& entry, const char* key, const std::string& where)
{
    if(!entry.contains(key))
    {
        throw std::invalid_argument(where + " lacks \"" + key + "\"");
    }
    const auto value = integerValue(entry.at(key));
    if(!value.has_value())
    {
        throw std::invalid_argument(where + ": \"" + key + "\" is not a 64-bit signed integer");
    }
    return *value;
}

// Reads `key` as a 64-bit signed integer: nullopt when absent, throwing (naming
// `where`) only when present and not one.
inline std::optional<int64_t>
    readOptionalInteger(const nlohmann::json& entry, const char* key, const std::string& where)
{
    if(!entry.contains(key))
    {
        return std::nullopt;
    }
    const auto value = integerValue(entry.at(key));
    if(!value.has_value())
    {
        throw std::invalid_argument(where + ": \"" + key + "\" is not a 64-bit signed integer");
    }
    return value;
}

// Parses `json` as a non-empty array of objects, one T per entry via `buildEntry(entry,
// where)` (`where` already names the array and index, e.g. "aliases[2]"), then rejects
// a uid (read by `uidOf`) that repeats across entries, naming it with
// `duplicateMessage`. Shared outer shape of the aliases/start_offsets/live_rows
// metadata-array parsers; each keeps its own field list and per-entry rules in
// `buildEntry`.
template <typename T, typename BuildEntry, typename UidOf, typename DuplicateMessage>
inline std::vector<T> parseUidKeyedArray(const nlohmann::json& json,
                                         const char* arrayName,
                                         BuildEntry buildEntry,
                                         UidOf uidOf,
                                         DuplicateMessage duplicateMessage)
{
    if(!json.is_array())
    {
        throw std::invalid_argument(std::string(arrayName) + " is not an array");
    }
    if(json.empty())
    {
        throw std::invalid_argument(std::string(arrayName) + " is empty");
    }

    std::vector<T> entries;
    std::unordered_set<int64_t> uids;
    for(std::size_t i = 0; i < json.size(); ++i)
    {
        const auto& entry = json[i];
        const std::string where = std::string(arrayName) + "[" + std::to_string(i) + "]";
        if(!entry.is_object())
        {
            throw std::invalid_argument(where + " is not an object");
        }
        T parsed = buildEntry(entry, where);
        if(!uids.insert(uidOf(parsed)).second)
        {
            throw std::invalid_argument(duplicateMessage(parsed));
        }
        entries.push_back(std::move(parsed));
    }
    return entries;
}

inline TensorAlias parseEntry(const nlohmann::json& entry, const std::string& baseWhere)
{
    std::string where = baseWhere;
    const auto known = [&](const char* key) {
        return entry.contains(key) ? integerValue(entry.at(key)) : std::nullopt;
    };
    const auto uid = known("uid");
    const auto ofUid = known("of");
    if(uid.has_value() || ofUid.has_value())
    {
        where += " (";
        where += uid.has_value() ? "view " + std::to_string(*uid) : std::string();
        where += uid.has_value() && ofUid.has_value() ? ", " : "";
        where += ofUid.has_value() ? "owner " + std::to_string(*ofUid) : std::string();
        where += ")";
    }

    rejectUnknownKeys(entry, where, {"uid", "of", "byte_offset"});

    TensorAlias alias;
    alias.uid = readRequiredInteger(entry, "uid", where);
    alias.ofUid = readRequiredInteger(entry, "of", where);
    alias.byteOffset = readRequiredInteger(entry, "byte_offset", where);

    if(alias.byteOffset < 0)
    {
        throw std::invalid_argument(describe(alias) + ": byte_offset "
                                    + std::to_string(alias.byteOffset) + " is negative");
    }
    if(alias.uid == alias.ofUid)
    {
        throw std::invalid_argument(describe(alias) + ": the view and the owner are one tensor");
    }
    return alias;
}

// Bytes per element of a data type that a view may have; nullopt for the rest.
inline std::optional<int64_t> elementSize(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 8;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
    case DataType::FP8_E8M0:
        return 1;
    case DataType::BOOLEAN:
        return static_cast<int64_t>(sizeof(bool));
    default:
        return std::nullopt;
    }
}

inline bool isSubByte(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    return dataType == DataType::FP4_E2M1 || dataType == DataType::FP6_E2M3
           || dataType == DataType::FP6_E3M2 || dataType == DataType::INT4;
}

// Sum and product of two non-negative values. Nullopt when either is negative or the
// result does not fit in an int64_t.
inline std::optional<int64_t> checkedAdd(int64_t a, int64_t b)
{
    if(a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b)
    {
        return std::nullopt;
    }
    return a + b;
}

inline std::optional<int64_t> checkedMultiply(int64_t a, int64_t b)
{
    if(a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<int64_t>::max() / a))
    {
        return std::nullopt;
    }
    return a * b;
}

// Bytes from the start of the allocation to the end of the last element:
// (1 + sum of (dim - 1) * stride) * element size. Every dim is at least 1 and every
// stride is at least 0. Nullopt when the count does not fit in an int64_t.
inline std::optional<int64_t> spanBytes(const std::vector<int64_t>& dims,
                                        const std::vector<int64_t>& strides,
                                        int64_t elementBytes)
{
    std::optional<int64_t> elements = 1;
    for(std::size_t i = 0; i < dims.size() && elements.has_value(); ++i)
    {
        const auto term = checkedMultiply(dims[i] - 1, strides[i]);
        elements = term.has_value() ? checkedAdd(*elements, *term) : std::nullopt;
    }
    return elements.has_value() ? checkedMultiply(*elements, elementBytes) : std::nullopt;
}

struct Layout
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

// The layout of one side of an alias, after the checks that apply to the view and the
// owner alike.
inline Layout checkedLayout(const TensorAlias& alias,
                            const char* role,
                            int64_t uid,
                            const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor,
                            const std::unordered_set<int64_t>& outputUids)
{
    const std::string prefix = describe(alias) + ": the " + role + " ";
    if(tensor == nullptr)
    {
        throw std::invalid_argument(prefix + "is not a tensor of the graph");
    }
    if(tensor->virtual_())
    {
        throw std::invalid_argument(prefix + "is virtual");
    }
    if(outputUids.count(uid) != 0)
    {
        throw std::invalid_argument(prefix + "is an output; aliases describe inputs only");
    }
    if(tensor->is_runtime_pass_by_value())
    {
        throw std::invalid_argument(prefix + "is passed by value at run time");
    }
    if(tensor->value_type() != hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
    {
        throw std::invalid_argument(prefix + "has a constant value");
    }
    if(tensor->ragged_offset_tensor_uid().has_value())
    {
        throw std::invalid_argument(prefix + "is ragged");
    }
    if(isSubByte(tensor->data_type()))
    {
        throw std::invalid_argument(prefix + "has the sub-byte data type "
                                    + EnumNameDataType(tensor->data_type()));
    }
    if(!elementSize(tensor->data_type()).has_value())
    {
        throw std::invalid_argument(prefix + "has the data type "
                                    + EnumNameDataType(tensor->data_type())
                                    + ", which has no element size");
    }

    const auto dims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(tensor->dims());
    const auto strides
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(tensor->strides());
    if(dims.size() != strides.size())
    {
        throw std::invalid_argument(prefix + "has " + std::to_string(dims.size()) + " dims but "
                                    + std::to_string(strides.size()) + " strides");
    }
    for(std::size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] < 0)
        {
            throw std::invalid_argument(prefix + "has a negative dim");
        }
        if(strides[i] < 0)
        {
            throw std::invalid_argument(prefix + "has a negative stride");
        }
    }
    return Layout{dims, strides};
}

inline void checkAlias(
    const TensorAlias& alias,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensors,
    const std::unordered_set<int64_t>& outputUids)
{
    const auto find = [&](int64_t uid) {
        const auto it = tensors.find(uid);
        return it == tensors.end() ? nullptr : it->second;
    };
    const auto* view = find(alias.uid);
    const auto* owner = find(alias.ofUid);
    const auto viewLayout = checkedLayout(alias, "view", alias.uid, view, outputUids);
    const auto ownerLayout = checkedLayout(alias, "owner", alias.ofUid, owner, outputUids);

    if(view->data_type() != owner->data_type())
    {
        throw std::invalid_argument(
            describe(alias) + ": the view's data type " + EnumNameDataType(view->data_type())
            + " differs from the owner's data type " + EnumNameDataType(owner->data_type()));
    }

    const int64_t bytes = *elementSize(view->data_type());
    if(alias.byteOffset % bytes != 0)
    {
        throw std::invalid_argument(
            describe(alias) + ": byte_offset " + std::to_string(alias.byteOffset)
            + " is not a multiple of the view's element size " + std::to_string(bytes));
    }

    const int64_t alignment = view->alignment();
    if(alignment > 0 && alias.byteOffset % alignment != 0)
    {
        throw std::invalid_argument(
            describe(alias) + ": byte_offset " + std::to_string(alias.byteOffset)
            + " is not a multiple of the view's alignment " + std::to_string(alignment));
    }
    const int64_t ownerAlignment = owner->alignment();
    if(alignment > 0 && (ownerAlignment <= 0 || ownerAlignment % alignment != 0))
    {
        throw std::invalid_argument(
            describe(alias) + ": the owner's alignment " + std::to_string(ownerAlignment)
            + " is not a multiple of the view's alignment " + std::to_string(alignment));
    }

    for(const int64_t dim : ownerLayout.dims)
    {
        if(dim == 0)
        {
            throw std::invalid_argument(describe(alias) + ": the owner has a zero dim");
        }
    }

    const auto ownerBytes = spanBytes(ownerLayout.dims, ownerLayout.strides, bytes);
    if(!ownerBytes.has_value())
    {
        throw std::invalid_argument(describe(alias)
                                    + ": the owner's size does not fit in a 64-bit byte count");
    }

    // A view with a zero dim reaches no bytes, but its pointer still lies in the owner.
    for(const int64_t dim : viewLayout.dims)
    {
        if(dim == 0)
        {
            if(alias.byteOffset > *ownerBytes)
            {
                std::string message = describe(alias);
                message.append(": byte_offset ")
                    .append(std::to_string(alias.byteOffset))
                    .append(" is past the end of the owner's ")
                    .append(std::to_string(*ownerBytes))
                    .append("-byte allocation");
                throw std::invalid_argument(message);
            }
            return;
        }
    }
    const auto viewBytes = spanBytes(viewLayout.dims, viewLayout.strides, bytes);
    const auto viewEnd
        = viewBytes.has_value() ? checkedAdd(alias.byteOffset, *viewBytes) : std::nullopt;
    if(!viewEnd.has_value())
    {
        throw std::invalid_argument(describe(alias)
                                    + ": the view's reach does not fit in a 64-bit byte count");
    }
    if(*viewEnd > *ownerBytes)
    {
        throw std::invalid_argument(describe(alias) + ": the view reaches byte "
                                    + std::to_string(*viewEnd) + ", past the end of the owner's "
                                    + std::to_string(*ownerBytes) + "-byte allocation");
    }
}

} // namespace tensor_alias_detail

// Parses metadata `aliases`. Throws std::invalid_argument when the value is not a
// non-empty array of {"uid", "of", "byte_offset"} objects with integer values, when a
// byte_offset is negative, when a view is its own owner, when a view appears twice, or
// when a view is also an owner.
inline std::vector<TensorAlias> parseTensorAliases(const nlohmann::json& json)
{
    const auto aliases = tensor_alias_detail::parseUidKeyedArray<TensorAlias>(
        json,
        "aliases",
        &tensor_alias_detail::parseEntry,
        [](const TensorAlias& alias) { return alias.uid; },
        [](const TensorAlias& alias) {
            return tensor_alias_detail::describe(alias)
                   + ": the view appears in more than one entry";
        });

    for(const auto& alias : aliases)
    {
        for(const auto& other : aliases)
        {
            if(other.ofUid == alias.uid)
            {
                throw std::invalid_argument(
                    tensor_alias_detail::describe(alias) + ": the view is also the owner of view "
                    + std::to_string(other.uid) + "; an alias cannot point at another alias");
            }
        }
    }
    return aliases;
}

inline nlohmann::json toJson(const std::vector<TensorAlias>& aliases)
{
    auto json = nlohmann::json::array();
    for(const auto& alias : aliases)
    {
        json.push_back(
            {{"uid", alias.uid}, {"of", alias.ofUid}, {"byte_offset", alias.byteOffset}});
    }
    return json;
}

// Checks each alias against the graph's tensors. Throws std::invalid_argument, naming
// the view, the owner and the rule, when a view or an owner is not an input tensor of
// the graph that holds its own data (it is virtual, an output, passed by value at run
// time, a constant, ragged or of a sub-byte type), when either has a negative dim, a
// negative stride, unequal dims and strides counts or no element size, when the two
// data types differ, when byte_offset is not a multiple of the view's element size or
// alignment, when the owner has a zero dim, or when the view reaches past the end of
// the owner's allocation. A view with a zero dim reaches no bytes, but its byte_offset
// must still be inside the owner's allocation. `graphFillDerivesInputs` is
// fillDerivesInputs() of the graph: a fill that derives inputs from other inputs needs
// each input as its own tensor, so any alias is rejected.
inline void checkTensorAliases(
    const std::vector<TensorAlias>& aliases,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensors,
    const std::vector<int64_t>& outputUids,
    bool graphFillDerivesInputs)
{
    const std::unordered_set<int64_t> outputs(outputUids.begin(), outputUids.end());
    for(const auto& alias : aliases)
    {
        tensor_alias_detail::checkAlias(alias, tensors, outputs);
    }
    if(graphFillDerivesInputs && !aliases.empty())
    {
        throw std::invalid_argument(
            tensor_alias_detail::describe(aliases.front())
            + ": the graph's input fill derives some inputs from other inputs, and that needs "
              "every input as its own tensor");
    }
}

} // namespace hipdnn_integration_tests
