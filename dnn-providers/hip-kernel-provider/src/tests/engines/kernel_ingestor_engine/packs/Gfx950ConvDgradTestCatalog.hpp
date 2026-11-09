// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestVectors.hpp"

/// @file Gfx950ConvDgradTestCatalog.hpp
/// The generated catalog of Gfx950ConvDgradTestVectors.hpp, decoded for the
/// hipkernel:Gfx950ConvDgradAot tests: each instance's KMD metadata as name -> int or
/// string (KMD defaults applied), read through dgrad::parseAotInstance as the native pack
/// reads a descriptor.
namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog
{

namespace dgrad = gfx950_conv_dgrad;
namespace vectors = dgrad_test_vectors;

using FieldValue = std::variant<int64_t, std::string>;
using Fields = std::map<std::string, FieldValue>;

/// "name=value;..." into fields, ints except vectors::STRING_FIELDS.
inline Fields parseFields(std::string_view text)
{
    Fields fields;
    size_t position = 0;
    while(position < text.size())
    {
        auto end = text.find(';', position);
        if(end == std::string_view::npos)
        {
            end = text.size();
        }
        const auto item = text.substr(position, end - position);
        const auto equals = item.find('=');
        const auto name = std::string(item.substr(0, equals));
        const auto value = std::string(item.substr(equals + 1));
        const bool isString
            = std::find(vectors::STRING_FIELDS.begin(), vectors::STRING_FIELDS.end(), name)
              != vectors::STRING_FIELDS.end();
        fields[name] = isString ? FieldValue(value) : FieldValue(std::stoll(value));
        position = end + 1;
    }
    return fields;
}

/// Fields as parseAotInstance reads metadata.
struct FieldsMetadata
{
    const Fields& fields;

    std::optional<int64_t> intField(std::string_view name) const
    {
        const auto it = fields.find(std::string(name));
        const auto* value = it == fields.end() ? nullptr : std::get_if<int64_t>(&it->second);
        return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
    }

    const std::string* stringField(std::string_view name) const
    {
        const auto it = fields.find(std::string(name));
        return it == fields.end() ? nullptr : std::get_if<std::string>(&it->second);
    }
};

/// One decoded catalog instance.
struct Entry
{
    std::string dtype;
    Fields fields;
    std::optional<dgrad::AotInstance> instance;

    int64_t integer(const std::string& name) const
    {
        return std::get<int64_t>(fields.at(name));
    }

    const std::string& text(const std::string& name) const
    {
        return std::get<std::string>(fields.at(name));
    }

    bool isHelper() const
    {
        return integer("kernel_family") == 2;
    }
};

/// The catalog instances of @p dtype, in the generated (and served-matrix) order.
inline const std::vector<Entry>& catalog(const std::string& dtype)
{
    static const auto s_catalog = [] {
        std::map<std::string, std::vector<Entry>> byDtype;
        for(const auto& instance : vectors::catalogInstances())
        {
            // The KMD defaults, then the fields the instance states.
            auto fields = parseFields(vectors::DEFAULT_FIELDS);
            for(auto& [name, value] : parseFields(instance.fields))
            {
                fields[name] = std::move(value);
            }
            Entry entry{instance.dtype, std::move(fields), std::nullopt};
            entry.instance = dgrad::parseAotInstance(FieldsMetadata{entry.fields});
            byDtype[instance.dtype].push_back(std::move(entry));
        }
        return byDtype;
    }();
    return s_catalog.at(dtype);
}

/// The first catalog instance of @p dtype @p pick accepts.
inline const Entry& firstEntry(const std::string& dtype,
                               const std::function<bool(const Entry&)>& pick)
{
    const auto& entries = catalog(dtype);
    const auto it = std::find_if(entries.begin(), entries.end(), pick);
    if(it == entries.end())
    {
        throw std::runtime_error("no catalog instance matches");
    }
    return *it;
}

inline dgrad::Problem toProblem(const vectors::ProblemValues& v)
{
    return dgrad::Problem{
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog
