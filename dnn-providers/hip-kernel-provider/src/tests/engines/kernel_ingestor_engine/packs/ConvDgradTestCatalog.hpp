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

#include "engines/kernel_ingestor_engine/packs/ConvDgradAotGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvDgradTestVectors.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestVectors.hpp"

/// @file ConvDgradTestCatalog.hpp
/// The generated catalogs of Gfx950ConvDgradTestVectors.hpp and
/// Gfx1250ConvDgradTestVectors.hpp, decoded for the dgrad AOT engine tests: each
/// instance's KMD metadata as name -> int or string (KMD defaults applied), read through
/// dgrad::parseAotInstance as the native pack of its engine reads a descriptor.
namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog
{

namespace dgrad = conv_dgrad_aot;
namespace vectors = dgrad_test_vectors;

using FieldValue = std::variant<int64_t, std::string>;
using Fields = std::map<std::string, FieldValue>;

/// "name=value;..." into fields, ints except @p stringFields.
template <typename StringFields>
Fields parseFields(std::string_view text, const StringFields& stringFields)
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
            = std::find(stringFields.begin(), stringFields.end(), name) != stringFields.end();
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

/// One engine's generated catalog by dtype: its instances (@p instances, KMD @p defaults,
/// @p stringFields of one *_test_vectors namespace), read for @p target.
template <typename Instances, typename StringFields>
std::map<std::string, std::vector<Entry>> decodeCatalog(const Instances& instances,
                                                        const char* defaults,
                                                        const StringFields& stringFields,
                                                        const dgrad::Target& target)
{
    std::map<std::string, std::vector<Entry>> byDtype;
    for(const auto& instance : instances)
    {
        // The KMD defaults, then the fields the instance states.
        auto fields = parseFields(defaults, stringFields);
        for(auto& [name, value] : parseFields(instance.fields, stringFields))
        {
            fields[name] = std::move(value);
        }
        Entry entry{instance.dtype, std::move(fields), std::nullopt};
        entry.instance = dgrad::parseAotInstance(FieldsMetadata{entry.fields}, target);
        byDtype[instance.dtype].push_back(std::move(entry));
    }
    return byDtype;
}

/// The catalog instances of @p dtype of @p target's engine, in the generated (and
/// served-matrix) order.
inline const std::vector<Entry>& catalog(const std::string& dtype,
                                         const dgrad::Target& target = dgrad::GFX950)
{
    static const auto s_gfx950 = decodeCatalog(vectors::catalogInstances(),
                                               vectors::DEFAULT_FIELDS,
                                               vectors::STRING_FIELDS,
                                               dgrad::GFX950);
    static const auto s_gfx1250 = decodeCatalog(gfx1250_dgrad_test_vectors::catalogInstances(),
                                                gfx1250_dgrad_test_vectors::DEFAULT_FIELDS,
                                                gfx1250_dgrad_test_vectors::STRING_FIELDS,
                                                dgrad::GFX1250);
    return (target.arch == dgrad::GFX1250.arch ? s_gfx1250 : s_gfx950).at(dtype);
}

/// The first catalog instance of @p dtype of @p target's engine @p pick accepts.
inline const Entry& firstEntry(const std::string& dtype,
                               const std::function<bool(const Entry&)>& pick,
                               const dgrad::Target& target = dgrad::GFX950)
{
    const auto& entries = catalog(dtype, target);
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

/// A problem as a test failure prints it.
inline std::string describe(const dgrad::Problem& p)
{
    return "{N" + std::to_string(p.n) + " C" + std::to_string(p.c) + " K" + std::to_string(p.k)
           + " " + std::to_string(p.hi) + "x" + std::to_string(p.wi) + " f" + std::to_string(p.y)
           + "x" + std::to_string(p.x) + " s" + std::to_string(p.strideH) + "x"
           + std::to_string(p.strideW) + " p" + std::to_string(p.padH) + "x"
           + std::to_string(p.padW) + " d" + std::to_string(p.dilationH) + "x"
           + std::to_string(p.dilationW) + " g" + std::to_string(p.groups) + "}";
}

/// The FNV-1a fold of a served-matrix row's plan digests, as
/// make_conv_dgrad_aot_test_vectors.py writes it: start at FNV_OFFSET, fold each served
/// plan's planDigest byte by byte, little-endian.
inline constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;

inline void foldPlanDigest(uint64_t& folded, uint64_t digest)
{
    for(int byte = 0; byte < 8; ++byte)
    {
        folded ^= (digest >> (8 * byte)) & 0xFFU;
        folded *= 0x100000001B3ULL;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog
