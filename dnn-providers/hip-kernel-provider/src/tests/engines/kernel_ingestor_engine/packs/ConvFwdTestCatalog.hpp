// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdAotTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx942ConvFwdTestVectors.hpp"

/// @file ConvFwdTestCatalog.hpp
/// The generated catalog of Gfx942ConvFwdTestVectors.hpp, decoded for the forward AOT
/// engine tests: each instance's KMD metadata as name -> int or string (KMD defaults
/// applied), as the KernelDefinition the engine loads from its descriptor and as the
/// conv::AotInstance or conv::DirectInstance its fields spell; and the launch plan the
/// geometry header derives for an instance on a problem, in the values the generator
/// (tools/make_conv_aot_test_vectors.py, plan_values) writes for the adapter's.
namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_test_catalog
{

namespace conv = gfx950_conv_fwd;

using FieldValue = std::variant<int64_t, std::string>;
using Fields = std::map<std::string, FieldValue>;
using Instance = std::variant<conv::AotInstance, conv::DirectInstance>;

/// "name=value;..." into fields, ints except @p stringFields.
template <typename StringFields>
Fields parseNamedFields(std::string_view text, const StringFields& stringFields)
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

/// One decoded catalog instance.
struct Entry
{
    std::string dtype;
    Fields fields;
    /// What the fields spell, as the engine's metadata decode reads them; nullopt for a
    /// kernel_family or direct_variant the geometry has no launch for.
    std::optional<Instance> instance;

    int64_t integer(const std::string& name) const
    {
        return std::get<int64_t>(fields.at(name));
    }

    const std::string& text(const std::string& name) const
    {
        return std::get<std::string>(fields.at(name));
    }

    /// The kind a failure names (implicit GEMM or the direct variant, the nongrouped atom,
    /// a group merge), then the instance id.
    std::string kind() const
    {
        auto kind
            = integer("kernel_family") == 0 ? std::string("implicit_gemm") : text("direct_variant");
        if(text("atom") != "none")
        {
            kind += " " + text("atom");
        }
        if(integer("kernel_family") == 0 && integer("group_merge") > 1)
        {
            kind += " gm" + std::to_string(integer("group_merge"));
        }
        return kind + " " + text("instance_id");
    }
};

/// The instance @p fields spell. The field-to-member mapping of the native pack's
/// metadata decode (aotImplicitGemmInstance, aotDirectInstance in Gfx950ConvFwdNative.cpp),
/// without its refusals: the engine's own decode is exercised through its kernel matcher.
inline std::optional<Instance> decodeInstance(const Entry& entry)
{
    if(entry.integer("kernel_family") == 0)
    {
        return conv::AotInstance{entry.integer("tile_m"),
                                 entry.integer("tile_n"),
                                 entry.integer("warp_m"),
                                 entry.integer("warp_n"),
                                 entry.integer("vector_size_a"),
                                 entry.integer("vector_size_b"),
                                 entry.integer("vector_size_c"),
                                 entry.integer("async_dma") == 1,
                                 entry.integer("async_chunk_a"),
                                 entry.integer("async_chunk_b"),
                                 entry.integer("grouped") == 1,
                                 entry.integer("group_merge")};
    }
    const auto variant = conv::parseDirectVariant(entry.text("direct_variant"));
    if(entry.integer("kernel_family") != 1 || !variant)
    {
        return std::nullopt;
    }
    conv::DirectInstance direct{{entry.integer("filter_h"),
                                 entry.integer("filter_w"),
                                 entry.integer("stride_h"),
                                 entry.integer("stride_w"),
                                 entry.integer("pad_h"),
                                 entry.integer("pad_w"),
                                 entry.integer("dilation_h"),
                                 entry.integer("dilation_w")},
                                *variant,
                                entry.integer("block_w"),
                                entry.integer("block_waves")};
    direct.blockH = entry.integer("block_h");
    direct.blockQ = entry.integer("block_q");
    direct.blockGroups = entry.integer("block_groups");
    direct.cpg = entry.integer("cpg");
    direct.kpg = entry.integer("kpg");
    direct.tileH = entry.integer("tile_h");
    direct.tileW = entry.integer("tile_w");
    direct.tileKOut = entry.integer("tile_k_out");
    direct.ck = entry.integer("ck");
    direct.wavesM = entry.integer("waves_m");
    direct.wavesN = entry.integer("waves_n");
    direct.atomTile = conv::nongroupedAtomTile(entry.text("atom")).value_or(0);
    direct.atomFragment = conv::nongroupedAtomFragment(entry.text("atom")).value_or(0);
    direct.groupMerge = entry.integer("group_merge");
    direct.splitC = entry.integer("split_c");
    direct.prefetchRows = entry.integer("prefetch_rows");
    direct.wFold = entry.integer("w_fold");
    return direct;
}

/// One engine's generated catalog by dtype: each instance's comma-separated values over
/// @p fieldNames, an empty value standing for the field's KMD default (@p defaults).
template <typename Instances, typename FieldNames, typename StringFields>
std::map<std::string, std::vector<Entry>> decodeCatalog(const Instances& instances,
                                                        const FieldNames& fieldNames,
                                                        const char* defaults,
                                                        const StringFields& stringFields)
{
    const auto defaultFields = parseNamedFields(defaults, stringFields);
    std::map<std::string, std::vector<Entry>> byDtype;
    for(const auto& instance : instances)
    {
        Fields fields;
        const std::string_view values(instance.values);
        size_t position = 0;
        for(const auto* name : fieldNames)
        {
            const auto end = std::min(values.find(',', position), values.size());
            const auto value = std::string(values.substr(position, end - position));
            position = end + 1;
            const bool isString
                = std::find(stringFields.begin(), stringFields.end(), std::string_view(name))
                  != stringFields.end();
            if(value.empty())
            {
                fields[name] = defaultFields.at(name);
            }
            else
            {
                fields[name] = isString ? FieldValue(value) : FieldValue(std::stoll(value));
            }
        }
        Entry entry{instance.dtype, std::move(fields), std::nullopt};
        entry.instance = decodeInstance(entry);
        byDtype[instance.dtype].push_back(std::move(entry));
    }
    return byDtype;
}

/// The gfx942 catalog instances of @p dtype, in the generated (and served-matrix) order.
inline const std::vector<Entry>& gfx942Catalog(const std::string& dtype)
{
    namespace vectors = gfx942_fwd_test_vectors;
    static const auto s_catalog = decodeCatalog(vectors::catalogInstances(),
                                                vectors::FIELDS,
                                                vectors::DEFAULT_FIELDS,
                                                vectors::STRING_FIELDS);
    return s_catalog.at(dtype);
}

/// @p entry as the engine's loader hands it a kernel of a prebuilt (kpack) descriptor:
/// every KMD field in its metadata.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition kernelOf(const Entry& entry)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = entry.text("instance_id");
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
    for(const auto& [name, value] : entry.fields)
    {
        if(const auto* integer = std::get_if<int64_t>(&value))
        {
            kernel.metadata[name] = *integer;
        }
        else
        {
            kernel.metadata[name] = std::get<std::string>(value);
        }
    }
    return kernel;
}

inline conv::Problem toProblem(const std::array<int64_t, 14>& v)
{
    return conv::Problem{
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13]};
}

/// A problem as a test failure prints it.
inline std::string describe(const conv::Problem& p)
{
    return "{N" + std::to_string(p.n) + " C" + std::to_string(p.c) + " K" + std::to_string(p.k)
           + " " + std::to_string(p.hi) + "x" + std::to_string(p.wi) + " f" + std::to_string(p.y)
           + "x" + std::to_string(p.x) + " s" + std::to_string(p.strideH) + "x"
           + std::to_string(p.strideW) + " p" + std::to_string(p.padH) + "x"
           + std::to_string(p.padW) + " d" + std::to_string(p.dilationH) + "x"
           + std::to_string(p.dilationW) + " g" + std::to_string(p.groups) + "}";
}

/// Bit @p index of a served-matrix bit string: four bits to a hex digit, the first
/// instance the most significant bit of the first digit.
inline bool bitAt(std::string_view hex, size_t index)
{
    const auto digit = hex.at(index / 4);
    const int nibble = digit <= '9' ? digit - '0' : digit - 'a' + 10;
    return ((nibble >> (3 - static_cast<int>(index % 4))) & 1) != 0;
}

/// FNV-1a 64 over @p values, each as 8 little-endian bytes, as the generator's _fnv.
inline constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;

inline void fnvFold(uint64_t& folded, uint64_t value)
{
    for(int byte = 0; byte < 8; ++byte)
    {
        folded ^= (value >> (8 * byte)) & 0xFFU;
        folded *= 0x100000001B3ULL;
    }
}

inline uint64_t planDigest(const std::vector<int64_t>& values)
{
    uint64_t digest = FNV_OFFSET;
    for(const auto value : values)
    {
        fnvFold(digest, static_cast<uint64_t>(value));
    }
    return digest;
}

/// One launch as plan_values writes it: grid x, y, z, block x, the argument count and the
/// AOT problem block, each argument as the u32 the kernel reads.
template <typename Kernargs>
void appendLaunch(std::vector<int64_t>& values,
                  const conv::LaunchGeometry& launch,
                  const Kernargs& kernargs)
{
    values.insert(values.end(), {launch.gridX, launch.gridY, launch.gridZ, launch.blockX});
    values.push_back(static_cast<int64_t>(kernargs.size()));
    for(const auto argument : kernargs)
    {
        values.push_back(static_cast<int64_t>(static_cast<uint32_t>(argument)));
    }
}

/// The launch plan of @p instance on @p problem as the geometry header derives it, in
/// plan_values' layout, or std::nullopt when the instance does not serve the problem.
/// The native pack's composition (aotInstanceLaunch and batchSplitLaunch in
/// Gfx950ConvFwdNative.cpp): a non-grouped or grouped-halo kernel runs a batch
/// conv::nongroupedBatchChunks splits one launch per chunk, each its images as a problem
/// of its own (refused for a split-C kernel); every other kernel needs the whole problem
/// in one launch, through conv::directInstanceLaunch and conv::directKernargs or
/// conv::aotInstanceLaunch and conv::implicitGemmKernargs.
inline std::optional<std::vector<int64_t>> geometryPlan(const conv::Problem& problem,
                                                        const Instance& instance)
{
    std::vector<int64_t> values;
    const auto* direct = std::get_if<conv::DirectInstance>(&instance);
    const auto chunks = conv::nongroupedBatchChunks(problem);
    if(direct != nullptr && !chunks.empty()
       && (direct->variant == conv::DirectVariant::NONGROUPED
           || direct->variant == conv::DirectVariant::GROUPED_HALO))
    {
        if(direct->splitC != 1)
        {
            return std::nullopt;
        }
        values.push_back(static_cast<int64_t>(chunks.size()));
        for(const auto images : chunks)
        {
            auto sub = problem;
            sub.n = images;
            const auto geometry = conv::deriveGeometry(sub);
            const auto launch
                = geometry ? conv::directInstanceLaunch(sub, *geometry, *direct) : std::nullopt;
            if(!launch)
            {
                return std::nullopt;
            }
            values.push_back(images);
            appendLaunch(values, *launch, conv::directKernargs(sub, *geometry));
        }
        values.insert(values.end(), {0, 0});
        return values;
    }
    const auto geometry = conv::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    values.insert(values.end(), {0, problem.n});
    if(direct != nullptr)
    {
        const auto launch = conv::directInstanceLaunch(problem, *geometry, *direct);
        if(!launch)
        {
            return std::nullopt;
        }
        appendLaunch(values, *launch, conv::directKernargs(problem, *geometry));
        values.push_back(conv::splitWorkspaceBytes(problem, *geometry, *direct));
        if(direct->splitC > 1)
        {
            const auto reduce = conv::splitReduceLaunch(problem, *geometry);
            values.insert(values.end(),
                          {1, reduce.gridX, reduce.gridY, reduce.gridZ, reduce.blockX});
        }
        else
        {
            values.push_back(0);
        }
        return values;
    }
    const auto launch
        = conv::aotInstanceLaunch(problem, *geometry, std::get<conv::AotInstance>(instance));
    const auto kernargs
        = launch ? conv::implicitGemmKernargs(problem, *geometry, *launch) : std::nullopt;
    if(!kernargs)
    {
        return std::nullopt;
    }
    appendLaunch(values, *launch, *kernargs);
    values.insert(values.end(), {0, 0});
    return values;
}

/// A packed NHWC forward graph for @p v (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW,
/// groups) in @p dtype, its tensors uids 1 (x), 2 (w) and 3 (y).
inline flatbuffers::FlatBufferBuilder convGraph(const std::array<int64_t, 14>& v,
                                                const std::string& dtype)
{
    return conv_fwd_aot_test::convGraph(
        v, conv_fwd_aot_test::dataTypeOf(dtype), "conv_fwd_test_catalog");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_test_catalog

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
