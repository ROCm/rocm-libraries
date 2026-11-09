// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaCdnaPacks.cpp
 * @brief A census of what the three CDNA SDPA engines actually load, read from the shards the
 *        census entries name rather than from anything built in-process.
 *
 * One kernel family per engine: the generic MFMA kernel's prefill objects
 * (hipkernel:flydsl_sdpa_cdna), gfx950's dual-wave kernel (..._cdna_dualwave) and the
 * generic kernel's split-KV decode objects (..._cdna_decode). One suite each, over the
 * shared checks below.
 *
 * Every case opens by reading loadedSet(), none skips, and none touches a device, so an
 * empty shard fails here instead of passing silently. The seam suites -- matchers, bindings
 * and the real launches -- are in TestFlydslSdpaCdnaEngine.cpp.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt, each suite over
 * the architectures its engine's objects ship for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

/// One tensor parameter: a pointer and FlyDSL's 40-byte rank-4 dynamic layout.
constexpr uint32_t POINTER_BYTES = 8;
constexpr uint32_t LAYOUT_BYTES = 40;
constexpr uint32_t I32_BYTES = 4;

/// One engine's contract, and the knobs its family does not carry because they hold one
/// value across it (`_instances.SDPA_CDNA_FAMILY_CONSTANTS`).
struct CdnaEngine
{
    const char* name;
    const char* graphMatcher;
    bool dualwave;
    std::map<std::string, int64_t> constants;
};

const CdnaEngine GENERIC{"hipkernel:flydsl_sdpa_cdna",
                         "hipkernel.flydsl_sdpa_cdna.graph_match",
                         false,
                         {{"any_diagonal", 1}, {"split_kv", 0}, {"pack_gqa", 0}}};
const CdnaEngine DUALWAVE{"hipkernel:flydsl_sdpa_cdna_dualwave",
                          "hipkernel.flydsl_sdpa_cdna_dualwave.graph_match",
                          true,
                          {{"kv_align", 0},
                           {"qk_mask", 0},
                           {"has_sink", 0},
                           {"dv_split", 1},
                           {"split_kv", 0},
                           {"pack_gqa", 0}}};
const CdnaEngine DECODE{"hipkernel:flydsl_sdpa_cdna_decode",
                        "hipkernel.flydsl_sdpa_cdna_decode.graph_match",
                        false,
                        {{"any_diagonal", 1}, {"kv_align", 0}, {"split_kv", 1}, {"pack_gqa", 1}}};

void expectOnePackWithAGraphMatchAndOneKernelScopedMatcher(const CdnaEngine& engine)
{
    const auto& set = loadedSet(engine.name);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, engine.graphMatcher);

    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

/// Every object built once, and per graph class one serve-all object -- one taking any key
/// length and any diagonal -- that outranks the rest, so the untuned pick never depends on a
/// key length or a diagonal and every other object is a measured companion below it.
void expectUniqueObjectsAndOneTopServeAllPerClass(const CdnaEngine& engine)
{
    const auto& set = loadedSet(engine.name);

    // Everything an object was built with, in one string: a duplicate would make the choice
    // between two objects an accident of ids.
    std::set<std::string> builds;
    // Per graph class -- (dtype, head_dim, head_dim_v, causal, has_bias, has_sink) -- the
    // priorities of the objects that serve every shape of it.
    using ClassKey = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>;
    std::map<ClassKey, std::vector<int64_t>> serveAll;
    const std::set<std::string> features = {"none", "rows", "seq_q"};
    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto field = [&](const char* name) {
                const auto constant = engine.constants.find(name);
                if(constant != engine.constants.end())
                {
                    EXPECT_EQ(kernel.metadata.count(name), 0U)
                        << kernel.name << " carries " << name << ", its family's constant";
                    return constant->second;
                }
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, name);
                EXPECT_NE(value, nullptr) << kernel.name << " has no " << name;
                return value == nullptr ? int64_t{-1} : *value;
            };
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            EXPECT_EQ(kernel.metadata.count("kernel"), 0U)
                << kernel.name << ": an engine is one kernel family, so it needs no kernel knob";
            const auto* feature
                = tryGetMetadataField<std::string>(kernel.metadata, "prefer_feature");
            if(feature != nullptr)
            {
                EXPECT_EQ(features.count(*feature), 1U) << kernel.name << ": " << *feature;
            }

            std::ostringstream build;
            for(const auto& [key, value] : kernel.metadata)
            {
                build << key << '=';
                std::visit(
                    [&build](const auto& v) {
                        if constexpr(std::is_same_v<std::decay_t<decltype(v)>,
                                                    std::vector<int64_t>>)
                        {
                            for(const auto element : v)
                            {
                                build << element << ',';
                            }
                        }
                        else
                        {
                            build << v;
                        }
                    },
                    value);
                build << ';';
            }
            builds.insert(build.str());

            const auto classKey = std::make_tuple(*dtype,
                                                  field("head_dim"),
                                                  field("head_dim_v"),
                                                  field("causal"),
                                                  field("has_bias"),
                                                  field("has_sink"));
            // Every class, so one whose objects all need an aligned key length (or offset 0)
            // is caught for having no object that serves any graph of it.
            serveAll[classKey];
            if(field("kv_align") == 0 && field("any_diagonal") == 1)
            {
                serveAll[classKey].push_back(static_cast<int64_t>(kernel.priority));
            }
            ++kernels;

            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }
    EXPECT_GT(kernels, 0U);
    EXPECT_EQ(kernels, builds.size()) << "two objects share one build";
    for(const auto& [key, priorities] : serveAll)
    {
        if(priorities.empty())
        {
            ADD_FAILURE() << std::get<0>(key) << " head_dim " << std::get<1>(key) << "/"
                          << std::get<2>(key) << " causal " << std::get<3>(key) << " bias "
                          << std::get<4>(key) << " sink " << std::get<5>(key)
                          << ": no object serves any key length and diagonal";
            continue;
        }
        const auto top = *std::max_element(priorities.begin(), priorities.end());
        EXPECT_EQ(std::count(priorities.begin(), priorities.end(), top), 1)
            << std::get<0>(key) << " head_dim " << std::get<1>(key) << "/" << std::get<2>(key)
            << " causal " << std::get<3>(key) << " bias " << std::get<4>(key) << " sink "
            << std::get<5>(key) << ": two serve-all objects tie at the top";
    }
}

/// Dual-wave: 12 tensors, then 8 i32, the f32 scale, the two head counts, lse_on,
/// diag_offset, seven operand strides and two bias strides. Generic (prefill and decode
/// alike): 5 tensors, 2 i32, 3 tensors, then 1 i32, the f32 scale, the two head counts, the
/// two token strides, lse_on, diag_offset and the same seven strides, the window, the bias,
/// its three strides, V's three strides and the Q/K head dim, then the split workspace's two
/// tensors and the split count, then the sink tensor, its head stride and the sink switch,
/// then the LSE's three strides. Every scalar slot is 4 bytes.
void expectEveryKernelDeclaresItsFamilysSignature(const CdnaEngine& engine)
{
    const auto& set = loadedSet(engine.name);

    using Slot = std::pair<std::string, uint32_t>;
    std::vector<Slot> slots;
    const auto tensors = [&slots](int count) {
        for(int i = 0; i < count; ++i)
        {
            slots.emplace_back("global_buffer", POINTER_BYTES);
            slots.emplace_back("by_value", LAYOUT_BYTES);
        }
    };
    const auto scalars = [&slots](int count) {
        for(int i = 0; i < count; ++i)
        {
            slots.emplace_back("by_value", I32_BYTES);
        }
    };
    if(engine.dualwave)
    {
        tensors(12);
        scalars(22);
    }
    else
    {
        tensors(5);
        scalars(2);
        tensors(3);
        scalars(16);
        tensors(1);
        scalars(7);
        tensors(2);
        scalars(1);
        tensors(1);
        scalars(2);
        scalars(3);
    }

    size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), slots.size()) << kernel.name;
            for(size_t i = 0; i < slots.size(); ++i)
            {
                EXPECT_EQ(signature[i].kind, slots[i].first) << kernel.name << " argument " << i;
                EXPECT_EQ(signature[i].size, slots[i].second) << kernel.name << " argument " << i;
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0U);
}

TEST(TestFlydslSdpaCdnaPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    expectOnePackWithAGraphMatchAndOneKernelScopedMatcher(GENERIC);
}

TEST(TestFlydslSdpaCdnaPacks, ShipsUniqueObjectsAndOneTopServeAllPerClass)
{
    expectUniqueObjectsAndOneTopServeAllPerClass(GENERIC);
}

/// Among the generic kernel's bucket objects (qk_mask, V following Q/K, no bias or sink) of
/// one arch, dtype and causal flag, each width's default outranks every wider width's, and
/// its companions rank below every width's default: a graph several buckets serve gets the
/// narrowest, which does the least wasted matrix work, unless autotune measures otherwise.
TEST(TestFlydslSdpaCdnaPacks, RanksTheNarrowerBucketFirst)
{
    const auto& set = loadedSet(GENERIC.name);
    using Key = std::tuple<std::string, std::string, int64_t>;
    std::map<Key, std::map<int64_t, std::vector<int64_t>>> byWidth;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto value = [&kernel](const char* name) {
                const auto* field = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return field == nullptr ? int64_t{-1} : *field;
            };
            const auto headDim = value("head_dim");
            if(value("qk_mask") != 1 || value("head_dim_v") != headDim || value("has_bias") != 0
               || value("has_sink") != 0)
            {
                continue;
            }
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                byWidth[{arch, *dtype, value("causal")}][headDim].push_back(
                    static_cast<int64_t>(kernel.priority));
            }
        }
    }
    for(const auto& [key, widths] : byWidth)
    {
        int64_t lowestDefault = std::numeric_limits<int64_t>::max();
        int64_t highestCompanion = std::numeric_limits<int64_t>::min();
        int64_t previous = std::numeric_limits<int64_t>::max();
        for(const auto& [width, priorities] : widths) // ascending width
        {
            const auto top = *std::max_element(priorities.begin(), priorities.end());
            EXPECT_LT(top, previous)
                << std::get<0>(key) << " " << std::get<1>(key) << " causal " << std::get<2>(key)
                << ": bucket " << width << " does not rank below the narrower one";
            previous = top;
            lowestDefault = std::min(lowestDefault, top);
            for(const auto priority : priorities)
            {
                if(priority != top)
                {
                    highestCompanion = std::max(highestCompanion, priority);
                }
            }
        }
        EXPECT_LT(highestCompanion, lowestDefault)
            << std::get<0>(key) << " " << std::get<1>(key) << " causal " << std::get<2>(key)
            << ": a bucket companion outranks a bucket default";
    }
}

TEST(TestFlydslSdpaCdnaPacks, EveryKernelDeclaresItsFamilysSignature)
{
    expectEveryKernelDeclaresItsFamilysSignature(GENERIC);
}

TEST(TestFlydslSdpaCdnaDualwavePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    expectOnePackWithAGraphMatchAndOneKernelScopedMatcher(DUALWAVE);
}

TEST(TestFlydslSdpaCdnaDualwavePacks, ShipsUniqueObjectsAndOneTopServeAllPerClass)
{
    expectUniqueObjectsAndOneTopServeAllPerClass(DUALWAVE);
}

TEST(TestFlydslSdpaCdnaDualwavePacks, EveryKernelDeclaresItsFamilysSignature)
{
    expectEveryKernelDeclaresItsFamilysSignature(DUALWAVE);
}

TEST(TestFlydslSdpaCdnaDecodePacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    expectOnePackWithAGraphMatchAndOneKernelScopedMatcher(DECODE);
}

TEST(TestFlydslSdpaCdnaDecodePacks, ShipsUniqueObjectsAndOneTopServeAllPerClass)
{
    expectUniqueObjectsAndOneTopServeAllPerClass(DECODE);
}

TEST(TestFlydslSdpaCdnaDecodePacks, EveryKernelDeclaresItsFamilysSignature)
{
    expectEveryKernelDeclaresItsFamilysSignature(DECODE);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
