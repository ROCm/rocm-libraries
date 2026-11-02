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
                           {"pack_gqa", 0},
                           {"bias_kstrided", 0},
                           {"has_dropout", 0},
                           {"v_mask", 0}}};
const CdnaEngine DECODE{"hipkernel:flydsl_sdpa_cdna_decode",
                        "hipkernel.flydsl_sdpa_cdna_decode.graph_match",
                        false,
                        {{"any_diagonal", 1},
                         {"kv_align", 0},
                         {"split_kv", 1},
                         {"pack_gqa", 1},
                         {"bias_kstrided", 0},
                         {"has_dropout", 0},
                         {"v_mask", 0}}};

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
    // Per graph class -- (dtype, head_dim, head_dim_v, causal, has_bias, has_sink,
    // has_dropout, v_mask) -- the priorities of the objects that serve every shape of it. A
    // pair bucket (v_mask) serves only graphs whose two head dims differ, a class of its own.
    using ClassKey
        = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
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
                                                  field("has_sink"),
                                                  field("has_dropout"),
                                                  field("v_mask"));
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
                          << std::get<4>(key) << " sink " << std::get<5>(key) << " dropout "
                          << std::get<6>(key) << " v_mask " << std::get<7>(key)
                          << ": no object serves any key length and diagonal";
            continue;
        }
        const auto top = *std::max_element(priorities.begin(), priorities.end());
        EXPECT_EQ(std::count(priorities.begin(), priorities.end(), top), 1)
            << std::get<0>(key) << " head_dim " << std::get<1>(key) << "/" << std::get<2>(key)
            << " causal " << std::get<3>(key) << " bias " << std::get<4>(key) << " sink "
            << std::get<5>(key) << " dropout " << std::get<6>(key) << " v_mask " << std::get<7>(key)
            << ": two serve-all objects tie at the top";
    }
}

/// Dual-wave: 12 tensors, then 8 i32, the f32 scale, the two head counts, lse_on,
/// diag_offset, seven operand strides and two bias strides. Generic (prefill and decode
/// alike): 5 tensors, 2 i32, 3 tensors, then 1 i32, the f32 scale, the two head counts, the
/// two token strides, lse_on, diag_offset and the same seven strides, the window, the bias,
/// its three strides, V's three strides and the Q/K head dim, then the split workspace's two
/// tensors and the split count, then the sink tensor, its head stride and the sink switch,
/// then the LSE's three strides, then the bias's key stride and f32 switch (vendored-kernel
/// modification 31), then four tensors (packed sequences' row offsets, the dropout seed and
/// offset) and four scalars (the offset's two by-value words, the keep threshold, the scale;
/// modification 32), then V's own group size, Hq / Hv (modification 42). Every scalar slot is
/// 4 bytes.
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
        scalars(2);
        tensors(4);
        scalars(4);
        scalars(1);
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

/// The exact-f32 objects (vendored-kernel modification 35) on every arch, on the generic kernel
/// only, each causal and not: the plain buckets up to 64, 128 and 256 (and on gfx942 and gfx90a
/// head_dim 256 exactly, both splitting O's columns there); and every bucket's builds with
/// dropout, a bias read one key at a time, and both (AOTriton's fp32 suite). No sink, V following
/// Q/K.
TEST(TestFlydslSdpaCdnaPacks, ShipsTheF32Objects)
{
    const auto& set = loadedSet(GENERIC.name);
    using Class = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
    std::map<std::string, std::multiset<Class>> classes;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            if(*dtype != "f32")
            {
                continue;
            }
            const auto value = [&kernel](const char* name) {
                const auto* field = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return field == nullptr ? int64_t{-1} : *field;
            };
            EXPECT_EQ(value("has_sink"), 0) << kernel.name;
            EXPECT_EQ(value("has_bias"), value("bias_kstrided")) << kernel.name;
            EXPECT_EQ(value("head_dim_v"), value("head_dim")) << kernel.name;
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                classes[arch].emplace(value("head_dim"),
                                      value("qk_mask"),
                                      value("causal"),
                                      value("has_dropout"),
                                      value("bias_kstrided"));
            }
        }
    }
    ASSERT_FALSE(classes.empty()) << "no f32 object loaded";
    for(const auto& [arch, shipped] : classes)
    {
        std::multiset<Class> expected;
        for(const int64_t causal : {0, 1})
        {
            for(const int64_t bucket : {64, 128, 256})
            {
                for(const auto& [dropout, kstrided] :
                    {std::pair<int64_t, int64_t>{0, 0}, {1, 0}, {0, 1}, {1, 1}})
                {
                    expected.emplace(bucket, 1, causal, dropout, kstrided);
                }
            }
            if(arch != "gfx950")
            {
                expected.emplace(256, 0, causal, 0, 0);
            }
        }
        EXPECT_EQ(shipped, expected) << arch;
    }
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
               || value("has_sink") != 0 || value("v_mask") != 0)
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

/// The buckets past 256 and the pair buckets, per arch and 16-bit dtype, causal and not: the
/// 512 bucket whose V follows Q/K with O split four ways (vendored-kernel modification 39), and
/// the pair buckets 64, 128, 256 and 512 whose V has a width of its own (modification 40), as
/// `_instances.SDPA_CDNA_PAIR_BUCKETS` and `SDPA_CDNA_WIDE_BUCKET_TIERS` list them. Each width
/// ships three ways, the generator's rows for them: plain, and with a bias (the parity gaps E4,
/// E5 and E6) read vectorized or one key at a time. None has a sink or dropout. Within one way
/// the pair buckets rank narrower first, and a bias read one key at a time ranks below the
/// vectorized read of the same bucket, which serves the same graphs.
TEST(TestFlydslSdpaCdnaPacks, ShipsTheWideAndPairBuckets)
{
    const auto& set = loadedSet(GENERIC.name);
    // (has_bias, bias_kstrided): the three ways each bucket ships.
    using Bias = std::pair<int64_t, int64_t>;
    const std::set<Bias> ways = {{0, 0}, {1, 0}, {1, 1}};
    using Key = std::tuple<std::string, std::string, int64_t, Bias>;
    std::map<Key, std::set<int64_t>> wide;
    std::map<Key, std::map<int64_t, int64_t>> pairs;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto value = [&kernel](const char* name) {
                const auto* field = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return field == nullptr ? int64_t{-1} : *field;
            };
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            const auto headDim = value("head_dim");
            const bool pair = value("v_mask") == 1;
            if(value("qk_mask") != 1 || (!pair && headDim <= 256))
            {
                continue;
            }
            const Bias bias{value("has_bias"), value("bias_kstrided")};
            EXPECT_EQ(ways.count(bias), 1U) << kernel.name << ": bias " << bias.first
                                            << " read one key at a time " << bias.second;
            EXPECT_EQ(value("head_dim_v"), headDim) << kernel.name;
            EXPECT_EQ(value("has_sink"), 0) << kernel.name;
            EXPECT_EQ(value("has_dropout"), 0) << kernel.name;
            EXPECT_EQ(value("dv_split"), headDim == 512 ? 4 : 1) << kernel.name;
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                const Key key{arch, *dtype, value("causal"), bias};
                if(pair)
                {
                    EXPECT_EQ(pairs[key].count(headDim), 0U) << kernel.name;
                    pairs[key][headDim] = static_cast<int64_t>(kernel.priority);
                }
                else
                {
                    EXPECT_EQ(wide[key].count(headDim), 0U) << kernel.name;
                    wide[key].insert(headDim);
                }
            }
        }
    }
    ASSERT_FALSE(pairs.empty()) << "no pair bucket loaded";
    std::set<std::tuple<std::string, std::string, int64_t>> classes;
    for(const auto& [key, widths] : pairs)
    {
        const auto& [arch, dtype, causal, bias] = key;
        std::ostringstream text;
        text << arch << " " << dtype << " causal " << causal << " bias " << bias.first
             << " read one key at a time " << bias.second;
        const auto where = text.str();
        if(dtype == "f32")
        {
            ADD_FAILURE() << where << ": an f32 pair bucket";
            continue;
        }
        classes.emplace(arch, dtype, causal);
        std::set<int64_t> shipped;
        int64_t previous = std::numeric_limits<int64_t>::max();
        for(const auto& [width, priority] : widths) // ascending width
        {
            shipped.insert(width);
            EXPECT_LT(priority, previous)
                << where << ": pair bucket " << width << " outranks a narrower";
            previous = priority;
        }
        EXPECT_EQ(shipped, (std::set<int64_t>{64, 128, 256, 512})) << where;
        EXPECT_EQ(wide[key], (std::set<int64_t>{512})) << where;
        if(bias == Bias{1, 1})
        {
            const auto vectorized = pairs.find(Key{arch, dtype, causal, Bias{1, 0}});
            ASSERT_NE(vectorized, pairs.end()) << where << ": no vectorized bias read";
            for(const auto& [width, priority] : widths)
            {
                const auto other = vectorized->second.find(width);
                ASSERT_NE(other, vectorized->second.end()) << where << " pair bucket " << width;
                EXPECT_LT(priority, other->second)
                    << where << ": pair bucket " << width << " outranks its vectorized read";
            }
        }
    }
    // Every class ships all three ways, and no wide bucket ships apart from a pair bucket.
    EXPECT_EQ(pairs.size(), classes.size() * ways.size());
    EXPECT_EQ(wide.size(), pairs.size());
}

/// The plain head_dim 256 prefill objects (16-bit, V as wide, no mask, bias, sink or dropout)
/// on the 64 KiB-LDS parts, gfx942 and gfx90a, stage K in 128-column chunks without the KV
/// global prefetch, which keeps their K and transposed-V tiles padded; every class ships them
/// (causal and not, the non-causal key-aligned build too). gfx950's 160 KiB holds the whole K
/// tile padded, so none of its objects chunks K there.
TEST(TestFlydslSdpaCdnaPacks, StagesTheHeadDim256KeysInChunksOn64KiBParts)
{
    const auto& set = loadedSet(GENERIC.name);
    using Class = std::tuple<std::string, int64_t, int64_t>;
    std::map<std::string, std::set<Class>> classes;
    size_t plain = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto value = [&kernel](const char* name) {
                const auto* field = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return field == nullptr ? int64_t{-1} : *field;
            };
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            if(*dtype == "f32" || value("head_dim") != 256 || value("head_dim_v") != 256
               || value("qk_mask") != 0 || value("has_bias") != 0 || value("has_sink") != 0
               || value("has_dropout") != 0)
            {
                continue;
            }
            ++plain;
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            for(const auto& arch : arches)
            {
                const bool smallLds = arch == "gfx942" || arch == "gfx90a";
                EXPECT_EQ(value("k_chunk"), smallLds ? 128 : 0) << arch << " " << kernel.name;
                if(smallLds)
                {
                    EXPECT_EQ(value("kv_gpfetch"), 0) << arch << " " << kernel.name;
                    classes[arch].emplace(*dtype, value("causal"), value("kv_align"));
                }
            }
        }
    }
    EXPECT_GT(plain, 0U) << "no plain head_dim 256 object loaded";
    const std::set<Class> every = {{"bf16", 0, 0},
                                   {"bf16", 0, 64},
                                   {"bf16", 1, 0},
                                   {"f16", 0, 0},
                                   {"f16", 0, 64},
                                   {"f16", 1, 0}};
    for(const auto& [arch, shipped] : classes)
    {
        EXPECT_EQ(shipped, every) << arch;
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

/// The dual-wave kernel's row tile follows the head dim: 256 rows (8 waves) at 64 and 128, 128
/// rows (4 waves, one per SIMD) at 256 (vendored-kernel modification 43). head_dim 256 ships in
/// both dtypes, causal and not, with no bias.
TEST(TestFlydslSdpaCdnaDualwavePacks, ServesHeadDim256OnFourWaves)
{
    const auto& set = loadedSet(DUALWAVE.name);
    std::set<std::pair<std::string, int64_t>> wide;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto value = [&kernel](const char* name) {
                const auto* field = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return field == nullptr ? int64_t{-1} : *field;
            };
            const auto* dtype = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
            ASSERT_NE(dtype, nullptr) << kernel.name;
            if(value("head_dim") != 256)
            {
                EXPECT_EQ(value("block_m"), 256) << kernel.name;
                continue;
            }
            EXPECT_EQ(value("block_m"), 128) << kernel.name;
            EXPECT_EQ(value("waves_per_eu"), 1) << kernel.name;
            EXPECT_EQ(value("has_bias"), 0) << kernel.name;
            wide.emplace(*dtype, value("causal"));
        }
    }
    const std::set<std::pair<std::string, int64_t>> every
        = {{"bf16", 0}, {"bf16", 1}, {"f16", 0}, {"f16", 1}};
    EXPECT_EQ(wide, every);
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
