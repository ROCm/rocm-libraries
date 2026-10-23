// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_bwd_merged actually loads, read from the shard
 *        the census entry names rather than from anything built in-process.
 *
 * The merged table is 208 objects: 126 without a bias and 82 with one. The 28 bias objects
 * that store dB carry the dBias reduce kernel as a second symbol of the same code object; a
 * store_db=0 object never writes dB and carries no reduce kernel. Every case
 * opens by reading loadedSet(), none skips and none touches a device, so an empty shard fails
 * here instead of passing silently. The seam suite, over rows written in the test, is
 * TestFlydslSdpaBwdEngine.cpp.
 *
 * A descriptor carries the argument list of the merged symbol only. The reduce kernel's
 * 88-byte list (`_PARAMS["dbias_reduce"]`) is checked when the object is generated and, as the
 * engine launches it, by the seam suite; here every dB-storing bias row names the reduce symbol.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt, over the
 * architectures the op ships for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_merged";
constexpr const char* GRAPH_MATCHER = "hipkernel.flydsl_sdpa_bwd_merged.graph_match";
constexpr const char* REDUCE_SYMBOL = "fmha_bwd_dbias_reduce_gfx950_kernel_1";

constexpr size_t MERGED_ROWS = 208;
constexpr size_t BIAS_ROWS = 82;
constexpr size_t REDUCE_ROWS = 28;

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

TEST(TestFlydslSdpaBwdPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(ENGINE_NAME);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, GRAPH_MATCHER);
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaBwdPacks, ShipsEveryMergedRowOnce)
{
    const auto& set = loadedSet(ENGINE_NAME);

    const std::set<int64_t> tiles{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};
    const std::set<std::string> dtypes{"f16", "bf16"};
    const std::set<std::string> dforms{"rt", "bk"};
    const std::set<std::string> roleOrders{"auto", "concat", "lpt"};

    std::set<std::string> names;
    size_t kernels = 0;
    size_t biasRows = 0;
    size_t reduceRows = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++kernels;
            names.insert(kernel.name);
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;

            const auto integer = [&](const char* field) {
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
                EXPECT_NE(value, nullptr) << kernel.name << " lacks the integer field " << field;
                return value == nullptr ? int64_t{0} : *value;
            };
            const auto text = [&](const char* field) {
                const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
                EXPECT_NE(value, nullptr) << kernel.name << " lacks the string field " << field;
                return value == nullptr ? std::string() : *value;
            };
            EXPECT_EQ(dtypes.count(text("dtype")), 1U) << kernel.name;
            EXPECT_EQ(tiles.count(integer("head_dim")), 1U) << kernel.name;
            EXPECT_GT(integer("head_dim_v"), 0) << kernel.name;
            EXPECT_EQ(dforms.count(text("dform")), 1U) << kernel.name;
            EXPECT_EQ(roleOrders.count(text("role_order")), 1U) << kernel.name;
            EXPECT_FALSE(text("floor").empty()) << kernel.name;
            for(const char* flag : {"causal", "has_bias", "dropout", "padded", "hints"})
            {
                const auto value = integer(flag);
                EXPECT_TRUE(value == 0 || value == 1) << kernel.name << " " << flag;
            }
            const auto storeDb = integer("store_db");
            EXPECT_TRUE(storeDb == -1 || storeDb == 0) << kernel.name;
            const auto gqaPack = integer("gqa_pack");
            EXPECT_TRUE(gqaPack == -1 || gqaPack == 1) << kernel.name;
            for(const char* geometry : {"block_threads", "block_kv", "block_m"})
            {
                EXPECT_GT(integer(geometry), 0) << kernel.name << " " << geometry;
            }

            // A dB-storing bias object carries the dBias reduce kernel; the class is exclusive
            // of causal.
            const bool hasBias = integer("has_bias") == 1;
            const bool storesDb = hasBias && storeDb != 0;
            EXPECT_FALSE(hasBias && integer("causal") == 1) << kernel.name;
            EXPECT_EQ(text("reduce_symbol"), storesDb ? std::string(REDUCE_SYMBOL) : std::string())
                << kernel.name;
            biasRows += hasBias ? 1 : 0;
            reduceRows += storesDb ? 1 : 0;
        }
    }

    EXPECT_EQ(kernels, MERGED_ROWS);
    EXPECT_EQ(names.size(), MERGED_ROWS) << "a name is shipped twice";
    EXPECT_EQ(biasRows, BIAS_ROWS);
    EXPECT_EQ(reduceRows, REDUCE_ROWS);
}

TEST(TestFlydslSdpaBwdPacks, EveryKernelDeclaresTheMergedSignature)
{
    const auto& set = loadedSet(ENGINE_NAME);

    // `_PARAMS["merged"]`: eleven pointers (Q K V B O DO DK DV DQ DB LSE), batch_size, four
    // seqinfo pointers, six i32 (varlen_bits .. window_right), the two philox pointers, the i64
    // offset, i32 threshold, f32 scale, four i32 (heads, head dims), f32 sm_scale, thirty i64
    // strides and i32 rt_flags: 63 arguments, 452 bytes, each at its natural offset.
    struct Run
    {
        const char* kind;
        uint32_t size;
        size_t count;
    };
    const std::array<Run, 12> runs{{{"global_buffer", 8, 11},
                                    {"by_value", 4, 1},
                                    {"global_buffer", 8, 4},
                                    {"by_value", 4, 6},
                                    {"global_buffer", 8, 2},
                                    {"by_value", 8, 1},
                                    {"by_value", 4, 1},
                                    {"by_value", 4, 1},
                                    {"by_value", 4, 4},
                                    {"by_value", 4, 1},
                                    {"by_value", 8, 30},
                                    {"by_value", 4, 1}}};
    std::vector<std::pair<const char*, uint32_t>> slots;
    std::vector<uint32_t> offsets;
    uint32_t offset = 0;
    for(const auto& run : runs)
    {
        for(size_t item = 0; item < run.count; ++item)
        {
            offset = (offset + run.size - 1) / run.size * run.size;
            slots.emplace_back(run.kind, run.size);
            offsets.push_back(offset);
            offset += run.size;
        }
    }
    ASSERT_EQ(slots.size(), 63U);
    ASSERT_EQ(offset, 452U);

    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), slots.size()) << kernel.name;
            for(size_t slot = 0; slot < slots.size(); ++slot)
            {
                EXPECT_EQ(signature[slot].kind, slots[slot].first)
                    << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].size, slots[slot].second)
                    << kernel.name << " slot " << slot;
                EXPECT_EQ(signature[slot].offset, offsets[slot]) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
