// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdSplitPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_bwd_split actually loads, read from the shard
 *        the census entry names rather than from anything built in-process.
 *
 * The engine is slotted: its packs belong to the slots prep, dkdv and dq, one op each. The
 * shipped rows are 86 prep (34 bf16, 34 f16, 18 fp32), 1319 dkdv (535, 472, 312) and 1010 dq
 * (350, 348, 312). The 320 dq rows that serve a dB-storing bias carry the dBias reduce kernel
 * as a second symbol of the same code object. Every case opens by reading loadedSet(), none
 * skips and none touches a device, so an empty shard fails here instead of passing silently.
 * The seam suite, over rows written in the test, is TestFlydslSdpaBwdSplitEngine.cpp.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt, over the
 * architectures the slot ops ship for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";
constexpr const char* GRAPH_MATCHER = "hipkernel.flydsl_sdpa_bwd_split.graph_match";
constexpr const char* SHARED_WORKSPACE_TOKEN = "ws.bytes";
constexpr const char* REDUCE_SYMBOL_16BIT = "fmha_bwd_dbias_reduce_gfx950_kernel_1";
constexpr const char* REDUCE_SYMBOL_FP32 = "fmha_bwd_dbias_reduce_f32_gfx950_kernel_1";

/// The rows one slot ships, per dtype, as the generators emit them from the S1 catalogue.
struct SlotCount
{
    const char* slot;
    size_t rows;
    size_t bf16;
    size_t f16;
    size_t fp32;
};

constexpr std::array<SlotCount, 3> SLOT_COUNTS{
    {{"prep", 86, 34, 34, 18}, {"dkdv", 1319, 535, 472, 312}, {"dq", 1010, 350, 348, 312}}};
constexpr size_t REDUCE_ROWS = 320;
/// prep rows that are the sink twin, with its two extra arguments.
constexpr size_t PREP_SINK_ROWS = 32;

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

const SlotDescriptorSet& slotOf(const DescriptorSet& set, const char* name)
{
    for(const auto& slot : set.slots)
    {
        if(slot.name == name)
        {
            return slot;
        }
    }
    throw std::runtime_error(std::string("no slot named ") + name);
}

size_t kernelCount(const SlotDescriptorSet& slot)
{
    size_t kernels = 0;
    for(const auto& pack : slot.packs)
    {
        kernels += pack.kernels.size();
    }
    return kernels;
}

TEST(TestFlydslSdpaBwdSplitPacks, IsOneSlottedEngineWithANativeGraphMatchAndAKernelMatcherPerSlot)
{
    const auto& set = loadedSet(ENGINE_NAME);

    EXPECT_EQ(set.engine.graphMatchNativeSymbol, GRAPH_MATCHER);
    EXPECT_EQ(set.engine.sharedWorkspaceToken, SHARED_WORKSPACE_TOKEN);
    // A slotted engine's selection state is per slot.
    EXPECT_TRUE(set.packs.empty());

    ASSERT_EQ(set.slots.size(), SLOT_COUNTS.size());
    for(size_t index = 0; index < SLOT_COUNTS.size(); ++index)
    {
        EXPECT_EQ(set.slots[index].name, SLOT_COUNTS[index].slot);
        EXPECT_FALSE(set.slots[index].packs.empty()) << SLOT_COUNTS[index].slot;
    }

    std::set<std::string> kernelMatchers;
    for(const auto& matcher : set.matchers)
    {
        EXPECT_EQ(matcher.scope, MatchScope::KERNEL) << matcher.matchSymbol;
        kernelMatchers.insert(matcher.matchSymbol);
    }
    EXPECT_EQ(kernelMatchers,
              (std::set<std::string>{"hipkernel.flydsl_sdpa_bwd_split.prep.kernel_match",
                                     "hipkernel.flydsl_sdpa_bwd_split.dkdv.kernel_match",
                                     "hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match"}));

    std::set<std::string> dispatches;
    for(const auto& dispatch : set.dispatches)
    {
        dispatches.insert(dispatch.dispatchSymbol);
    }
    EXPECT_EQ(dispatches,
              (std::set<std::string>{"hipkernel.flydsl_sdpa_bwd_split.prep.dispatch",
                                     "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch",
                                     "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch"}));
}

TEST(TestFlydslSdpaBwdSplitPacks, ShipsEverySplitRowOnceAndNoKernelInTwoSlots)
{
    const auto& set = loadedSet(ENGINE_NAME);

    const std::set<int64_t> tiles{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};
    const std::set<std::string> dtypes{"f16", "bf16", "fp32"};
    const std::set<std::string> dforms{"rt", "bk"};

    std::set<std::string> names;
    std::set<DescriptorId> ids;
    size_t total = 0;
    size_t reduceRows = 0;
    for(const auto& expected : SLOT_COUNTS)
    {
        SCOPED_TRACE(expected.slot);
        const std::string slotName = expected.slot;
        const auto& slot = slotOf(set, expected.slot);
        std::map<std::string, size_t> perDtype;
        size_t kernels = 0;
        for(const auto& pack : slot.packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                ++kernels;
                names.insert(kernel.name);
                ids.insert(kernel.id);
                EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;

                const auto integer = [&](const char* field) {
                    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
                    EXPECT_NE(value, nullptr)
                        << kernel.name << " lacks the integer field " << field;
                    return value == nullptr ? int64_t{0} : *value;
                };
                const auto text = [&](const char* field) {
                    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
                    EXPECT_NE(value, nullptr) << kernel.name << " lacks the string field " << field;
                    return value == nullptr ? std::string() : *value;
                };
                const auto dtype = text("dtype");
                EXPECT_EQ(dtypes.count(dtype), 1U) << kernel.name;
                ++perDtype[dtype];
                EXPECT_EQ(tiles.count(integer("head_dim")), 1U) << kernel.name;
                EXPECT_GT(integer("head_dim_v"), 0) << kernel.name;
                EXPECT_EQ(dforms.count(text("dform")), 1U) << kernel.name;
                EXPECT_FALSE(text("floor").empty()) << kernel.name;
                for(const char* flag : {"causal", "has_bias", "dropout", "padded", "hints"})
                {
                    const auto value = integer(flag);
                    EXPECT_TRUE(value == 0 || value == 1) << kernel.name << " " << flag;
                }
                const bool hasBias = integer("has_bias") == 1;
                EXPECT_FALSE(hasBias && integer("causal") == 1) << kernel.name;

                if(slotName == "prep")
                {
                    const auto sink = integer("sink");
                    EXPECT_TRUE(sink == -1 || sink == 1) << kernel.name;
                }
                else if(slotName == "dkdv")
                {
                    const auto gqaPack = integer("gqa_pack");
                    EXPECT_TRUE(gqaPack == -1 || gqaPack == 1) << kernel.name;
                    EXPECT_GT(integer("block_threads"), 0) << kernel.name;
                    EXPECT_GT(integer("block_kv"), 0) << kernel.name;
                }
                else
                {
                    const auto storeDb = integer("store_db");
                    EXPECT_TRUE(storeDb == -1 || storeDb == 0) << kernel.name;
                    EXPECT_GT(integer("block_threads"), 0) << kernel.name;
                    EXPECT_GT(integer("block_m"), 0) << kernel.name;
                    const auto stage = text("bias_stage");
                    EXPECT_TRUE(stage == "auto" || stage == "direct") << kernel.name;

                    // A dB-storing bias object carries the dBias reduce kernel of its dtype.
                    const bool storesDb = hasBias && storeDb != 0;
                    const auto reduce = text("reduce_symbol");
                    const std::string expectedReduce
                        = !storesDb ? std::string()
                                    : std::string(dtype == "fp32" ? REDUCE_SYMBOL_FP32
                                                                  : REDUCE_SYMBOL_16BIT);
                    EXPECT_EQ(reduce, expectedReduce) << kernel.name;
                    reduceRows += reduce.empty() ? 0 : 1;
                }
            }
        }
        EXPECT_EQ(kernels, expected.rows);
        EXPECT_EQ(kernelCount(slot), expected.rows);
        EXPECT_EQ(perDtype["bf16"], expected.bf16);
        EXPECT_EQ(perDtype["f16"], expected.f16);
        EXPECT_EQ(perDtype["fp32"], expected.fp32);
        total += kernels;
    }

    EXPECT_EQ(names.size(), total) << "a name is shipped twice";
    EXPECT_EQ(ids.size(), total) << "a kernel id is shipped twice or in two slots";
    EXPECT_EQ(reduceRows, REDUCE_ROWS);
}

TEST(TestFlydslSdpaBwdSplitPacks, EveryKernelDeclaresItsSlotsSignature)
{
    const auto& set = loadedSet(ENGINE_NAME);

    // A run of @p count consecutive arguments of one kind and size.
    struct Run
    {
        const char* kind;
        uint32_t size;
        size_t count;
    };
    using Signature = std::vector<std::pair<const char*, uint32_t>>;
    const auto expand = [](const std::vector<Run>& runs) {
        Signature slots;
        for(const auto& run : runs)
        {
            for(size_t index = 0; index < run.count; ++index)
            {
                slots.emplace_back(run.kind, run.size);
            }
        }
        return slots;
    };

    // `_PARAMS["prep"]`: seven pointers (O DO LSE LSE_ws Delta seqinfo_q0 seqinfo_q1), five i32,
    // six i64 strides: 18 arguments. The sink twin adds the `sink` and `sinkterm` pointers.
    const auto prep = expand({{"global_buffer", 8, 7}, {"by_value", 4, 5}, {"by_value", 8, 6}});
    const auto prepSink = expand(
        {{"global_buffer", 8, 7}, {"by_value", 4, 5}, {"by_value", 8, 6}, {"global_buffer", 8, 2}});
    // `_PARAMS["dkdv"]` and `_PARAMS["dq"]`: nine pointers and four seqinfo pointers, six i32
    // (varlen_bits .. window_right), the two philox pointers, the i64 offset, i32 threshold, f32
    // scale, four i32 (heads, head dims), f32 sm_scale, twenty-one i64 strides and i32
    // rt_flags: 51 arguments each.
    const auto attention = expand({{"global_buffer", 8, 13},
                                   {"by_value", 4, 6},
                                   {"global_buffer", 8, 2},
                                   {"by_value", 8, 1},
                                   {"by_value", 4, 1},
                                   {"by_value", 4, 1},
                                   {"by_value", 4, 4},
                                   {"by_value", 4, 1},
                                   {"by_value", 8, 21},
                                   {"by_value", 4, 1}});

    const auto offsetsOf = [](const Signature& slots) {
        std::vector<uint32_t> offsets;
        uint32_t offset = 0;
        for(const auto& [kind, size] : slots)
        {
            offset = (offset + size - 1) / size * size;
            offsets.push_back(offset);
            offset += size;
        }
        return offsets;
    };

    for(const auto& expected : SLOT_COUNTS)
    {
        SCOPED_TRACE(expected.slot);
        const std::string slotName = expected.slot;
        size_t sinkTwins = 0;
        size_t kernels = 0;
        for(const auto& pack : slotOf(set, expected.slot).packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                ++kernels;
                const auto* sink = tryGetMetadataField<int64_t>(kernel.metadata, "sink");
                const bool twin = slotName == "prep" && sink != nullptr && *sink == 1;
                sinkTwins += twin ? 1 : 0;
                const Signature* expectedSlots = &attention;
                if(slotName == "prep")
                {
                    expectedSlots = twin ? &prepSink : &prep;
                }
                const auto& slots = *expectedSlots;
                const auto offsets = offsetsOf(slots);
                const auto& signature = kernel.source.signature;
                ASSERT_EQ(signature.size(), slots.size()) << kernel.name;
                for(size_t index = 0; index < slots.size(); ++index)
                {
                    EXPECT_EQ(signature[index].kind, slots[index].first)
                        << kernel.name << " slot " << index;
                    EXPECT_EQ(signature[index].size, slots[index].second)
                        << kernel.name << " slot " << index;
                    EXPECT_EQ(signature[index].offset, offsets[index])
                        << kernel.name << " slot " << index;
                }
            }
        }
        EXPECT_EQ(kernels, expected.rows);
        EXPECT_EQ(sinkTwins, slotName == "prep" ? PREP_SINK_ROWS : size_t{0});
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
