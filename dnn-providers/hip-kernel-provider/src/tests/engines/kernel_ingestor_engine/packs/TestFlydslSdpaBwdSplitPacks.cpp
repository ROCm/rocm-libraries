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

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdSplitPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa_bwd_split actually loads, read from the shard
 *        the census entry names rather than from anything built in-process.
 *
 * The engine is slotted: its packs belong to the slots prep, dkdv and dq, one op each. What a
 * shard holds depends on the architecture it was packed for, which the census entry supplies
 * (HIPDNN_TEST_EXPECTED_ARCH) and which selects the inventory below; the descriptors never
 * choose their own expectation.
 *
 * - gfx950: 86 prep (34 bf16, 34 f16, 18 fp32), 1319 dkdv (535, 472, 312) and 1010 dq (350,
 *   348, 312) rows over tiles 32..512 in the rt and bk dQ forms, prep with sink twins. The 320
 *   dq rows that serve a dB-storing bias name the dBias reduce kernel by symbol, a second
 *   entry point of the same code object.
 * - gfx11-generic (every gfx11 member's shard): 33 prep, 198 dkdv and 198 dq rows (11 rungs
 *   16..256 times three dtypes, bk form only, no sink). The 66 dq rows that serve a dB-storing
 *   bias are a multi-launch pack: its `sources` hold the dQ kernel and the dBias reduce kernel
 *   as two code objects, the reduce one of three companions shared by dtype.
 *
 * Every case opens by reading loadedSet(), none skips and none touches a device, so an empty
 * shard fails here instead of passing silently. The seam suite, over rows written in the test,
 * is TestFlydslSdpaBwdSplitEngine.cpp.
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
constexpr const char* DQ_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch";
constexpr const char* DQ_REDUCE_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch";
constexpr const char* DQ_LAUNCH = "dq";
constexpr const char* REDUCE_LAUNCH = "dbias_reduce";

/// The rows one slot ships, per dtype, as the generators emit them.
struct SlotCount
{
    const char* slot;
    size_t rows;
    size_t bf16;
    size_t f16;
    size_t fp32;
};

/// What the shard of one architecture family holds.
struct SplitContent
{
    std::array<SlotCount, 3> slots;
    /// The dQ-accumulation tiles (`head_dim`) the rows are built for, ascending.
    std::vector<int64_t> tiles;
    std::set<std::string> dforms;
    /// prep rows that are the sink twin, with its two extra arguments.
    size_t prepSinkRows;
    /// Values of prep's `sink` field and dkdv's `gqa_pack` field.
    std::set<int64_t> sinkValues;
    std::set<int64_t> gqaPackValues;
    /// dq rows that serve a dB-storing bias and so name the dBias reduce kernel.
    size_t reduceRows;
    /// Whether those rows carry the reduce kernel as a launch of `sources` (a multi-launch
    /// object) rather than as a second symbol of the dQ code object.
    bool reduceInSources;
    /// The dBias reduce kernel's symbol, per dtype width.
    const char* reduceSymbol16Bit;
    const char* reduceSymbolFp32;
    /// Distinct code objects the reduce launch of `sources` names (the dq op's companions).
    size_t companions;
    /// The dkdv slot's launch grid, as the KMD names it.
    const char* dkdvGrid;
    /// Whether prep's `floor` is the numeric rung below its tile (rather than a label).
    bool prepFloorIsRungBelow;
};

const SplitContent GFX950_CONTENT{
    {{{"prep", 86, 34, 34, 18}, {"dkdv", 1319, 535, 472, 312}, {"dq", 1010, 350, 348, 312}}},
    {32, 64, 96, 128, 160, 192, 224, 256, 384, 512},
    {"rt", "bk"},
    32,
    {-1, 1},
    {-1, 1},
    320,
    false,
    "fmha_bwd_dbias_reduce_gfx950_kernel_1",
    "fmha_bwd_dbias_reduce_f32_gfx950_kernel_1",
    0,
    "kheads_by_kvblocks_by_seqs",
    false};

const SplitContent GFX11_CONTENT{
    {{{"prep", 33, 11, 11, 11}, {"dkdv", 198, 66, 66, 66}, {"dq", 198, 66, 66, 66}}},
    {16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256},
    {"bk"},
    0,
    {0},
    {0},
    66,
    true,
    "fmha_bwd_dbias_reduce_gfx11_kernel_0",
    "fmha_bwd_dbias_reduce_f32_gfx11_kernel_0",
    3,
    "kvblocks_by_kheads_by_seqs",
    true};

/// The inventory for the architecture the census entry names; the architecture is never read
/// from the descriptors being counted. nullptr for an architecture the engine has no
/// content for.
const SplitContent* contentOf(const std::string& arch)
{
    if(arch == "gfx950")
    {
        return &GFX950_CONTENT;
    }
    // Every gfx11 member is served by the one gfx11-generic content.
    if(arch.rfind("gfx11", 0) == 0)
    {
        return &GFX11_CONTENT;
    }
    return nullptr;
}

class TestFlydslSdpaBwdSplitPacks : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        ASSERT_FALSE(arch.empty())
            << "HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has no expectation to "
               "check against; the architecture is supplied by the census entry.";
        _arch = arch;
        _content = contentOf(arch);
        ASSERT_NE(_content, nullptr) << "no split SDPA backward inventory is recorded for " << arch;
    }

    const SplitContent& content() const
    {
        return *_content;
    }

    const std::string& arch() const
    {
        return _arch;
    }

private:
    std::string _arch;
    const SplitContent* _content = nullptr;
};

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

TEST_F(TestFlydslSdpaBwdSplitPacks, IsOneSlottedEngineWithANativeGraphMatchAndAKernelMatcherPerSlot)
{
    const auto& set = loadedSet(ENGINE_NAME);

    EXPECT_EQ(set.engine.graphMatchNativeSymbol, GRAPH_MATCHER);
    EXPECT_EQ(set.engine.sharedWorkspaceToken, SHARED_WORKSPACE_TOKEN);
    // A slotted engine's selection state is per slot.
    EXPECT_TRUE(set.packs.empty());

    ASSERT_EQ(set.slots.size(), content().slots.size());
    for(size_t index = 0; index < content().slots.size(); ++index)
    {
        EXPECT_EQ(set.slots[index].name, content().slots[index].slot);
        EXPECT_FALSE(set.slots[index].packs.empty()) << content().slots[index].slot;
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

    // The multi-launch dq dispatch is shipped exactly where dq rows carry `sources`.
    std::set<std::string> expectedDispatches{"hipkernel.flydsl_sdpa_bwd_split.prep.dispatch",
                                             "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch",
                                             DQ_DISPATCH};
    if(content().reduceInSources)
    {
        expectedDispatches.insert(DQ_REDUCE_DISPATCH);
    }
    std::set<std::string> dispatches;
    for(const auto& dispatch : set.dispatches)
    {
        dispatches.insert(dispatch.dispatchSymbol);
    }
    EXPECT_EQ(dispatches, expectedDispatches);
}

TEST_F(TestFlydslSdpaBwdSplitPacks, ShipsEverySplitRowOnceAndNoKernelInTwoSlots)
{
    const auto& set = loadedSet(ENGINE_NAME);

    const std::set<int64_t> tiles(content().tiles.begin(), content().tiles.end());
    const std::set<std::string> dtypes{"f16", "bf16", "fp32"};

    // The numeric floor of a prep row is the rung below its tile; the lowest rung has none.
    std::map<int64_t, std::string> rungBelow;
    for(size_t index = 0; index < content().tiles.size(); ++index)
    {
        rungBelow[content().tiles[index]]
            = index == 0 ? std::string("0") : std::to_string(content().tiles[index - 1]);
    }

    std::set<std::string> names;
    std::set<DescriptorId> ids;
    size_t total = 0;
    size_t reduceRows = 0;
    size_t sourcesRows = 0;
    std::set<std::string> companions;
    for(const auto& expected : content().slots)
    {
        SCOPED_TRACE(expected.slot);
        const std::string slotName = expected.slot;
        const auto& slot = slotOf(set, expected.slot);
        std::map<std::string, size_t> perDtype;
        size_t kernels = 0;
        for(const auto& pack : slot.packs)
        {
            // A pack is wholly single-launch or wholly multi-launch: its dispatch handler
            // reads one or the other.
            const bool multiLaunch = !pack.kernels.empty() && !pack.kernels.front().sources.empty();
            for(const auto& kernel : pack.kernels)
            {
                ++kernels;
                names.insert(kernel.name);
                ids.insert(kernel.id);
                EXPECT_EQ(!kernel.sources.empty(), multiLaunch) << kernel.name;

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
                const auto headDim = integer("head_dim");
                EXPECT_EQ(tiles.count(headDim), 1U) << kernel.name;
                EXPECT_GT(integer("head_dim_v"), 0) << kernel.name;
                EXPECT_EQ(content().dforms.count(text("dform")), 1U) << kernel.name;
                const auto floorLabel = text("floor");
                EXPECT_FALSE(floorLabel.empty()) << kernel.name;
                for(const char* flag : {"causal", "has_bias", "dropout", "padded", "hints"})
                {
                    const auto value = integer(flag);
                    EXPECT_TRUE(value == 0 || value == 1) << kernel.name << " " << flag;
                }
                const bool hasBias = integer("has_bias") == 1;
                EXPECT_FALSE(hasBias && integer("causal") == 1) << kernel.name;

                if(slotName == "prep")
                {
                    EXPECT_EQ(content().sinkValues.count(integer("sink")), 1U) << kernel.name;
                    EXPECT_EQ(text("grid"), "qblocks_by_heads_by_seqs") << kernel.name;
                    if(content().prepFloorIsRungBelow && rungBelow.count(headDim) == 1)
                    {
                        EXPECT_EQ(floorLabel, rungBelow[headDim]) << kernel.name;
                    }
                }
                else if(slotName == "dkdv")
                {
                    EXPECT_EQ(content().gqaPackValues.count(integer("gqa_pack")), 1U)
                        << kernel.name;
                    EXPECT_GT(integer("block_threads"), 0) << kernel.name;
                    EXPECT_GT(integer("block_kv"), 0) << kernel.name;
                    EXPECT_EQ(text("grid"), content().dkdvGrid) << kernel.name;
                }
                else
                {
                    const auto storeDb = integer("store_db");
                    EXPECT_GT(integer("block_threads"), 0) << kernel.name;
                    EXPECT_GT(integer("block_m"), 0) << kernel.name;
                    EXPECT_EQ(text("grid"), "qheads_by_qblocks_by_seqs") << kernel.name;
                    const auto stage = text("bias_stage");

                    // A dB-storing bias object carries the dBias reduce kernel of its dtype.
                    const bool storesDb = hasBias && storeDb != 0;
                    const auto reduce = text("reduce_symbol");
                    const std::string expectedSymbol
                        = !storesDb ? std::string()
                                    : std::string(dtype == "fp32" ? content().reduceSymbolFp32
                                                                  : content().reduceSymbol16Bit);
                    EXPECT_EQ(reduce, expectedSymbol) << kernel.name;
                    reduceRows += reduce.empty() ? 0 : 1;

                    if(content().reduceInSources)
                    {
                        // The reduce pack: bias rows only, each storing dBias and staging none.
                        EXPECT_EQ(multiLaunch, hasBias) << kernel.name;
                        EXPECT_EQ(storeDb, hasBias ? 1 : -1) << kernel.name;
                        EXPECT_EQ(stage, hasBias ? "none" : "auto") << kernel.name;
                        if(multiLaunch)
                        {
                            ++sourcesRows;
                            EXPECT_EQ(kernel.sources.size(), 2U) << kernel.name;
                            EXPECT_EQ(kernel.source.kind, KernelSourceKind::EMBEDDED_SOURCE)
                                << kernel.name << ": `source` is left default beside `sources`";
                            for(const char* launch : {DQ_LAUNCH, REDUCE_LAUNCH})
                            {
                                const auto entry = kernel.sources.find(launch);
                                ASSERT_NE(entry, kernel.sources.end())
                                    << kernel.name << " lacks the launch " << launch;
                                EXPECT_EQ(entry->second.kind, KernelSourceKind::KPACK)
                                    << kernel.name << " " << launch;
                                EXPECT_FALSE(entry->second.sha256.empty())
                                    << kernel.name << " " << launch;
                            }
                            const auto& reduceEntry = kernel.sources.at(REDUCE_LAUNCH);
                            EXPECT_EQ(reduceEntry.symbol, reduce) << kernel.name;
                            companions.insert(reduceEntry.sha256);
                        }
                    }
                    else
                    {
                        EXPECT_TRUE(stage == "auto" || stage == "direct") << kernel.name;
                        EXPECT_TRUE(storeDb == -1 || storeDb == 0) << kernel.name;
                    }
                }
                if(kernel.sources.empty())
                {
                    EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
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
    EXPECT_EQ(reduceRows, content().reduceRows);
    EXPECT_EQ(sourcesRows, content().reduceInSources ? content().reduceRows : size_t{0});
    EXPECT_EQ(companions.size(), content().companions);
}

TEST_F(TestFlydslSdpaBwdSplitPacks, EveryKernelDeclaresItsSlotsSignature)
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

    const auto expectSignature =
        [&](const std::string& name, const KernelSource& source, const Signature& slots) {
            const auto offsets = offsetsOf(slots);
            const auto& signature = source.signature;
            ASSERT_EQ(signature.size(), slots.size()) << name;
            for(size_t index = 0; index < slots.size(); ++index)
            {
                EXPECT_EQ(signature[index].kind, slots[index].first) << name << " slot " << index;
                EXPECT_EQ(signature[index].size, slots[index].second) << name << " slot " << index;
                EXPECT_EQ(signature[index].offset, offsets[index]) << name << " slot " << index;
            }
        };

    // The dBias reduce kernel's own argument list: the dBias scratch and the dBias pointer,
    // then the extents and strides of the reduction (see `_PARAMS["dbias_reduce"]`).
    std::set<std::string> reduceShapes;

    for(const auto& expected : content().slots)
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
                if(kernel.sources.empty())
                {
                    expectSignature(kernel.name, kernel.source, *expectedSlots);
                    continue;
                }
                // A multi-launch kernel: the dQ launch has the dq slot's argument list.
                expectSignature(kernel.name, kernel.sources.at(DQ_LAUNCH), *expectedSlots);
                std::string shape;
                for(const auto& argument : kernel.sources.at(REDUCE_LAUNCH).signature)
                {
                    shape += argument.kind + std::string(":") + std::to_string(argument.size)
                             + std::string(",");
                }
                reduceShapes.insert(shape);
            }
        }
        EXPECT_EQ(kernels, expected.rows);
        EXPECT_EQ(sinkTwins, slotName == "prep" ? content().prepSinkRows : size_t{0});
    }
    // One argument list serves every reduce launch of the shard.
    EXPECT_EQ(reduceShapes.size(), content().reduceInSources ? 1U : 0U);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
