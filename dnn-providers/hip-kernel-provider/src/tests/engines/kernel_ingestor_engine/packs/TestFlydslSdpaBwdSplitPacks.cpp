// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
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

#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdNoSerialization.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdSchemaGate.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdVerdictTable.hpp"
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
 * - gfx950: 86 prep (34 bf16, 34 f16, 18 fp32), 1313 dkdv (547, 484, 282) and 962 dq (369,
 *   368, 225) rows over tiles 32..512 in the rt and bk dQ forms, prep with sink twins. The 301
 *   dq rows that serve a dB-storing bias name the dBias reduce kernel by symbol, a second
 *   entry point of the same code object: the fp32 one for an fp32 query or an f32 bias. Every
 *   dq row is a multi-launch pack: its `sources` hold the dQ kernel, the dBias reduce kernel
 *   on a dB-storing row (an entry naming the row's own object and the reduce symbol) and the
 *   tail-zero kernel, one of three tail-zero objects shared by dtype.
 * - gfx11-generic (every gfx11 member's shard): 33 prep, 222 dkdv (78, 78, 66) and 230 dq (82,
 *   82, 66) rows (11 rungs 16..256 times three dtypes, bk form only, no sink, plus the V-head
 *   and f32-bias rows). The 74 dq rows that serve a dB-storing bias carry `sources` of the dQ
 *   kernel, the dBias reduce kernel and the tail-zero kernel as code objects, the reduce one of
 *   three companions shared by dtype; the other dq rows (including the f32-bias rows built
 *   without the dBias store) carry the dQ and tail-zero kernels, the latter one of three
 *   companions shared by dtype.
 *
 * Two cases hold the gfx11 shard to the verdict table (flydsl/tests/data/sdpa_bwd_gfx11_verdicts.csv):
 * every row's graph is run through graph_match and every slot's kernel_match over the installed
 * rows, and the served rows' slot catalogs are checked for unique enrolled knob tuples. A gfx950
 * entry does not read the table.
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
using hipdnn_flatbuffers_sdk::data_objects::DataType;

constexpr const char* ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";
constexpr const char* GRAPH_MATCHER = "hipkernel.flydsl_sdpa_bwd_split.graph_match";
constexpr const char* SHARED_WORKSPACE_TOKEN = "ws.bytes";
constexpr const char* DQ_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch";
constexpr const char* DQ_REDUCE_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch";
constexpr const char* DQ_U2_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch";
constexpr const char* DKDV_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch";
constexpr const char* DKDV_U2_DISPATCH = "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch";
constexpr const char* DQ_LAUNCH = "dq";
constexpr const char* DKDV_LAUNCH = "dkdv";
constexpr const char* DVONLY_LAUNCH = "dvonly";
constexpr const char* REDUCE_LAUNCH = "dbias_reduce";
constexpr const char* TAILZERO_LAUNCH = "tailzero";

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
    /// Whether those rows carry the reduce kernel as a launch of `sources` (the `dq_reduce` pack);
    /// every shipped shard does.
    bool reduceInSources;
    /// Whether the reduce rows are the bias rows that store dBias and stage no bias (gfx11), rather
    /// than any bias row of the dq table.
    bool reduceRowsStageNone;
    /// The dBias reduce kernel's symbol, per dtype width.
    const char* reduceSymbol16Bit;
    const char* reduceSymbolFp32;
    /// Distinct standalone code objects the reduce launch of `sources` names (the dq op's
    /// companions); 0 where the reduce is a second kernel of the row's own object.
    size_t companions;
    /// The dkdv slot's launch grid, as the KMD names it.
    const char* dkdvGrid;
    /// Whether prep's `floor` is the numeric rung below its tile (rather than a label).
    bool prepFloorIsRungBelow;
    /// The content directory name of the verdict table's arch family (the manifests the table's
    /// hashes name); nullptr where the table does not apply.
    const char* verdictContent;
    /// Tail-zero companion entries the dq rows' `sources` name (one per dtype; entries of two
    /// dtypes may resolve to the same object, whose bytes do not depend on the dtype), and the
    /// kernel symbol they carry.
    size_t tailzeroEntries;
    const char* tailzeroSymbol;
};

const SplitContent GFX950_CONTENT{
    {{{"prep", 86, 34, 34, 18}, {"dkdv", 1313, 547, 484, 282}, {"dq", 962, 369, 368, 225}}},
    {32, 64, 96, 128, 160, 192, 224, 256, 384, 512},
    {"rt", "bk"},
    32,
    {-1, 1},
    {-1, 1},
    301,
    true,
    false,
    "fmha_bwd_dbias_reduce_gfx950_kernel_1",
    "fmha_bwd_dbias_reduce_f32_gfx950_kernel_1",
    0,
    "kheads_by_kvblocks_by_seqs",
    false,
    nullptr,
    3,
    "fmha_bwd_tailzero_gfx950_kernel_0"};

const SplitContent GFX11_CONTENT{
    {{{"prep", 33, 11, 11, 11}, {"dkdv", 222, 78, 78, 66}, {"dq", 230, 82, 82, 66}}},
    {16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256},
    {"bk"},
    0,
    {0},
    {0},
    74,
    true,
    true,
    "fmha_bwd_dbias_reduce_gfx11_kernel_0",
    "fmha_bwd_dbias_reduce_f32_gfx11_kernel_0",
    3,
    "kvblocks_by_kheads_by_seqs",
    true,
    "gfx11-generic",
    3,
    "fmha_bwd_tailzero_gfx11_kernel_0"};

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

/// The dispatch symbol each dispatch descriptor id of @p set resolves to.
std::map<DescriptorId, std::string> dispatchSymbolsOf(const DescriptorSet& set)
{
    std::map<DescriptorId, std::string> symbols;
    for(const auto& dispatch : set.dispatches)
    {
        symbols[dispatch.id] = dispatch.dispatchSymbol;
    }
    return symbols;
}

/// Whether @p kernel is a row of V heads of its own (`u2_role` not `none`).
bool isU2Row(const KernelDescriptor& kernel)
{
    const auto it = kernel.metadata.find("u2_role");
    if(it == kernel.metadata.end())
    {
        return false;
    }
    const auto* role = std::get_if<std::string>(&it->second);
    return role != nullptr && *role != "none";
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

    // The multi-launch dq dispatch is shipped exactly where dq rows carry `sources`; the two
    // V-head dispatches exactly where a row of V heads of its own is shipped.
    std::set<std::string> expectedDispatches{
        "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch", DKDV_DISPATCH, DQ_DISPATCH};
    if(content().reduceInSources)
    {
        expectedDispatches.insert(DQ_REDUCE_DISPATCH);
    }
    for(const auto& pack : slotOf(set, "dkdv").packs)
    {
        if(std::any_of(pack.kernels.begin(), pack.kernels.end(), isU2Row))
        {
            expectedDispatches.insert(DKDV_U2_DISPATCH);
        }
    }
    for(const auto& pack : slotOf(set, "dq").packs)
    {
        if(std::any_of(pack.kernels.begin(), pack.kernels.end(), isU2Row))
        {
            expectedDispatches.insert(DQ_U2_DISPATCH);
        }
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
    const auto dispatchSymbols = dispatchSymbolsOf(set);

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
    std::map<std::string, std::set<std::string>> tailzeroByDtype;
    std::map<std::string, std::set<std::string>> tailzeroHvByDtype;
    // The `lds` and the `direct` bias rows of the dq slot, per dtype: the chunk-contract rule
    // keeps an aligned 16-bit bias off a `direct` row that has a staged alternative, and a row
    // above the largest staged tile has none.
    struct DirectRow
    {
        std::string name;
        std::string dtype;
        int64_t tile;
        std::string stage;
    };
    std::vector<DirectRow> dqBiasRows;
    std::map<std::string, int64_t> largestLdsTile;
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
            // A row of V heads of its own is bound by the V-head dispatch of its slot and no
            // other row is: the pack's dispatch is the one the rows' roles name.
            const auto& packDispatch = dispatchSymbols.at(pack.dispatchId);
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
                for(const char* flag : {"causal", "has_bias", "has_dropout", "padded", "hints"})
                {
                    const auto value = integer(flag);
                    EXPECT_TRUE(value == 0 || value == 1) << kernel.name << " " << flag;
                }
                const bool hasBias = integer("has_bias") == 1;
                EXPECT_FALSE(hasBias && integer("causal") == 1) << kernel.name;

                // C1: the f32-bias flag and the V-head role of dkdv and dq rows. An f32 bias
                // rides a 16-bit query; a row of V heads of its own has no bias and is not
                // fp32; the role is the slot's own.
                if(slotName != "prep")
                {
                    const auto biasF32 = integer("bias_f32");
                    EXPECT_TRUE(biasF32 == 0 || biasF32 == 1) << kernel.name;
                    EXPECT_FALSE(biasF32 == 1 && (!hasBias || dtype == "fp32")) << kernel.name;
                    const auto role = text("u2_role");
                    EXPECT_TRUE(role == "none" || role == (slotName == "dkdv" ? "dk" : "dq"))
                        << kernel.name << " u2_role " << role;
                    EXPECT_FALSE(role != "none" && (hasBias || dtype == "fp32")) << kernel.name;
                    if(slotName == "dkdv")
                    {
                        EXPECT_EQ(packDispatch, role == "dk" ? DKDV_U2_DISPATCH : DKDV_DISPATCH)
                            << kernel.name;
                        // dK-only and dV-only launches exactly on a V-head row.
                        EXPECT_EQ(!kernel.sources.empty(), role == "dk") << kernel.name;
                        if(role == "dk")
                        {
                            EXPECT_EQ(kernel.sources.size(), 2U) << kernel.name;
                            EXPECT_EQ(kernel.sources.count(DKDV_LAUNCH), 1U) << kernel.name;
                            EXPECT_EQ(kernel.sources.count(DVONLY_LAUNCH), 1U) << kernel.name;
                        }
                    }
                    else
                    {
                        if(role == "dq")
                        {
                            EXPECT_EQ(packDispatch, DQ_U2_DISPATCH) << kernel.name;
                        }
                        else
                        {
                            EXPECT_TRUE(packDispatch == DQ_DISPATCH
                                        || packDispatch == DQ_REDUCE_DISPATCH)
                                << kernel.name << " is dispatched by " << packDispatch;
                        }
                    }
                }

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
                    const auto resolved = text("bias_stage_resolved");
                    EXPECT_TRUE(resolved == "lds" || resolved == "direct"
                                || resolved == "direct_only" || resolved == "na")
                        << kernel.name << " bias_stage_resolved " << resolved;
                    EXPECT_EQ(resolved == "na", !hasBias) << kernel.name;
                    if(hasBias)
                    {
                        // Only a 16-bit bias with no f32 flag is staged; every row that cannot
                        // be (gfx11, f32 bias, fp32 query) is direct_only.
                        const bool stageable = dtype != "fp32" && integer("bias_f32") == 0
                                               && arch().rfind("gfx11", 0) != 0;
                        EXPECT_FALSE(resolved == "lds" && !stageable) << kernel.name;
                        EXPECT_FALSE(resolved == "direct" && !stageable) << kernel.name;
                        EXPECT_FALSE(!stageable && resolved != "direct_only") << kernel.name;
                        if(stageable)
                        {
                            dqBiasRows.push_back({kernel.name, dtype, headDim, resolved});
                            if(resolved == "lds")
                            {
                                largestLdsTile[dtype] = std::max(largestLdsTile[dtype], headDim);
                            }
                        }
                    }

                    // A dB-storing bias object carries the dBias reduce kernel of its scratch
                    // width: the fp32 one for an fp32 query or an f32 bias, the 16-bit one
                    // otherwise. A bias row built without the dBias store carries none.
                    const bool storesDb = hasBias && storeDb != 0;
                    const auto reduce = text("reduce_symbol");
                    const bool f32Scratch = dtype == "fp32" || integer("bias_f32") == 1;
                    const std::string expectedSymbol
                        = !storesDb ? std::string()
                                    : std::string(f32Scratch ? content().reduceSymbolFp32
                                                             : content().reduceSymbol16Bit);
                    EXPECT_EQ(reduce, expectedSymbol) << kernel.name;
                    reduceRows += reduce.empty() ? 0 : 1;

                    // Every dq row is a multi-launch pack: the dQ kernel, the tail-zero
                    // kernel and, on a pack that carries the reduce kernel as a launch, the
                    // dBias reduce kernel of a dB-storing bias row.
                    EXPECT_TRUE(multiLaunch) << kernel.name;
                    const bool reduceLaunch = content().reduceInSources && storesDb;
                    if(content().reduceRowsStageNone)
                    {
                        // The reduce pack: bias rows storing dBias and staging none, except the
                        // f32-bias rows built without the dBias store (`store_db` 0), which
                        // launch no reduce kernel.
                        if(hasBias)
                        {
                            EXPECT_TRUE(storeDb == 1 || (storeDb == 0 && integer("bias_f32") == 1))
                                << kernel.name << " store_db " << storeDb;
                        }
                        else
                        {
                            EXPECT_EQ(storeDb, -1) << kernel.name;
                        }
                        EXPECT_EQ(stage, hasBias ? "none" : "auto") << kernel.name;
                    }
                    else
                    {
                        EXPECT_TRUE(stage == "auto" || stage == "direct") << kernel.name;
                        EXPECT_TRUE(storeDb == -1 || storeDb == 0) << kernel.name;
                    }
                    std::vector<const char*> launches{DQ_LAUNCH, TAILZERO_LAUNCH};
                    if(reduceLaunch)
                    {
                        launches.push_back(REDUCE_LAUNCH);
                    }
                    ++sourcesRows;
                    EXPECT_EQ(kernel.sources.size(), launches.size()) << kernel.name;
                    EXPECT_EQ(kernel.source.kind, KernelSourceKind::EMBEDDED_SOURCE)
                        << kernel.name << ": `source` is left default beside `sources`";
                    for(const char* launch : launches)
                    {
                        const auto entry = kernel.sources.find(launch);
                        ASSERT_NE(entry, kernel.sources.end())
                            << kernel.name << " lacks the launch " << launch;
                        EXPECT_EQ(entry->second.kind, KernelSourceKind::KPACK)
                            << kernel.name << " " << launch;
                        EXPECT_FALSE(entry->second.sha256.empty()) << kernel.name << " " << launch;
                    }
                    if(reduceLaunch)
                    {
                        const auto& reduceEntry = kernel.sources.at(REDUCE_LAUNCH);
                        EXPECT_EQ(reduceEntry.symbol, reduce) << kernel.name;
                        if(content().companions != 0)
                        {
                            companions.insert(reduceEntry.sha256);
                        }
                        else
                        {
                            // The reduce is a second kernel of the row's own object.
                            const auto& dqEntry = kernel.sources.at(DQ_LAUNCH);
                            EXPECT_EQ(reduceEntry.sha256, dqEntry.sha256) << kernel.name;
                            EXPECT_EQ(reduceEntry.tocKey, dqEntry.tocKey) << kernel.name;
                        }
                    }
                    const auto& zeroEntry = kernel.sources.at(TAILZERO_LAUNCH);
                    if(text("u2_role") == "dq")
                    {
                        // The V-head object, one per dtype entry like the other.
                        std::string hvSymbol = content().tailzeroSymbol;
                        hvSymbol.replace(hvSymbol.find("tailzero"), 8, "tailzero_hv");
                        EXPECT_EQ(zeroEntry.symbol, hvSymbol) << kernel.name;
                        tailzeroHvByDtype[dtype].insert(zeroEntry.sha256);
                    }
                    else
                    {
                        EXPECT_EQ(zeroEntry.symbol, content().tailzeroSymbol) << kernel.name;
                        tailzeroByDtype[dtype].insert(zeroEntry.sha256);
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

    // The tile rule: a stageable bias row above the largest staged tile of its dtype has no
    // staged alternative and is direct_only; a row at or below it that is direct has one.
    for(const auto& row : dqBiasRows)
    {
        SCOPED_TRACE(row.name);
        if(row.tile > largestLdsTile[row.dtype])
        {
            EXPECT_EQ(row.stage, "direct_only");
        }
        else
        {
            EXPECT_NE(row.stage, "direct_only");
        }
    }
    EXPECT_EQ(names.size(), total) << "a name is shipped twice";
    EXPECT_EQ(ids.size(), total) << "a kernel id is shipped twice or in two slots";
    EXPECT_EQ(reduceRows, content().reduceRows);
    EXPECT_EQ(sourcesRows, content().slots[2].rows);
    EXPECT_EQ(companions.size(), content().companions);
    EXPECT_EQ(tailzeroByDtype.size(), content().tailzeroEntries);
    for(const auto& [dtype, objects] : tailzeroByDtype)
    {
        EXPECT_EQ(objects.size(), 1U) << dtype << ": one tail-zero object per dtype entry";
        EXPECT_EQ(objects.count(std::string()), 0U) << dtype << ": the entry names no object";
    }
    for(const auto& [dtype, objects] : tailzeroHvByDtype)
    {
        EXPECT_NE(dtype, "fp32") << "a V-head dq row of fp32 is declined";
        EXPECT_EQ(objects.size(), 1U) << dtype << ": one V-head tail-zero object per dtype";
        EXPECT_EQ(objects.count(std::string()), 0U) << dtype << ": the entry names no object";
    }
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

    // The offset arguments follow the seqinfo pointers: an i64 multiplier for Q and K, then an
    // (offset pointer, i64 multiplier) pair for each other tensor the kernel addresses.
    const auto pairs = [](size_t count) {
        std::vector<Run> runs;
        for(size_t index = 0; index < count; ++index)
        {
            runs.push_back({"global_buffer", 8, 1});
            runs.push_back({"by_value", 8, 1});
        }
        return runs;
    };
    const auto join = [](std::vector<Run> head, const std::vector<Run>& tail) {
        head.insert(head.end(), tail.begin(), tail.end());
        return head;
    };
    // `_PARAMS["prep"]`: seven pointers (O DO LSE LSE_ws Delta seqinfo_q0 seqinfo_q1), the
    // offset pairs of O and dO, the stats' offset pointer, multiplier and three i64 strides,
    // five i32, six i64 strides: 27 arguments. The sink twin adds the `sink` and `sinkterm`
    // pointers.
    const auto prepRuns = join(
        join({{"global_buffer", 8, 7}}, pairs(2)),
        {{"global_buffer", 8, 1}, {"by_value", 8, 4}, {"by_value", 4, 5}, {"by_value", 8, 6}});
    const auto prep = expand(prepRuns);
    const auto prepSink = expand(join(prepRuns, {{"global_buffer", 8, 2}}));
    // `_PARAMS["dkdv"]` and `_PARAMS["dq"]`: nine pointers and four seqinfo pointers, the Q and
    // K multipliers, the offset pairs, six i32 (varlen_bits .. window_right), the two philox
    // pointers, the i64 offset, i32 threshold, f32 scale, four i32 (heads, head dims), f32
    // sm_scale, twenty-one i64 strides and i32 rt_flags. dkdv adds the pairs of V, dO, dK and
    // dV (61 arguments), dq those of V, dO and dQ (59).
    const auto attentionTail = std::vector<Run>{{"by_value", 4, 6},
                                                {"global_buffer", 8, 2},
                                                {"by_value", 8, 1},
                                                {"by_value", 4, 1},
                                                {"by_value", 4, 1},
                                                {"by_value", 4, 4},
                                                {"by_value", 4, 1},
                                                {"by_value", 8, 21},
                                                {"by_value", 4, 1}};
    const auto dkdv = expand(
        join(join({{"global_buffer", 8, 13}, {"by_value", 8, 2}}, pairs(4)), attentionTail));
    const auto dq = expand(
        join(join({{"global_buffer", 8, 13}, {"by_value", 8, 2}}, pairs(3)), attentionTail));
    // `_PARAMS["dkdv_u2"]` and `_PARAMS["dq_u2"]`: the same lists with `num_head_v`, one more
    // i32, after `num_head_k`.
    const auto attentionTailU2 = std::vector<Run>{{"by_value", 4, 6},
                                                  {"global_buffer", 8, 2},
                                                  {"by_value", 8, 1},
                                                  {"by_value", 4, 1},
                                                  {"by_value", 4, 1},
                                                  {"by_value", 4, 5},
                                                  {"by_value", 4, 1},
                                                  {"by_value", 8, 21},
                                                  {"by_value", 4, 1}};
    const auto dkdvU2 = expand(
        join(join({{"global_buffer", 8, 13}, {"by_value", 8, 2}}, pairs(4)), attentionTailU2));
    const auto dqU2 = expand(
        join(join({{"global_buffer", 8, 13}, {"by_value", 8, 2}}, pairs(3)), attentionTailU2));
    // `_PARAMS["tailzero"]`: the dQ, dK and dV pointers, the two length pointers, seven i32
    // extents, nine i64 strides and the i32 mask. The V-head variant (`PARAMS_HV`) has eight
    // extents, `Hv` after `Hkv`.
    const auto tailzero = expand(
        {{"global_buffer", 8, 5}, {"by_value", 4, 7}, {"by_value", 8, 9}, {"by_value", 4, 1}});
    const auto tailzeroHv = expand(
        {{"global_buffer", 8, 5}, {"by_value", 4, 8}, {"by_value", 8, 9}, {"by_value", 4, 1}});

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
                const auto* role = tryGetMetadataField<std::string>(kernel.metadata, "u2_role");
                const bool u2 = role != nullptr && *role != "none";
                const Signature* expectedSlots = slotName == "dkdv" ? &dkdv : &dq;
                if(u2)
                {
                    expectedSlots = slotName == "dkdv" ? &dkdvU2 : &dqU2;
                }
                if(slotName == "prep")
                {
                    expectedSlots = twin ? &prepSink : &prep;
                }
                if(kernel.sources.empty())
                {
                    expectSignature(kernel.name, kernel.source, *expectedSlots);
                    continue;
                }
                if(slotName == "dkdv")
                {
                    // A dkdv row of V heads of its own: the dK-only and dV-only launches take
                    // one list.
                    expectSignature(kernel.name, kernel.sources.at(DKDV_LAUNCH), *expectedSlots);
                    expectSignature(
                        kernel.name + " dvonly", kernel.sources.at(DVONLY_LAUNCH), *expectedSlots);
                    continue;
                }
                // A multi-launch kernel: the dQ launch has the dq slot's argument list, the
                // tail-zero launch the tail-zero kernel's (the V-head one for a dq_u2 row).
                expectSignature(kernel.name, kernel.sources.at(DQ_LAUNCH), *expectedSlots);
                expectSignature(kernel.name + " tailzero",
                                kernel.sources.at(TAILZERO_LAUNCH),
                                u2 ? tailzeroHv : tailzero);
                if(kernel.sources.count(REDUCE_LAUNCH) == 0)
                {
                    continue;
                }
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

// ---------------------------------------------------------------------------
// The verdict table over the installed gfx11 shard
// ---------------------------------------------------------------------------

constexpr std::array<FlydslSdpaBwdSplitSlot, 3> SLOT_ORDER{
    FlydslSdpaBwdSplitSlot::PREP, FlydslSdpaBwdSplitSlot::DKDV, FlydslSdpaBwdSplitSlot::DQ};

/// The spec hashes of a content directory's manifests, which name the objects the table's
/// `prep`, `dkdv`, `dq` and `reduce` columns cite: instance name to hash per slot, and the
/// dBias reduce companions' code-object digests to hash.
struct ManifestHashes
{
    std::array<std::map<std::string, std::string>, 3> hashOfName;
    std::array<std::map<std::string, std::string>, 3> nameOfHash;
    std::map<std::string, std::string> companionHashOfDigest;
};

ManifestHashes loadManifestHashes(const std::string& content)
{
    ManifestHashes hashes;
    const std::array<const char*, 3> ops{"sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"};
    for(size_t slot = 0; slot < ops.size(); ++slot)
    {
        const std::string path
            = (verdictDataRoot() / ops[slot] / content / "manifest.json").string();
        std::ifstream in(path);
        if(!in.good())
        {
            throw std::runtime_error("cannot open the manifest " + path);
        }
        const auto manifest = nlohmann::json::parse(in);
        for(const auto& instance : manifest.at("instances"))
        {
            const std::string name = instance.at("name");
            const std::string hash = instance.at("spec_hash");
            hashes.hashOfName[slot][name] = hash;
            hashes.nameOfHash[slot][hash] = name;
        }
        for(const auto& companion : manifest.value("companions", nlohmann::json::array()))
        {
            hashes.companionHashOfDigest[companion.at("sha256")] = companion.at("hash");
        }
    }
    return hashes;
}

/// What the engine says about one table row over the installed shard.
struct RowMatch
{
    /// Whether graph_match binds the graph.
    bool graphAccepted = false;
    /// The shared matcher's decline cause, when it declines.
    std::optional<std::string> sharedCause;
    /// Per slot, the installed kernels whose kernel_match serves the graph.
    std::array<std::vector<const KernelDefinition*>, 3> matched;
};

/// Every kernel the installed shard ships for each slot, across the slot's packs, as the
/// definitions the engine's matchers read.
std::array<std::vector<KernelDefinition>, 3> installedKernels(const DescriptorSet& set)
{
    std::array<std::vector<KernelDefinition>, 3> kernels;
    for(size_t slot = 0; slot < SLOT_ORDER.size(); ++slot)
    {
        for(const auto& pack : slotOf(set, slotName(SLOT_ORDER[slot])).packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                kernels[slot].push_back(KernelDefinition{kernel.id,
                                                         pack.id,
                                                         pack.dispatchId,
                                                         kernel.source,
                                                         kernel.metadata,
                                                         kernel.priority,
                                                         kernel.arch,
                                                         kernel.originDirectory,
                                                         kernel.name,
                                                         kernel.treeRoot,
                                                         kernel.sources,
                                                         nullptr,
                                                         nullptr});
            }
        }
    }
    return kernels;
}

RowMatch matchRow(const VerdictRow& row,
                  const std::string& arch,
                  const std::array<std::vector<KernelDefinition>, 3>& installed)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = 32;
    const GraphFixture fixture(buildSdpaBwdGraph(specOfRow(row)), properties);
    const auto context = fixture.context();

    RowMatch result;
    const auto bound = matchesGraph(splitSymbols(SLOT_ORDER[0]), context);
    result.graphAccepted = bound.has_value();
    if(!bound.has_value())
    {
        const auto shared
            = flydsl_sdpa_bwd::matchSdpaBackward(context, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        if(const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&shared))
        {
            result.sharedCause = decline->cause;
        }
        return result;
    }
    for(size_t slot = 0; slot < SLOT_ORDER.size(); ++slot)
    {
        const auto symbols = splitSymbols(SLOT_ORDER[slot]);
        for(const auto& kernel : installed[slot])
        {
            if(matchesKernel(symbols, context, kernel, *bound))
            {
                result.matched[slot].push_back(&kernel);
            }
        }
    }
    return result;
}

/// The matches of every row of the verdict table over the installed shard, computed once for
/// the cases of one process.
struct TableMatches
{
    std::vector<VerdictRow> rows;
    std::vector<RowMatch> matches;
    ManifestHashes hashes;
    /// What `matches` points into.
    std::array<std::vector<KernelDefinition>, 3> installed;
};

const TableMatches& tableMatches(const std::string& arch, const char* content)
{
    static const TableMatches s_table = [&] {
        TableMatches built;
        built.rows = loadVerdictRows();
        built.hashes = loadManifestHashes(content);
        built.installed = installedKernels(loadedSet(ENGINE_NAME));
        built.matches.reserve(built.rows.size());
        for(const auto& row : built.rows)
        {
            built.matches.push_back(matchRow(row, arch, built.installed));
        }
        return built;
    }();
    return s_table;
}

std::string namesOf(const std::vector<const KernelDefinition*>& kernels, size_t limit = 4)
{
    std::string text;
    for(size_t index = 0; index < kernels.size() && index < limit; ++index)
    {
        text += (index == 0 ? "" : ", ") + kernels[index]->name;
    }
    return kernels.size() > limit ? text + ", ..." : text;
}

/// The kernel the engine selects among @p matched: the highest priority, then the lowest
/// descriptor id (the kernel ranking when a pack ships no heuristic), or nullptr for none.
const KernelDefinition* selectedOf(const std::vector<const KernelDefinition*>& matched)
{
    const auto best
        = std::min_element(matched.begin(), matched.end(), [](const auto* lhs, const auto* rhs) {
              return lhs->priority != rhs->priority ? lhs->priority > rhs->priority
                                                    : lhs->kernelId < rhs->kernelId;
          });
    return best == matched.end() ? nullptr : *best;
}

/// Why the engine's verdict on @p row disagrees with the table, or nullopt when it agrees.
std::optional<std::string>
    verdictDisagreement(const VerdictRow& row, const RowMatch& match, const ManifestHashes& hashes)
{
    if(row.served)
    {
        if(!match.graphAccepted)
        {
            return "table says served; graph_match declined ["
                   + match.sharedCause.value_or("no shared-matcher decline (SP-WS)") + "]";
        }
        const std::array<const std::string*, 3> expected{&row.prep, &row.dkdv, &row.dq};
        for(size_t slot = 0; slot < SLOT_ORDER.size(); ++slot)
        {
            const std::string slotLabel = slotName(SLOT_ORDER[slot]);
            const auto name = hashes.nameOfHash[slot].find(*expected[slot]);
            if(name == hashes.nameOfHash[slot].end())
            {
                return slotLabel + " hash " + *expected[slot] + " is not in the manifest";
            }
            // The expected row must be the one the engine selects, not merely one that matches.
            const auto* selected = selectedOf(match.matched[slot]);
            if(selected == nullptr || selected->name != name->second)
            {
                return slotLabel + " " + *expected[slot] + " (" + name->second
                       + ") is not the selected row; "
                       + (selected == nullptr ? std::string("none") : selected->name)
                       + " is selected of " + std::to_string(match.matched[slot].size())
                       + " matched: " + namesOf(match.matched[slot]);
            }
            if(slotLabel == "dq" && !row.reduce.empty())
            {
                const auto companion = selected->sources.find(REDUCE_LAUNCH);
                if(companion == selected->sources.end())
                {
                    return "dq " + name->second + " has no " + REDUCE_LAUNCH
                           + " launch for the reduce companion " + row.reduce;
                }
                const auto hash = hashes.companionHashOfDigest.find(companion->second.sha256);
                if(hash == hashes.companionHashOfDigest.end() || hash->second != row.reduce)
                {
                    return "dq " + name->second + " names reduce companion "
                           + (hash == hashes.companionHashOfDigest.end() ? "<not in manifest>"
                                                                         : hash->second)
                           + ", table says " + row.reduce;
                }
            }
        }
        return std::nullopt;
    }

    if(row.rule == NO_ROW_RULE)
    {
        if(!match.graphAccepted)
        {
            return "table says no_row (graph_match accepts); graph_match declined ["
                   + match.sharedCause.value_or("no shared-matcher decline (SP-WS)") + "]";
        }
        const bool anyEmpty = std::any_of(match.matched.begin(),
                                          match.matched.end(),
                                          [](const auto& kernels) { return kernels.empty(); });
        if(!anyEmpty)
        {
            return "table says no_row; every slot has a matching row";
        }
        return std::nullopt;
    }

    const auto cause = ruleCauses().find(row.rule);
    if(cause == ruleCauses().end())
    {
        return "table rule " + row.rule + " has no cause mapping";
    }
    if(match.graphAccepted)
    {
        return "table says declined " + row.rule + " [" + cause->second + "]; graph_match accepted";
    }
    if(!match.sharedCause.has_value())
    {
        return "table says declined " + row.rule + " [" + cause->second
               + "]; graph_match declined without a shared-matcher decline (SP-WS)";
    }
    if(*match.sharedCause != cause->second)
    {
        return "table says " + row.rule + " [" + cause->second + "]; the engine declines ["
               + *match.sharedCause + "]";
    }
    return std::nullopt;
}

TEST_F(TestFlydslSdpaBwdSplitPacks, EveryVerdictRowIsServedOrDeclinedAsTheTableSays)
{
    if(content().verdictContent == nullptr)
    {
        // The table is gfx11-generic content's: this architecture's shard is held to its own
        // inventory by the cases above.
        SUCCEED();
        return;
    }
    const auto& table = tableMatches(arch(), content().verdictContent);
    ASSERT_FALSE(table.rows.empty());

    size_t served = 0;
    size_t declined = 0;
    size_t noRow = 0;
    std::vector<std::string> failures;
    for(size_t index = 0; index < table.rows.size(); ++index)
    {
        const auto& row = table.rows[index];
        served += row.served ? 1 : 0;
        noRow += (!row.served && row.rule == NO_ROW_RULE) ? 1 : 0;
        declined += (!row.served && row.rule != NO_ROW_RULE) ? 1 : 0;
        if(const auto why = verdictDisagreement(row, table.matches[index], table.hashes))
        {
            failures.push_back(row.id + ": " + *why);
        }
    }

    std::cout << "verdict rows " << table.rows.size() << " on " << arch() << ": served " << served
              << ", declined " << declined << ", no_row " << noRow << ", skipped 0, disagreeing "
              << failures.size() << '\n';
    std::ostringstream report;
    for(const auto& failure : failures)
    {
        report << "\n  " << failure;
    }
    EXPECT_TRUE(failures.empty()) << failures.size()
                                  << " rows disagree with the table:" << report.str();
}

TEST_F(TestFlydslSdpaBwdSplitPacks, EveryServedRowsSlotCatalogsHaveUniqueEnrolledKnobTuples)
{
    if(content().verdictContent == nullptr)
    {
        SUCCEED();
        return;
    }
    const auto& set = loadedSet(ENGINE_NAME);
    std::array<std::vector<std::string>, 3> knobs;
    for(size_t slot = 0; slot < SLOT_ORDER.size(); ++slot)
    {
        const std::string label = slotName(SLOT_ORDER[slot]);
        const auto declared
            = std::find_if(set.engine.slots.begin(),
                           set.engine.slots.end(),
                           [&](const auto& candidate) { return candidate.name == label; });
        ASSERT_NE(declared, set.engine.slots.end()) << label;
        knobs[slot] = declared->knobs;
        EXPECT_EQ(enrolledKnobs(label), declared->knobs)
            << label << ": the engine descriptor's enrolled knobs changed; update enrolledKnobs()";
    }

    const auto& table = tableMatches(arch(), content().verdictContent);
    size_t checked = 0;
    std::vector<std::string> failures;
    for(size_t index = 0; index < table.rows.size(); ++index)
    {
        if(!table.rows[index].served)
        {
            continue;
        }
        ++checked;
        for(size_t slot = 0; slot < SLOT_ORDER.size(); ++slot)
        {
            if(const auto pair
               = ambiguousEnrolledTuple(table.matches[index].matched[slot], knobs[slot]))
            {
                failures.push_back(table.rows[index].id + " " + slotName(SLOT_ORDER[slot]) + ": "
                                   + *pair);
            }
        }
    }
    std::cout << "enrolled-tuple check over " << checked << " served rows on " << arch() << ": "
              << failures.size() << " ambiguous" << '\n';
    std::ostringstream report;
    for(const auto& failure : failures)
    {
        report << "\n  " << failure;
    }
    EXPECT_TRUE(failures.empty()) << failures.size() << " ambiguous slot catalogs:" << report.str();
}

// The ingestor declines a graph whose schema floor is above the engine's `sdk_version` before
// the matcher runs. The matcher serves ragged tensors, non-default alignment and ragged offset
// multipliers, so the engine declares the schema that introduced the last of them: an SDK that
// raises the multiplier floor again fails here until the matcher has been reviewed for it.
// ---------------------------------------------------------------------------
// The probe-graph table, the coverage constants and the chunk contract over the shard
// ---------------------------------------------------------------------------

/// One row of flydsl/tests/data/sdpa_bwd_probe_graphs.csv: the graph (ten space-separated
/// `key=value` tokens: hq hk hv dqk dv bias pitch mask dropout dbias) and the spec hashes of the rows the
/// Python model serves it with, best first; empty for a declined graph.
struct ProbeRow
{
    std::string arch;
    std::string kernel;
    std::string dtype;
    std::vector<std::string> graph;
    std::vector<std::string> served;
};

std::vector<std::string> splitOn(const std::string& text, char separator)
{
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(text);
    while(std::getline(stream, part, separator))
    {
        parts.push_back(part);
    }
    return parts;
}

/// The ten values of a graph cell, in the key order of the table; throws on a token that is not
/// the next key's `key=value`.
std::vector<std::string> parseProbeGraph(const std::string& cell)
{
    constexpr std::array<const char*, 10> KEYS{
        "hq", "hk", "hv", "dqk", "dv", "bias", "pitch", "mask", "dropout", "dbias"};
    const auto tokens = splitOn(cell, ' ');
    if(tokens.size() != KEYS.size())
    {
        throw std::runtime_error("a probe graph has " + std::to_string(tokens.size())
                                 + " tokens: " + cell);
    }
    std::vector<std::string> values;
    for(size_t i = 0; i < KEYS.size(); ++i)
    {
        const std::string prefix = std::string(KEYS[i]) + "=";
        if(tokens[i].rfind(prefix, 0) != 0)
        {
            std::string message = "probe graph token ";
            message += tokens[i];
            message += " is not ";
            message += prefix;
            message += "<value>: ";
            message += cell;
            throw std::runtime_error(message);
        }
        values.push_back(tokens[i].substr(prefix.size()));
    }
    return values;
}

/// The rows of the table; throws when the file is missing or malformed, so a consumer fails
/// rather than checking nothing. A table with a header and no rows is valid.
std::vector<ProbeRow> loadProbeRows()
{
    const auto path = (verdictDataRoot() / "sdpa_bwd_probe_graphs.csv").string();
    std::ifstream in(path);
    if(!in.good())
    {
        throw std::runtime_error("cannot open the probe-graph table " + path);
    }
    std::string line;
    std::getline(in, line);
    if(line != "arch,kernel,dtype,graph,served_rows")
    {
        throw std::runtime_error("the probe-graph table has an unexpected header: " + line);
    }
    std::vector<ProbeRow> rows;
    while(std::getline(in, line))
    {
        if(line.empty())
        {
            continue;
        }
        const auto cells = splitCsvLine(line);
        if(cells.size() != 5)
        {
            throw std::runtime_error("a probe-graph row has " + std::to_string(cells.size())
                                     + " cells: " + line);
        }
        ProbeRow row;
        row.arch = cells[0];
        row.kernel = cells[1];
        row.dtype = cells[2];
        row.graph = parseProbeGraph(cells[3]);
        row.served = splitOn(cells[4], ';');
        rows.push_back(std::move(row));
    }
    return rows;
}

/// The graph of a probe row at the fixed shape (batch 1, 64 query and key rows): `aligned`
/// pitches are the whole vector of the bias's dtype, `unaligned` ones 65 elements.
SdpaBwdSpec specOfProbe(const ProbeRow& probe)
{
    const auto& t = probe.graph;
    SdpaBwdSpec spec;
    if(probe.dtype == "bf16")
    {
        spec.dtype = DataType::BFLOAT16;
    }
    else if(probe.dtype == "fp32")
    {
        spec.dtype = DataType::FLOAT;
    }
    else
    {
        spec.dtype = DataType::HALF;
    }
    spec.batch = 1;
    spec.seqQ = 64;
    spec.seqKv = 64;
    spec.heads = std::stoll(t[0]);
    spec.kvHeads = std::stoll(t[1]);
    const auto valueHeads = std::stoll(t[2]);
    spec.headDimQk = std::stoll(t[3]);
    spec.headDimV = std::stoll(t[4]);
    if(valueHeads != spec.kvHeads)
    {
        spec = spec.withEdit([=](SdpaBwdGraph& g) {
            g.tensor("v").dims[1] = valueHeads;
            g.tensor("dv").dims[1] = valueHeads;
        });
    }
    if(t[7] == "causal")
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; });
    }
    if(t[8] == "1")
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { addDropout(g, 0.2F, DataType::INT64, false); });
    }
    if(t[5] != "none")
    {
        const bool withDbias = t[9] == "1";
        const int64_t pitch = t[6] == "unaligned" ? 65 : 0;
        std::optional<DataType> biasType;
        if(t[5] == "f32")
        {
            biasType = DataType::FLOAT;
        }
        spec = spec.withEdit([=](SdpaBwdGraph& g) {
            addBroadcastBias(g, false, false, withDbias, biasType, pitch);
        });
    }
    return spec;
}

/// The spec hashes of the manifests of @p content, or nothing where the build stages none
/// for this architecture.
std::optional<ManifestHashes> tryLoadManifestHashes(const char* content)
{
    try
    {
        return loadManifestHashes(content);
    }
    catch(const std::runtime_error&)
    {
        return std::nullopt;
    }
}

/// The probe graphs of the table, run through graph_match and the slot's kernel_match over the
/// installed shard: the rows served, ranked by priority (high first) then spec hash, must be
/// the rows the model names. Where the build stages the manifests that give a row its spec
/// hash (gfx11), the ranked lists are compared; elsewhere only whether any row serves.
TEST_F(TestFlydslSdpaBwdSplitPacks, EveryProbeGraphIsServedByTheRowsTheModelNames)
{
    const std::string tableArch = arch() == "gfx950" ? "gfx950" : "gfx11-generic";
    const auto rows = loadProbeRows();
    const auto installed = installedKernels(loadedSet(ENGINE_NAME));
    const auto hashes = tryLoadManifestHashes(tableArch.c_str());
    const std::array<const char*, 3> kernels{"prep", "dkdv", "dq"};

    size_t checked = 0;
    std::vector<std::string> failures;
    for(const auto& row : rows)
    {
        if(row.arch != tableArch)
        {
            continue;
        }
        const auto slotIndex = static_cast<size_t>(
            std::find(kernels.begin(), kernels.end(), row.kernel) - kernels.begin());
        ASSERT_LT(slotIndex, kernels.size()) << row.kernel;
        ++checked;

        DeviceProperties properties;
        properties.gcnArchName = arch();
        properties.warpSize = 32;
        const GraphFixture fixture(buildSdpaBwdGraph(specOfProbe(row)), properties);
        const auto context = fixture.context();
        std::vector<const KernelDefinition*> matched;
        if(const auto bound = matchesGraph(splitSymbols(SLOT_ORDER[0]), context))
        {
            for(const auto& kernel : installed[slotIndex])
            {
                if(matchesKernel(splitSymbols(SLOT_ORDER[slotIndex]), context, kernel, *bound))
                {
                    matched.push_back(&kernel);
                }
            }
        }
        const std::string label = row.arch + " " + row.kernel + " " + row.dtype + " ["
                                  + row.graph[0] + "/" + row.graph[1] + "/" + row.graph[2] + " d"
                                  + row.graph[3] + "/" + row.graph[4] + " " + row.graph[5] + " "
                                  + row.graph[6] + " " + row.graph[7] + " dr" + row.graph[8] + " db"
                                  + row.graph[9] + "]";
        if(!hashes.has_value())
        {
            if(matched.empty() != (row.served.empty()))
            {
                failures.push_back(label + ": the model serves " + std::to_string(row.served.size())
                                   + " rows, the shard " + std::to_string(matched.size()));
            }
            continue;
        }
        std::vector<std::pair<int64_t, std::string>> ranked;
        for(const auto* kernel : matched)
        {
            const auto hash = hashes->hashOfName[slotIndex].find(kernel->name);
            ranked.emplace_back(kernel->priority,
                                hash == hashes->hashOfName[slotIndex].end()
                                    ? "<" + kernel->name + ">"
                                    : hash->second);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
        });
        std::vector<std::string> served;
        served.reserve(ranked.size());
        for(const auto& [priority, hash] : ranked)
        {
            served.push_back(hash);
        }
        if(served != row.served)
        {
            failures.push_back(label + ": the shard serves " + std::to_string(served.size())
                               + " rows, the model " + std::to_string(row.served.size()));
        }
    }
    std::cout << "probe rows checked on " << arch() << ": " << checked << ", disagreeing "
              << failures.size() << '\n';
    std::ostringstream report;
    for(const auto& failure : failures)
    {
        report << "\n  " << failure;
    }
    EXPECT_TRUE(failures.empty()) << failures.size()
                                  << " probe graphs disagree with the model:" << report.str();
}

/// The dq rows that serve a bias graph, by what the chunk contract says: an aligned 16-bit bias
/// is read by a staged row or by a row no staged one backs, an unaligned one by a direct row,
/// an f32 bias only by a row no staged one backs.
TEST_F(TestFlydslSdpaBwdSplitPacks,
       TheChunkContractKeepsAnAlignedBiasOffDirectRowsWithAStagedAlternative)
{
    const auto installed = installedKernels(loadedSet(ENGINE_NAME));
    const auto resolvedOf = [](const KernelDefinition& kernel) {
        const auto* stage
            = tryGetMetadataField<std::string>(kernel.metadata, "bias_stage_resolved");
        return stage == nullptr ? std::string() : *stage;
    };
    size_t served = 0;
    std::vector<std::string> failures;
    for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
    {
        for(const int64_t headDim : content().tiles)
        {
            for(const std::string form : {"aligned", "unaligned", "f32"})
            {
                for(const bool withDbias : {false, true})
                {
                    ProbeRow probe;
                    probe.dtype = dtype == DataType::HALF ? "f16" : "bf16";
                    const std::string dim = std::to_string(headDim);
                    probe.graph = {"4",
                                   "4",
                                   "4",
                                   dim,
                                   dim,
                                   form == "f32" ? "f32" : "same",
                                   form == "unaligned" ? "unaligned" : "aligned",
                                   "none",
                                   "0",
                                   withDbias ? "1" : "0"};
                    DeviceProperties properties;
                    properties.gcnArchName = arch();
                    properties.warpSize = 32;
                    const GraphFixture fixture(buildSdpaBwdGraph(specOfProbe(probe)), properties);
                    const auto context = fixture.context();
                    const auto bound = matchesGraph(splitSymbols(SLOT_ORDER[2]), context);
                    if(!bound.has_value())
                    {
                        continue; // an unaligned dBias is declined (BI-9)
                    }
                    for(const auto& kernel : installed[2])
                    {
                        if(!matchesKernel(splitSymbols(SLOT_ORDER[2]), context, kernel, *bound))
                        {
                            continue;
                        }
                        ++served;
                        const auto stage = resolvedOf(kernel);
                        bool allowed = stage == "direct_only";
                        if(form == "aligned")
                        {
                            allowed = allowed || stage == "lds";
                        }
                        else if(form == "unaligned")
                        {
                            allowed = allowed || stage == "direct";
                        }
                        if(!allowed)
                        {
                            std::string violation = kernel.name;
                            violation += " (" + stage;
                            violation += ") serves a " + form;
                            violation += " bias at head dim " + dim;
                            failures.push_back(std::move(violation));
                        }
                    }
                }
            }
        }
    }
    std::cout << "chunk-contract check over " << served << " served rows on " << arch() << ": "
              << failures.size() << " violations" << '\n';
    std::ostringstream report;
    for(const auto& failure : failures)
    {
        report << "\n  " << failure;
    }
    EXPECT_TRUE(failures.empty()) << report.str();
}

/// The floor of a row as the engine reads it: `g1` is 0, `tail` the rung of the dtype's ladder
/// below the tile, anything else the number.
int64_t floorOfRow(const KernelDefinition& kernel, int64_t tile)
{
    if(const auto* number = tryGetMetadataField<int64_t>(kernel.metadata, "floor"))
    {
        return *number;
    }
    const auto* text = tryGetMetadataField<std::string>(kernel.metadata, "floor");
    if(text == nullptr || *text == "g1")
    {
        return 0;
    }
    if(*text == "tail")
    {
        // The 16-bit ladder (`_sdpa_bwd_spec.LADDER`); fp32 rows carry neither family.
        int64_t below = 0;
        for(const int64_t rung : {32, 64, 96, 128, 160, 192, 224, 256, 384, 512})
        {
            if(rung < tile)
            {
                below = rung;
            }
        }
        return below;
    }
    return std::stoll(*text);
}

/// The head dims (Qk == V == d, d up to 512) the shipped rows of a family serve in both the
/// dkdv and the dq slot, for each of f16 and bf16 and each class the family has.
std::set<int64_t> shippedCoverage(const std::array<std::vector<KernelDefinition>, 3>& installed,
                                  bool f32Bias)
{
    std::set<int64_t> covered;
    const std::vector<std::pair<int64_t, int64_t>> classes
        = f32Bias ? std::vector<std::pair<int64_t, int64_t>>{{0, 1}}
                  : std::vector<std::pair<int64_t, int64_t>>{{0, 0}, {1, 0}};
    for(int64_t d = 1; d <= 512; ++d)
    {
        bool every = true;
        for(const char* dtype : {"f16", "bf16"})
        {
            for(const auto& [causal, hasBias] : classes)
            {
                for(const size_t slot : {size_t{1}, size_t{2}})
                {
                    bool any = false;
                    for(const auto& kernel : installed[slot])
                    {
                        const auto* kernelDtype
                            = tryGetMetadataField<std::string>(kernel.metadata, "dtype");
                        const auto* role
                            = tryGetMetadataField<std::string>(kernel.metadata, "u2_role");
                        const auto* biasF32
                            = tryGetMetadataField<int64_t>(kernel.metadata, "bias_f32");
                        const auto* kernelCausal
                            = tryGetMetadataField<int64_t>(kernel.metadata, "causal");
                        const auto* kernelBias
                            = tryGetMetadataField<int64_t>(kernel.metadata, "has_bias");
                        const auto* tile
                            = tryGetMetadataField<int64_t>(kernel.metadata, "head_dim");
                        const auto* padded
                            = tryGetMetadataField<int64_t>(kernel.metadata, "padded");
                        if(kernelDtype == nullptr || *kernelDtype != dtype || role == nullptr
                           || biasF32 == nullptr || kernelCausal == nullptr || kernelBias == nullptr
                           || tile == nullptr || padded == nullptr)
                        {
                            continue;
                        }
                        const bool inFamily = f32Bias ? *biasF32 == 1 : *role != "none";
                        const bool serves = *padded == 0
                                                ? d == *tile
                                                : d > floorOfRow(kernel, *tile) && d <= *tile;
                        if(inFamily && *kernelCausal == causal && *kernelBias == hasBias && serves)
                        {
                            any = true;
                            break;
                        }
                    }
                    every = every && any;
                }
            }
        }
        if(every)
        {
            covered.insert(d);
        }
    }
    return covered;
}

/// The constants graph_match's coverage line states (FlydslSdpaBwdCommon.hpp, the parity
/// bands) are the head dims the loaded shard's f32-bias and `Hk != Hv` rows serve.
TEST_F(TestFlydslSdpaBwdSplitPacks, TheCoverageConstantsAreTheHeadDimsTheShippedParityRowsServe)
{
    const auto installed = installedKernels(loadedSet(ENGINE_NAME));
    std::set<int64_t> constants;
    const bool gfx11 = arch().rfind("gfx11", 0) == 0;
    const auto addBands = [&](const auto& bands) {
        for(const auto& band : bands)
        {
            for(int64_t d = band.floor + 1; d <= band.tile; ++d)
            {
                constants.insert(d);
            }
        }
    };
    if(gfx11)
    {
        addBands(flydsl_sdpa_bwd::GFX11_PARITY_BANDS);
    }
    else
    {
        addBands(flydsl_sdpa_bwd::GFX950_PARITY_BANDS);
    }
    for(const bool f32Bias : {true, false})
    {
        SCOPED_TRACE(f32Bias ? "f32 bias" : "V heads of their own");
        const auto shipped = shippedCoverage(installed, f32Bias);
        EXPECT_FALSE(shipped.empty()) << "the shard ships no row of the family";
        EXPECT_EQ(shipped, constants);
    }
}

TEST_F(TestFlydslSdpaBwdSplitPacks, DeclaresASchemaAtLeastTheRaggedOffsetMultiplierFloor)
{
    const auto& set = loadedSet(ENGINE_NAME);
    const auto& floor
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, false, false, false, true);
    EXPECT_FALSE(set.engine.sdkVersion < floor) << set.engine.sdkVersion.str();
}

// Graphs stamped as the backend stamps them reach the matcher through GenericPlanBuilder and
// through the provider's ApplicabilityManager, the engine registered at its UED's
// `sdk_version` as Container does, and the matcher's verdicts hold on both: ragged tensors with
// and without a multiplier and an aligned operand are served; a misaligned operand, ragged
// stats beside a dense Q, a ragged bias and a multiplier below 1 are declined by the matcher,
// not by a schema gate.
TEST_F(TestFlydslSdpaBwdSplitPacks, ServesStampedRaggedGraphsAndDeclinesTheFieldsTheMatcherRejects)
{
    const auto& set = loadedSet(ENGINE_NAME);
    const auto failures = schema_gate::failures(set, arch());
    std::string report;
    for(const auto& failure : failures)
    {
        report += "\n  " + failure;
    }
    EXPECT_TRUE(failures.empty()) << report;
    // The cases only matter if they can see each gate: an engine of the schema these engines
    // shipped with, as descriptor and as registration, declines the same graph.
    const auto invisible = schema_gate::gateIsInvisible(set, arch());
    EXPECT_FALSE(invisible.has_value()) << invisible.value_or("");
}

// Pins that neither engine advertises or performs plan saving. Remove or invert when RFC 0021
// slot-program saving lands (split) or the merged dispatch handler implements save/restore.
TEST_F(TestFlydslSdpaBwdSplitPacks, RefusesPlanSerialization)
{
    expectEngineRefusesPlanSerialization(loadedSet(ENGINE_NAME));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
