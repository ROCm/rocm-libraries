// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx942AttentionDenseKnobs.cpp
 * @brief hipkernel:Gfx942AttentionDense end to end through the frontend on a gfx942
 *        device: a forced block_m/block_n selects exactly the policy-arm kernel of that
 *        tile, that kernel launches with its own geometry and computes what the CPU
 *        reference computes, and a knob pair no kernel has is refused before any plan.
 *
 * The case table is generated so it runs every (head size, dtype, mask, window, LSE, tile)
 * class the catalog compiles, at a shape the tile divides; a coverage case asserts the
 * table and the installed catalog agree. An LSE row binds the frontend's stats output
 * (generate_stats, a packed [B, Hq, Sq, 1] FLOAT tensor), selects the class's emit_lse
 * twin, and grades the LSE against the CPU reference at the default FLOAT SDPA
 * tolerance, printing its largest error as `LSE_ERROR <case> max_abs=.. max_rel=..`. Head counts vary across rows because the body
 * reads them at runtime. Window rows run Sq = Skv = 2W, so the window really excludes
 * keys, and spell the graph's left_bound as W (the reference keeps left_bound keys,
 * diagonal included). Hand-written rows add cold selection, B > 1, Sq != Skv, both alignments, the
 * deprecated causal flag and the default and a non-default scale.
 *
 * Each row also names the launch grid it forces: the one-CTA-per-block grid at a block
 * order (qb_minor, bt_hkv_minor, hq_minor_swz) or the persistent grid at a decode, forced
 * through the persistent, nonpersist_decode and persist_decode knobs (the string ones by
 * ordinal). Persistent rows cover W < NP, W % NP != 0, W >> NP, GQA 40/8 and 28/4, B > 1,
 * an odd query-block count, Sq != Skv, every window and LSE; a cold row expects the
 * persistent arm exactly where the engine's score prefers it on this device (D128 without
 * a narrow window, and 304 CTAs a multiple of its CU count). A graph with Hq = 12 is never
 * offered an hq_minor_swz kernel, which would drop and repeat work items there.
 *
 * Each expected kernel id is resolved at run time from the production descriptors: the
 * one kernel of the class and grid whose lds_row_pad, pv_priority, lds_num_buffers,
 * v_reg_prefetch, causal_diag_split, o_store_width, pv_loop_order, use_cfvst,
 * use_exp2_fast and lds_k_group_pad are the policy arms, which the engine's score ranks
 * first among the arms sharing a tile and grid.
 *
 * A saved plan of one case, and of its LSE twin, is loaded with a new handle in both save
 * forms and must compute the original graph's CPU reference without selecting a kernel
 * again.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx942AttentionDense";

constexpr const char* SERVED_ARCH = "gfx942";

constexpr const char* BLOCK_M_KNOB = "block_m";
constexpr const char* BLOCK_N_KNOB = "block_n";
constexpr const char* PERSISTENT_KNOB = "persistent";
constexpr const char* NONPERSIST_DECODE_KNOB = "nonpersist_decode";
constexpr const char* PERSIST_DECODE_KNOB = "persist_decode";

/// The persistent CTA count every persistent variant bakes, and the narrow-window bound
/// past which the engine's cold score prefers it at D128.
constexpr int64_t NUM_PERSISTENT = 304;
constexpr int64_t NARROW_WINDOW = 256;

/// The policy arms of the two knobs the engine does not expose (the score's first rank).
constexpr int64_t POLICY_LDS_ROW_PAD = 8;
constexpr int64_t POLICY_PV_PRIORITY = 0;

/// The policy arms of the body levers no case forces: no diagonal split, the 4-element O
/// store, the d_major PV loop and the default K-group pad. use_cfvst and use_exp2_fast
/// resolve per dtype, head size and grid (rocKE's _use_cfvst / _use_exp2_fast).
constexpr int64_t POLICY_CAUSAL_DIAG_SPLIT = 0;
constexpr int64_t POLICY_O_STORE_WIDTH = 4;
constexpr const char* POLICY_PV_LOOP_ORDER = "d_major";
constexpr int64_t POLICY_LDS_K_GROUP_PAD = 8;
constexpr int64_t CFVST_HEAD_SIZE = 128;

int64_t policyUseCfvst(int64_t headSize)
{
    return headSize == CFVST_HEAD_SIZE ? 1 : 0;
}

int64_t policyUseExp2Fast(bool bf16, int64_t headSize, bool persistent)
{
    return bf16 && headSize == CFVST_HEAD_SIZE && !persistent ? 0 : 1;
}

/// The heuristic path's selection line (GenericPlanBuilder::buildPlan), which names the
/// kernel id as the kdp spells it: lowercase, hyphenated.
constexpr const char* SELECTED_KERNEL_MARKER = "' selected kernel ";
constexpr size_t KERNEL_ID_LENGTH = 36;

// The knob case that the save and load cases build. Every launch value it saves is one a
// swap or a dropped value would show: batch 2, seqlen_q 1024 against seqlen_kv 1536, 4
// query heads over 1 key/value head, a 512-key window and the scale 1/sqrt(64), so a
// loaded plan that applies the default scale 1.0 fails the comparison.
constexpr const char* SAVE_LOAD_CASE = "Shape_WindowTopLeftCrossLength";

/// The same case binding the stats output: its plan saves the stats uid and restores the
/// 11-argument kernel.
constexpr const char* SAVE_LOAD_LSE_CASE = "Shape_WindowTopLeftCrossLength_Lse";

/// A persistent case (bt_hkv_minor, GQA 40/8, B 3, W = 480 work items over 304 CTAs) and
/// its LSE twin: the plan saves the grid and restores the 14- and 15-argument kernels with
/// the fast-division pairs recomputed.
constexpr const char* SAVE_LOAD_PERSISTENT_CASE = "Persist_Gqa40x8WNotAMultipleOfNp";
constexpr const char* SAVE_LOAD_PERSISTENT_LSE_CASE = "Persist_Gqa40x8WNotAMultipleOfNp_Lse";

/// The suffix of an LSE row's name.
constexpr const char* LSE_SUFFIX = "_Lse";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t STATS_UID = 5;

/// Every spelling of the masks the engine serves.
enum class Mask
{
    NO_MASK,
    BOUNDS_TOP_LEFT,
    BOUNDS_BOTTOM_RIGHT,
    CAUSAL_MASK_FLAG,
    /// left_bound W, right_bound 0: a causal window of W keys.
    WINDOW_TOP_LEFT,
    WINDOW_BOTTOM_RIGHT
};

struct GraphShape
{
    DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    Mask mask;
    /// The window in keys for the WINDOW_* masks; 0 otherwise.
    int64_t window;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
    /// Leaves attn_scale_value unset, so the engine must apply 1.0 (no scaling).
    bool omitScale = false;
    std::optional<float> scaleOverride = std::nullopt;
    /// Binds the stats (LSE) output, so the emit_lse twin serves the graph.
    bool withStats = false;
};

struct Tile
{
    int64_t blockM;
    int64_t blockN;
};

/// A launch grid: the one-CTA-per-block grid at a block order, or the persistent grid at a
/// decode.
struct Grid
{
    bool persistent = false;
    /// The nonpersist_decode, or on the persistent grid the persist_decode.
    std::string order = "qb_minor";
};

Grid persistentGrid(const char* decode)
{
    return Grid{true, decode};
}

Grid blockOrder(const char* order)
{
    return Grid{false, order};
}

/// One graph and the tile and grid the engine must serve it with. A forced case sets
/// block_m, block_n and the grid knobs; a cold case sets no knob, and the tile and grid are
/// the ones the engine's own ranking picks (coldGrid for a case built with the default).
struct KnobCase
{
    std::string name;
    GraphShape shape;
    bool forced;
    Tile tile;
    Grid grid = {};
};

/// One graph and a knob pair whose values the engine advertises for it one by one while
/// no kernel it admits carries both.
struct UnsatisfiableCase
{
    std::string name;
    GraphShape shape;
    Tile forcedTile;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const UnsatisfiableCase& unsatisfiableCase, std::ostream* os)
{
    *os << unsatisfiableCase.name;
}

GraphShape makeShape(DataType dataType,
                     int64_t headSize,
                     int64_t queryHeads,
                     int64_t kvHeads,
                     Mask mask,
                     int64_t window,
                     int64_t batch,
                     int64_t seqQ,
                     int64_t seqKv)
{
    return {dataType, headSize, queryHeads, kvHeads, mask, window, batch, seqQ, seqKv};
}

/// The tiles the reduced catalog keeps for a cell (configs/gfx942_attention_dense.yaml).
/// Window cells keep their floor -- the 256/64 default and the dividing tiles 64/64 and
/// 256/32 -- and the causal and unmasked cells keep the tiles the reduction sweep ranked.
std::vector<Tile> tilesFor(int64_t headSize, DataType dataType, bool causal, int64_t window)
{
    std::vector<Tile> tiles{{64, 64}, {256, 32}, {256, 64}};
    if(window != 0)
    {
        return tiles;
    }
    tiles.push_back({128, 64});
    if(headSize == 64 && dataType == DataType::BFLOAT16)
    {
        tiles.push_back({256, 128});
    }
    if(headSize == 64 && !causal)
    {
        tiles.push_back({512, 64});
    }
    return tiles;
}

/// The catalog's mask cells: causal, unmasked, and causal windows of these sizes.
const std::vector<int64_t>& catalogWindows()
{
    static const std::vector<int64_t> s_windows{128, 512, 1024, 2048};
    return s_windows;
}

std::string tileName(const Tile& tile)
{
    return "Bm" + std::to_string(tile.blockM) + "Bn" + std::to_string(tile.blockN);
}

/// @p knobCase's twin that binds the stats output, named with LSE_SUFFIX.
KnobCase lseTwin(KnobCase knobCase)
{
    knobCase.name += LSE_SUFFIX;
    knobCase.shape.withStats = true;
    return knobCase;
}

/// Every catalog class once, forced to its tile, each followed by its LSE twin, then the
/// hand-written cases.
std::vector<KnobCase> knobCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr bool FORCED = true;
    constexpr bool COLD = false;

    // MHA, GQA and MQA head configurations, rotated across the class rows.
    const std::array<std::pair<int64_t, int64_t>, 4> heads{{{8, 8}, {16, 2}, {8, 1}, {4, 4}}};

    std::vector<KnobCase> cases;
    // A class row and its LSE twin: the same shape, so the two differ only in the output.
    const auto addClass = [&cases](KnobCase knobCase) {
        cases.push_back(knobCase);
        cases.push_back(lseTwin(std::move(knobCase)));
    };
    std::size_t row = 0;
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        for(const auto dataType : {BF16, FP16})
        {
            const std::string prefix = std::string(headSize == 64 ? "D64_" : "D128_")
                                       + (dataType == BF16 ? "Bf16_" : "Fp16_");
            for(const auto& tile : tilesFor(headSize, dataType, true, 0))
            {
                const auto& [hq, hkv] = heads[row++ % std::size(heads)];
                addClass(
                    {prefix + "Causal_" + tileName(tile),
                     makeShape(dataType, headSize, hq, hkv, Mask::BOUNDS_TOP_LEFT, 0, 1, 512, 512),
                     FORCED,
                     tile});
            }
            for(const auto& tile : tilesFor(headSize, dataType, false, 0))
            {
                const auto& [hq, hkv] = heads[row++ % std::size(heads)];
                addClass({prefix + "NoMask_" + tileName(tile),
                          makeShape(dataType, headSize, hq, hkv, Mask::NO_MASK, 0, 1, 512, 512),
                          FORCED,
                          tile});
            }
            for(const int64_t window : catalogWindows())
            {
                const int64_t seq = std::max<int64_t>(512, 2 * window);
                for(const auto& tile : tilesFor(headSize, dataType, true, window))
                {
                    addClass(
                        {prefix + "Window" + std::to_string(window) + "_" + tileName(tile),
                         makeShape(
                             dataType, headSize, 1, 1, Mask::WINDOW_TOP_LEFT, window, 1, seq, seq),
                         FORCED,
                         tile});
                }
            }
            // The grids past qb_minor the catalog carries at the 256/64 tile: bt_hkv_minor
            // in the causal cell, and at D128 the persistent base arm of every mask, at the
            // decode the registry pins (bt_hkv_minor where the decodes balance causal work
            // at the build shape, qb_major for no mask and the 128-key window).
            const Tile base{256, 64};
            addClass({prefix + "Causal_BtHkvMinor",
                      makeShape(dataType, headSize, 16, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 512),
                      FORCED,
                      base,
                      blockOrder("bt_hkv_minor")});
            if(headSize != 128)
            {
                continue;
            }
            addClass({prefix + "Causal_Persistent",
                      makeShape(dataType, headSize, 8, 8, Mask::BOUNDS_TOP_LEFT, 0, 1, 512, 512),
                      FORCED,
                      base,
                      persistentGrid("bt_hkv_minor")});
            addClass({prefix + "NoMask_Persistent",
                      makeShape(dataType, headSize, 16, 4, Mask::NO_MASK, 0, 2, 512, 512),
                      FORCED,
                      base,
                      persistentGrid("qb_major")});
            for(const int64_t window : catalogWindows())
            {
                const int64_t seq = std::max<int64_t>(512, 2 * window);
                addClass({prefix + "Window" + std::to_string(window) + "_Persistent",
                          makeShape(
                              dataType, headSize, 8, 2, Mask::WINDOW_TOP_LEFT, window, 1, seq, seq),
                          FORCED,
                          base,
                          persistentGrid(window < 512 ? "qb_major" : "bt_hkv_minor")});
            }
        }
    }

    // hq_minor_swz, both grids, on bf16 D128 causal: Hq a multiple of 8, and an even
    // query-block count under causal balancing on the persistent grid.
    addClass({"D128_Bf16_Causal_HqMinorSwz",
              makeShape(BF16, 128, 16, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
              FORCED,
              Tile{256, 64},
              blockOrder("hq_minor_swz")});
    addClass({"D128_Bf16_Causal_PersistentHqMinorSwz",
              makeShape(BF16, 128, 16, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
              FORCED,
              Tile{256, 64},
              persistentGrid("hq_minor_swz")});

    // The persistent grid over its work-item count W = nqb * Hq * B against NP = 304 CTAs.
    // W % NP != 0 and W > NP, GQA 40/8, B > 1; then W >> NP (3.4 rounds), GQA 16; then GQA
    // 28/4 with an odd query-block count; then Sq != Skv, causal and unmasked.
    addClass({"Persist_Gqa40x8WNotAMultipleOfNp",
              makeShape(BF16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 3, 1024, 1024),
              FORCED,
              Tile{256, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Persist_WFarAboveNp",
              makeShape(BF16, 128, 128, 8, Mask::BOUNDS_TOP_LEFT, 0, 4, 512, 512),
              FORCED,
              Tile{256, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Persist_Gqa28x4OddQueryBlocks",
              makeShape(FP16, 128, 28, 4, Mask::BOUNDS_TOP_LEFT, 0, 2, 768, 768),
              FORCED,
              Tile{256, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Persist_CausalCrossLength",
              makeShape(BF16, 128, 8, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 1024),
              FORCED,
              Tile{256, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Persist_NoMaskCrossLength",
              makeShape(FP16, 128, 12, 4, Mask::NO_MASK, 0, 3, 512, 1280),
              FORCED,
              Tile{256, 64},
              persistentGrid("qb_major")});

    // Cold: no knob, so the 256/64 policy kernel where it divides, and otherwise the
    // smallest dividing tile, which the score ranks next; on the grid coldGrid names.
    cases.push_back({"Cold_Causal",
                     makeShape(BF16, 128, 32, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
                     COLD,
                     Tile{256, 64}});
    cases.push_back({"Cold_Seq384",
                     makeShape(FP16, 64, 8, 8, Mask::NO_MASK, 0, 1, 384, 384),
                     COLD,
                     Tile{64, 64}});
    // Shape breadth on a forced tile: B > 1, Sq != Skv, both alignments at equal lengths,
    // the deprecated causal flag, top-left windows at unequal lengths: Sq > Skv by less
    // than W, so the last 64 rows keep only part of their window, then Skv > Sq.
    cases.push_back({"Shape_Batch3CrossLength",
                     makeShape(FP16, 128, 16, 2, Mask::NO_MASK, 0, 3, 256, 768),
                     FORCED,
                     Tile{256, 64}});
    cases.push_back({"Shape_BottomRightEqualLengths",
                     makeShape(BF16, 64, 8, 8, Mask::BOUNDS_BOTTOM_RIGHT, 0, 2, 512, 512),
                     FORCED,
                     Tile{128, 64}});
    cases.push_back({"Shape_CausalFlag",
                     makeShape(FP16, 128, 8, 2, Mask::CAUSAL_MASK_FLAG, 0, 1, 512, 512),
                     FORCED,
                     Tile{256, 32}});
    cases.push_back({"Shape_WindowBottomRightEqualLengths",
                     makeShape(BF16, 128, 4, 2, Mask::WINDOW_BOTTOM_RIGHT, 128, 1, 512, 512),
                     FORCED,
                     Tile{256, 64}});
    cases.push_back({"Shape_WindowTopLeftQueriesPastKeys",
                     makeShape(BF16, 128, 8, 2, Mask::WINDOW_TOP_LEFT, 128, 2, 512, 448),
                     FORCED,
                     Tile{256, 64}});
    cases.push_back({"Shape_WindowTopLeftCrossLength",
                     makeShape(FP16, 64, 4, 1, Mask::WINDOW_TOP_LEFT, 512, 2, 1024, 1536),
                     FORCED,
                     Tile{256, 64}});
    // The LSE index (b * Hq + h) * Sq + s at B > 1, grouped heads and Sq != Skv, and the
    // cold ranking over the LSE twins.
    cases.push_back(lseTwin(cases.back()));
    cases.push_back(lseTwin({"Shape_Batch3CrossLength",
                             makeShape(FP16, 128, 16, 2, Mask::NO_MASK, 0, 3, 256, 768),
                             FORCED,
                             Tile{256, 64}}));
    cases.push_back(lseTwin({"Cold_Causal",
                             makeShape(BF16, 128, 32, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
                             COLD,
                             Tile{256, 64}}));
    // The scale: absent means 1.0; a non-default value must reach the kernel.
    auto noScale = makeShape(BF16, 128, 8, 8, Mask::BOUNDS_TOP_LEFT, 0, 1, 256, 256);
    noScale.omitScale = true;
    cases.push_back({"Scale_Omitted", noScale, FORCED, Tile{256, 64}});
    auto otherScale = makeShape(FP16, 64, 8, 8, Mask::NO_MASK, 0, 1, 256, 256);
    otherScale.scaleOverride = 0.03F;
    cases.push_back({"Scale_NonDefault", otherScale, FORCED, Tile{64, 64}});
    return cases;
}

/// D128 at 512 admits 64/64, 128/64, 256/32 and 256/64, so block_m=128 and block_n=32
/// are each advertised while no D128 kernel is 128/32.
std::vector<UnsatisfiableCase> unsatisfiableCases()
{
    return {
        {"D128_Bm128Bn32",
         makeShape(DataType::HALF, 128, 8, 8, Mask::NO_MASK, 0, 1, 512, 512),
         Tile{128, 32}},
    };
}

/// BSHD: token-major, head varying fastest. The kernel bakes this layout.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t seq, int64_t headSize)
{
    return {seq * heads * headSize, headSize, heads * headSize, 1};
}

std::shared_ptr<TensorAttributes> makeBshdTensor(int64_t uid,
                                                 const std::string& name,
                                                 DataType dataType,
                                                 int64_t batch,
                                                 int64_t heads,
                                                 int64_t seq,
                                                 int64_t headSize)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({batch, heads, seq, headSize})
        .set_stride(bshdStrides(heads, seq, headSize))
        .set_data_type(dataType);
    return tensor;
}

struct SdpaGraph
{
    std::shared_ptr<Graph> graph;
    std::shared_ptr<TensorAttributes> output;
    /// The stats (LSE) output, or nullptr for a graph that binds none.
    std::shared_ptr<TensorAttributes> stats;
};

/// A single SDPA-forward node with BSHD Q/K/V/O. O is laid out explicitly: the frontend's
/// default for an unset output is packed BHSD, which the engine declines. A shape with
/// stats binds the stats output as the frontend does, FLOAT [B, Hq, Sq, 1], packed.
SdpaGraph buildSdpaGraph(const std::string& name, const GraphShape& shape)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(name)
        .set_io_data_type(shape.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = makeBshdTensor(
        Q_UID, "Q", shape.dataType, shape.batch, shape.queryHeads, shape.seqQ, shape.headSize);
    auto k = makeBshdTensor(
        K_UID, "K", shape.dataType, shape.batch, shape.kvHeads, shape.seqKv, shape.headSize);
    auto v = makeBshdTensor(
        V_UID, "V", shape.dataType, shape.batch, shape.kvHeads, shape.seqKv, shape.headSize);

    SdpaAttributes attributes;
    attributes.set_name(name);
    if(!shape.omitScale)
    {
        attributes.set_attn_scale(
            shape.scaleOverride.value_or(1.0f / std::sqrt(static_cast<float>(shape.headSize))));
    }
    switch(shape.mask)
    {
    case Mask::NO_MASK:
        break;
    case Mask::BOUNDS_TOP_LEFT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        break;
    case Mask::BOUNDS_BOTTOM_RIGHT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        break;
    case Mask::CAUSAL_MASK_FLAG:
        attributes.causal_mask = true;
        break;
    case Mask::WINDOW_TOP_LEFT:
        attributes.set_diagonal_band_left_bound(shape.window);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        break;
    case Mask::WINDOW_BOTTOM_RIGHT:
        attributes.set_diagonal_band_left_bound(shape.window);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        break;
    default:
        throw std::invalid_argument("buildSdpaGraph: unhandled Mask value");
    }
    if(shape.withStats)
    {
        attributes.set_generate_stats(true);
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(shape.dataType)
        .set_dim({shape.batch, shape.queryHeads, shape.seqQ, shape.headSize})
        .set_stride(bshdStrides(shape.queryHeads, shape.seqQ, shape.headSize));

    auto stats = outputs[1];
    if(stats != nullptr)
    {
        stats->set_uid(STATS_UID)
            .set_name("Stats")
            .set_output(true)
            .set_data_type(DataType::FLOAT)
            .set_dim({shape.batch, shape.queryHeads, shape.seqQ, 1})
            .set_stride({shape.queryHeads * shape.seqQ, shape.seqQ, 1, 1});
    }
    return {graph, o, stats};
}

/// The forward SDPA tolerance every SdpaFwd bundle of this engine is graded at
/// (config/hipkernel_Gfx942AttentionDense.toml names no override).
float sdpaForwardTolerance(DataType dataType)
{
    return dataType == DataType::HALF
               ? hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>()
               : hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<
                     hipdnn_data_sdk::types::bfloat16>();
}

/// The FLOAT SDPA forward tolerance every stats output is graded at (the bundles' default;
/// config/hipkernel_Gfx942AttentionDense.toml names no override).
constexpr float LSE_TOLERANCE = hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<float>();

/// Grades the LSE as the default FLOAT validator does, and prints its largest absolute and
/// relative error against the CPU reference as `LSE_ERROR <case> max_abs=.. max_rel=..`, so
/// a run records how far inside LSE_TOLERANCE the kernel lands.
class RecordingLseValidator : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    explicit RecordingLseValidator(std::string caseName)
        : _caseName(std::move(caseName))
        , _validator(hipdnn_test_sdk::utilities::createAllCloseValidator(
              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, LSE_TOLERANCE, LSE_TOLERANCE))
    {
    }

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        // Both are the packed host copies of one tensor, so they compare element by element.
        if(reference.dims() == implementation.dims()
           && reference.strides() == implementation.strides()
           && reference.elementSpace() == reference.elementCount())
        {
            const auto* expected = static_cast<const float*>(reference.rawHostData());
            const auto* actual = static_cast<const float*>(implementation.rawHostData());
            double maxAbs = 0.0;
            double maxRel = 0.0;
            for(std::size_t i = 0; i < reference.elementCount(); ++i)
            {
                const auto want = static_cast<double>(expected[i]);
                const double error = std::fabs(static_cast<double>(actual[i]) - want);
                maxAbs = std::max(maxAbs, error);
                maxRel = std::max(maxRel, error / std::max(std::fabs(want), 1e-30));
            }
            std::cout << "LSE_ERROR " << _caseName << " max_abs=" << maxAbs << " max_rel=" << maxRel
                      << " tolerance=" << static_cast<double>(LSE_TOLERANCE) << '\n';
        }
        return _validator->allClose(reference, implementation);
    }

private:
    std::string _caseName;
    std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> _validator;
};

/// The knob value that pins @p value of the string field @p field: its ordinal, the index
/// into the field's sorted value set over the engine's packs (RFC 0019 13.2).
int64_t ordinalOf(const std::string& field, const std::string& value);

std::vector<KnobSetting> knobSettingsFor(const Tile& tile, const Grid& grid = {})
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_M_KNOB, tile.blockM);
    settings.emplace_back(BLOCK_N_KNOB, tile.blockN);
    settings.emplace_back(PERSISTENT_KNOB, int64_t{grid.persistent ? 1 : 0});
    const std::string field = grid.persistent ? PERSIST_DECODE_KNOB : NONPERSIST_DECODE_KNOB;
    settings.emplace_back(field, ordinalOf(field, grid.order));
    return settings;
}

/// A kernel metadata field and the value a case expects it to carry.
using MetadataKey = std::vector<std::pair<std::string, hipdnn_plugin_sdk::ingestor::MetadataValue>>;

std::string describeKey(const MetadataKey& key)
{
    std::string text;
    for(const auto& [field, value] : key)
    {
        text += (text.empty() ? "" : " ") + field + "=";
        std::visit(
            [&text](const auto& alternative) {
                using Alternative = std::decay_t<decltype(alternative)>;
                if constexpr(std::is_same_v<Alternative, std::string>)
                {
                    text += alternative;
                }
                else if constexpr(std::is_arithmetic_v<Alternative>)
                {
                    text += std::to_string(alternative);
                }
            },
            value);
    }
    return text;
}

/// The production descriptors HIPDNN_DESCRIPTOR_RUNTIME_DIR names, loaded once.
const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& runtimeCatalog()
{
    static const auto s_catalog = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(
        std::filesystem::path(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")));
    return s_catalog;
}

/// Calls @p visit for every kernel runtimeCatalog() serves this engine with on SERVED_ARCH.
template <typename Visit>
void forEachServedKernel(Visit&& visit)
{
    const auto& catalog = runtimeCatalog();
    for(const auto& entry : catalog.packs)
    {
        const auto& pack = entry.second;
        const auto& arch = pack.descriptor.arch;
        const auto engine = catalog.engines.find(pack.descriptor.engineId);
        if(pack.conflicted || engine == catalog.engines.end()
           || engine->second.descriptor.name != ENGINE_NAME
           || std::find(arch.begin(), arch.end(), SERVED_ARCH) == arch.end())
        {
            continue;
        }
        for(const auto& kernel : pack.descriptor.kernels)
        {
            visit(kernel);
        }
    }
}

std::string catalogDtypeName(DataType dataType)
{
    return dataType == DataType::HALF ? "FP16" : "BF16";
}

int64_t catalogCausal(Mask mask)
{
    return mask == Mask::NO_MASK ? 0 : 1;
}

/// The id of the one rank-zero kernel this engine's gfx942 packs carry for @p shape's
/// semantic fields and @p tile, as the selection line spells it.
std::optional<std::string>
    expectedKernelId(const GraphShape& shape, const Tile& tile, const Grid& grid)
{
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        ADD_FAILURE() << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production "
                         "descriptors to read the expected kernel id from";
        return std::nullopt;
    }

    const MetadataKey key{
        {"dtype", catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"causal", catalogCausal(shape.mask)},
        {"ragged", int64_t{0}},
        {"sliding_window", shape.window},
        {"block_m", tile.blockM},
        {"block_n", tile.blockN},
        {"lds_row_pad", POLICY_LDS_ROW_PAD},
        {"pv_priority", POLICY_PV_PRIORITY},
        {"emit_lse", int64_t{shape.withStats ? 1 : 0}},
        {"persistent", int64_t{grid.persistent ? 1 : 0}},
        {grid.persistent ? "persist_decode" : "nonpersist_decode", grid.order},
        {"lds_num_buffers", int64_t{1}},
        {"v_reg_prefetch", int64_t{0}},
        {"causal_diag_split", POLICY_CAUSAL_DIAG_SPLIT},
        {"o_store_width", POLICY_O_STORE_WIDTH},
        {"pv_loop_order", std::string(POLICY_PV_LOOP_ORDER)},
        {"lds_k_group_pad", POLICY_LDS_K_GROUP_PAD},
        {"use_cfvst", policyUseCfvst(shape.headSize)},
        {"use_exp2_fast",
         policyUseExp2Fast(shape.dataType != DataType::HALF, shape.headSize, grid.persistent)},
    };

    std::vector<std::string> ids;
    forEachServedKernel([&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        const bool carriesKey = std::all_of(key.begin(), key.end(), [&kernel](const auto& field) {
            const auto value = kernel.metadata.find(field.first);
            return value != kernel.metadata.end() && value->second == field.second;
        });
        if(carriesKey)
        {
            ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
        }
    });

    if(ids.size() != 1)
    {
        ADD_FAILURE() << ids.size() << " " << ENGINE_NAME << " kernel(s) for " << SERVED_ARCH
                      << " carry " << describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                      << "'; exactly one must";
        return std::nullopt;
    }
    return ids.front();
}

int64_t ordinalOf(const std::string& field, const std::string& value)
{
    std::set<std::string> values;
    forEachServedKernel(
        [&field, &values](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            const auto found = kernel.metadata.find(field);
            if(found != kernel.metadata.end())
            {
                if(const auto* text = std::get_if<std::string>(&found->second))
                {
                    values.insert(*text);
                }
            }
        });
    const auto at = values.find(value);
    EXPECT_NE(at, values.end()) << "no " << ENGINE_NAME << " kernel carries " << field << "="
                                << value;
    return static_cast<int64_t>(std::distance(values.begin(), at));
}

/// The device's compute-unit count, which the engine's cold score reads.
int deviceMultiProcessorCount()
{
    int device = 0;
    int count = 0;
    if(hipGetDevice(&device) != hipSuccess
       || hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, device)
              != hipSuccess)
    {
        ADD_FAILURE() << "cannot read the device's compute-unit count";
        return 0;
    }
    return count;
}

/// The grid a cold case selects: the persistent arm at the decode the registry pins for the
/// mask where the engine's score prefers it -- D128, no window or one wider than
/// NARROW_WINDOW keys, and NUM_PERSISTENT a multiple of this device's CU count -- and the
/// qb_minor grid otherwise.
Grid coldGrid(const GraphShape& shape)
{
    const int cus = deviceMultiProcessorCount();
    if(shape.headSize != 128 || (shape.window != 0 && shape.window <= NARROW_WINDOW) || cus <= 0
       || NUM_PERSISTENT % cus != 0)
    {
        return Grid{};
    }
    // The registry pins its decode at the build shape (Sq = Skv = 512).
    const bool causal = shape.mask != Mask::NO_MASK;
    const bool balanced = causal && (shape.window <= 0 || shape.window >= 512);
    return persistentGrid(balanced ? "bt_hkv_minor" : "qb_major");
}

/// The kernel id the plugin's selection line names, or nullopt when no plan was built.
std::optional<std::string>
    selectedKernelId(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string marker = std::string("engine '") + ENGINE_NAME + SELECTED_KERNEL_MARKER;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at != std::string::npos)
        {
            return log.message.substr(at + marker.size(), KERNEL_ID_LENGTH);
        }
    }
    return std::nullopt;
}

/// Prints `KNOB_SELECTED <case> <kernelId>` when the case ends, however it ends.
class SelectedKernelReport
{
public:
    SelectedKernelReport(std::string caseName, const ScopedPluginLogCapture& capture)
        : _caseName(std::move(caseName))
        , _capture(capture)
    {
    }

    ~SelectedKernelReport()
    {
        std::cout << "KNOB_SELECTED " << _caseName << " "
                  << selectedKernelId(_capture.recorder()).value_or("none") << '\n';
    }

    SelectedKernelReport(const SelectedKernelReport&) = delete;
    SelectedKernelReport& operator=(const SelectedKernelReport&) = delete;
    SelectedKernelReport(SelectedKernelReport&&) = delete;
    SelectedKernelReport& operator=(SelectedKernelReport&&) = delete;

private:
    std::string _caseName;
    const ScopedPluginLogCapture& _capture;
};

bool advertises(const std::vector<Knob>& knobs, const std::string& knobId, int64_t value)
{
    const auto knob = std::find_if(knobs.begin(), knobs.end(), [&](const Knob& candidate) {
        return candidate.knobId() == knobId;
    });
    if(knob == knobs.end())
    {
        return false;
    }
    const auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
    return constraint != nullptr && constraint->getValidValues().count(value) > 0;
}

} // namespace

/// Runs only on the architecture the engine ships for.
template <typename TestCaseType>
class IntegrationGpuGfx942AttentionDenseBase
    : public IntegrationGraphVerificationHarness<float, TestCaseType>
{
protected:
    using Harness = IntegrationGraphVerificationHarness<float, TestCaseType>;

    void SetUp() override
    {
        Harness::SetUp();
        if(this->IsSkipped() || this->HasFatalFailure())
        {
            return;
        }

        const auto arch = hip_kernel_provider_common::getDeviceString(this->_stream);
        if(arch != SERVED_ARCH)
        {
            GTEST_SKIP() << ENGINE_NAME << " ships for " << SERVED_ARCH << " only; this device is "
                         << arch;
        }
    }

    /// Offsets the seed by UID so Q, K and V carry different data.
    void initializeBundle(const Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   Harness::DEFAULT_MIN,
                                   Harness::DEFAULT_MAX,
                                   seed + static_cast<unsigned int>(tensorPair.first));
        }
    }

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    void buildOperationGraphTheEngineOffersToServe(Graph& graph)
    {
        auto result = graph.build_operation_graph(this->_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        const bool offered
            = result.code == ErrorCode::OK
              && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId())
                     != rankedEngineIds.end();
        ASSERT_TRUE(offered) << ENGINE_NAME << " did not offer to serve this graph on "
                             << SERVED_ARCH << " (" << result.err_msg
                             << "). Its packed descriptors load through "
                             << "HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                             << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                             << "'; this configuration must pack the production descriptors for "
                             << SERVED_ARCH << ".";
    }

    /// A cache root private to ONE test case, so no other case's winner record is reused.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx942-dense-knobs-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

using IntegrationGpuGfx942AttentionDenseKnobs = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseKnobFilter
    = IntegrationGpuGfx942AttentionDenseBase<UnsatisfiableCase>;
using IntegrationGpuGfx942AttentionDenseKnobCoverage
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseSwzLegality
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;

/// Every row runs one compiled kernel, and what the compiler specializes on is the head
/// size, dtype, causal flag, window, LSE and tile. The table must run every such class the
/// catalog carries at its policy arms, and nothing the catalog lacks.
TEST_F(IntegrationGpuGfx942AttentionDenseKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production descriptors to "
           "read the catalog's kernel classes from";

    using KernelClass = std::tuple<int64_t,
                                   std::string,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   std::string>;

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        if(!knobCase.forced)
        {
            continue;
        }
        tableClasses.emplace(knobCase.shape.headSize,
                             catalogDtypeName(knobCase.shape.dataType),
                             catalogCausal(knobCase.shape.mask),
                             knobCase.shape.window,
                             int64_t{knobCase.shape.withStats ? 1 : 0},
                             knobCase.tile.blockM,
                             knobCase.tile.blockN,
                             int64_t{knobCase.grid.persistent ? 1 : 0},
                             knobCase.grid.order);
    }

    std::set<KernelClass> catalogClasses;
    forEachServedKernel([&catalogClasses](
                            const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        const auto intField = [&kernel](const char* field) -> const int64_t* {
            const auto value = kernel.metadata.find(field);
            return value == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&value->second);
        };
        const auto dtype = kernel.metadata.find("dtype");
        const auto* dtypeName
            = dtype == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&dtype->second);
        const auto* headSize = intField("head_size");
        const auto* causal = intField("causal");
        const auto* window = intField("sliding_window");
        const auto* emitLse = intField("emit_lse");
        const auto* blockM = intField("block_m");
        const auto* blockN = intField("block_n");
        const auto* persistent = intField("persistent");
        const auto stringField = [&kernel](const char* field) -> const std::string* {
            const auto value = kernel.metadata.find(field);
            return value == kernel.metadata.end() ? nullptr
                                                  : std::get_if<std::string>(&value->second);
        };
        const auto* order = persistent != nullptr && *persistent == 1
                                ? stringField("persist_decode")
                                : stringField("nonpersist_decode");
        if(dtypeName == nullptr || headSize == nullptr || causal == nullptr || window == nullptr
           || emitLse == nullptr || blockM == nullptr || blockN == nullptr || persistent == nullptr
           || order == nullptr)
        {
            ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                          << " carries no string dtype and grid decode, and integer head_size, "
                             "causal, sliding_window, emit_lse, block_m, block_n and persistent";
            return;
        }
        catalogClasses.emplace(*headSize,
                               *dtypeName,
                               *causal,
                               *window,
                               *emitLse,
                               *blockM,
                               *blockN,
                               *persistent,
                               *order);
    });

    const auto onlyIn = [](const std::set<KernelClass>& from, const std::set<KernelClass>& other) {
        std::string text;
        for(const auto& kernelClass : from)
        {
            if(other.count(kernelClass) == 0)
            {
                const auto& [headSize,
                             dtype,
                             causal,
                             window,
                             emitLse,
                             blockM,
                             blockN,
                             persistent,
                             order]
                    = kernelClass;
                text += " (D" + std::to_string(headSize) + ", " + dtype + ", causal "
                        + std::to_string(causal) + ", window " + std::to_string(window)
                        + ", emit_lse " + std::to_string(emitLse) + ", " + std::to_string(blockM)
                        + "/" + std::to_string(blockN);
                text += ", persistent " + std::to_string(persistent) + " ";
                text += order;
                text += ")";
            }
        }
        return text.empty() ? std::string(" none") : text;
    };
    EXPECT_EQ(tableClasses, catalogClasses)
        << "in the catalog but not knobCases():" << onlyIn(catalogClasses, tableClasses)
        << "\nin knobCases() but not the catalog:" << onlyIn(tableClasses, catalogClasses)
        << "\nunder HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the
/// output is then checked against the CPU reference, which catches a kernel launched with
/// any geometry, window or head count but its own.
TEST_P(IntegrationGpuGfx942AttentionDenseKnobs, SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const SelectedKernelReport report(testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    const auto settings = testCase.forced ? knobSettingsFor(testCase.tile, testCase.grid)
                                          : std::vector<KnobSetting>{};
    auto result = sdpa.graph->create_execution_plan_ext(engineId(), settings);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = selectedKernelId(recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE_NAME << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const auto grid = testCase.forced ? testCase.grid : coldGrid(testCase.shape);
    const auto expectedId = expectedKernelId(testCase.shape, testCase.tile, grid);
    ASSERT_TRUE(expectedId.has_value());
    EXPECT_EQ(*selected, *expectedId) << "Captured logs:\n" << recorder.getRecordedLogsAsString();

    // Rank 0: the policy-arm kernel heads the ranking, rather than being reached past a
    // kernel that failed to load.
    const auto selectionLine = std::string(SELECTED_KERNEL_MARKER) + *expectedId + " at rank 0";
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    ASSERT_EQ(sdpa.stats != nullptr, testCase.shape.withStats);
    if(sdpa.stats != nullptr)
    {
        registerValidator(
            context, sdpa.stats, std::make_unique<RecordingLseValidator>(testCase.name));
    }
    verifyBuiltGraph(context, /*seed=*/0);
}

/// A pair of advertised knob values that no kernel carries reaches the plugin's knob
/// filter, whose refusal is the only thing between it and a plan built from nothing.
TEST_P(IntegrationGpuGfx942AttentionDenseKnobFilter, RefusesAKnobPairNoKernelCarriesAndBuildsNoPlan)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    std::vector<Knob> knobs;
    auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_TRUE(advertises(knobs, BLOCK_M_KNOB, testCase.forcedTile.blockM))
        << BLOCK_M_KNOB << "=" << testCase.forcedTile.blockM
        << " is not advertised for this graph, so the frontend refuses it before the plugin";
    ASSERT_TRUE(advertises(knobs, BLOCK_N_KNOB, testCase.forcedTile.blockN))
        << BLOCK_N_KNOB << "=" << testCase.forcedTile.blockN
        << " is not advertised for this graph, so the frontend refuses it before the plugin";

    result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.forcedTile));
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;

    const std::string refusal
        = std::string("engine '") + ENGINE_NAME + "' has no kernel satisfying";
    EXPECT_NE(result.err_msg.find(refusal), std::string::npos)
        << "expected '" << refusal << "' in: " << result.err_msg;

    result = sdpa.graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    int64_t planEngineId = 0;
    EXPECT_NE(sdpa.graph->get_execution_plan_engine_id(planEngineId).code, ErrorCode::OK)
        << "a plan backed by engine " << planEngineId << " exists";

    EXPECT_FALSE(selectedKernelId(recorder).has_value()) << "Captured logs:\n"
                                                         << recorder.getRecordedLogsAsString();
}

/// hq_minor_swz at Hq = 12 drops and repeats work items, so the engine never offers its
/// kernels to such a graph: neither grid's hq_minor_swz ordinal is advertised (while a
/// graph with Hq = 16 is offered both), and the graph forced to the persistent grid at the
/// swz kernels' tile is served by the bt_hkv_minor arm and matches the CPU reference.
TEST_F(IntegrationGpuGfx942AttentionDenseSwzLegality, NeverOffersAnHqMinorSwzKernelAtHq12)
{
    const auto twelve
        = makeShape(DataType::BFLOAT16, 128, 12, 4, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    const auto sixteen
        = makeShape(DataType::BFLOAT16, 128, 16, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    const int64_t gridSwz = ordinalOf(NONPERSIST_DECODE_KNOB, "hq_minor_swz");
    const int64_t persistentSwz = ordinalOf(PERSIST_DECODE_KNOB, "hq_minor_swz");
    for(const auto& [shape, offered] : {std::pair{sixteen, true}, std::pair{twelve, false}})
    {
        auto sdpa = buildSdpaGraph("SwzLegality", shape);
        ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));
        std::vector<Knob> knobs;
        const auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(advertises(knobs, NONPERSIST_DECODE_KNOB, gridSwz), offered)
            << "Hq " << shape.queryHeads;
        EXPECT_EQ(advertises(knobs, PERSIST_DECODE_KNOB, persistentSwz), offered)
            << "Hq " << shape.queryHeads;
    }

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();
    auto sdpa = buildSdpaGraph("SwzLegalityHq12", twelve);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_M_KNOB, int64_t{256});
    settings.emplace_back(BLOCK_N_KNOB, int64_t{64});
    settings.emplace_back(PERSISTENT_KNOB, int64_t{1});
    auto result = sdpa.graph->create_execution_plan_ext(engineId(), settings);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto selected = selectedKernelId(recorder);
    const auto expected = expectedKernelId(twelve, Tile{256, 64}, persistentGrid("bt_hkv_minor"));
    ASSERT_TRUE(selected.has_value() && expected.has_value());
    EXPECT_EQ(*selected, *expected) << "Captured logs:\n" << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(twelve.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

class IntegrationGpuGfx942AttentionDenseSaveLoad
    : public IntegrationGpuGfx942AttentionDenseBase<KnobCase>
{
protected:
    // Builds the forced-tile graph of the knob case named caseName, validates one run and
    // saves the plan in form. Then it destroys the graph, loads the plan with a new handle and
    // validates two runs of the loaded plan. Every run uses the CPU reference of the original
    // graph, for O and, where the case binds one, the stats output.
    void roundTripInAFreshHandle(SaveForm form, const std::string& caseName)
    {
        const auto cases = knobCases();
        const auto found
            = std::find_if(cases.begin(), cases.end(), [&caseName](const KnobCase& candidate) {
                  return candidate.name == caseName;
              });
        ASSERT_NE(found, cases.end()) << "no knob case is named " << caseName;
        ASSERT_TRUE(found->forced);
        const KnobCase& knobCase = *found;

        auto sdpa = buildSdpaGraph(knobCase.name, knobCase.shape);
        {
            const ScopedPluginLogCapture capture(this);
            ASSERT_NO_FATAL_FAILURE(this->buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

            auto result = sdpa.graph->create_execution_plan_ext(
                engineId(), knobSettingsFor(knobCase.tile, knobCase.grid));
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            result = sdpa.graph->check_support();
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            result = sdpa.graph->build_plans();
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

            // The load below must log no selection. This check proves that the capture
            // records selection lines.
            ASSERT_TRUE(capture.recorder().hasLogContaining(detail::SELECTED_KERNEL_LOG))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }

        int64_t servingEngineId = 0;
        auto result = sdpa.graph->get_execution_plan_engine_id(servingEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngineId, engineId());

        std::vector<BehaviorNote> notes;
        result = sdpa.graph->get_behavior_notes_for_engine(engineId(), notes);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(notes.begin(),
                            notes.end(),
                            BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
                  notes.end());

        VerificationRun run;
        {
            GraphVerificationContext context(*sdpa.graph);
            const float tolerance = sdpaForwardTolerance(knobCase.shape.dataType);
            ASSERT_NO_FATAL_FAILURE(
                this->registerValidator(context, sdpa.output, tolerance, tolerance));
            ASSERT_EQ(sdpa.stats != nullptr, knobCase.shape.withStats);
            if(sdpa.stats != nullptr)
            {
                ASSERT_NO_FATAL_FAILURE(this->registerValidator(
                    context, sdpa.stats, std::make_unique<RecordingLseValidator>(caseName)));
            }
            ASSERT_NO_FATAL_FAILURE(this->prepareVerification(context, /*seed=*/0, run));
        }
        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *sdpa.graph, this->_handle));

        std::vector<uint8_t> saved;
        {
            const ScopedPluginLogCapture capture(this);
            auto save = detail::saveInForm(*sdpa.graph, form);
            ASSERT_EQ(save.second.code, ErrorCode::OK) << save.second.err_msg;
            saved = std::move(save.first);
            EXPECT_FALSE(capture.recorder().hasLogContaining(
                "does not support execution plan serialization"))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }
        if(form == SaveForm::GRAPH_AND_PLAN)
        {
            int contents = 0;
            ASSERT_EQ(
                hipdnnBackendGetSerializedBinaryContents_ext(saved.data(), saved.size(), &contents),
                HIPDNN_STATUS_SUCCESS);
            ASSERT_NE(contents & HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN, 0)
                << "the saved graph holds no execution plan";
        }

        int64_t originalWorkspaceSize = 0;
        result = sdpa.graph->get_workspace_size(originalWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const std::weak_ptr<Graph> original = sdpa.graph;
        sdpa = SdpaGraph{};
        ASSERT_TRUE(original.expired()) << "another owner keeps the original graph alive";

        const ScopedPluginLogCapture capture(this);
        ASSERT_NO_FATAL_FAILURE(detail::replaceHandleWithAFreshOne(this->_handle, this->_stream));

        auto loaded = std::make_shared<Graph>();
        result = detail::loadInForm(*loaded, this->_handle, saved, form);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        EXPECT_TRUE(capture.recorder().hasLogContaining(detail::CREATING_CONTAINER_LOG))
            << "the new handle did not get a new provider container. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
        detail::expectNoKernelSelectionLogged(capture.recorder());

        int64_t loadedEngineId = 0;
        result = loaded->get_execution_plan_engine_id(loadedEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedEngineId, engineId());

        int64_t loadedWorkspaceSize = 0;
        result = loaded->get_workspace_size(loadedWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedWorkspaceSize, originalWorkspaceSize);

        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *loaded, this->_handle));
        ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, *loaded, this->_handle));
    }
};

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_CASE);
}

// The saved stats uid selects the 11-argument kernel on load, and the loaded plan writes the
// LSE to the same output.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAnLseGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_LSE_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAnLsePlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_LSE_CASE);
}

// The saved grid selects the persistent kernel's arguments on load, and the loaded plan
// recomputes the fast-division pairs and the 304-CTA grid from the saved shape.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAPersistentGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_PERSISTENT_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAPersistentPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_PERSISTENT_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAPersistentLsePlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_PERSISTENT_LSE_CASE);
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_PERSISTENT_LSE_CASE);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionDenseKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionDenseKnobFilter,
                         ::testing::ValuesIn(unsatisfiableCases()),
                         [](const ::testing::TestParamInfo<UnsatisfiableCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
