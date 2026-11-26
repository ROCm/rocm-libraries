// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
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
 *        device: a forced tile, grid and levers select exactly one kernel, that kernel
 *        launches with its own geometry and computes what the CPU reference computes, and a
 *        knob pair no kernel has is refused before any plan.
 *
 * The case table is generated so it runs every (head size, dtype, mask, window, LSE, tile)
 * class the catalog compiles, at a shape the tile divides; a coverage case asserts the
 * table and the installed catalog agree. An LSE row binds the frontend's stats output
 * (generate_stats, a packed [B, Hq, Sq, 1] FLOAT tensor), selects the class's emit_lse
 * twin, and grades the LSE against the CPU reference at the default FLOAT SDPA
 * tolerance, printing its largest error as `LSE_ERROR <case> max_abs=.. max_rel=..`. Head
 * counts vary across rows because the body reads them at runtime. Window rows run
 * Sq = Skv = 2W, so the window really excludes keys, and spell the graph's left_bound as W
 * (the reference keeps left_bound keys, diagonal included). Hand-written rows add cold
 * selection, B > 1, Sq != Skv, both alignments, the deprecated causal flag and the default
 * and a non-default scale.
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
 * Each forced row pins the body levers (lds_row_pad, pv_priority, lds_num_buffers,
 * v_reg_prefetch, causal_diag_split, o_store_width, pv_loop_order, use_cfvst, use_exp2_fast,
 * lds_k_group_pad) of its class's KNOB_CLASSES row, and its expected kernel id is resolved at
 * run time from the production descriptors: the one kernel of the class and grid carrying
 * those levers.
 *
 * An additive-bias row binds an fp32 attn_mask next to Q, K and V: a full [B, Hq, Sq, Skv]
 * tensor, or one that broadcasts over the batch, the heads or the queries. The catalog ships a
 * bias kernel only on the qb_minor grid, with no window and the causal or no mask, and sweeps
 * only its tile, so a bias row is forced to its tile and grid alone and its expected kernel is
 * the one of its class carrying use_additive_bias = 1. A plain row's kernel carries 0, which a
 * catalog record that omits the field also means. A bias graph is never offered the
 * persistent grid, nor served when it also has a sliding window: no kernel carries both.
 *
 * A saved plan of one case, and of its LSE twin, is loaded with a new handle in both save
 * forms and must compute the original graph's CPU reference without selecting a kernel
 * again. A bias plan saves the bias uid and its three strides, and a broadcast bias saves a
 * stride of 0 on each extent-1 axis, so the loaded plan must read each stride of its own axis.
 *
 * Tiers: the per-kernel rows (about 400 s on MI300X) are instantiated as `Comprehensive/`,
 * which HIP_MLOPS_ENGINE_test_categories.yaml keeps out of the quick, standard and ffm-quick
 * tiers; the coverage, filter, swizzle-legality and save/load cases run in every tier.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr ServedEngine ENGINE{"hipkernel:Gfx942AttentionDense", "gfx942"};

constexpr const char* BLOCK_M_KNOB = "block_m";
constexpr const char* BLOCK_N_KNOB = "block_n";
constexpr const char* PERSISTENT_KNOB = "persistent";
constexpr const char* NONPERSIST_DECODE_KNOB = "nonpersist_decode";
constexpr const char* PERSIST_DECODE_KNOB = "persist_decode";

/// The persistent CTA count every persistent variant bakes, and the narrow-window bound
/// past which the engine's cold score prefers it at D128.
constexpr int64_t NUM_PERSISTENT = 304;
constexpr int64_t NARROW_WINDOW = 256;

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
// The ALiBi knob cases the save and load cases build. An ALiBi plan saves under the alibi
// contract, and only that name restores the flag: a plan restored without it would launch
// the ALiBi kernel one argument short and score without the bias.
constexpr const char* ALIBI_SAVE_LOAD_CASE = "Alibi_D128_Bf16Hq40Kv8_Sq256Skv384";
constexpr const char* ALIBI_SAVE_LOAD_PERSISTENT_LSE_CASE = "Alibi_Persist_D128_Fp16Hq40Kv8_Lse";
// A windowed ALiBi plan, plain and with LSE: restored under the alibi contract, it must launch
// the windowed ALiBi kernel with the slopes pointer and match the reference.
constexpr const char* ALIBI_WINDOW_SAVE_LOAD_CASE = "Alibi_Window512_D64_Fp16Hq12Kv4_Sq640Skv1024";
constexpr const char* ALIBI_WINDOW_SAVE_LOAD_LSE_CASE
    = "Alibi_Window512_D64_Fp16Hq12Kv4_Sq640Skv1024_Lse";

/// Bias cases (B = 2 or 3, Hq > 1 and Sq != Skv so no axis stride can stand for another), and
/// their LSE twins: a full [B, Hq, Sq, Skv] bias on a GQA graph, and a [B, 1, Sq, Skv] bias
/// whose head axis broadcasts through a stride of 0 between two nonzero strides. The plan
/// saves the bias uid and its strides and restores the 14-argument kernel (15 with the LSE).
constexpr const char* SAVE_LOAD_BIAS_CASE = "Bias_D64_Bf16Causal_Full_CrossLength";
constexpr const char* SAVE_LOAD_BIAS_LSE_CASE = "Bias_D64_Bf16Causal_Full_CrossLength_Lse";
constexpr const char* SAVE_LOAD_BROADCAST_BIAS_CASE = "Bias_D128_Fp16NoMask_PerBatch";
constexpr const char* SAVE_LOAD_BROADCAST_BIAS_LSE_CASE = "Bias_D128_Fp16NoMask_PerBatch_Lse";

/// The suffix of an LSE row's name.
constexpr const char* LSE_SUFFIX = "_Lse";

using Mask = SdpaMask;
using GraphShape = IngestorSdpaGraphSpec;
using BiasLayout = IngestorSdpaBias;

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

/// A kernel's body levers. Each is an engine knob, so pinning them with the tile and grid
/// selects exactly one kernel of a class.
struct Levers
{
    int64_t ldsRowPad;
    int64_t pvPriority;
    int64_t ldsNumBuffers;
    int64_t vRegPrefetch;
    int64_t causalDiagSplit;
    int64_t oStoreWidth;
    std::string_view pvLoopOrder;
    int64_t useCfvst;
    int64_t useExp2Fast;
    int64_t ldsKGroupPad;
};

/// One graph and the tile and grid the engine must serve it with. A forced case sets
/// block_m, block_n, the grid knobs and its kernel's levers; a cold case sets no knob, and
/// the tile and grid are the ones the engine's own ranking picks (coldGrid for a case
/// built with the default).
struct KnobCase
{
    std::string name;
    GraphShape shape;
    bool forced;
    Tile tile;
    Grid grid = {};
    std::optional<Levers> levers = std::nullopt;
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
    return {dataType, headSize, queryHeads, kvHeads, mask, batch, seqQ, seqKv, window};
}

/// @p shape with an fp32 additive bias of layout @p bias as its attn_mask.
GraphShape withBias(GraphShape shape, BiasLayout bias)
{
    shape.bias = bias;
    return shape;
}

/// The catalog's use_additive_bias of a graph: 1 when it carries an attn_mask.
int64_t catalogBias(const GraphShape& shape)
{
    return shape.bias == BiasLayout::NONE ? 0 : 1;
}

/// The part of a bias row's name that spells its bias layout.
std::string biasLayoutName(BiasLayout bias)
{
    const std::array<std::pair<BiasLayout, const char*>, 4> names{
        {{BiasLayout::FULL, "Full"},
         {BiasLayout::PER_BATCH, "PerBatch"},
         {BiasLayout::PER_HEAD, "PerHead"},
         {BiasLayout::KEY_ROW, "KeyRow"}}};
    const auto found = std::find_if(
        names.begin(), names.end(), [bias](const auto& entry) { return entry.first == bias; });
    return found == names.end() ? "NoBias" : found->second;
}

std::string tileName(const Tile& tile)
{
    return "Bm" + std::to_string(tile.blockM) + "Bn" + std::to_string(tile.blockN);
}

/// @p shape with alibi_mask set (hipDNN's ALiBi needs the top-left causal mask it carries).
GraphShape withAlibi(GraphShape shape)
{
    shape.alibi = true;
    return shape;
}

/// @p knobCase's twin that binds the stats output, named with LSE_SUFFIX.
KnobCase lseTwin(KnobCase knobCase)
{
    knobCase.name += LSE_SUFFIX;
    knobCase.shape.withStats = true;
    return knobCase;
}

/// One kernel class of the shipped selection: what a knob case forces one graph onto.
struct KnobClass
{
    int64_t headSize;
    std::string_view dtype;
    bool causal;
    int64_t window;
    bool emitLse;
    bool alibi;
    int64_t blockM;
    int64_t blockN;
    bool persistent;
    std::string_view order;
    Levers levers;
    /// use_additive_bias of the kernel. Last, so a generated row that omits it is a plain one.
    bool useAdditiveBias = false;
};

// KNOB_CLASSES: every class of configs/gfx942_attention_dense.yaml and its ALiBi satellite
// (configs/gfx942_attention_dense_alibi.yaml), generated by
// gen_gfx942_attention_dense_knob_classes.py (tests/test_attention_dense_goldens.py fails
// when it is stale).
#include "IntegrationGpuGfx942AttentionDenseKnobClasses.inc"

/// "QbMinor" for "qb_minor".
std::string camelName(std::string_view snake)
{
    std::string out;
    bool upper = true;
    for(const char c : snake)
    {
        if(c == '_')
        {
            upper = true;
            continue;
        }
        out += upper ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c;
        upper = false;
    }
    return out;
}

/// The graph a class row runs: hq_minor_swz needs Hq a multiple of 8 and, on the
/// persistent grid, an even query-block count; a window cell runs Sq = Skv = max(512, 2W)
/// with one head, or with ALiBi 3 query heads over 1, so the slopes differ per head and
/// take hipDNN's second series; the other rows rotate MHA, GQA and MQA heads. A bias row
/// (qb_minor, no window) runs B = 2 and rotates the four bias layouts, a step slower than the
/// heads so the two vary independently.
KnobCase classCase(const KnobClass& knobClass, std::size_t row)
{
    const std::array<std::pair<int64_t, int64_t>, 4> heads{{{8, 8}, {16, 2}, {8, 1}, {4, 4}}};
    const std::array<BiasLayout, 4> biasLayouts{
        BiasLayout::FULL, BiasLayout::PER_BATCH, BiasLayout::PER_HEAD, BiasLayout::KEY_ROW};
    const auto dataType = knobClass.dtype == "BF16" ? DataType::BFLOAT16 : DataType::HALF;
    const bool swz = knobClass.order == "hq_minor_swz";
    auto [hq, hkv] = heads[row % heads.size()];
    int64_t batch = 1;
    int64_t seq = 512;
    if(swz)
    {
        hq = 16;
        hkv = 2;
        batch = 2;
        seq = 1024;
    }
    if(knobClass.window != 0)
    {
        hq = knobClass.alibi ? 3 : 1;
        hkv = 1;
        seq = std::max<int64_t>(512, 2 * knobClass.window);
    }
    BiasLayout bias = BiasLayout::NONE;
    if(knobClass.useAdditiveBias)
    {
        bias = biasLayouts[(row / heads.size()) % biasLayouts.size()];
        batch = 2;
    }
    Mask mask = knobClass.causal ? Mask::BOUNDS_TOP_LEFT : Mask::NO_MASK;
    std::string cell = knobClass.causal ? "Causal" : "NoMask";
    if(knobClass.window != 0)
    {
        mask = Mask::WINDOW_TOP_LEFT;
        cell = "Window" + std::to_string(knobClass.window);
    }
    if(knobClass.alibi)
    {
        cell += "Alibi";
    }
    std::string name = std::string(knobClass.headSize == 64 ? "D64_" : "D128_")
                       + (dataType == DataType::BFLOAT16 ? "Bf16_" : "Fp16_") + cell + "_"
                       + tileName(Tile{knobClass.blockM, knobClass.blockN});
    if(knobClass.persistent || knobClass.order != "qb_minor")
    {
        name
            += std::string(knobClass.persistent ? "_Persistent" : "_") + camelName(knobClass.order);
    }
    if(bias != BiasLayout::NONE)
    {
        name += std::string("_Bias") + biasLayoutName(bias);
    }
    KnobCase knobCase{
        name,
        withBias(
            makeShape(
                dataType, knobClass.headSize, hq, hkv, mask, knobClass.window, batch, seq, seq),
            bias),
        true,
        Tile{knobClass.blockM, knobClass.blockN},
        Grid{knobClass.persistent, std::string(knobClass.order)},
        knobClass.levers};
    knobCase.shape.alibi = knobClass.alibi;
    return knobClass.emitLse ? lseTwin(std::move(knobCase)) : knobCase;
}

/// The levers of the first KNOB_CLASSES row of @p knobCase's class (bias included), if it has
/// one.
std::optional<Levers> classLevers(const KnobCase& knobCase)
{
    const auto& shape = knobCase.shape;
    const std::string_view dtype = shape.dataType == DataType::BFLOAT16 ? "BF16" : "FP16";
    const bool causal = shape.mask != Mask::NO_MASK;
    for(const auto& row : KNOB_CLASSES)
    {
        if(row.headSize == shape.headSize && row.dtype == dtype && row.causal == causal
           && row.window == shape.window && row.emitLse == shape.withStats
           && row.alibi == shape.alibi && row.blockM == knobCase.tile.blockM
           && row.blockN == knobCase.tile.blockN && row.persistent == knobCase.grid.persistent
           && row.order == knobCase.grid.order
           && row.useAdditiveBias == (shape.bias != BiasLayout::NONE))
        {
            return row.levers;
        }
    }
    return std::nullopt;
}

/// Every catalog class once (KNOB_CLASSES), forced to its tile and grid, then the
/// hand-written cases.
std::vector<KnobCase> knobCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr bool FORCED = true;
    constexpr bool COLD = false;

    std::vector<KnobCase> cases;
    // A class row and its LSE twin: the same shape, so the two differ only in the output.
    const auto addClass = [&cases](KnobCase knobCase) {
        cases.push_back(knobCase);
        cases.push_back(lseTwin(std::move(knobCase)));
    };
    for(std::size_t row = 0; row < KNOB_CLASSES.size(); ++row)
    {
        cases.push_back(classCase(KNOB_CLASSES.at(row), row));
    }

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

    // ALiBi with hipDNN's slopes (SdpaSemanticsRegister SEM-021), each with its LSE twin:
    // non-power-of-two Hq, which takes hipDNN's second slope series, GQA, B > 1 and
    // Sq != Skv (an odd multiple of 128 too), on the qb_minor grid and the persistent grid,
    // and the deprecated causal flag, at 128/64.
    addClass({"Alibi_D128_Bf16Hq40Kv8_Sq256Skv384",
              withAlibi(makeShape(BF16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 256, 384)),
              FORCED,
              Tile{128, 64}});
    addClass({"Alibi_D64_Fp16Hq12Kv4_Sq384Skv640CausalFlag",
              withAlibi(makeShape(FP16, 64, 12, 4, Mask::CAUSAL_MASK_FLAG, 0, 1, 384, 640)),
              FORCED,
              Tile{128, 64}});
    addClass({"Alibi_Persist_D128_Fp16Hq40Kv8",
              withAlibi(makeShape(FP16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 3, 1024, 1024)),
              FORCED,
              Tile{128, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Alibi_Persist_D128_Bf16Hq12Kv4_Sq512Skv1024",
              withAlibi(makeShape(BF16, 128, 12, 4, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 1024)),
              FORCED,
              Tile{128, 64},
              persistentGrid("bt_hkv_minor")});
    // hipDNN defines ALiBi with a left window too: the windowed ALiBi kernels (qb_minor grid
    // only) at GQA with Sq < Skv, and at Sq > Skv by less than W, so the last rows keep only
    // part of their window.
    addClass({"Alibi_Window512_D64_Fp16Hq12Kv4_Sq640Skv1024",
              withAlibi(makeShape(FP16, 64, 12, 4, Mask::WINDOW_TOP_LEFT, 512, 2, 640, 1024)),
              FORCED,
              Tile{128, 64}});
    addClass({"Alibi_Window128_D128_Bf16Hq40Kv8_Sq640Skv576",
              withAlibi(makeShape(BF16, 128, 40, 8, Mask::WINDOW_TOP_LEFT, 128, 1, 640, 576)),
              FORCED,
              Tile{128, 64}});
    // The 256/64 tile: each grid it ships (bt_hkv_minor also with the causal flag).
    addClass({"Alibi256_D128_Bf16Hq40Kv8_Sq512Skv768",
              withAlibi(makeShape(BF16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 768)),
              FORCED,
              Tile{256, 64},
              Grid{false, "bt_hkv_minor"}});
    addClass({"Alibi256_D64_Fp16Hq12Kv4_Sq768Skv1024CausalFlag",
              withAlibi(makeShape(FP16, 64, 12, 4, Mask::CAUSAL_MASK_FLAG, 0, 1, 768, 1024)),
              FORCED,
              Tile{256, 64},
              Grid{false, "bt_hkv_minor"}});
    addClass({"Alibi256_Persist_D128_Fp16Hq40Kv8",
              withAlibi(makeShape(FP16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 3, 1024, 1024)),
              FORCED,
              Tile{256, 64},
              persistentGrid("bt_hkv_minor")});
    addClass({"Alibi256_Window512_D64_Bf16Hq12Kv4_Sq768Skv1024",
              withAlibi(makeShape(BF16, 64, 12, 4, Mask::WINDOW_TOP_LEFT, 512, 2, 768, 1024)),
              FORCED,
              Tile{256, 64}});
    cases.push_back(
        {"Cold_AlibiWindow1024D128",
         withAlibi(makeShape(BF16, 128, 8, 2, Mask::WINDOW_TOP_LEFT, 1024, 1, 2048, 2048)),
         COLD,
         Tile{256, 64}});
    cases.push_back(
        {"Cold_AlibiD128",
         withAlibi(makeShape(BF16, 128, 40, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024)),
         COLD,
         Tile{256, 64}});
    // The score ranks the qb_minor grid first, and the D64 ALiBi kernels without a window
    // ship on it at 128/64 only.
    cases.push_back({"Cold_AlibiD64",
                     withAlibi(makeShape(FP16, 64, 6, 6, Mask::BOUNDS_TOP_LEFT, 0, 1, 512, 512)),
                     COLD,
                     Tile{128, 64}});

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
    // Cold on a graph whose K and V are one tensor: three unique inputs instead of four, so a
    // ranker that read positional tensor features could not evaluate it.
    auto sharedKv = makeShape(BF16, 128, 32, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    sharedKv.sharedKv = true;
    cases.push_back({"Cold_SharedKv", sharedKv, COLD, Tile{256, 64}});
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

    // Additive bias (an fp32 attn_mask), forced to 256/64, which every bias cell ships as its
    // default tile on the qb_minor grid (the generated rows above run the other tiles). Each
    // bias layout runs at B > 1: a full bias on a GQA graph with Sq != Skv and the causal mask
    // (its top-left diagonal against a longer key axis), a bias broadcast over the heads
    // ([B, 1, Sq, Skv], PerBatch), both with an LSE twin, one broadcast over the batch
    // ([1, Hq, Sq, Skv], PerHead), and one row of keys shared by every query and head. A cold
    // row takes the engine's own ranking, which has no persistent bias kernel to prefer at
    // D128.
    constexpr Tile BIAS_TILE{256, 64};
    cases.push_back({SAVE_LOAD_BIAS_CASE,
                     withBias(makeShape(BF16, 64, 4, 1, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 1024),
                              BiasLayout::FULL),
                     FORCED,
                     BIAS_TILE});
    cases.push_back(lseTwin(cases.back()));
    cases.push_back({SAVE_LOAD_BROADCAST_BIAS_CASE,
                     withBias(makeShape(FP16, 128, 16, 2, Mask::NO_MASK, 0, 3, 256, 768),
                              BiasLayout::PER_BATCH),
                     FORCED,
                     BIAS_TILE});
    cases.push_back(lseTwin(cases.back()));
    cases.push_back({"Bias_D64_Fp16Causal_PerHead",
                     withBias(makeShape(FP16, 64, 8, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 512),
                              BiasLayout::PER_HEAD),
                     FORCED,
                     BIAS_TILE});
    cases.push_back(
        {"Bias_D128_Bf16NoMask_KeyRow",
         withBias(makeShape(BF16, 128, 12, 4, Mask::NO_MASK, 0, 2, 512, 512), BiasLayout::KEY_ROW),
         FORCED,
         BIAS_TILE});
    cases.push_back({"Cold_Bias_D128_Bf16Causal_Full",
                     withBias(makeShape(BF16, 128, 8, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
                              BiasLayout::FULL),
                     COLD,
                     BIAS_TILE});
    cases.push_back(lseTwin(cases.back()));

    // A hand-written forced case pins the levers of the first shipped kernel of its class,
    // which need not be the policy arm: the reduction keeps what it measured.
    for(auto& knobCase : cases)
    {
        if(knobCase.forced && !knobCase.levers.has_value())
        {
            knobCase.levers = classLevers(knobCase);
        }
    }
    // Kernels of one class that differ only in their levers get the lever values appended.
    std::set<std::string> names;
    for(auto& knobCase : cases)
    {
        if(!names.insert(knobCase.name).second && knobCase.levers.has_value())
        {
            const auto& l = *knobCase.levers;
            knobCase.name += "_Pad" + std::to_string(l.ldsRowPad) + "Pri"
                             + std::to_string(l.pvPriority) + "Vrp" + std::to_string(l.vRegPrefetch)
                             + "Cds" + std::to_string(l.causalDiagSplit) + "Kgp"
                             + std::to_string(l.ldsKGroupPad) + "Cfv" + std::to_string(l.useCfvst)
                             + "Exp" + std::to_string(l.useExp2Fast);
            names.insert(knobCase.name);
        }
    }
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

std::vector<KnobSetting> knobSettingsFor(const Tile& tile,
                                         const Grid& grid = {},
                                         const std::optional<Levers>& levers = std::nullopt)
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_M_KNOB, tile.blockM);
    settings.emplace_back(BLOCK_N_KNOB, tile.blockN);
    settings.emplace_back(PERSISTENT_KNOB, int64_t{grid.persistent ? 1 : 0});
    const std::string field = grid.persistent ? PERSIST_DECODE_KNOB : NONPERSIST_DECODE_KNOB;
    settings.emplace_back(field, ordinalOf(field, grid.order));
    if(levers.has_value())
    {
        settings.emplace_back("lds_row_pad", levers->ldsRowPad);
        settings.emplace_back("pv_priority", levers->pvPriority);
        settings.emplace_back("lds_num_buffers", levers->ldsNumBuffers);
        settings.emplace_back("v_reg_prefetch", levers->vRegPrefetch);
        settings.emplace_back("causal_diag_split", levers->causalDiagSplit);
        settings.emplace_back("o_store_width", levers->oStoreWidth);
        settings.emplace_back("pv_loop_order",
                              ordinalOf("pv_loop_order", std::string(levers->pvLoopOrder)));
        settings.emplace_back("use_cfvst", levers->useCfvst);
        settings.emplace_back("use_exp2_fast", levers->useExp2Fast);
        settings.emplace_back("lds_k_group_pad", levers->ldsKGroupPad);
    }
    return settings;
}

/// The use_additive_bias a served kernel carries: 0 when its record omits the field, which
/// is the KMD's default, and -1 when the field is not an integer, so it matches no graph.
int64_t carriedBias(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel)
{
    const auto value = kernel.metadata.find("use_additive_bias");
    if(value == kernel.metadata.end())
    {
        return 0;
    }
    const auto* bias = std::get_if<int64_t>(&value->second);
    return bias == nullptr ? -1 : *bias;
}

/// Whether @p kernel carries every field of @p key and the use_additive_bias @p bias.
bool carriesKeyAndBias(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
                       const detail::MetadataKey& key,
                       int64_t bias)
{
    return detail::carriesKey(kernel, key) && carriedBias(kernel) == bias;
}

/// The id of the one kernel this engine serves that carries @p key and the use_additive_bias
/// @p bias, as the selection line spells it, or nullopt (with a recorded failure) when that
/// is not exactly one. detail::onlyServedKernelCarrying with a bias a record may omit.
std::optional<std::string> onlyServedKernelCarryingBias(const detail::MetadataKey& key,
                                                        int64_t bias)
{
    const auto runtimeDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    if(runtimeDir.empty())
    {
        ADD_FAILURE() << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production "
                         "descriptors to read the expected kernel id from";
        return std::nullopt;
    }

    std::vector<std::string> ids;
    detail::forEachServedKernel(
        ENGINE, [&key, bias, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            if(carriesKeyAndBias(kernel, key, bias))
            {
                ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
            }
        });

    if(ids.size() != 1)
    {
        ADD_FAILURE() << ids.size() << " " << ENGINE.name << " kernel(s) for " << ENGINE.arch
                      << " carry " << detail::describeKey(key) << " and use_additive_bias=" << bias
                      << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='" << runtimeDir
                      << "'; exactly one must";
        return std::nullopt;
    }
    return ids.front();
}

/// The id of the one kernel this engine's gfx942 packs carry for @p shape's semantic fields
/// (its bias included), @p tile, @p grid and, when given, @p levers, as the selection line
/// spells it, or nullopt (with a recorded failure) when that is not exactly one. A bias cell
/// sweeps only its tile, so a bias row needs no levers; a plain class has lever variants and
/// does.
std::optional<std::string> expectedKernelId(const GraphShape& shape,
                                            const Tile& tile,
                                            const Grid& grid,
                                            const std::optional<Levers>& levers)
{
    detail::MetadataKey key{
        {"dtype", detail::catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"causal", detail::catalogCausal(shape.mask)},
        {"ragged", int64_t{0}},
        {"sliding_window", shape.window},
        {"block_m", tile.blockM},
        {"block_n", tile.blockN},
        {"emit_lse", int64_t{shape.withStats ? 1 : 0}},
        {"persistent", int64_t{grid.persistent ? 1 : 0}},
        {grid.persistent ? "persist_decode" : "nonpersist_decode", grid.order},
        {"use_alibi", int64_t{shape.alibi ? 1 : 0}},
    };
    if(levers.has_value())
    {
        key.insert(key.end(),
                   {
                       {"lds_row_pad", levers->ldsRowPad},
                       {"pv_priority", levers->pvPriority},
                       {"lds_num_buffers", levers->ldsNumBuffers},
                       {"v_reg_prefetch", levers->vRegPrefetch},
                       {"causal_diag_split", levers->causalDiagSplit},
                       {"o_store_width", levers->oStoreWidth},
                       {"pv_loop_order", std::string(levers->pvLoopOrder)},
                       {"lds_k_group_pad", levers->ldsKGroupPad},
                       {"use_cfvst", levers->useCfvst},
                       {"use_exp2_fast", levers->useExp2Fast},
                   });
    }
    return onlyServedKernelCarryingBias(key, catalogBias(shape));
}

/// The metadata fields a kernel serving @p shape must carry, whatever its tile, grid and levers.
/// Its use_additive_bias is not among them: a record may omit that field (see carriedBias), so
/// selectedCarries matches it separately.
detail::MetadataKey semanticKey(const GraphShape& shape)
{
    return {
        {"dtype", detail::catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"causal", detail::catalogCausal(shape.mask)},
        {"sliding_window", shape.window},
        {"emit_lse", int64_t{shape.withStats ? 1 : 0}},
        {"use_alibi", int64_t{shape.alibi ? 1 : 0}},
    };
}

/// Whether the served kernel with id @p kernelId carries @p key and the use_additive_bias @p bias.
bool selectedCarries(const std::string& kernelId, const detail::MetadataKey& key, int64_t bias)
{
    bool found = false;
    bool matches = false;
    detail::forEachServedKernel(ENGINE,
                                [&](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
                                    if(hipdnn_plugin_sdk::ingestor::toString(kernel.id) != kernelId)
                                    {
                                        return;
                                    }
                                    found = true;
                                    matches = carriesKeyAndBias(kernel, key, bias);
                                });
    EXPECT_TRUE(found) << "no served kernel has id " << kernelId;
    if(found && !matches)
    {
        ADD_FAILURE() << "kernel " << kernelId << " is not of the class "
                      << detail::describeKey(key) << " use_additive_bias=" << bias;
    }
    return found && matches;
}

/// Whether the served kernel with id @p kernelId carries @p shape's semantic fields (its bias
/// included), @p tile and @p grid, whatever its levers.
bool selectedIsOfClass(const std::string& kernelId,
                       const GraphShape& shape,
                       const Tile& tile,
                       const Grid& grid)
{
    auto key = semanticKey(shape);
    key.emplace_back("block_m", tile.blockM);
    key.emplace_back("block_n", tile.blockN);
    key.emplace_back("persistent", int64_t{grid.persistent ? 1 : 0});
    key.emplace_back(grid.persistent ? "persist_decode" : "nonpersist_decode", grid.order);
    return selectedCarries(kernelId, key, catalogBias(shape));
}

int64_t ordinalOf(const std::string& field, const std::string& value)
{
    std::set<std::string> values;
    detail::forEachServedKernel(
        ENGINE, [&field, &values](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
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
    EXPECT_NE(at, values.end()) << "no " << ENGINE.name << " kernel carries " << field << "="
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
/// qb_minor grid otherwise, always for a bias graph, whose catalog has no persistent kernel,
/// and for a windowed ALiBi graph, whose kernels ship on it only.
Grid coldGrid(const GraphShape& shape)
{
    const int cus = deviceMultiProcessorCount();
    if(shape.bias != BiasLayout::NONE || shape.headSize != 128
       || (shape.window != 0 && shape.window <= NARROW_WINDOW) || cus <= 0
       || NUM_PERSISTENT % cus != 0 || (shape.alibi && shape.window != 0))
    {
        return Grid{};
    }
    // The registry pins its decode at the build shape (Sq = Skv = 512).
    const bool causal = shape.mask != Mask::NO_MASK;
    const bool balanced = causal && (shape.window <= 0 || shape.window >= 512);
    return persistentGrid(balanced ? "bt_hkv_minor" : "qb_major");
}

} // namespace

template <typename TestCaseType>
class IntegrationGpuGfx942AttentionDenseBase : public ServedEngineHarness<TestCaseType>
{
protected:
    IntegrationGpuGfx942AttentionDenseBase()
        : ServedEngineHarness<TestCaseType>(ENGINE, "gfx942-dense-knobs-case")
    {
    }
};

using IntegrationGpuGfx942AttentionDenseKnobs = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseKnobFilter
    = IntegrationGpuGfx942AttentionDenseBase<UnsatisfiableCase>;
using IntegrationGpuGfx942AttentionDenseKnobCoverage
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseSwzLegality
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseBiasGrid = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseBiasWindow
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;

/// Every row runs one compiled kernel, and what the compiler specializes on is the head
/// size, dtype, causal flag, window, LSE, ALiBi, tile and additive bias. The table must run
/// every such class the catalog carries, and nothing the catalog lacks.
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
                                   std::string,
                                   int64_t,
                                   int64_t>;

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        if(!knobCase.forced)
        {
            continue;
        }
        tableClasses.emplace(knobCase.shape.headSize,
                             detail::catalogDtypeName(knobCase.shape.dataType),
                             detail::catalogCausal(knobCase.shape.mask),
                             knobCase.shape.window,
                             int64_t{knobCase.shape.withStats ? 1 : 0},
                             knobCase.tile.blockM,
                             knobCase.tile.blockN,
                             int64_t{knobCase.grid.persistent ? 1 : 0},
                             knobCase.grid.order,
                             int64_t{knobCase.shape.alibi ? 1 : 0},
                             catalogBias(knobCase.shape));
    }

    std::set<KernelClass> catalogClasses;
    detail::forEachServedKernel(
        ENGINE, [&catalogClasses](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            const auto intField = [&kernel](const char* field) -> const int64_t* {
                const auto value = kernel.metadata.find(field);
                return value == kernel.metadata.end() ? nullptr
                                                      : std::get_if<int64_t>(&value->second);
            };
            const auto dtype = kernel.metadata.find("dtype");
            const auto* dtypeName = dtype == kernel.metadata.end()
                                        ? nullptr
                                        : std::get_if<std::string>(&dtype->second);
            const auto* headSize = intField("head_size");
            const auto* causal = intField("causal");
            const auto* window = intField("sliding_window");
            const auto* emitLse = intField("emit_lse");
            const auto* blockM = intField("block_m");
            const auto* blockN = intField("block_n");
            const auto* persistent = intField("persistent");
            // A record without use_alibi is a plain kernel (the KMD default 0).
            const auto* alibi = intField("use_alibi");
            const auto stringField = [&kernel](const char* field) -> const std::string* {
                const auto value = kernel.metadata.find(field);
                return value == kernel.metadata.end() ? nullptr
                                                      : std::get_if<std::string>(&value->second);
            };
            const auto* order = persistent != nullptr && *persistent == 1
                                    ? stringField("persist_decode")
                                    : stringField("nonpersist_decode");
            if(dtypeName == nullptr || headSize == nullptr || causal == nullptr || window == nullptr
               || emitLse == nullptr || blockM == nullptr || blockN == nullptr
               || persistent == nullptr || order == nullptr)
            {
                ADD_FAILURE()
                    << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
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
                                   *order,
                                   alibi == nullptr ? int64_t{0} : *alibi,
                                   carriedBias(kernel));
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
                             order,
                             alibi,
                             bias]
                    = kernelClass;
                text += " (D" + std::to_string(headSize) + ", " + dtype + ", causal "
                        + std::to_string(causal) + ", window " + std::to_string(window)
                        + ", emit_lse " + std::to_string(emitLse) + ", " + std::to_string(blockM)
                        + "/" + std::to_string(blockN);
                text += ", persistent " + std::to_string(persistent) + " ";
                text += order;
                text += ", alibi " + std::to_string(alibi);
                text += ", bias " + std::to_string(bias) + ")";
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
    const detail::SelectedKernelReport report(ENGINE, testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    const auto settings = testCase.forced
                              ? knobSettingsFor(testCase.tile, testCase.grid, testCase.levers)
                              : std::vector<KnobSetting>{};
    auto result = sdpa.graph->create_execution_plan_ext(engineId(), settings);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = detail::selectedKernelId(ENGINE, recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE.name << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const auto grid = testCase.forced ? testCase.grid : coldGrid(testCase.shape);
    std::string headId = *selected;
    if(testCase.forced)
    {
        // Tile, grid and levers pinned: exactly one kernel. A bias cell sweeps only its tile,
        // so a bias row needs no levers; a plain class has lever variants and does.
        ASSERT_TRUE(testCase.levers.has_value() || testCase.shape.bias != BiasLayout::NONE)
            << "no KNOB_CLASSES row is of this forced case's class, so nothing pins its levers";
        const auto expectedId
            = expectedKernelId(testCase.shape, testCase.tile, grid, testCase.levers);
        ASSERT_TRUE(expectedId.has_value());
        EXPECT_EQ(*selected, *expectedId) << "Captured logs:\n"
                                          << recorder.getRecordedLogsAsString();
        headId = *expectedId;
    }
    else
    {
        // Cold: a trained ranker bound for this arch picks whichever kernel it ranks fastest,
        // so only the shape's semantic class is pinned (the CPU reference below still checks
        // the launch); one that could not rank fell back to declared kernel order, which is a
        // failure unless the pack left the ranker out because its model.bin was never pulled.
        // Without a trained ranker the engine's score picks the tile and grid; which
        // of the class's lever variants it ranks first is the score's own policy (its goldens
        // pin that).
        const auto decidedBy = detail::rankingDecidedBy(ENGINE, recorder);
        testing::expectRankerDecided(decidedBy,
                                     detail::boundRankerLeftOutAsUnpulled(ENGINE),
                                     recorder.getRecordedLogsAsString());
        if(decidedBy.has_value())
        {
            EXPECT_TRUE(selectedCarries(
                *selected, semanticKey(testCase.shape), catalogBias(testCase.shape)))
                << "Captured logs:\n"
                << recorder.getRecordedLogsAsString();
        }
        else
        {
            EXPECT_TRUE(selectedIsOfClass(*selected, testCase.shape, testCase.tile, grid))
                << "Captured logs:\n"
                << recorder.getRecordedLogsAsString();
        }
    }

    // Rank 0: the selected kernel heads the ranking, rather than being reached past a
    // kernel that failed to load.
    const auto selectionLine = std::string(detail::SELECTED_KERNEL_LOG) + headId + " at rank 0";
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = detail::sdpaForwardTolerance(testCase.shape.dataType);
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

    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    std::vector<Knob> knobs;
    auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_TRUE(detail::advertises(knobs, BLOCK_M_KNOB, testCase.forcedTile.blockM))
        << BLOCK_M_KNOB << "=" << testCase.forcedTile.blockM
        << " is not advertised for this graph, so the frontend refuses it before the plugin";
    ASSERT_TRUE(detail::advertises(knobs, BLOCK_N_KNOB, testCase.forcedTile.blockN))
        << BLOCK_N_KNOB << "=" << testCase.forcedTile.blockN
        << " is not advertised for this graph, so the frontend refuses it before the plugin";

    result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.forcedTile));
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;

    const std::string refusal
        = std::string("engine '") + ENGINE.name + "' has no kernel satisfying";
    EXPECT_NE(result.err_msg.find(refusal), std::string::npos)
        << "expected '" << refusal << "' in: " << result.err_msg;

    result = sdpa.graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    int64_t planEngineId = 0;
    EXPECT_NE(sdpa.graph->get_execution_plan_engine_id(planEngineId).code, ErrorCode::OK)
        << "a plan backed by engine " << planEngineId << " exists";

    EXPECT_FALSE(detail::selectedKernelId(ENGINE, recorder).has_value())
        << "Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}

/// hq_minor_swz at Hq = 12 drops and repeats work items, so the engine never offers its
/// kernels to such a graph: neither grid's hq_minor_swz ordinal is advertised (while a
/// graph with Hq = 16 is offered both; fp16 D128 causal ships both swz arms), and the graph
/// forced to the persistent grid at the swz kernels' tile is served by the bt_hkv_minor arm
/// and matches the CPU reference.
TEST_F(IntegrationGpuGfx942AttentionDenseSwzLegality, NeverOffersAnHqMinorSwzKernelAtHq12)
{
    const auto twelve
        = makeShape(DataType::HALF, 128, 12, 4, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    const auto sixteen
        = makeShape(DataType::HALF, 128, 16, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    const int64_t gridSwz = ordinalOf(NONPERSIST_DECODE_KNOB, "hq_minor_swz");
    const int64_t persistentSwz = ordinalOf(PERSIST_DECODE_KNOB, "hq_minor_swz");
    for(const auto& [shape, offered] : {std::pair{sixteen, true}, std::pair{twelve, false}})
    {
        auto sdpa = detail::buildSdpaGraph("SwzLegality", shape);
        ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));
        std::vector<Knob> knobs;
        const auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(detail::advertises(knobs, NONPERSIST_DECODE_KNOB, gridSwz), offered)
            << "Hq " << shape.queryHeads;
        EXPECT_EQ(detail::advertises(knobs, PERSIST_DECODE_KNOB, persistentSwz), offered)
            << "Hq " << shape.queryHeads;
    }

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();
    auto sdpa = detail::buildSdpaGraph("SwzLegalityHq12", twelve);
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

    const auto selected = detail::selectedKernelId(ENGINE, recorder);
    ASSERT_TRUE(selected.has_value());
    EXPECT_TRUE(selectedIsOfClass(*selected, twelve, Tile{256, 64}, persistentGrid("bt_hkv_minor")))
        << "Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = detail::sdpaForwardTolerance(twelve.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// The catalog ships no persistent bias kernel, so a bias graph is offered the one-CTA-per-
/// block grid only: persistent = 0 is advertised for a D128 graph with a bias and persistent = 1
/// is not, while the same graph without the bias is offered both.
TEST_F(IntegrationGpuGfx942AttentionDenseBiasGrid, NeverOffersThePersistentGridToABiasGraph)
{
    const auto plain
        = makeShape(DataType::BFLOAT16, 128, 8, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024);
    const auto biased = withBias(plain, BiasLayout::FULL);
    for(const auto& [shape, persistentOffered] : {std::pair{plain, true}, std::pair{biased, false}})
    {
        auto sdpa = detail::buildSdpaGraph("BiasGrid", shape);
        ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));
        std::vector<Knob> knobs;
        const auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_TRUE(detail::advertises(knobs, PERSISTENT_KNOB, 0)) << "bias " << catalogBias(shape);
        EXPECT_EQ(detail::advertises(knobs, PERSISTENT_KNOB, 1), persistentOffered)
            << "bias " << catalogBias(shape);
    }
}

/// The catalog ships no bias kernel with a sliding window (its windowed kernels carry no
/// bias, and its bias kernels no window), so the engine is not offered a graph with both: no
/// plan is built, rather than the bias dropped by a window kernel or the window dropped by a
/// bias kernel. The same shape with only the window and with only the bias is offered, so
/// the graph is not refused for anything else.
TEST_F(IntegrationGpuGfx942AttentionDenseBiasWindow, DoesNotServeABiasGraphWithASlidingWindow)
{
    const auto windowed
        = makeShape(DataType::BFLOAT16, 128, 8, 2, Mask::WINDOW_TOP_LEFT, 128, 2, 512, 512);
    const auto biased
        = withBias(makeShape(DataType::BFLOAT16, 128, 8, 2, Mask::BOUNDS_TOP_LEFT, 0, 2, 512, 512),
                   BiasLayout::FULL);
    const auto both = withBias(windowed, BiasLayout::FULL);
    for(const auto& [name, shape, offered] : {std::tuple{"WindowOnly", windowed, true},
                                              std::tuple{"BiasOnly", biased, true},
                                              std::tuple{"BiasAndWindow", both, false}})
    {
        SCOPED_TRACE(name);
        auto sdpa = detail::buildSdpaGraph(name, shape);
        auto result = sdpa.graph->build_operation_graph(this->_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> rankedEngineIds;
        result = sdpa.graph->get_ranked_engine_ids(rankedEngineIds);
        const bool listed = result.code == ErrorCode::OK
                            && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId())
                                   != rankedEngineIds.end();
        EXPECT_EQ(listed, offered)
            << ENGINE.name << (offered ? " did not offer" : " offered") << " to serve this graph";
        if(!offered)
        {
            // Forcing the engine does not build a plan either.
            result = sdpa.graph->create_execution_plan_ext(engineId(), {});
            EXPECT_NE(result.code, ErrorCode::OK) << "a plan was created for " << ENGINE.name;
        }
    }
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

        auto sdpa = detail::buildSdpaGraph(knobCase.name, knobCase.shape);
        {
            const ScopedPluginLogCapture capture(this);
            ASSERT_NO_FATAL_FAILURE(this->buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

            auto result = sdpa.graph->create_execution_plan_ext(
                engineId(), knobSettingsFor(knobCase.tile, knobCase.grid, knobCase.levers));
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
            const float tolerance = detail::sdpaForwardTolerance(knobCase.shape.dataType);
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

// An ALiBi plan carries no ALiBi launch value; only its contract name restores the flag. A
// loaded plan that launched without ALiBi, or with v1's argument list, fails the reference.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAnAlibiGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, ALIBI_SAVE_LOAD_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAnAlibiPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, ALIBI_SAVE_LOAD_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAPersistentLseAlibiPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, ALIBI_SAVE_LOAD_PERSISTENT_LSE_CASE);
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, ALIBI_SAVE_LOAD_PERSISTENT_LSE_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresAWindowedAlibiPlanInAFreshHandle)
{
    for(const auto* name : {ALIBI_WINDOW_SAVE_LOAD_CASE, ALIBI_WINDOW_SAVE_LOAD_LSE_CASE})
    {
        SCOPED_TRACE(name);
        roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, name);
        roundTripInAFreshHandle(SaveForm::PLAN_ONLY, name);
    }
}

// A bias plan saves the bias uid and its strides, and restores the 14-argument kernel (the
// bias pointer and its batch, head and query strides after the plain arguments): a loaded
// plan that dropped the bias, or read the stride of another axis, fails the comparison.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABiasGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_BIAS_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABiasPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_BIAS_CASE);
}

// The bias and the stats output together: the plan saves both uids and restores the
// 15-argument kernel.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABiasLsePlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_BIAS_LSE_CASE);
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_BIAS_LSE_CASE);
}

// A bias broadcast over the heads saves a stride of 0 between two nonzero strides.
TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABroadcastBiasGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_BROADCAST_BIAS_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABroadcastBiasPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_BROADCAST_BIAS_CASE);
}

TEST_F(IntegrationGpuGfx942AttentionDenseSaveLoad, RestoresABroadcastBiasLsePlanInAFreshHandle)
{
    roundTripInAFreshHandle(SaveForm::GRAPH_AND_PLAN, SAVE_LOAD_BROADCAST_BIAS_LSE_CASE);
    roundTripInAFreshHandle(SaveForm::PLAN_ONLY, SAVE_LOAD_BROADCAST_BIAS_LSE_CASE);
}

INSTANTIATE_TEST_SUITE_P(Comprehensive,
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
