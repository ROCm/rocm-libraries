// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)                          \
    && (defined(HKP_DETERMINISM_ROCKE_GFX950_DENSE) || defined(HKP_DETERMINISM_ROCKE_GFX942_DENSE) \
        || defined(HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_2D)                                        \
        || defined(HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_3D)                                        \
        || defined(HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_2D)                                        \
        || defined(HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_3D))

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismRocke.cpp
 * @brief The determinism rows of the rocKE attention engines hipkernel:Gfx950AttentionDense,
 *        Gfx942AttentionDense, Gfx942AttentionUnified2D, Gfx942AttentionUnified3d,
 *        Gfx950AttentionUnified2D and Gfx950AttentionUnified3D. CMake defines
 *        HKP_DETERMINISM_ROCKE_<ENGINE> for each engine the product root offers, and each
 *        table below exists only under its definition.
 *
 * Every row is a forward graph through the frontend (dense BSHD Q/K/V/O, the scale 1/sqrt(D)),
 * pinned to its engine with the knobs that select its kernel class. The rocKE forward
 * generators carry no atomics and no dynamic work counter; the only cross-workgroup combine is
 * the fixed-order reduce of the 3D segment pair (the T0 trace of the rocKE engines). A row set
 * therefore has to reach every distinct launch structure, not every kernel: the persistent
 * grid-stride loop (W below and above the CTA count), the one-CTA-per-item grids at each block
 * order, the 2D tiled grids at each geometry (several q-blocks per sequence, a tail q-block, many
 * sequences), and the 3D segment and reduce pair (a segment with no key, many blocks per
 * segment, the smallest and the largest num_segments). Every row's shape is one the engine's
 * matcher accepts and its knob combination is one its shipped catalog carries; the sources are
 * named above each table.
 *
 * Rows the engines decline are left out on purpose: a ragged, padded, paged or dropout graph (no
 * rocKE attention engine serves one: the unified engines build their own identity cache), the
 * dense engines' tail q-block or kv-block (Sq % block_m and Skv % block_n must be 0), and a
 * top-left causal decode graph without LSE (its one live key makes the output independent of Q,
 * so the sensitivity launch could not move it).
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::integration::FrontendSdpaSpec;
using hipdnn_frontend::DataType;
using hipdnn_frontend::DiagonalAlignment;

[[maybe_unused]] constexpr DataType BF16 = DataType::BFLOAT16;
[[maybe_unused]] constexpr DataType FP16 = DataType::HALF;

/// The causal band of a row: none, or the right bound 0 anchored at the top-left or the
/// bottom-right corner (they coincide at equal lengths).
enum class Band
{
    NONE,
    TOP_LEFT,
    BOTTOM_RIGHT,
};

/// One forward graph: Q [B, Hq, Sq, D] and K, V [B, Hkv, Skv, D], all BSHD.
struct Shape
{
    DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    Band band;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
    /// Binds the stats (LSE) output.
    bool withStats;
    /// A causal window of this many keys (left bound), top-left anchored; none when unset.
    std::optional<int64_t> window;
};

[[maybe_unused]] Shape shapeOf(DataType dataType,
                               int64_t headSize,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               Band band,
                               int64_t batch,
                               int64_t seqQ,
                               int64_t seqKv,
                               bool withStats = false,
                               std::optional<int64_t> window = std::nullopt)
{
    return Shape{
        dataType, headSize, queryHeads, kvHeads, band, batch, seqQ, seqKv, withStats, window};
}

[[maybe_unused]] FrontendSdpaSpec specOf(const std::string& name, const Shape& shape)
{
    FrontendSdpaSpec spec;
    spec.name = name;
    spec.dataType = shape.dataType;
    spec.batch = shape.batch;
    spec.queryHeads = shape.queryHeads;
    spec.kvHeads = shape.kvHeads;
    spec.seqQ = shape.seqQ;
    spec.seqKv = shape.seqKv;
    spec.headSize = shape.headSize;
    spec.leftBound.reset();
    spec.rightBound.reset();
    if(shape.band != Band::NONE)
    {
        spec.rightBound = 0;
        spec.alignment = shape.band == Band::TOP_LEFT ? DiagonalAlignment::TOP_LEFT
                                                      : DiagonalAlignment::BOTTOM_RIGHT;
    }
    if(shape.window.has_value())
    {
        spec.leftBound = shape.window;
    }
    spec.withStats = shape.withStats;
    return spec;
}

/// The letters-and-digits token of a shape, the tail of every row's case name.
[[maybe_unused]] std::string shapeToken(const Shape& shape)
{
    std::string token = shape.dataType == FP16 ? "Fp16" : "Bf16";
    token += "Hd" + std::to_string(shape.headSize) + "Hq" + std::to_string(shape.queryHeads) + "Kv"
             + std::to_string(shape.kvHeads);
    if(shape.band == Band::NONE)
    {
        token += "NoMask";
    }
    else if(shape.band == Band::TOP_LEFT)
    {
        token += "TopLeft";
    }
    else
    {
        token += "BottomRight";
    }
    token += "B" + std::to_string(shape.batch) + "S" + std::to_string(shape.seqQ) + "x"
             + std::to_string(shape.seqKv);
    if(shape.window.has_value())
    {
        token += "Win" + std::to_string(*shape.window);
    }
    if(shape.withStats)
    {
        token += "Lse";
    }
    return token;
}

/// An engine and the one architecture its rows run on.
struct Served
{
    const char* engine;
    const char* arch;
};

[[maybe_unused]] void addRow(std::vector<DeterminismRow>& rows,
                             const Served& served,
                             const std::string& caseName,
                             KnobClass knobClass,
                             KnobList knobs,
                             const Shape& shape)
{
    rows.push_back(forwardRow(RowMeta{served.engine,
                                      caseName,
                                      std::vector<std::string>{served.arch},
                                      Sku::ANY,
                                      Disposition::PROVEN,
                                      knobClass,
                                      std::move(knobs),
                                      false},
                              specOf(caseName, shape)));
}

std::vector<DeterminismRow> rockeRows()
{
    std::vector<DeterminismRow> rows;
#ifdef HKP_DETERMINISM_ROCKE_GFX950_DENSE
    {
        // One launch per (q-block, head, batch), no persistent grid and no kv split (the gfx950
        // KMD has neither field), so the launch structure does not vary; what varies is the
        // compiled kernel. The rows are the shapes and tiles of the repo's own knob suite
        // (IntegrationGpuGfx950AttentionDenseKnobs.cpp knobCases, which holds its table to the
        // catalog's kernel classes in CoversEveryCatalogKernelClass and head configurations in
        // CoversEveryCatalogHeadConfiguration): one row per (head size, dtype, causal, block_m,
        // block_n) class, 52 classes, and every one of the 33 (head size, Hq, Hkv) configurations
        // the catalog bakes (the matcher compares head counts). The tile is pinned through the
        // block_m and block_n knobs; the knob suite's mma_core_mode and attn_scale variants
        // select no other kernel and are left out. The graph asks for no LSE (no catalog kernel
        // writes one) and Sq, Skv divide the tile (the matcher requires it), so there is no tail
        // row. A top-left or bottom-right band is the same kernel (causal 1).
        const Served served{"hipkernel:Gfx950AttentionDense", "gfx950"};
        const auto addDense = [&](DataType dataType,
                                  int64_t headSize,
                                  int64_t queryHeads,
                                  int64_t kvHeads,
                                  Band band,
                                  int64_t batch,
                                  int64_t seqQ,
                                  int64_t seqKv,
                                  int64_t blockM,
                                  int64_t blockN) {
            const auto shape
                = shapeOf(dataType, headSize, queryHeads, kvHeads, band, batch, seqQ, seqKv);
            addRow(rows,
                   served,
                   shapeToken(shape) + "Bm" + std::to_string(blockM) + "Bn"
                       + std::to_string(blockN),
                   KnobClass::TILE_SCHEDULE,
                   {{"block_m", blockM}, {"block_n", blockN}},
                   shape);
        };
        addDense(BF16, 64, 8, 8, Band::NONE, 1, 256, 256, 128, 32);
        addDense(FP16, 64, 16, 2, Band::TOP_LEFT, 2, 128, 256, 128, 64);
        addDense(BF16, 64, 8, 1, Band::NONE, 2, 384, 128, 128, 128);
        addDense(FP16, 64, 8, 8, Band::TOP_LEFT, 1, 256, 256, 256, 32);
        addDense(BF16, 64, 16, 16, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 128);
        addDense(FP16, 64, 12, 12, Band::NONE, 1, 512, 256, 256, 256);
        addDense(BF16, 64, 10, 10, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 64);
        addDense(FP16, 128, 8, 2, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 32);
        addDense(BF16, 128, 8, 1, Band::BOTTOM_RIGHT, 1, 128, 128, 128, 64);
        addDense(FP16, 128, 4, 4, Band::NONE, 3, 128, 384, 128, 128);
        addDense(BF16, 128, 16, 2, Band::NONE, 2, 256, 480, 256, 32);
        addDense(BF16, 128, 9, 9, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 128);
        addDense(FP16, 128, 8, 8, Band::NONE, 1, 256, 256, 256, 64);
        addDense(BF16, 64, 16, 2, Band::TOP_LEFT, 2, 256, 512, 256, 64);
        addDense(BF16, 128, 8, 1, Band::NONE, 3, 384, 256, 128, 32);
        addDense(FP16, 64, 8, 8, Band::BOTTOM_RIGHT, 2, 512, 512, 256, 64);
        addDense(BF16, 128, 9, 9, Band::BOTTOM_RIGHT, 2, 256, 256, 256, 64);
        addDense(BF16, 64, 20, 20, Band::NONE, 1, 256, 256, 128, 64);
        addDense(BF16, 64, 32, 4, Band::NONE, 1, 256, 256, 256, 32);
        addDense(BF16, 64, 32, 8, Band::NONE, 1, 256, 256, 256, 128);
        addDense(BF16, 64, 32, 32, Band::NONE, 1, 256, 256, 256, 256);
        addDense(BF16, 64, 64, 8, Band::TOP_LEFT, 1, 256, 256, 128, 32);
        addDense(BF16, 64, 128, 128, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 64);
        addDense(BF16, 128, 16, 16, Band::NONE, 1, 256, 256, 128, 64);
        addDense(BF16, 128, 24, 24, Band::NONE, 1, 256, 256, 128, 128);
        addDense(BF16, 128, 28, 4, Band::NONE, 1, 256, 256, 256, 128);
        addDense(BF16, 128, 32, 4, Band::TOP_LEFT, 1, 256, 256, 128, 32);
        addDense(BF16, 128, 32, 8, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 128);
        addDense(BF16, 128, 32, 32, Band::TOP_LEFT, 1, 256, 256, 256, 32);
        addDense(FP16, 128, 40, 8, Band::NONE, 1, 256, 256, 128, 32);
        addDense(FP16, 128, 40, 10, Band::NONE, 1, 256, 256, 128, 64);
        addDense(FP16, 128, 40, 40, Band::NONE, 1, 256, 256, 256, 32);
        addDense(FP16, 128, 48, 8, Band::NONE, 1, 256, 256, 256, 128);
        addDense(FP16, 128, 64, 4, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 64);
        addDense(FP16, 128, 64, 8, Band::TOP_LEFT, 1, 256, 256, 128, 128);
        addDense(FP16, 128, 64, 64, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 32);
        addDense(FP16, 128, 96, 8, Band::TOP_LEFT, 1, 256, 256, 256, 128);
        addDense(FP16, 128, 128, 8, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 32);
        addDense(BF16, 64, 8, 1, Band::TOP_LEFT, 1, 256, 256, 128, 128);
        addDense(BF16, 64, 8, 8, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 32);
        addDense(BF16, 64, 10, 10, Band::TOP_LEFT, 1, 256, 256, 256, 256);
        addDense(FP16, 64, 12, 12, Band::NONE, 1, 256, 256, 128, 64);
        addDense(FP16, 64, 16, 2, Band::NONE, 1, 256, 256, 128, 128);
        addDense(FP16, 64, 16, 16, Band::NONE, 1, 256, 256, 256, 32);
        addDense(FP16, 64, 20, 20, Band::NONE, 1, 256, 256, 256, 128);
        addDense(FP16, 64, 8, 1, Band::BOTTOM_RIGHT, 1, 256, 256, 128, 32);
        addDense(FP16, 64, 8, 8, Band::TOP_LEFT, 1, 256, 256, 128, 128);
        addDense(FP16, 64, 10, 10, Band::BOTTOM_RIGHT, 1, 256, 256, 256, 128);
        addDense(FP16, 64, 12, 12, Band::TOP_LEFT, 1, 256, 256, 256, 256);
        addDense(BF16, 64, 8, 8, Band::NONE, 1, 256, 256, 256, 64);
        addDense(FP16, 64, 16, 16, Band::NONE, 1, 256, 256, 256, 64);
        addDense(BF16, 128, 8, 8, Band::NONE, 1, 256, 256, 256, 64);
        addDense(FP16, 128, 16, 16, Band::TOP_LEFT, 2, 256, 512, 256, 64);
        addDense(FP16, 64, 32, 8, Band::NONE, 1, 384, 256, 128, 32);

        // The ranker default: no knob, so the shape-blind scorer picks the baseline 256/64 tile.
        const auto cold = shapeOf(BF16, 64, 8, 8, Band::NONE, 1, 256, 256);
        addRow(rows, served, shapeToken(cold) + "Cold", KnobClass::TILE_SCHEDULE, {}, cold);
    }
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX942_DENSE
    {
        // The kernel classes of IntegrationGpuGfx942AttentionDenseKnobClasses.inc, one row per
        // launch-structure class (the inc's 224 rows differ in body levers that change
        // instruction order and layout only; the T0 trace found no atomic, counter or second
        // writer in any of them): the one-CTA-per-block grids at each block order, the
        // persistent grid at each decode with W work items below, not a multiple of and above
        // its 304 CTAs, causal_diag_split, the LSE twin, the windowed cells and the ranker
        // default. Each row pins block_m, block_n, persistent and the decode; the levers are
        // left to the engine's score, so a row may run any kernel of its class, except the
        // lever rows. The string knobs are addressed by ordinal, the index into the sorted value
        // set of the field over the engine's kernels
        // (KernelIngestorStateManager::buildOrdinalDomains), with KMD defaults filled in:
        // nonpersist_decode {bt_hkv_minor, hq_minor_swz, qb_minor}, persist_decode {auto,
        // bt_hkv_minor, hkv_major, hq_minor_swz, qb_major}. Shapes follow the knob suite's
        // classCase and hand-written persistent cases
        // (IntegrationGpuGfx942AttentionDenseKnobs.cpp); hq_minor_swz needs Hq a multiple of 8
        // and, persistent, an even query-block count. The gfx942 packs divide Sq by block_m and
        // Skv by block_n as the gfx950 ones do, so there is no tail row; the cold Seq384 row is
        // served by the 64/64 tile.
        const Served served{"hipkernel:Gfx942AttentionDense", "gfx942"};
        constexpr int64_t QB_MINOR = 2;
        constexpr int64_t BT_HKV_MINOR = 0;
        constexpr int64_t HQ_MINOR_SWZ = 1;
        constexpr int64_t PERSIST_BT_HKV_MINOR = 1;
        constexpr int64_t PERSIST_HKV_MAJOR = 2;
        constexpr int64_t PERSIST_HQ_MINOR_SWZ = 3;
        constexpr int64_t PERSIST_QB_MAJOR = 4;
        const auto grid = [](int64_t blockM, int64_t blockN, int64_t decode) -> KnobList {
            return {{"block_m", blockM},
                    {"block_n", blockN},
                    {"persistent", 0},
                    {"nonpersist_decode", decode}};
        };
        const auto persistent = [](int64_t decode) -> KnobList {
            return {
                {"block_m", 256}, {"block_n", 64}, {"persistent", 1}, {"persist_decode", decode}};
        };
        const auto withLever = [](KnobList knobs, const char* lever, int64_t value) {
            knobs.emplace_back(lever, value);
            return knobs;
        };
        const auto add = [&](const std::string& name, KnobList knobs, const Shape& shape) {
            addRow(rows,
                   served,
                   name + shapeToken(shape),
                   KnobClass::STRUCTURE,
                   std::move(knobs),
                   shape);
        };

        // One CTA per q-block, head and batch at each block order.
        add("NonpersistQbMinor",
            grid(256, 64, QB_MINOR),
            shapeOf(BF16, 128, 16, 2, Band::TOP_LEFT, 1, 512, 512));
        add("NonpersistBtHkvMinor",
            grid(256, 64, BT_HKV_MINOR),
            shapeOf(BF16, 128, 16, 2, Band::TOP_LEFT, 2, 512, 512));
        add("NonpersistHqMinorSwz",
            grid(256, 64, HQ_MINOR_SWZ),
            shapeOf(FP16, 128, 16, 2, Band::TOP_LEFT, 2, 1024, 1024));
        add("NonpersistHqMinorSwz",
            grid(256, 64, HQ_MINOR_SWZ),
            shapeOf(BF16, 64, 16, 2, Band::NONE, 2, 1024, 1024));
        add("NonpersistQbMinor",
            grid(256, 64, QB_MINOR),
            shapeOf(FP16, 64, 8, 8, Band::NONE, 1, 512, 512));
        // The smaller tiles: more q-blocks per sequence.
        add("NonpersistTile64x64",
            grid(64, 64, QB_MINOR),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("NonpersistTile128x64",
            grid(128, 64, QB_MINOR),
            shapeOf(FP16, 64, 8, 8, Band::TOP_LEFT, 1, 512, 512));
        add("NonpersistTile256x32",
            grid(256, 32, QB_MINOR),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        // causal_diag_split splits the diagonal tile's work inside the CTA.
        auto diagSplit = grid(256, 64, QB_MINOR);
        diagSplit.emplace_back("causal_diag_split", 1);
        add("NonpersistCausalDiagSplit",
            std::move(diagSplit),
            shapeOf(BF16, 64, 8, 8, Band::TOP_LEFT, 1, 512, 512));
        // The body levers of a kernel of one class (instruction order, LDS padding, the exp2
        // path): the T0 trace found no writer difference among them, and their LDS schedule is
        // outside it, so the device repeat is the check on each.
        add("LeverPvPriority1",
            withLever(grid(256, 64, QB_MINOR), "pv_priority", 1),
            shapeOf(BF16, 64, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverPvPriority2",
            withLever(grid(256, 64, QB_MINOR), "pv_priority", 2),
            shapeOf(BF16, 64, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverPvPriority3",
            withLever(grid(256, 64, QB_MINOR), "pv_priority", 3),
            shapeOf(BF16, 64, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverLdsKGroupPad24",
            withLever(grid(256, 64, QB_MINOR), "lds_k_group_pad", 24),
            shapeOf(BF16, 64, 8, 8, Band::NONE, 1, 512, 512, true));
        add("LeverUseExp2Fast0",
            withLever(grid(256, 64, QB_MINOR), "use_exp2_fast", 0),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverUseExp2Fast1",
            withLever(grid(256, 64, QB_MINOR), "use_exp2_fast", 1),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        // The LSE twin writes the stats from one lane per row.
        add("NonpersistQbMinor",
            grid(256, 64, QB_MINOR),
            shapeOf(BF16, 128, 16, 2, Band::TOP_LEFT, 2, 512, 512, true));
        add("NonpersistHqMinorSwz",
            grid(256, 64, HQ_MINOR_SWZ),
            shapeOf(BF16, 64, 16, 2, Band::NONE, 2, 1024, 1024, true));

        // The persistent grid: 304 CTAs stride over W = q-blocks * Hq * B work items. W is 16
        // (CTAs left idle), 32, 72, 128, 168, 480 (not a multiple of 304) and 1024 (3.4 rounds).
        add("PersistQbMajorWBelowNp",
            persistent(PERSIST_QB_MAJOR),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("PersistQbMajorCrossLength",
            persistent(PERSIST_QB_MAJOR),
            shapeOf(FP16, 128, 12, 4, Band::NONE, 3, 512, 1280));
        add("PersistQbMajorWAboveNp",
            persistent(PERSIST_QB_MAJOR),
            shapeOf(FP16, 128, 40, 8, Band::NONE, 3, 1024, 1024));
        add("PersistHkvMajor",
            persistent(PERSIST_HKV_MAJOR),
            shapeOf(BF16, 128, 40, 8, Band::NONE, 3, 1024, 1024));
        add("PersistBtHkvMinorWNotAMultipleOfNp",
            persistent(PERSIST_BT_HKV_MINOR),
            shapeOf(BF16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));
        add("PersistBtHkvMinorWFarAboveNp",
            persistent(PERSIST_BT_HKV_MINOR),
            shapeOf(BF16, 128, 128, 8, Band::TOP_LEFT, 4, 512, 512));
        add("PersistBtHkvMinorOddQueryBlocks",
            persistent(PERSIST_BT_HKV_MINOR),
            shapeOf(FP16, 128, 28, 4, Band::TOP_LEFT, 2, 768, 768));
        add("PersistBtHkvMinorCrossLength",
            persistent(PERSIST_BT_HKV_MINOR),
            shapeOf(BF16, 128, 8, 2, Band::TOP_LEFT, 2, 512, 1024));
        auto persistDiagSplit = persistent(PERSIST_BT_HKV_MINOR);
        persistDiagSplit.emplace_back("causal_diag_split", 1);
        add("PersistBtHkvMinorCausalDiagSplit",
            std::move(persistDiagSplit),
            shapeOf(BF16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));
        add("PersistHqMinorSwz",
            persistent(PERSIST_HQ_MINOR_SWZ),
            shapeOf(FP16, 128, 16, 2, Band::TOP_LEFT, 2, 1024, 1024));
        add("PersistHkvMajor",
            persistent(PERSIST_HKV_MAJOR),
            shapeOf(BF16, 128, 40, 8, Band::NONE, 3, 1024, 1024, true));
        // The causal pair-fold work order (even q-block count, W = 480 > 304 CTAs).
        add("PersistPairFoldHkvMajor",
            persistent(PERSIST_HKV_MAJOR),
            shapeOf(BF16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));
        add("PersistPairFoldHqMinorSwz",
            persistent(PERSIST_HQ_MINOR_SWZ),
            shapeOf(BF16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));
        add("PersistPairFoldHqMinorSwz",
            persistent(PERSIST_HQ_MINOR_SWZ),
            shapeOf(FP16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));
        add("PersistHqMinorSwz",
            persistent(PERSIST_HQ_MINOR_SWZ),
            shapeOf(FP16, 128, 16, 2, Band::TOP_LEFT, 2, 1024, 1024, true));

        // The persistent body levers: the LDS row padding of the qb_major kernels and the V
        // register prefetch of the causal bt_hkv_minor kernel.
        add("LeverLdsRowPad4",
            withLever(persistent(PERSIST_QB_MAJOR), "lds_row_pad", 4),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverLdsRowPad12",
            withLever(persistent(PERSIST_QB_MAJOR), "lds_row_pad", 12),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("LeverLdsRowPad24",
            withLever(persistent(PERSIST_QB_MAJOR), "lds_row_pad", 24),
            shapeOf(BF16, 128, 8, 8, Band::NONE, 1, 512, 512));
        add("PersistBtHkvMinorVRegPrefetch",
            withLever(persistent(PERSIST_BT_HKV_MINOR), "v_reg_prefetch", 1),
            shapeOf(BF16, 128, 40, 8, Band::TOP_LEFT, 3, 1024, 1024));

        // The windowed cells (Sq = Skv, the window really excludes keys): the left bound W with
        // the right bound 0, top-left anchored, on the one-CTA-per-block grid.
        add("Window",
            grid(256, 64, QB_MINOR),
            shapeOf(BF16, 128, 1, 1, Band::TOP_LEFT, 1, 512, 512, false, 128));
        add("Window",
            grid(256, 64, QB_MINOR),
            shapeOf(BF16, 128, 1, 1, Band::TOP_LEFT, 1, 1024, 1024, true, 512));
        add("Window",
            grid(256, 64, QB_MINOR),
            shapeOf(FP16, 64, 1, 1, Band::TOP_LEFT, 1, 4096, 4096, false, 2048));

        // The ranker default: no knob; the scorer picks the persistent arm at D128 where the
        // CTA count suits the device, and the smallest tile that divides a short sequence.
        add("Cold", {}, shapeOf(BF16, 128, 32, 8, Band::TOP_LEFT, 2, 1024, 1024));
        add("Cold", {}, shapeOf(FP16, 64, 8, 8, Band::NONE, 1, 384, 384));
    }
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_2D
    {
        // One CTA per (kv head, q-block) with a binary-search sequence lookup, a single launch,
        // and no cross-CTA combine (T0 trace). The classes are the 14 geometry x head size x
        // dtype kernel classes the reduced catalog ships (the forced table of
        // IntegrationGpuGfx942AttentionUnified2DKnobs.cpp knobCases, each on a head pair whose
        // GQA ratio the class ships), forced through the ten geometry and codegen-stack knobs at
        // B 2, Sq 96, Skv 256: two sequences, several q-blocks per sequence, and a tail q-block
        // where block_m (num_warps * block_m_per_warp) does not divide 96. Then the cold rows the
        // engine's own ranking serves: B 3, several q-blocks, a tail (Sq 769) and B 512. The
        // catalog carries the bottom-right causal kernel only, so every band is bottom-right; the
        // graph asks for no LSE (no gfx942 kernel writes one) and no paged or padded K/V (the
        // engine builds the identity cache itself, and Skv is a whole number of 16-token blocks).
        const Served served{"hipkernel:Gfx942AttentionUnified2D", "gfx942"};
        struct Geometry
        {
            const char* name;
            int64_t tileSize;
            int64_t numWarps;
            int64_t blockMPerWarp;
            int64_t useMfma32x32x8;
            int64_t useTransposedQk32x32;
            int64_t useKSlicedRing;
            int64_t useQDirectGlobal;
            int64_t useConflictFreeVStore;
            int64_t useTransposedMaskLimit;
            int64_t useTransposedInvariantHoist;
        };
        constexpr Geometry D64_TRANSPOSED{"Transposed4x32T64", 64, 4, 32, 1, 1, 0, 0, 0, 0, 0};
        constexpr Geometry D64_PRODUCTION{"Production4x32T64", 64, 4, 32, 1, 1, 1, 1, 1, 1, 1};
        constexpr Geometry D64_NARROW{"Narrow2x16T64", 64, 2, 16, 0, 0, 0, 0, 0, 0, 0};
        constexpr Geometry D128_T_2X32{"Transposed2x32T32", 32, 2, 32, 1, 1, 0, 0, 0, 0, 0};
        constexpr Geometry D128_T_4X32{"Transposed4x32T32", 32, 4, 32, 1, 1, 0, 0, 0, 0, 0};
        constexpr Geometry D128_NARROW_4X16{"Narrow4x16T32", 32, 4, 16, 0, 0, 0, 0, 0, 0, 0};
        constexpr Geometry D128_NARROW_2X32{"Narrow2x32T32", 32, 2, 32, 0, 0, 0, 0, 0, 0, 0};
        constexpr Geometry D128_NARROW_2X16{"Narrow2x16T16", 16, 2, 16, 0, 0, 0, 0, 0, 0, 0};
        constexpr Geometry D128_PRODUCTION{"Production4x32T64Ring", 64, 4, 32, 1, 1, 1, 0, 1, 1, 1};
        const auto forced = [&](const Geometry& g,
                                DataType dataType,
                                int64_t headSize,
                                int64_t queryHeads,
                                int64_t kvHeads) {
            const auto shape
                = shapeOf(dataType, headSize, queryHeads, kvHeads, Band::BOTTOM_RIGHT, 2, 96, 256);
            addRow(rows,
                   served,
                   g.name + shapeToken(shape),
                   KnobClass::TILE_SCHEDULE,
                   {{"tile_size", g.tileSize},
                    {"num_warps", g.numWarps},
                    {"block_m_per_warp", g.blockMPerWarp},
                    {"use_mfma_32x32x8", g.useMfma32x32x8},
                    {"use_transposed_qk_32x32", g.useTransposedQk32x32},
                    {"use_k_sliced_ring", g.useKSlicedRing},
                    {"use_q_direct_global", g.useQDirectGlobal},
                    {"use_conflict_free_v_store", g.useConflictFreeVStore},
                    {"use_transposed_mask_limit", g.useTransposedMaskLimit},
                    {"use_transposed_invariant_hoist", g.useTransposedInvariantHoist}},
                   shape);
        };
        forced(D64_TRANSPOSED, BF16, 64, 8, 8);
        forced(D64_TRANSPOSED, FP16, 64, 24, 6);
        forced(D64_PRODUCTION, BF16, 64, 64, 8);
        forced(D64_PRODUCTION, FP16, 64, 64, 8);
        forced(D64_NARROW, BF16, 64, 8, 8);
        forced(D64_NARROW, FP16, 64, 8, 8);
        forced(D128_T_2X32, BF16, 128, 48, 3);
        forced(D128_T_2X32, FP16, 128, 64, 8);
        forced(D128_T_4X32, BF16, 128, 20, 10);
        forced(D128_NARROW_4X16, BF16, 128, 16, 2);
        forced(D128_NARROW_2X32, BF16, 128, 16, 4);
        forced(D128_NARROW_2X16, BF16, 128, 64, 8);
        forced(D128_NARROW_2X16, FP16, 128, 8, 8);
        forced(D128_PRODUCTION, FP16, 128, 20, 10);

        // The ranker default: no knob (the knob suite's cold cases, each with the kernel it
        // expects in its name there).
        const auto cold = [&](const std::string& name, const Shape& shape) {
            addRow(rows, served, name + shapeToken(shape), KnobClass::TILE_SCHEDULE, {}, shape);
        };
        cold("ColdProductionStack", shapeOf(BF16, 64, 32, 8, Band::BOTTOM_RIGHT, 3, 512, 512));
        cold("ColdProductionStack", shapeOf(FP16, 128, 64, 8, Band::BOTTOM_RIGHT, 1, 512, 2048));
        cold("ColdNarrowShortPrefill", shapeOf(BF16, 128, 64, 8, Band::BOTTOM_RIGHT, 1, 768, 2048));
        cold("ColdDefaultGeometryTail",
             shapeOf(BF16, 128, 64, 8, Band::BOTTOM_RIGHT, 1, 769, 2048));
        cold("ColdNarrowShortPrefill", shapeOf(BF16, 64, 8, 8, Band::BOTTOM_RIGHT, 2, 768, 1024));
        cold("ColdProductionStack", shapeOf(FP16, 64, 8, 8, Band::BOTTOM_RIGHT, 2, 768, 1024));
        cold("ColdManySequences", shapeOf(BF16, 128, 8, 2, Band::BOTTOM_RIGHT, 512, 16, 64));
    }
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_2D
    {
        // One CTA per (kv head, q-block), a single launch, no cross-CTA combine (T0 trace). The
        // classes are the kernel classes of IntegrationGpuGfx950AttentionUnifiedKnobs.cpp
        // knobCases2D, which CoversEveryCatalogKernelClass holds to the catalog: the default
        // narrow 4x16 geometry at every dtype x mask on the head configurations it cycles
        // through, the narrow 2x16 and the transposed 32x4 geometries in the cells that ship
        // them, and the Llama-3-70B prefill cell with both default geometries. Forced rows run
        // B 2, Sq 96, Skv 256: two sequences, several q-blocks per sequence and a tail q-block
        // where block_m (num_warps * block_m_per_warp) does not divide 96. The LSE twin of the
        // default geometry (every cell ships one) writes the stats head-major. No row has Sq 1:
        // decode is Unified3D's. The engine builds its own identity cache, so the rows have no
        // paged or padded K/V, and Skv is a whole number of 16-token blocks.
        const Served served{"hipkernel:Gfx950AttentionUnified2D", "gfx950"};
        struct Geometry
        {
            const char* name;
            int64_t tileSize;
            int64_t numWarps;
            int64_t blockMPerWarp;
            int64_t useMfma32x32;
            int64_t useTransposedQk32x32;
        };
        constexpr Geometry NARROW_4X16{"Narrow4x16T64", 64, 4, 16, 0, 0};
        constexpr Geometry NARROW_2X16{"Narrow2x16T32", 32, 2, 16, 0, 0};
        constexpr Geometry TRANSPOSED_32X4{"Transposed32x4T64", 64, 4, 32, 1, 1};
        const auto geometryKnobs = [](const Geometry& g) -> KnobList {
            return {{"tile_size", g.tileSize},
                    {"num_warps", g.numWarps},
                    {"block_m_per_warp", g.blockMPerWarp},
                    {"use_mfma_32x32", g.useMfma32x32},
                    {"use_transposed_qk_32x32", g.useTransposedQk32x32}};
        };
        const auto forced = [&](const Geometry& g, const Shape& shape) {
            addRow(rows,
                   served,
                   g.name + shapeToken(shape),
                   KnobClass::TILE_SCHEDULE,
                   geometryKnobs(g),
                   shape);
        };
        const auto prefill = [](DataType dataType,
                                int64_t headSize,
                                int64_t queryHeads,
                                int64_t kvHeads,
                                Band band,
                                bool withStats = false) {
            return shapeOf(dataType, headSize, queryHeads, kvHeads, band, 2, 96, 256, withStats);
        };
        forced(NARROW_4X16, prefill(BF16, 128, 8, 2, Band::NONE));
        forced(NARROW_4X16, prefill(BF16, 64, 16, 2, Band::TOP_LEFT));
        forced(NARROW_4X16, prefill(BF16, 128, 32, 8, Band::BOTTOM_RIGHT));
        forced(NARROW_4X16, prefill(FP16, 64, 8, 8, Band::NONE));
        forced(NARROW_4X16, prefill(FP16, 128, 8, 1, Band::TOP_LEFT));
        forced(NARROW_4X16, prefill(FP16, 64, 32, 4, Band::BOTTOM_RIGHT));
        forced(NARROW_2X16, prefill(BF16, 128, 32, 8, Band::TOP_LEFT));
        forced(NARROW_2X16, prefill(BF16, 128, 32, 8, Band::BOTTOM_RIGHT));
        forced(TRANSPOSED_32X4, prefill(BF16, 128, 24, 24, Band::NONE));
        forced(TRANSPOSED_32X4, prefill(BF16, 128, 32, 8, Band::TOP_LEFT));
        forced(TRANSPOSED_32X4, prefill(BF16, 128, 32, 8, Band::BOTTOM_RIGHT));
        forced(TRANSPOSED_32X4, prefill(FP16, 128, 32, 8, Band::TOP_LEFT));
        // The Llama-3-70B prefill cell: B 1, Sq = Skv = 512, a multiple of every block_m.
        forced(NARROW_4X16, shapeOf(BF16, 128, 64, 8, Band::TOP_LEFT, 1, 512, 512));
        forced(TRANSPOSED_32X4, shapeOf(BF16, 128, 64, 8, Band::TOP_LEFT, 1, 512, 512));
        // The LSE twin of the default geometry writes the stats head-major [B, Hq, Sq, 1].
        forced(NARROW_4X16, prefill(BF16, 128, 32, 8, Band::BOTTOM_RIGHT, true));
        forced(NARROW_4X16, prefill(FP16, 64, 8, 8, Band::NONE, true));
        forced(NARROW_4X16, prefill(BF16, 64, 16, 2, Band::TOP_LEFT, true));

        // The ranker default: no knob, so the default narrow geometry; and B 512, past the old
        // baked sequence search.
        const auto cold = [&](const std::string& name, const Shape& shape) {
            addRow(rows, served, name + shapeToken(shape), KnobClass::TILE_SCHEDULE, {}, shape);
        };
        cold("ColdDefaultGeometry", shapeOf(BF16, 128, 8, 2, Band::TOP_LEFT, 3, 512, 512));
        cold("ColdManySequences", shapeOf(BF16, 128, 8, 2, Band::TOP_LEFT, 512, 16, 64));
    }
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX942_UNIFIED_3D
    {
        // Single-token decode (Sq 1), two launches on one stream: the segment kernel (grid B x
        // Hkv x num_segments, every (sequence, head, segment) slot written by exactly one CTA, an
        // empty segment writes the neutral entry) and the reduce kernel (grid B x Hq, a fixed
        // butterfly over the lanes and a sequential sum over the segments, O stored once). The
        // shipped (block_size, num_segments) arms are those of
        // projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_3d.yaml (the
        // generator's selection, which tests/test_registry_regeneration.py holds to the shipped
        // descriptors): every cell ships block 16 with num_segments 2 and 128 and between them a
        // default arm (7 at D64, 4 at D128, 2 at D256), blocks 32 and 64 with 8, 16, 32, 64 and
        // 128, and D128 adds 9, 19, 32, 38 and 64 at block 16. The rows pin the minimum, the
        // default arm, the maximum and the shipped cells the knob suite runs
        // (IntegrationGpuGfx942AttentionUnified3dKnobs.cpp decodeCases), plus: Skv shorter than
        // the segments cover (segments with no key), Skv with many blocks per segment, all
        // segments holding keys, a Skv that is not a whole number of blocks, and each block size.
        // Every graph is bottom-right causal (at Sq 1 it hides no key), BSHD, with no LSE (no
        // gfx942 decode kernel writes one); the ranker default has no knob.
        const Served served{"hipkernel:Gfx942AttentionUnified3d", "gfx942"};
        const auto add = [&](const std::string& name,
                             int64_t blockSize,
                             int64_t numSegments,
                             const Shape& shape) {
            addRow(rows,
                   served,
                   name + shapeToken(shape),
                   KnobClass::STRUCTURE,
                   {{"block_size", blockSize}, {"num_segments", numSegments}},
                   shape);
        };
        const auto decode = [](DataType dataType,
                               int64_t headSize,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               int64_t batch,
                               int64_t seqKv) {
            return shapeOf(
                dataType, headSize, queryHeads, kvHeads, Band::BOTTOM_RIGHT, batch, 1, seqKv);
        };
        add("SegmentsMin", 16, 2, decode(BF16, 128, 32, 8, 3, 1024));
        add("SegmentsDefaultArm", 16, 4, decode(BF16, 128, 32, 8, 3, 1024));
        add("SegmentsMax", 16, 128, decode(BF16, 128, 32, 8, 2, 8192));
        add("SegmentsWithNoKey", 16, 128, decode(BF16, 128, 32, 8, 3, 64));
        // num_segments 64 is one unguarded reduce pass; 76 (a D256 arm) needs the guarded
        // second pass, and at Skv 1000 (63 blocks) its trailing segments hold no key.
        add("SegmentsOneReducePass", 16, 64, decode(BF16, 128, 32, 8, 2, 2048));
        add("SegmentsGuardedReducePass", 16, 76, decode(FP16, 256, 8, 8, 1, 4096));
        add("SegmentsGuardedReducePassNoKey", 16, 76, decode(FP16, 256, 8, 8, 1, 1000));
        add("ManyBlocksPerSegment", 16, 2, decode(FP16, 128, 32, 8, 2, 8192));
        add("SegmentsNotAWholeBlock", 16, 38, decode(BF16, 128, 32, 8, 2, 1000));
        add("MqaSegmentsWithNoKey", 16, 7, decode(FP16, 64, 8, 1, 3, 33));
        add("RuntimeHeads", 16, 38, decode(BF16, 128, 24, 6, 3, 777));
        add("DefaultArmHd64", 16, 7, decode(BF16, 64, 8, 8, 2, 1024));
        add("RatioTwo", 16, 4, decode(BF16, 128, 16, 8, 2, 1024));
        add("RatioSixteen", 16, 4, decode(FP16, 128, 16, 1, 2, 1024));
        add("Block32", 32, 16, decode(BF16, 128, 32, 8, 2, 1024));
        add("Block64", 64, 8, decode(FP16, 128, 32, 8, 2, 2048));
        add("Block64SegmentsMax", 64, 128, decode(BF16, 128, 32, 8, 1, 8192));
        add("Hd256Block32", 32, 16, decode(FP16, 256, 8, 8, 1, 4096));

        const auto cold = [&](const std::string& name, const Shape& shape) {
            addRow(rows, served, name + shapeToken(shape), KnobClass::STRUCTURE, {}, shape);
        };
        cold("Cold", decode(FP16, 128, 32, 8, 3, 1024));
        cold("Cold", decode(BF16, 64, 8, 1, 2, 33));
    }
#endif
#ifdef HKP_DETERMINISM_ROCKE_GFX950_UNIFIED_3D
    {
        // Single-token decode (Sq 1), two launches on one stream: the segment kernel (grid B x
        // Hkv x num_segments, every slot written by exactly one CTA, an empty segment writes the
        // neutral entry) and the reduce kernel (grid B x Hq, one wave64, a fixed 64-lane
        // butterfly, O stored once). Every cell ships num_segments 16 and an LSE twin of it; the
        // seg32 kernels ship in the hd128 (Hq 32, Hkv 4) and (64, 4) cells and the seg64 kernels in
        // (32, 4), (32, 8), (64, 4), (64, 8) and (128, 8), both bottom-right only (the
        // `gfx950_attention_unified_3d.yaml` selection, which tests/test_registry_regeneration.py
        // holds to the shipped descriptors; the knob suite's knobCases3D runs the same classes).
        // Rows: each shipped num_segments, a Skv with fewer key blocks than segments (segments
        // with no key) and one with many blocks per segment, B 1, 2 and 3 sequences, a no-mask
        // graph (served by the bottom-right kernel) and the LSE twin, including a top-left one.
        // A top-left graph without LSE is left out: at Sq 1 its one live key makes the output
        // independent of Q. Every row is BSHD with block size 16, one of the head configurations
        // the catalog bakes.
        const Served served{"hipkernel:Gfx950AttentionUnified3D", "gfx950"};
        const auto add = [&](const std::string& name, int64_t numSegments, const Shape& shape) {
            addRow(rows,
                   served,
                   name + shapeToken(shape),
                   KnobClass::STRUCTURE,
                   {{"num_segments", numSegments}},
                   shape);
        };
        const auto decode = [](DataType dataType,
                               int64_t headSize,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               Band band,
                               int64_t batch,
                               int64_t seqKv,
                               bool withStats = false) {
            return shapeOf(
                dataType, headSize, queryHeads, kvHeads, band, batch, 1, seqKv, withStats);
        };
        constexpr auto BR = Band::BOTTOM_RIGHT;
        add("Segments16", 16, decode(BF16, 128, 8, 2, BR, 3, 1024));
        add("Segments16", 16, decode(FP16, 128, 32, 8, BR, 3, 1024));
        add("Segments16", 16, decode(BF16, 64, 8, 1, BR, 3, 1024));
        add("Segments16", 16, decode(FP16, 64, 16, 2, BR, 3, 1024));
        add("Segments32", 32, decode(BF16, 128, 32, 4, BR, 3, 1024));
        add("Segments64", 64, decode(BF16, 128, 32, 8, BR, 3, 1024));
        add("Segments64", 64, decode(BF16, 128, 128, 8, BR, 3, 1024));
        add("SegmentsWithNoKey", 64, decode(BF16, 128, 32, 8, BR, 3, 64));
        add("SegmentsWithNoKey", 16, decode(BF16, 128, 8, 2, BR, 2, 16));
        add("ManyBlocksPerSegment", 16, decode(BF16, 128, 8, 2, BR, 1, 16384));
        add("ManyBlocksPerSegment", 32, decode(BF16, 128, 32, 4, BR, 2, 8192));
        add("NoMask", 16, decode(BF16, 128, 8, 2, Band::NONE, 2, 512));
        // The LSE twin of the seg16 kernel: the reduce also writes the stats.
        add("Segments16", 16, decode(BF16, 128, 8, 2, BR, 3, 1024, true));
        add("Segments16", 16, decode(FP16, 64, 16, 2, Band::TOP_LEFT, 3, 1024, true));
        add("SegmentsWithNoKey", 16, decode(BF16, 128, 8, 2, BR, 2, 32, true));

        // The ranker default: no knob (seg16), and B 1024.
        const auto cold = [&](const std::string& name, const Shape& shape) {
            addRow(rows, served, name + shapeToken(shape), KnobClass::STRUCTURE, {}, shape);
        };
        cold("Cold", decode(BF16, 128, 32, 8, BR, 3, 1024));
        cold("ColdManySequences", decode(FP16, 64, 8, 1, BR, 1024, 256));
    }
#endif
    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(rockeRows());

} // namespace

#endif
