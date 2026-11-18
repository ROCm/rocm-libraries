// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx942AttentionUnified2DCatalogTiles.cpp
 * @brief Census of what the shipped hipkernel:Gfx942AttentionUnified2D catalog declares,
 *        judged by the rules the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateFor() accepts the record's
 * geometry through isSupportedGfx942Unified2DGeometry(); a kernel it declines ships but is
 * never selected. The engine's scorer ranks each head size's default geometry first, which
 * means something only while every cell ships that geometry. The gfx942 kernel computes
 * the bottom-right causal band only. And graph_match serves only the GQA groups the catalog
 * ships (GFX942_UNIFIED_2D_SHIPPED_RATIOS). This suite reads the kernels through the descriptor
 * loader and holds the catalog to all four.
 *
 * TestGfx942AttentionUnified2DDecodeHandoff checks the decode boundary between the two gfx942
 * unified engines: Unified2D declines every decode graph whose GQA group Unified3d ships and
 * leaves it to Unified3d, so every such cell the 2D catalog ships must have a Unified3d kernel
 * with the same head_size, ratio and dtype.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionUnified2DPacks, and is registered with its own
 * hkp_register_census_tests() calls in src/tests/CMakeLists.txt (the hand-off suite only when
 * both engines ship).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

TEST(TestGfx942AttentionUnified2DCatalogTiles, EveryShippedKernelsGeometryIsOneTheEngineLaunches)
{
    const auto& set = loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            const auto headSize = integerField(kernel, "head_size");
            const auto blockSize = integerField(kernel, "block_size");
            const auto tileSize = integerField(kernel, "tile_size");
            const auto numWarps = integerField(kernel, "num_warps");
            const auto blockMPerWarp = integerField(kernel, "block_m_per_warp");
            const auto useMfma32x32x8 = integerField(kernel, "use_mfma_32x32x8");
            const auto useTransposed = integerField(kernel, "use_transposed_qk_32x32");
            const auto ratio = integerField(kernel, "num_queries_per_kv");
            ASSERT_TRUE(headSize && blockSize && tileSize && numWarps && blockMPerWarp
                        && useMfma32x32x8 && useTransposed && ratio)
                << "kernel '" << kernel.name << "' does not carry the integer geometry metadata";

            const Gfx942Unified2DGeometryFields fields{*headSize,
                                                       *blockSize,
                                                       *tileSize,
                                                       *numWarps,
                                                       *blockMPerWarp,
                                                       *useMfma32x32x8 != 0,
                                                       *useTransposed != 0};
            EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields))
                << "kernel '" << kernel.name << "' declares head_size " << *headSize
                << ", block_size " << *blockSize << ", tile_size " << *tileSize << ", num_warps "
                << *numWarps << ", block_m_per_warp " << *blockMPerWarp
                << ", a geometry the engine declines to launch";
            EXPECT_GE(*ratio, 1) << "kernel '" << kernel.name << "'";
            EXPECT_LE(*ratio, GFX942_UNIFIED_2D_MAX_QUERIES_PER_KV)
                << "kernel '" << kernel.name << "'";
        }
    }
    // A catalog that loaded no kernel would pass the loop above vacuously.
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

/// The scorer ranks the head size's default geometry below the production stack: transposed
/// 32x32x8 at 32 rows per warp, with 4 warps and a 64-token tile at D64 and 2 warps and a
/// 32-token tile at D128, none of the codegen stack on. Every (head size, ratio, dtype) cell
/// must ship it, as the floor the production stack is measured against.
TEST(TestGfx942AttentionUnified2DCatalogTiles, EveryCellShipsItsHeadSizesDefaultGeometry)
{
    using Cell = std::tuple<int64_t, int64_t, std::string>;
    std::map<Cell, bool> hasDefault;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto headSize = integerField(kernel, "head_size");
            const auto ratio = integerField(kernel, "num_queries_per_kv");
            const auto dtype = stringField(kernel, "dtype");
            const auto tileSize = integerField(kernel, "tile_size");
            const auto numWarps = integerField(kernel, "num_warps");
            const auto blockMPerWarp = integerField(kernel, "block_m_per_warp");
            const auto useMfma32x32x8 = integerField(kernel, "use_mfma_32x32x8");
            const auto useTransposed = integerField(kernel, "use_transposed_qk_32x32");
            const auto maskLimit = integerField(kernel, "use_transposed_mask_limit");
            const auto hoist = integerField(kernel, "use_transposed_invariant_hoist");
            ASSERT_TRUE(headSize && ratio && dtype && tileSize && numWarps && blockMPerWarp
                        && useMfma32x32x8 && useTransposed && maskLimit && hoist)
                << "kernel '" << kernel.name << "' does not carry its cell and geometry";

            const bool isDefault = *useMfma32x32x8 == 1 && *useTransposed == 1
                                   && *blockMPerWarp == 32 && *maskLimit == 0 && *hoist == 0
                                   && ((*headSize == 64 && *tileSize == 64 && *numWarps == 4)
                                       || (*headSize == 128 && *tileSize == 32 && *numWarps == 2));
            auto& entry = hasDefault[Cell{*headSize, *ratio, *dtype}];
            entry = entry || isDefault;
        }
    }
    ASSERT_FALSE(hasDefault.empty()) << "the loaded set holds no kernels";
    for(const auto& [cell, found] : hasDefault)
    {
        const auto& [headSize, ratio, dtype] = cell;
        EXPECT_TRUE(found) << "the head_size " << headSize << ", ratio " << ratio << ", " << dtype
                           << " cell ships no kernel of its head size's default geometry";
    }
}

/// rocKE's production stack (the mask limit and invariant hoist, with conflict-free V stores
/// and, where rocKE uses it, the K ring) wins every fp16 cell and every bf16 D64 cell of the
/// stage 6 sweep, so each of those cells ships it; at bf16 D128 the default geometry beat it
/// on every graph, so that stack is left out. A catalog that dropped one would regress the
/// ground-truth rows it serves to the slower default geometry. A stack is its flags
/// together: the mask limit and the hoist are on or off as a pair, and the K ring needs
/// conflict-free V stores (the spec's own validation).
TEST(TestGfx942AttentionUnified2DCatalogTiles, EveryCellRockeServesWithItsProductionStackShipsIt)
{
    using Cell = std::tuple<int64_t, int64_t, std::string>;
    std::map<Cell, bool> hasProduction;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto headSize = integerField(kernel, "head_size");
            const auto ratio = integerField(kernel, "num_queries_per_kv");
            const auto dtype = stringField(kernel, "dtype");
            const auto ring = integerField(kernel, "use_k_sliced_ring");
            const auto qDirect = integerField(kernel, "use_q_direct_global");
            const auto cfvst = integerField(kernel, "use_conflict_free_v_store");
            const auto maskLimit = integerField(kernel, "use_transposed_mask_limit");
            const auto hoist = integerField(kernel, "use_transposed_invariant_hoist");
            ASSERT_TRUE(headSize && ratio && dtype && ring && qDirect && cfvst && maskLimit
                        && hoist)
                << "kernel '" << kernel.name << "' does not carry its codegen stack";

            EXPECT_EQ(*maskLimit, *hoist)
                << "kernel '" << kernel.name << "' turns on one of the mask limit and the hoist";
            EXPECT_TRUE(*ring == 0 || *cfvst == 1)
                << "kernel '" << kernel.name << "' has the K ring without conflict-free V stores";
            auto& entry = hasProduction[Cell{*headSize, *ratio, *dtype}];
            entry = entry || *maskLimit == 1;
        }
    }
    ASSERT_FALSE(hasProduction.empty()) << "the loaded set holds no kernels";
    for(const auto& [cell, found] : hasProduction)
    {
        const auto& [headSize, ratio, dtype] = cell;
        const bool rockeServesProduction = dtype == "FP16" || headSize == 64;
        if(rockeServesProduction)
        {
            EXPECT_TRUE(found) << "the head_size " << headSize << ", ratio " << ratio << ", "
                               << dtype << " cell ships no production-stack kernel";
        }
    }
}

/// The gfx942 tiled 2D kernel computes only the bottom-right causal band, without a window.
TEST(TestGfx942AttentionUnified2DCatalogTiles, EveryShippedKernelIsBottomRightCausalWithoutAWindow)
{
    constexpr int64_t BOTTOM_RIGHT_CAUSAL = 2;
    std::size_t checked = 0;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            EXPECT_EQ(integerField(kernel, "mask_type"), BOTTOM_RIGHT_CAUSAL)
                << "kernel '" << kernel.name << "'";
            EXPECT_EQ(integerField(kernel, "sliding_window"), 0)
                << "kernel '" << kernel.name << "'";
        }
    }
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

/// graph_match declines every GQA group but those GFX942_UNIFIED_2D_SHIPPED_RATIOS lists, so the
/// list must be exactly the groups the catalog ships: a shipped group missing from it would
/// never be served, and a listed group with no kernel would pass graph_match and then empty
/// the catalog at kernel_match.
TEST(TestGfx942AttentionUnified2DCatalogTiles, TheShippedGqaRatioListIsTheCatalogsRatios)
{
    std::set<int64_t> shipped;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto ratio = integerField(kernel, "num_queries_per_kv");
            ASSERT_TRUE(ratio.has_value())
                << "kernel '" << kernel.name << "' carries no num_queries_per_kv";
            shipped.insert(*ratio);
        }
    }
    const std::set<int64_t> listed(GFX942_UNIFIED_2D_SHIPPED_RATIOS.begin(),
                                   GFX942_UNIFIED_2D_SHIPPED_RATIOS.end());
    EXPECT_EQ(shipped, listed);
}

/// The cell a decode graph is served in: (head_size, num_queries_per_kv, dtype).
using HeadCell = std::tuple<int64_t, int64_t, std::string>;

/// A kernel's cell, or nullopt when it does not carry all three fields.
std::optional<HeadCell> headCell(const ingestor::KernelDescriptor& kernel)
{
    const auto headSize = integerField(kernel, "head_size");
    const auto ratio = integerField(kernel, "num_queries_per_kv");
    const auto dtype = stringField(kernel, "dtype");
    if(!headSize || !ratio || !dtype)
    {
        return std::nullopt;
    }
    return HeadCell{*headSize, *ratio, *dtype};
}

/// The decode boundary between the two gfx942 unified engines: Unified2D declines every decode
/// graph whose GQA group the Unified3d catalog ships (gfx942Unified3dServesGqaRatio: 1, 2, 4, 8
/// and 16) and leaves it to Unified3d, so every such cell the 2D catalog ships must have a
/// Unified3d kernel with the same head_size, ratio and dtype. A reduction that dropped the 3D
/// cell would otherwise leave those decode graphs with no unified engine at all.
TEST(TestGfx942AttentionUnified2DDecodeHandoff, EveryDecodeCellHandedToUnified3dHasAUnified3dKernel)
{
    std::set<HeadCell> unified3dCells;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto cell = headCell(kernel);
            ASSERT_TRUE(cell.has_value())
                << "Unified3d kernel '" << kernel.name << "' does not carry its cell";
            unified3dCells.insert(*cell);
        }
    }
    ASSERT_FALSE(unified3dCells.empty()) << "the Unified3d set holds no kernels";

    std::set<HeadCell> handedOff;
    for(const auto& pack : loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto cell = headCell(kernel);
            ASSERT_TRUE(cell.has_value())
                << "Unified2D kernel '" << kernel.name << "' does not carry its cell";
            // The same rule Unified2D's graph match applies before declining decode.
            if(!gfx942Unified3dServesGqaRatio(std::get<1>(*cell)))
            {
                continue;
            }
            handedOff.insert(*cell);
            const auto& [headSize, ratio, dtype] = *cell;
            EXPECT_EQ(unified3dCells.count(*cell), 1U)
                << "Unified2D declines decode for head_size " << headSize << ", a GQA group of "
                << ratio << ", " << dtype << " (kernel '" << kernel.name
                << "') and hands it to Unified3d, which ships no kernel for that cell";
        }
    }
    // Every group the 2D catalog ships is one Unified3d serves, so every cell is handed off:
    // none handed off means the rule or a catalog changed.
    EXPECT_FALSE(handedOff.empty()) << "no Unified2D cell is handed to Unified3d at decode";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
