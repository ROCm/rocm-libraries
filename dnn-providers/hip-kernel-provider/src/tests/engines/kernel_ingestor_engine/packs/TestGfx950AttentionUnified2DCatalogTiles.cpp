// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950AttentionUnified2DCatalogTiles.cpp
 * @brief Census of the geometry the shipped hipkernel:Gfx950AttentionUnified2D catalog
 *        declares, judged by the checks the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateFor() accepts the record's
 * geometry through isSupportedGfx950Unified2DGeometry(); a kernel it declines ships but is
 * never selected. This suite reads the kernels through the descriptor loader and requires
 * every one to pass that same check, so a catalog entry the C++ engine cannot launch (an
 * unbuildable warp/row/tile combination, or a field missing from the record) fails here
 * instead of silently never running.
 *
 * TestGfx950AttentionUnified2DDecodeHandoff checks the decode boundary between the two
 * unified engines: Unified2D declines every decode graph whose GQA group divides
 * GFX950_UNIFIED_3D_BLOCK_M and leaves it to Unified3D, so every such cell the 2D catalog
 * ships must have a Unified3D kernel with the same head_size, head counts and dtype. A
 * reduction that drops the 3D cell would otherwise leave those graphs with no unified engine.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx950AttentionUnified2DPacks, and is registered with its own
 * hkp_register_census_tests() calls in src/tests/CMakeLists.txt (the handoff suite only
 * when both engines ship).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// (head_size, num_query_heads, num_kv_heads, dtype): the cell a decode graph is served in.
using HeadCell = std::tuple<int64_t, int64_t, int64_t, std::string>;

/// A kernel's head cell, or nullopt when it does not carry all four fields.
std::optional<HeadCell> headCell(const ingestor::KernelDescriptor& kernel)
{
    const auto headSize = integerField(kernel, "head_size");
    const auto numQueryHeads = integerField(kernel, "num_query_heads");
    const auto numKvHeads = integerField(kernel, "num_kv_heads");
    const auto dtype = stringField(kernel, "dtype");
    if(!headSize || !numQueryHeads || !numKvHeads || !dtype)
    {
        return std::nullopt;
    }
    return HeadCell{*headSize, *numQueryHeads, *numKvHeads, *dtype};
}

TEST(TestGfx950AttentionUnified2DCatalogTiles, EveryShippedKernelsGeometryIsOneTheEngineLaunches)
{
    const auto& set = loadedSet(GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME);

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
            const auto useMfma32x32 = integerField(kernel, "use_mfma_32x32");
            const auto useTransposed = integerField(kernel, "use_transposed_qk_32x32");
            ASSERT_TRUE(headSize && blockSize && tileSize && numWarps && blockMPerWarp
                        && useMfma32x32 && useTransposed)
                << "kernel '" << kernel.name << "' does not carry the integer geometry metadata";

            const Gfx950Unified2DGeometryFields fields{*headSize,
                                                       *blockSize,
                                                       *tileSize,
                                                       *numWarps,
                                                       *blockMPerWarp,
                                                       *useMfma32x32 != 0,
                                                       *useTransposed != 0};
            EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(fields))
                << "kernel '" << kernel.name << "' declares head_size " << *headSize
                << ", block_size " << *blockSize << ", tile_size " << *tileSize << ", num_warps "
                << *numWarps << ", block_m_per_warp " << *blockMPerWarp
                << ", a geometry the engine declines to launch";
        }
    }
    // A catalog that loaded no kernel would pass the loop above vacuously.
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

TEST(TestGfx950AttentionUnified2DDecodeHandoff, EveryDecodeCellHandedToUnified3DHasAUnified3DKernel)
{
    std::set<HeadCell> unified3dCells;
    for(const auto& pack : loadedSet(GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto cell = headCell(kernel);
            ASSERT_TRUE(cell.has_value())
                << "Unified3D kernel '" << kernel.name << "' does not carry its head cell";
            unified3dCells.insert(*cell);
        }
    }
    ASSERT_FALSE(unified3dCells.empty()) << "the Unified3D set holds no kernels";

    std::size_t handedOff = 0;
    for(const auto& pack : loadedSet(GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto cell = headCell(kernel);
            ASSERT_TRUE(cell.has_value())
                << "Unified2D kernel '" << kernel.name << "' does not carry its head cell";
            const auto& [headSize, numQueryHeads, numKvHeads, dtype] = *cell;
            ASSERT_GT(numKvHeads, 0) << "kernel '" << kernel.name << "'";
            // The same rule Unified2D's graph match applies before declining decode.
            if(!gfx950Unified3DServesGqaGroup(numQueryHeads, numKvHeads))
            {
                continue;
            }
            ++handedOff;
            EXPECT_TRUE(unified3dCells.count(*cell) != 0)
                << "Unified2D declines decode for head_size " << headSize << ", " << numQueryHeads
                << "/" << numKvHeads << " heads, " << dtype << " (kernel '" << kernel.name
                << "') and hands it to Unified3D, which ships no kernel for that cell";
        }
    }
    // Most shipped GQA groups divide 16; none handed off means the rule or catalog changed.
    EXPECT_GT(handedOff, 0U) << "no Unified2D cell is handed to Unified3D at decode";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
