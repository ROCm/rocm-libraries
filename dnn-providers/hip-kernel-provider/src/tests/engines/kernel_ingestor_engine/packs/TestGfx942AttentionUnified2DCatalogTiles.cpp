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
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx942AttentionUnified2DCatalogTiles.cpp
 * @brief Census of what the shipped hipkernel:Gfx942AttentionUnified2D catalog declares,
 *        judged by the rules the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateFor() accepts the record's
 * geometry through isSupportedGfx942Unified2DGeometry(); a kernel it declines ships but is
 * never selected. The engine's scorer ranks each head size's default geometry first, which
 * means something only while every cell ships that geometry. And the gfx942 kernel computes
 * the bottom-right causal band only. This suite reads the kernels through the descriptor
 * loader and holds the catalog to all three.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionUnified2DPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// A kernel's integer metadata field, or nullopt when it is absent or another type: the cases
/// candidateFor() declines.
std::optional<int64_t> integerField(const ingestor::KernelDescriptor& kernel, const char* field)
{
    const auto it = kernel.metadata.find(field);
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/// A kernel's string metadata field, or nullopt when it is absent or another type.
std::optional<std::string> stringField(const ingestor::KernelDescriptor& kernel, const char* field)
{
    const auto it = kernel.metadata.find(field);
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

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

/// The scorer ranks the head size's default geometry first: transposed 32x32x8 at 32 rows per
/// warp, with 4 warps and a 64-token tile at D64 and 2 warps and a 32-token tile at D128.
/// Every (head size, ratio, dtype) cell must ship it, or that ranking prefers nothing.
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
            ASSERT_TRUE(headSize && ratio && dtype && tileSize && numWarps && blockMPerWarp
                        && useMfma32x32x8 && useTransposed)
                << "kernel '" << kernel.name << "' does not carry its cell and geometry";

            const bool isDefault = *useMfma32x32x8 == 1 && *useTransposed == 1
                                   && *blockMPerWarp == 32
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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
