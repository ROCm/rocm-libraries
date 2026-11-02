// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950AttentionUnified2DCatalogTiles.cpp
 * @brief Census of the geometry and batch capacity the shipped
 *        hipkernel:Gfx950AttentionUnified2D catalog declares, judged by the checks the engine
 *        itself applies.
 *
 * The engine launches a loaded kernel only when its candidateFor() accepts the record's
 * geometry through isSupportedGfx950Unified2DGeometry(); a kernel it declines ships but is
 * never selected. This suite reads the kernels through the descriptor loader and requires
 * every one to pass that same check, so a catalog entry the C++ engine cannot launch (an
 * unbuildable warp/row/tile combination, or a field missing from the record) fails here
 * instead of silently never running.
 *
 * It also pins the batch capacity: the kernel's binary search resolves at most 2^iters - 1
 * sequences, and a catalog regenerated with a smaller `enumerate.fixed.batch` would ship
 * binaries that quietly serve fewer sequences than the shipped ones. Raise the floor below
 * deliberately, with the manifest.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx950AttentionUnified2DPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// The batch every shipped kernel must serve at least: the manifest's `enumerate.fixed.batch`.
constexpr int64_t MINIMUM_BATCH_CAPACITY = 511;

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

TEST(TestGfx950AttentionUnified2DCatalogTiles, EveryShippedKernelServesTheManifestsBatchCapacity)
{
    const auto& set = loadedSet(GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            const auto numSeqs = integerField(kernel, "num_seqs");
            ASSERT_TRUE(numSeqs.has_value())
                << "kernel '" << kernel.name << "' does not carry an integer num_seqs";
            EXPECT_GT(*numSeqs, 0) << "kernel '" << kernel.name << "'";
            EXPECT_GE(gfx950Unified2DBatchCapacity(*numSeqs), MINIMUM_BATCH_CAPACITY)
                << "kernel '" << kernel.name << "' bakes num_seqs " << *numSeqs
                << ", which resolves fewer than " << MINIMUM_BATCH_CAPACITY << " sequences";
        }
    }
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
