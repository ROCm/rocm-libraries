// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseCatalogTileTestSupport.hpp"

/**
 * @file TestGfx950AttentionDenseCatalogTiles.cpp
 * @brief Census of the tiles the shipped hipkernel:Gfx950AttentionDense catalog declares,
 *        judged by the check the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateTile() accepts the kernel's
 * head_size, block_m and block_n through isSupportedGfx950AttentionDenseTile(); a kernel
 * it declines ships but is never selected. This suite reads the kernels through the
 * descriptor loader and requires every one to pass that same check, so a catalog entry
 * the C++ engine cannot launch (an LDS-oversized or non-dividing tile, a missing field)
 * fails here instead of silently never running.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx950AttentionDensePacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestGfx950AttentionDenseCatalogTiles, EveryShippedKernelsTileIsOneTheEngineLaunches)
{
    expectEveryShippedAttentionDenseTileIsLaunchable(GFX950_ATTENTION_DENSE_ENGINE_NAME,
                                                     &isSupportedGfx950AttentionDenseTile);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
