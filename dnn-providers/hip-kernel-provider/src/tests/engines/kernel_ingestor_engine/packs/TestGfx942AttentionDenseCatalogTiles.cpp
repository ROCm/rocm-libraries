// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseCatalogTileTestSupport.hpp"

/**
 * @file TestGfx942AttentionDenseCatalogTiles.cpp
 * @brief Census of the tiles the shipped hipkernel:Gfx942AttentionDense catalog declares,
 *        judged by the check the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateTile() accepts the kernel's
 * head_size, block_m and block_n through isSupportedGfx942AttentionDenseTile(); a kernel
 * it declines ships but is never selected. This suite reads the kernels through the
 * descriptor loader and requires every one to pass that same check, so a catalog entry
 * the C++ engine cannot launch (a non-dividing tile, a missing field)
 * fails here instead of silently never running.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionDensePacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestGfx942AttentionDenseCatalogTiles, EveryShippedKernelsTileIsOneTheEngineLaunches)
{
    expectEveryShippedAttentionDenseTileIsLaunchable(GFX942_ATTENTION_DENSE_ENGINE_NAME,
                                                     &isSupportedGfx942AttentionDenseTile);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
