// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

// The build's HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS, set on this file by CMake.
#ifndef HKP_REQUIRE_DVC_DESCRIPTORS
#error "add this file to the HKP_REQUIRE_DVC_DESCRIPTORS list in src/tests/CMakeLists.txt"
#endif

/**
 * @file TestGfx942AttentionUnified3dModelBinding.cpp
 * @brief Census of the trained models hipkernel:Gfx942AttentionUnified3d binds in its UED
 *        for gfx942 (ModelBindingTestSupport.hpp holds the checks).
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionUnified3dPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

// The macro is a compile-time 0 or 1, so the comparison is constant in each build.
// NOLINTNEXTLINE(misc-redundant-expression)
constexpr bool REQUIRE_DVC = HKP_REQUIRE_DVC_DESCRIPTORS != 0;

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL2KernelRankerLoads)
{
    expectTheBoundL2KernelRankerLoads(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME, REQUIRE_DVC);
}

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    expectTheBoundL1EnginePredictorLoadsAndMatchesThisBuild(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME,
                                                            REQUIRE_DVC);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
