// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradModelBindingTestSupport.hpp"

/**
 * @file TestGfx1250ConvDgradAotModelBinding.cpp
 * @brief Census of the trained kernel ranker hipkernel:Gfx1250ConvDgradAot binds in its UED:
 *        it must survive the loader's pre-flight, and it must score every kernel the packed
 *        catalog ships. The checks, and why they fail the build, are
 *        ConvDgradModelBindingTestSupport.hpp's.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx1250ConvDgradAotPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_model_binding_test;

constexpr CensusEngine GFX1250_CONV_DGRAD_AOT{"hipkernel:Gfx1250ConvDgradAot", "gfx1250", 32};

TEST(TestGfx1250ConvDgradAotModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(GFX1250_CONV_DGRAD_AOT);
}

TEST(TestGfx1250ConvDgradAotModelBinding, TheBoundL2KernelRankerScoresEveryShippedKernel)
{
    expectBoundRankerScoresEveryShippedKernel(GFX1250_CONV_DGRAD_AOT);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
