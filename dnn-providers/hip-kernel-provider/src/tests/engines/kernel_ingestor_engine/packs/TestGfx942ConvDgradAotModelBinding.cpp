// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradModelBindingTestSupport.hpp"

/**
 * @file TestGfx942ConvDgradAotModelBinding.cpp
 * @brief Census of the trained kernel ranker hipkernel:Gfx942ConvDgradAot binds in its UED:
 *        it must survive the loader's pre-flight, and it must score every kernel the packed
 *        catalog ships. The checks, and why they fail the build, are
 *        ConvDgradModelBindingTestSupport.hpp's.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942ConvDgradAotPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_model_binding_test;

constexpr CensusEngine GFX942_CONV_DGRAD_AOT{"hipkernel:Gfx942ConvDgradAot", "gfx942", 64};

TEST(TestGfx942ConvDgradAotModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(GFX942_CONV_DGRAD_AOT);
}

TEST(TestGfx942ConvDgradAotModelBinding, TheBoundL2KernelRankerScoresEveryShippedKernel)
{
    expectBoundRankerScoresEveryShippedKernel(GFX942_CONV_DGRAD_AOT);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
