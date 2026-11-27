// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx942AttentionUnified2DModelBinding.cpp
 * @brief Census of the trained models hipkernel:Gfx942AttentionUnified2D binds in its UED
 *        for gfx942: the L2 kernel ranker (tflops) the packed tree holds must survive the
 *        loader's pre-flight, and a bound engine predictor (L1) must record the selector
 *        revision this build computes.
 *
 * The loader refuses a ranker whose knob axes or feature hash do not match this build and
 * only logs it; the engine then ranks by the native scorer as if no model shipped, and
 * every other census case stays green. This suite fails the build instead.
 *
 * No L1 ships today. Its case stays live: the day the UED binds one, it holds that model to
 * the revision this build computes.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionUnified2DPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// The architecture key the shipped UHDs declare.
constexpr const char* ARCH = "gfx942";

TEST(TestGfx942AttentionUnified2DModelBinding, TheBoundL2TflopsRankerLoads)
{
    expectBoundRankerLoads(loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME), "tflops", ARCH);
}

TEST(TestGfx942AttentionUnified2DModelBinding, TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    expectBoundPredictorMatchesThisBuild(
        loadedSet(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME), "tflops", ARCH);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
