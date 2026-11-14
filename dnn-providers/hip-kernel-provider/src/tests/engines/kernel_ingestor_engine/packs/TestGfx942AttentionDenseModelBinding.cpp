// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx942AttentionDenseModelBinding.cpp
 * @brief Census of the trained models hipkernel:Gfx942AttentionDense binds in its UED
 *        for gfx942: each one the packed tree holds must survive the loader's pre-flight,
 *        and a bound engine predictor (L1) must record the selector revision this build
 *        computes.
 *
 * The engine selector's revision digests, among other things, every launch's code-object
 * hash. Those are compiled at pack time, so a build that compiles them differently (a
 * compiler bump, a rocKE change) reports another revision, and GenericEngine then refuses
 * an L1 (PredictionStatus::UNAVAILABLE) with nothing but a log line. Engine selection
 * falls back to priority as if no model shipped, and every other census case stays green.
 * This suite reads the revision the way makeEngine() does and fails the build instead.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionDensePacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// The metric and architecture key the shipped UHDs declare.
constexpr const char* METRIC = "tflops";
constexpr const char* ARCH = "gfx942";

TEST(TestGfx942AttentionDenseModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(loadedSet(GFX942_ATTENTION_DENSE_ENGINE_NAME), METRIC, ARCH);
}

TEST(TestGfx942AttentionDenseModelBinding, TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    expectBoundPredictorMatchesThisBuild(
        loadedSet(GFX942_ATTENTION_DENSE_ENGINE_NAME), METRIC, ARCH);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
