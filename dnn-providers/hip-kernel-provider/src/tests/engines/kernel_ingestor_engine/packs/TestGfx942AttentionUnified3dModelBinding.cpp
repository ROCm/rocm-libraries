// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx942AttentionUnified3dModelBinding.cpp
 * @brief Census of the trained ranker hipkernel:Gfx942AttentionUnified3d binds in its UED for
 *        gfx942: the one the packed tree holds must survive the loader's pre-flight. Its
 *        engine-prediction (L1) models are censused with every other engine's by
 *        TestShippedEnginePredictors.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942AttentionUnified3dPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// The metric and architecture key the shipped UHDs declare.
constexpr const char* METRIC = "tflops";
constexpr const char* ARCH = "gfx942";

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME), METRIC, ARCH);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
