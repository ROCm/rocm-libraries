// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx942AttentionUnified3dModelBinding.cpp
 * @brief Census of the trained models hipkernel:Gfx942AttentionUnified3d binds in its UED
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
 * TestGfx942AttentionUnified3dPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// The metric and architecture key the shipped UHDs declare.
constexpr const char* METRIC = "tflops";
constexpr const char* ARCH = "gfx942";

const ingestor::HeuristicDescriptor* find(const ModelMap& models)
{
    return findModel(models, METRIC, ARCH);
}

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME), METRIC, ARCH);
}

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    const auto& set = loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME);

    const auto ids = boundIds(set.engine.predictEngine, ARCH);
    if(ids.empty())
    {
        // No L1 ships. The check stays live: the day a UED binds one, the branches below
        // hold it to the revision this build computes.
        EXPECT_EQ(find(set.enginePredictionsByMetric), nullptr)
            << "the UED binds no " << ARCH << " predict_engine model, yet one loaded";
        return;
    }

    if(!packedTreeObligesLoad(ids, "L1 predict_engine"))
    {
        return;
    }

    const auto* model = find(set.enginePredictionsByMetric);
    ASSERT_NE(model, nullptr) << "no " << METRIC << " predict_engine model for " << ARCH
                              << " survived the loader's pre-flight; the engine reports no "
                                 "prediction";
    expectModelArtifactPresent(*model, "L1 predict_engine");

    // The same call makeEngine() makes, on the same loaded set.
    EXPECT_EQ(model->trainedAgainstSelectorRevision, ingestor::engineSelectorRevision(set))
        << "GenericEngine refuses this model and reports no prediction: the shipped UHD was "
           "trained against another selector revision than this build computes. Retrain the "
           "L1 against this build, or find what changed the code objects.";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
