// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

// The build's HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS, set on this file by CMake.
#ifndef HKP_REQUIRE_DVC_DESCRIPTORS
#define HKP_REQUIRE_DVC_DESCRIPTORS 0
#endif

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
 * A model whose `model.bin` was never pulled (`dvc pull` not run) is packed without its
 * UHD, with a warning, unless the build sets HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS.
 * The census follows that: a bound model whose UHD the packed tree lacks passes in a
 * non-strict build (the pack already warned) and fails in a strict one. A bound model
 * whose UHD the tree holds must load, whatever the mode.
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

using ModelMap = std::map<std::string, std::map<std::string, ingestor::HeuristicDescriptor>>;

const ingestor::HeuristicDescriptor* find(const ModelMap& models)
{
    const auto byArch = models.find(METRIC);
    if(byArch == models.end())
    {
        return nullptr;
    }
    const auto model = byArch->second.find(ARCH);
    return model == byArch->second.end() ? nullptr : &model->second;
}

/// The ids a UED role map binds for ARCH.
std::vector<std::string>
    boundIds(const std::map<std::string, std::vector<ingestor::DescriptorId>>& role)
{
    std::vector<std::string> ids;
    const auto byArch = role.find(ARCH);
    if(byArch != role.end())
    {
        for(const auto& id : byArch->second)
        {
            ids.push_back(hipdnn_flatbuffers_sdk::utilities::formatUuid(id));
        }
    }
    return ids;
}

/// Whether the packed tree this census reads holds a UHD file naming `id`. A model left
/// out of the pack (its `model.bin` never pulled) has no UHD there.
bool packedTreeHoldsUhd(const std::string& id)
{
    const char* root = std::getenv("HIPDNN_DESCRIPTOR_DIR");
    if(root == nullptr || *root == '\0')
    {
        return true; // no explicit root to inspect: treat the model as shipped
    }
    for(const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if(!entry.is_regular_file() || entry.path().string().find(".uhd.json") == std::string::npos)
        {
            continue;
        }
        const std::ifstream file(entry.path());
        std::stringstream text;
        text << file.rdbuf();
        if(text.str().find(id) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

/// Whether the case must check that the models `ids` name loaded. False when the packed
/// tree lacks a UHD: in a non-strict build that is a model left out with a pack-time
/// warning, so the case notes it and passes; in a strict build the pack should have
/// failed, so the case records a failure.
bool packedTreeObligesLoad(const std::vector<std::string>& ids, const char* role)
{
    for(const auto& id : ids)
    {
        if(packedTreeHoldsUhd(id))
        {
            continue;
        }
        // The macro is a compile-time 0 or 1, so one arm is dead in each build.
        // NOLINTNEXTLINE(misc-redundant-expression)
        if(HKP_REQUIRE_DVC_DESCRIPTORS != 0)
        {
            ADD_FAILURE() << "the " << role << " model " << id
                          << " has no UHD in the packed tree, and "
                             "HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS is ON; the pack "
                             "should have failed";
        }
        else
        {
            std::cout << "[  NOTE    ] the " << role << " model " << id
                      << " has no UHD in the packed tree (its model.bin was not pulled); the "
                         "pack warned and the engine runs with that role off\n";
        }
        return false;
    }
    return true;
}

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL2KernelRankerLoads)
{
    const auto& set = loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME);

    const auto ids = boundIds(set.engine.sortKernelCatalog);
    ASSERT_FALSE(ids.empty()) << "the UED binds no " << ARCH << " sort_kernel_catalog ranker";

    if(!packedTreeObligesLoad(ids, "L2 sort_kernel_catalog"))
    {
        return;
    }

    // The loader prunes a ranker it refuses into unavailableHeuristicArches; a pruned one
    // is absent from the map, so presence is the whole check.
    EXPECT_NE(find(set.heuristicsByMetric), nullptr)
        << "no " << METRIC << " sort_kernel_catalog ranker for " << ARCH
        << " survived the loader's pre-flight; kernels rank by priority";
}

TEST(TestGfx942AttentionUnified3dModelBinding, TheBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    const auto& set = loadedSet(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME);

    const auto ids = boundIds(set.engine.predictEngine);
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

    // The same call makeEngine() makes, on the same loaded set.
    EXPECT_EQ(model->trainedAgainstSelectorRevision, ingestor::engineSelectorRevision(set))
        << "GenericEngine refuses this model and reports no prediction: the shipped UHD was "
           "trained against another selector revision than this build computes. Retrain the "
           "L1 against this build, or find what changed the code objects.";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
