// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
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
 * UHD, with a warning, unless the build sets HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS;
 * hkp_pack then also writes hkp-unpulled-models.record at the packed arch root, naming each
 * UHD it left out. The census accepts a bound model whose UHD the packed tree lacks only in
 * a non-strict build and only when that record names it. Any other absence (a packer that
 * lost the UHD or its model.bin) fails in every build, as does an absence in a strict
 * build. A bound model whose UHD the tree holds must load and its model file must be
 * present (the loader only warns about an absent one), whatever the mode.
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

/// The skip record hkp_pack writes at the packed arch root when it leaves a UHD out for an
/// un-pulled model (hkp_pack.pipeline.UNPULLED_RECORD).
constexpr const char* UNPULLED_RECORD = "hkp-unpulled-models.record";

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
    const std::string root = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
    if(root.empty())
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

/// Whether hkp_pack's skip record in the packed tree names `id` as a UHD it left out for
/// an un-pulled model. The packer writes the record only when it skips, so a UHD or model
/// the packer lost for any other reason is in no record.
bool packedTreeRecordsUnpulled(const std::string& id)
{
    const std::string root = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
    if(root.empty())
    {
        return false;
    }
    std::ifstream file(std::filesystem::path(root) / UNPULLED_RECORD);
    if(!file)
    {
        return false;
    }
    const auto record = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if(!record.is_object() || !record.contains("unpulled_models")
       || !record["unpulled_models"].is_array())
    {
        return false;
    }
    for(const auto& entry : record["unpulled_models"])
    {
        if(entry.is_object() && entry.value("uhd_id", std::string()) == id)
        {
            return true;
        }
    }
    return false;
}

/// Whether the case must check that the models `ids` name loaded. False only when the
/// packed tree lacks a UHD that the pack's skip record names: a model left out because
/// its `model.bin` was never pulled, which the pack warned about. A UHD absent for any
/// other reason (nothing in the record), or absent in a strict build (where the pack
/// should have failed), is a failure.
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
        else if(!packedTreeRecordsUnpulled(id))
        {
            ADD_FAILURE() << "the " << role << " model " << id
                          << " has no UHD in the packed tree, and " << UNPULLED_RECORD
                          << " does not name it, so its model.bin was not left out as "
                             "un-pulled: the packer lost the UHD or its model";
        }
        else
        {
            std::cout << "[  NOTE    ] the " << role << " model " << id
                      << " has no UHD in the packed tree (its model.bin was not pulled, as "
                      << UNPULLED_RECORD
                      << " records); the pack warned and the engine runs with that role "
                         "off\n";
        }
        return false;
    }
    return true;
}

/// The loader only warns about a UHD whose model file is absent and keeps the model, so
/// "it loaded" does not mean its `model.bin` shipped. A packer that drops the file from the
/// packed tree would otherwise pass: check the file the loaded descriptor names.
void expectModelArtifactPresent(const ingestor::HeuristicDescriptor& model, const char* role)
{
    std::error_code error;
    const auto path = std::filesystem::path(model.baseDir) / model.modelArtifactPath;
    EXPECT_TRUE(std::filesystem::is_regular_file(path, error))
        << "the " << role << " model " << hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id)
        << " loaded, but its model file " << path.string()
        << " is absent from the packed tree: the packer lost it";
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
    if(const auto* model = find(set.heuristicsByMetric))
    {
        expectModelArtifactPresent(*model, "L2 sort_kernel_catalog");
    }
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
    expectModelArtifactPresent(*model, "L1 predict_engine");

    // The same call makeEngine() makes, on the same loaded set.
    EXPECT_EQ(model->trainedAgainstSelectorRevision, ingestor::engineSelectorRevision(set))
        << "GenericEngine refuses this model and reports no prediction: the shipped UHD was "
           "trained against another selector revision than this build computes. Retrain the "
           "L1 against this build, or find what changed the code objects.";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
