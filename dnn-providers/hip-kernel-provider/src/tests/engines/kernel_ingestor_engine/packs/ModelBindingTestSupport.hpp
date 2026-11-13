// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

// The build's HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS, set by CMake on every census file
// that includes this header (src/tests/CMakeLists.txt).
#ifndef HKP_REQUIRE_DVC_DESCRIPTORS
#define HKP_REQUIRE_DVC_DESCRIPTORS 0
#endif

/**
 * @file ModelBindingTestSupport.hpp
 * @brief What the ModelBinding census suites share: the trained models a UED binds, and
 *        whether the packed tree the census reads must hold them.
 *
 * A model whose `model.bin` was never pulled (`dvc pull` not run) is packed without its
 * UHD, with a warning, unless the build sets HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS;
 * hkp_pack then also writes hkp-unpulled-models.record at the packed arch root, naming each
 * UHD it left out. A census accepts a bound model whose UHD the packed tree lacks only in
 * a non-strict build and only when that record names it. Any other absence (a packer that
 * lost the UHD or its model.bin) fails in every build, as does an absence in a strict
 * build. A bound model whose UHD the tree holds must load and its model file must be
 * present (the loader only warns about an absent one), whatever the mode.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The skip record hkp_pack writes at the packed arch root when it leaves a UHD out for an
/// un-pulled model (hkp_pack.pipeline.UNPULLED_RECORD).
constexpr const char* UNPULLED_RECORD = "hkp-unpulled-models.record";

using ModelMap = std::map<std::string,
                          std::map<std::string, hipdnn_plugin_sdk::ingestor::HeuristicDescriptor>>;

/// The model @p models holds for @p metric under arch key @p archKey, or null.
inline const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor*
    findModel(const ModelMap& models, const std::string& metric, const std::string& archKey)
{
    const auto byArch = models.find(metric);
    if(byArch == models.end())
    {
        return nullptr;
    }
    const auto model = byArch->second.find(archKey);
    return model == byArch->second.end() ? nullptr : &model->second;
}

/// The ids a UED role map binds for arch key @p archKey.
inline std::vector<std::string> boundIds(
    const std::map<std::string, std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId>>& role,
    const std::string& archKey)
{
    std::vector<std::string> ids;
    const auto byArch = role.find(archKey);
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
inline bool packedTreeHoldsUhd(const std::string& id)
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
inline bool packedTreeRecordsUnpulled(const std::string& id)
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
inline bool packedTreeObligesLoad(const std::vector<std::string>& ids, const char* role)
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
inline void
    expectModelArtifactPresent(const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor& model,
                               const char* role)
{
    std::error_code error;
    const auto path = std::filesystem::path(model.baseDir) / model.modelArtifactPath;
    EXPECT_TRUE(std::filesystem::is_regular_file(path, error))
        << "the " << role << " model " << hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id)
        << " loaded, but its model file " << path.string()
        << " is absent from the packed tree: the packer lost it";
}

/// The L2 sort_kernel_catalog ranker @p set's UED binds under @p archKey, checked the way
/// every ModelBinding census checks it: the UED binds one, and when the packed tree obliges
/// it to load (packedTreeObligesLoad), one for @p metric survived the loader's pre-flight,
/// it is one the UED binds, and its model file is present. Null when none loaded (a
/// failure) or none had to.
inline const hipdnn_plugin_sdk::ingestor::HeuristicDescriptor*
    expectBoundRankerLoads(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                           const std::string& metric,
                           const std::string& archKey)
{
    constexpr const char* ROLE = "L2 sort_kernel_catalog";
    const auto ids = boundIds(set.engine.sortKernelCatalog, archKey);
    if(ids.empty())
    {
        ADD_FAILURE() << "the UED binds no " << archKey << " sort_kernel_catalog ranker";
        return nullptr;
    }
    if(!packedTreeObligesLoad(ids, ROLE))
    {
        return nullptr;
    }

    // The loader prunes a ranker it refuses into unavailableHeuristicArches; a pruned one
    // is absent from the map, so presence is the load check.
    const auto* model = findModel(set.heuristicsByMetric, metric, archKey);
    if(model == nullptr)
    {
        ADD_FAILURE() << "no " << metric << " sort_kernel_catalog ranker under " << archKey
                      << " survived the loader's pre-flight; kernels rank by priority";
        return nullptr;
    }
    EXPECT_NE(
        std::find(ids.begin(), ids.end(), hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id)),
        ids.end())
        << "the loaded " << metric << " ranker is not one the UED binds";
    expectModelArtifactPresent(*model, ROLE);
    return model;
}

/// The L1 predict_engine model @p set's UED binds under @p archKey, checked the way every
/// ModelBinding census checks it: with none bound, none loaded; with one bound and the packed
/// tree obliging it to load (packedTreeObligesLoad), one for @p metric survived the loader's
/// pre-flight, its model file is present, and it records the selector revision this build
/// computes. The engine selector's revision digests every launch's code-object hash and the
/// L2 rankers, so a compiler bump, a catalog change or a retrained L2 makes GenericEngine
/// refuse the L1 (PredictionStatus::UNAVAILABLE) with nothing but a log line; this fails the
/// build instead.
inline void
    expectBoundPredictorMatchesThisBuild(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                                         const std::string& metric,
                                         const std::string& archKey)
{
    constexpr const char* ROLE = "L1 predict_engine";
    const auto ids = boundIds(set.engine.predictEngine, archKey);
    if(ids.empty())
    {
        // No L1 ships. The check stays live: the day a UED binds one, the branches below
        // hold it to the revision this build computes.
        EXPECT_EQ(findModel(set.enginePredictionsByMetric, metric, archKey), nullptr)
            << "the UED binds no " << archKey << " predict_engine model, yet one loaded";
        return;
    }
    if(!packedTreeObligesLoad(ids, ROLE))
    {
        return;
    }

    const auto* model = findModel(set.enginePredictionsByMetric, metric, archKey);
    ASSERT_NE(model, nullptr) << "no " << metric << " predict_engine model under " << archKey
                              << " survived the loader's pre-flight; the engine reports no "
                                 "prediction";
    expectModelArtifactPresent(*model, ROLE);

    // The revision digests the L2 too. With a bound L2 left out of the pack as un-pulled
    // (the L2 case reports that), the runtime refuses the L1 for that reason, not because the
    // L1 is stale, so the comparison below would blame the wrong model.
    for(const auto& ranker : boundIds(set.engine.sortKernelCatalog, archKey))
    {
        // The macro is a compile-time 0 or 1, so one arm is dead in each build.
        // NOLINTNEXTLINE(misc-redundant-expression)
        if(HKP_REQUIRE_DVC_DESCRIPTORS == 0 && !packedTreeHoldsUhd(ranker)
           && packedTreeRecordsUnpulled(ranker))
        {
            std::cout << "[  NOTE    ] the L2 sort_kernel_catalog model " << ranker
                      << " was not pulled, so this build's selector revision cannot match the "
                         "L1's and the engine runs with the L1 off too\n";
            return;
        }
    }

    // The same call makeEngine() makes, on the same loaded set.
    EXPECT_EQ(model->trainedAgainstSelectorRevision,
              hipdnn_plugin_sdk::ingestor::engineSelectorRevision(set))
        << "GenericEngine refuses this model and reports no prediction: the shipped UHD was "
           "trained against another selector revision than this build computes. Retrain the "
           "L1 against this build, or find what changed the code objects.";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
