// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestShippedEnginePredictors.cpp
 * @brief Census of every engine-prediction (L1) model the shard ships, whatever the engine:
 *        each one bound for the census architecture loads, and each one this architecture's
 *        engine would serve records the selector revision this build computes for it.
 *
 * GenericEngine refuses an L1 trained against another selector revision (PredictionStatus::
 * UNAVAILABLE) with one log line, and selection then ranks the engine in static order as if
 * no model shipped. The revision digests every ranker, pack kernel and native symbol, so
 * any of those moving -- a retrained L2, a recompiled code object, a merge from another
 * branch -- does this silently. This suite fails the build instead, and writes this build's
 * selector so the change can be found and the model re-stamped (`uhd_gen restamp`) or
 * retrained.
 *
 * A shard carries its UED's models for every architecture, but each architecture's selector
 * is its own shard's (its packs are that architecture's), so a model is held to this shard's
 * revision only under the census architecture's key or `default`, the keys this shard's
 * engine serves; the other architectures' entries check theirs.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr const char* ROLE = "L1 predict_engine";

/// The architecture this census entry reads the shard of.
std::string censusArch()
{
    return hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
}

/// Where a mismatched engine's selector is written: the directory the runtime writes to
/// when one is set, so both builds' files land side by side, else a per-arch temporary one.
std::filesystem::path selectorDumpDir()
{
    const auto configured = hipdnn_data_sdk::utilities::getEnv(ingestor::SELECTOR_DUMP_DIR_ENV);
    if(!configured.empty())
    {
        return configured;
    }
    return std::filesystem::temp_directory_path() / "hipdnn-selectors" / censusArch();
}

TEST(TestShippedEnginePredictors, EveryBoundEnginePredictorForThisArchLoads)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    for(const auto& set : discoverDescriptorSets())
    {
        const auto ids = boundIds(set.engine.predictEngine, arch);
        if(ids.empty() || !packedTreeObligesLoad(ids, ROLE))
        {
            continue;
        }
        bool loaded = false;
        for(const auto& [metric, byArch] : set.enginePredictionsByMetric)
        {
            const auto* model = findModel(set.enginePredictionsByMetric, metric, arch);
            if(model == nullptr)
            {
                continue;
            }
            loaded = true;
            EXPECT_NE(std::find(ids.begin(),
                                ids.end(),
                                hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id)),
                      ids.end())
                << "engine '" << set.engine.name << "': the loaded " << metric << " " << ROLE
                << " model for " << arch << " is not one its UED binds";
            expectModelArtifactPresent(*model, ROLE);
        }
        EXPECT_TRUE(loaded) << "engine '" << set.engine.name << "' binds " << ids.size() << " "
                            << arch << " " << ROLE
                            << " model(s), but none survived the loader's pre-flight; the "
                               "engine reports no prediction";
    }
}

TEST(TestShippedEnginePredictors, EveryServedEnginePredictorMatchesThisBuildsSelector)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    for(const auto& set : discoverDescriptorSets())
    {
        if(set.enginePredictionsByMetric.empty())
        {
            continue;
        }
        // The same call makeEngine() makes, on the same loaded set.
        const auto revision = ingestor::engineSelectorRevision(set);
        std::string written;
        for(const auto& [metric, byArch] : set.enginePredictionsByMetric)
        {
            for(const auto& [archKey, model] : byArch)
            {
                if(archKey != arch && archKey != "default")
                {
                    continue;
                }
                if(model.trainedAgainstSelectorRevision == revision)
                {
                    continue;
                }
                if(written.empty())
                {
                    try
                    {
                        written = ingestor::writeEngineSelector(set, selectorDumpDir()).string();
                    }
                    catch(const std::exception& error)
                    {
                        written = std::string("(not written: ") + error.what() + ")";
                    }
                }
                ADD_FAILURE()
                    << "engine '" << set.engine.name << "' " << metric << "/" << archKey << " "
                    << ROLE << " model " << hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id)
                    << " (in " << model.baseDir.string() << ") records selector revision "
                    << model.trainedAgainstSelectorRevision << ", but this build computes "
                    << revision
                    << ". GenericEngine refuses it and ranks the engine without an estimate.\n"
                    << "This build's selector: " << written << "\n"
                    << "Write the training build's with " << ingestor::SELECTOR_DUMP_DIR_ENV
                    << "=<dir> and compare. If only kernel source paths moved, or the engine's "
                       "measured labels did not, re-stamp:\n"
                    << "  python -m uhd_gen restamp <model .uhd.json> --selector-before "
                       "<training build's selector> --selector-after "
                    << written
                    << " [--rows-before <L1 corpus at the old revision> --rows-after <L1 "
                       "corpus at the new revision>] --write\n"
                    << "Otherwise retrain the L1 against this build.";
            }
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
