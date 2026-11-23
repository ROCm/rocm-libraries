// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <iostream>
#include <set>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "ShippedEnginePredictorSources.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestShippedEnginePredictors.cpp
 * @brief Census of every engine-prediction (L1) model the shard ships, whatever the engine
 *        family: each one bound for the census architecture, or under `default`, loads and
 *        records the selector revision this build computes for its engine.
 *
 * GenericEngine refuses an L1 trained against another selector revision (PredictionStatus::
 * UNAVAILABLE) with one log line, and selection then ranks the engine as if no model
 * shipped. The revision digests every ranker, pack kernel and native symbol, so a retrained
 * L2, a recompiled code object or a source change does this silently. This suite fails the
 * build instead, for every engine the build packs: FlyDSL, rocKE when it is enabled, and
 * any family added later, with no list to keep in step.
 *
 * Engines whose source tree binds L1 models but which this build does not pack (rocKE with
 * HIPKERNELPROVIDER_ENABLE_ROCKE off, for one) cannot be checked here. They are listed as a
 * NOTE, so a run says which shipped models it did not cover. The list is read from the
 * source descriptors at configure time (ShippedEnginePredictorSources.hpp).
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// The architecture this census entry reads the shard of.
std::string censusArch()
{
    return hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
}

/// The metrics of the L1 models that loaded under @p archKey, or `tflops` when none did, so
/// a bound model that failed to load is still reported (as not having survived the loader).
std::set<std::string> loadedMetrics(const ingestor::DescriptorSet& set, const std::string& archKey)
{
    std::set<std::string> metrics;
    for(const auto& [metric, byArch] : set.enginePredictionsByMetric)
    {
        if(byArch.count(archKey) != 0)
        {
            metrics.insert(metric);
        }
    }
    if(metrics.empty())
    {
        metrics.insert("tflops");
    }
    return metrics;
}

/// Whether the source tree binds only native estimates (compiled scorers, which record no
/// selector revision) under @p archKey of @p engine.
bool bindsOnlyNativeEstimates(std::string_view engine, std::string_view archKey)
{
    for(const auto& binding : shipped_engine_predictor_sources::BINDINGS)
    {
        if(binding.engine == engine && binding.archKey == archKey)
        {
            return !binding.trained;
        }
    }
    return false;
}

TEST(TestShippedEnginePredictors, EveryEnginePredictorThisShardShipsMatchesThisBuild)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    std::set<std::string> packed;
    size_t checked = 0;
    for(const auto& set : discoverDescriptorSets())
    {
        packed.insert(set.engine.name);
        // A shard carries its UED's models for every architecture, but each architecture's
        // selector is its own shard's (its packs are that architecture's), so a model is
        // held to this shard's revision only under the census architecture or `default`.
        // An engine binding nothing under a key is held to loading nothing there, so the check
        // is live the day a UED starts binding one. A native estimate is code, not a trained
        // model, so it has no revision to hold.
        for(const std::string& archKey : {arch, std::string("default")})
        {
            if(bindsOnlyNativeEstimates(set.engine.name, archKey))
            {
                continue;
            }
            SCOPED_TRACE("engine '" + set.engine.name + "', predict_engine under " + archKey);
            for(const auto& metric : loadedMetrics(set, archKey))
            {
                expectBoundPredictorMatchesThisBuild(set, metric, archKey);
            }
            if(!boundIds(set.engine.predictEngine, archKey).empty())
            {
                ++checked;
            }
        }
    }

    for(const auto& binding : shipped_engine_predictor_sources::BINDINGS)
    {
        if(binding.trained && (binding.archKey == arch || binding.archKey == "default")
           && packed.count(std::string(binding.engine)) == 0)
        {
            std::cout << "[  NOTE    ] engine '" << binding.engine
                      << "' binds a trained L1 model under " << binding.archKey
                      << " in the source tree but is not packed in this build, so this census "
                         "does not check it\n";
        }
    }
    std::cout << "[  NOTE    ] checked " << checked
              << " (engine, architecture key) trained L1 binding(s) on " << arch << "\n";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
