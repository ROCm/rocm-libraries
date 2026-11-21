// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestShippedEnginePredictors.cpp
 * @brief Census of every engine-prediction (L1) model the shard ships, whatever the engine:
 *        each one this architecture's engine would serve loads, records the selector
 *        revision this build computes, passes the runtime's own acceptance, and, for a forward
 *        SDPA engine, estimates each representative graph it serves in TFLOPS.
 *
 * GenericEngine refuses an L1 trained against another selector revision (PredictionStatus::
 * UNAVAILABLE) with one log line, and selection then ranks the engine in static order as if
 * no model shipped. The revision digests every ranker, pack kernel and native symbol, so
 * any of those moving -- a retrained L2, a recompiled code object, a merge from another
 * branch -- does this silently. The runtime then checks the model itself when it is first
 * asked (uhd::prediction_detail::validateBinding and model()): an uncalibrated score
 * answers every query INVALID, and a model the artifact cannot back answers UNAVAILABLE.
 * Neither the loader nor the revision sees either, and a score transform that does not
 * match the model passes all of them while reporting a value that is not in TFLOPS. This
 * suite fails the build instead, so the L1 is retrained against this build.
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
namespace uhd = hipdnn_plugin_sdk::uhd;
namespace graphs = hip_kernel_provider::kernel_ingestor_engine::testing::attention_dense_graphs;
using PredictionStatus = hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;

constexpr const char* ROLE = "L1 predict_engine";
/// The metric the representative-graph estimate is checked in.
constexpr const char* TFLOPS = "tflops";
/// What an L1 of a forward SDPA engine reads: the forward SDPA node's work.
constexpr const char* SDPA_FORWARD_FEATURE = "$graph.flops_by_type.SdpaAttributes";

/// The architecture this census entry reads the shard of.
std::string censusArch()
{
    return hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
}

/// The arch keys whose L1 the census architecture's engine serves.
std::array<std::string, 2> servedKeys(const std::string& arch)
{
    return {arch, "default"};
}

/// The metrics of the L1 models loaded under @p archKey.
std::vector<std::string> loadedMetrics(const ingestor::DescriptorSet& set,
                                       const std::string& archKey)
{
    std::vector<std::string> metrics;
    for(const auto& [metric, byArch] : set.enginePredictionsByMetric)
    {
        if(byArch.count(archKey) != 0)
        {
            metrics.push_back(metric);
        }
    }
    return metrics;
}

/// The device a census architecture's representative graph is estimated on: the board the
/// architecture's shipped L1s were measured on, as hipGetDeviceProperties reports it, and
/// that board's dense bf16/fp16 matrix peak in TFLOPS, which no estimate may exceed.
struct CensusDevice
{
    ingestor::DeviceProperties properties;
    double peakTflops = 0.0;
};

std::optional<CensusDevice> censusDevice(const std::string& arch)
{
    // NOLINTBEGIN(readability-magic-numbers)
    const auto device = [&arch](int cus,
                                int warp,
                                std::size_t memory,
                                int busWidth,
                                int memoryClockKhz,
                                std::size_t lds,
                                double peak) {
        CensusDevice result;
        result.properties.gcnArchName = arch;
        result.properties.warpSize = warp;
        result.properties.multiProcessorCount = cus;
        result.properties.totalGlobalMem = memory;
        result.properties.memoryBusWidth = busWidth;
        result.properties.memoryClockRate = memoryClockKhz;
        result.properties.sharedMemPerBlock = lds;
        result.peakTflops = peak;
        return result;
    };
    if(arch == "gfx942") // MI300X
    {
        return device(304, 64, 205822885888, 8192, 1300000, 65536, 1307.4);
    }
    if(arch == "gfx950") // MI355X
    {
        return device(256, 64, 309220868096, 8192, 1900000, 163840, 2516.6);
    }
    if(arch == "gfx1100") // Radeon RX 7900 XTX
    {
        return device(48, 32, 25753026560, 384, 1250000, 65536, 122.8);
    }
    if(arch == "gfx1101") // Radeon RX 7800 XT
    {
        return device(30, 32, 17163091968, 256, 1219000, 65536, 74.6);
    }
    if(arch == "gfx1151") // Radeon 8060S
    {
        return device(20, 32, 103079215104, 256, 4000000, 65536, 59.4);
    }
    if(arch == "gfx1201") // Radeon RX 9070 XT
    {
        return device(32, 32, 17095983104, 256, 1258000, 65536, 194.7);
    }
    // NOLINTEND(readability-magic-numbers)
    return std::nullopt;
}

/// The forward SDPA graphs an engine's estimate is checked on: each engine is estimated on
/// every one its graph match serves: a large prefill and a large decode, the problems the
/// engines are built for, a small prefill, and a small decode, which no engine runs at even
/// 1 TFLOPS. A `log` model whose score is read back through the wrong transform reports its
/// raw score, ln(TFLOPS), which is negative on the small decode, and the runtime answers that
/// INVALID.
std::vector<std::pair<std::string, graphs::GraphSpec>> representativeGraphs()
{
    // NOLINTBEGIN(readability-magic-numbers)
    graphs::GraphSpec prefill;
    prefill.batch = 4;
    prefill.numQueryHeads = 32;
    prefill.numKvHeads = 32;
    prefill.seqLenQ = 4096;
    prefill.seqLenKv = 4096;

    auto gqaWithStats = graphs::withStats(prefill);
    gqaWithStats.numKvHeads = 8;

    graphs::GraphSpec smallPrefill;
    smallPrefill.batch = 1;
    smallPrefill.numQueryHeads = 8;
    smallPrefill.numKvHeads = 8;
    smallPrefill.seqLenQ = 128;
    smallPrefill.seqLenKv = 128;

    graphs::GraphSpec decode;
    decode.batch = 64;
    decode.numQueryHeads = 64;
    decode.numKvHeads = 4;
    decode.seqLenQ = 1;
    decode.seqLenKv = 8192;
    decode.leftBound.reset();
    decode.rightBound.reset();

    auto smallDecode = decode;
    smallDecode.batch = 1;
    smallDecode.numQueryHeads = 8;
    smallDecode.numKvHeads = 8;
    smallDecode.seqLenKv = 256;
    // NOLINTEND(readability-magic-numbers)

    return {{"bf16 D128 B4 H32 S4096 causal prefill", prefill},
            {"bf16 D128 B4 Hq32 Hkv8 S4096 causal prefill with stats", gqaWithStats},
            {"bf16 D128 B1 H8 S128 causal prefill", smallPrefill},
            {"bf16 D128 B64 Hq64 Hkv4 Skv8192 decode", decode},
            {"bf16 D128 B1 H8 Skv256 decode", smallDecode}};
}

/// Whether @p descriptor estimates forward SDPA graphs, by what its features read.
bool estimatesSdpaForward(const ingestor::HeuristicDescriptor& descriptor)
{
    const auto config = ingestor::UhdKernelHeuristic::configFrom(descriptor);
    return std::any_of(
        config.featuresSignature.begin(), config.featuresSignature.end(), [](const auto& entry) {
            return entry.is_string() && entry.template get<std::string>() == SDPA_FORWARD_FEATURE;
        });
}

/// The model GenericEngine's binding would serve the census architecture for @p metric: the
/// architecture's own key over `default`. Null when neither loaded.
const ingestor::HeuristicDescriptor* servedModel(const ingestor::DescriptorSet& set,
                                                 const std::string& metric,
                                                 const std::string& arch)
{
    for(const auto& key : servedKeys(arch))
    {
        if(const auto* model = findModel(set.enginePredictionsByMetric, metric, key))
        {
            return model;
        }
    }
    return nullptr;
}

TEST(TestShippedEnginePredictors, EveryServedEnginePredictorLoadsAndMatchesThisBuild)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    for(const auto& set : discoverDescriptorSets())
    {
        for(const auto& key : servedKeys(arch))
        {
            const auto ids = boundIds(set.engine.predictEngine, key);
            if(ids.empty())
            {
                continue;
            }
            SCOPED_TRACE("engine '" + set.engine.name + "', its " + key + " " + ROLE + " model");
            const auto metrics = loadedMetrics(set, key);
            if(metrics.empty())
            {
                // Only the architecture's own binding must load: a `default` one may be a
                // native placeholder that the loader disables where no scorer is registered
                // for it (the FlyDSL engines bind theirs so).
                EXPECT_FALSE(key == arch && packedTreeObligesLoad(ids, ROLE))
                    << "the UED binds " << ids.size() << " " << key << " " << ROLE
                    << " model(s), but none survived the loader's pre-flight; the engine "
                       "reports no prediction";
                continue;
            }
            for(const auto& metric : metrics)
            {
                const auto* model = findModel(set.enginePredictionsByMetric, metric, key);
                const auto id = hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id);
                SCOPED_TRACE(metric + " model " + id + " in " + model->baseDir.string());
                EXPECT_NE(std::find(ids.begin(), ids.end(), id), ids.end())
                    << "the loaded model is not one the UED binds";
                expectBoundPredictorMatchesThisBuild(set, metric, key);
            }
        }
    }
}

TEST(TestShippedEnginePredictors, EveryServedEnginePredictorPassesTheRuntimeAcceptance)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    for(const auto& set : discoverDescriptorSets())
    {
        for(const auto& key : servedKeys(arch))
        {
            for(const auto& metric : loadedMetrics(set, key))
            {
                const auto& model = *findModel(set.enginePredictionsByMetric, metric, key);
                const auto config = ingestor::UhdKernelHeuristic::configFrom(model);
                SCOPED_TRACE("engine '" + set.engine.name + "', its " + key + " " + metric + " "
                             + ROLE + " model "
                             + hipdnn_flatbuffers_sdk::utilities::formatUuid(model.id) + " in "
                             + model.baseDir.string());

                // What uhd::predictEngine asks of the model before it evaluates it.
                try
                {
                    uhd::prediction_detail::validateBinding(config, set.engine.name, arch, metric);
                }
                catch(const std::exception& error)
                {
                    ADD_FAILURE() << "the runtime refuses the binding, so every query of the "
                                     "engine's estimate answers INVALID: "
                                  << error.what() << ". Retrain the L1 against this build.";
                }
                const auto compiled = uhd::prediction_detail::model(config);
                EXPECT_EQ(compiled->status, PredictionStatus::AVAILABLE)
                    << "the runtime cannot bring the model up, so the engine reports no "
                       "estimate: "
                    << compiled->reason;
            }
        }
    }
}

TEST(TestShippedEnginePredictors, EveryServedSdpaForwardPredictorEstimatesItsRepresentativeGraphs)
{
    const auto arch = censusArch();
    ASSERT_FALSE(arch.empty()) << "a census entry names its architecture";

    registerNativeIngestorSymbols();
    for(const auto& set : discoverDescriptorSets())
    {
        const auto* model = servedModel(set, TFLOPS, arch);
        if(model == nullptr || !estimatesSdpaForward(*model))
        {
            continue;
        }
        SCOPED_TRACE("engine '" + set.engine.name + "', its " + TFLOPS + " " + ROLE + " model "
                     + hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id));
        const auto device = censusDevice(arch);
        ASSERT_TRUE(device.has_value()) << "the engine serves a forward SDPA estimate on " << arch
                                        << ", which censusDevice() names no board for";

        // GenericEngine's binding, built as its constructor builds it: every loaded model,
        // each under the arch key and metric it was bound with.
        uhd::EngineModelBinding binding;
        for(const auto& [metric, byArch] : set.enginePredictionsByMetric)
        {
            for(const auto& [key, bound] : byArch)
            {
                binding.bind(metric, key, ingestor::UhdKernelHeuristic::configFrom(bound));
            }
        }
        const auto stateManager
            = ingestor::makeStateManager<::Handle>(set, set.engine.graphMatchNativeSymbol);

        int served = 0;
        for(const auto& [name, spec] : representativeGraphs())
        {
            const auto builder = graphs::buildGraph(spec);
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
                builder.GetBufferPointer(), builder.GetSize());
            const ingestor::MatchContext context{graph, 0, device->properties};
            const auto bound = stateManager->graphBindings(context);
            if(!bound.has_value())
            {
                continue;
            }
            ++served;

            // GenericPlanBuilder::predictionFeatures with no constraint pinned.
            auto features
                = hipdnn_plugin_sdk::heuristics::problemFeatures(graph, device->properties);
            ingestor::detail::bindGraphMatchBindings(features, *bound, stateManager->isSlotted());
            const auto estimate = binding.predict(0,
                                                  set.engine.name,
                                                  ingestor::engineSelectorRevision(set),
                                                  TFLOPS,
                                                  arch,
                                                  features,
                                                  /*evaluate=*/true);
            EXPECT_EQ(estimate.status, PredictionStatus::AVAILABLE)
                << "the engine answers no estimate for the " << name
                << " graph it serves: " << estimate.reason
                << ". A score read back through another transform than the model was trained "
                   "with (its score.transform) can leave the TFLOPS range";
            EXPECT_TRUE(std::isfinite(estimate.value) && estimate.value > 0.0
                        && estimate.value <= device->peakTflops)
                << "the " << name << " graph is estimated at " << estimate.value
                << " TFLOPS, outside (0, " << device->peakTflops
                << "], this board's matrix peak: the score is not read back into TFLOPS as "
                   "the model was trained (its score.transform), or the model is not the "
                   "engine's";
        }
        EXPECT_GT(served, 0) << "the engine serves none of representativeGraphs(); add the "
                                "forward SDPA graph it is built for";
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
