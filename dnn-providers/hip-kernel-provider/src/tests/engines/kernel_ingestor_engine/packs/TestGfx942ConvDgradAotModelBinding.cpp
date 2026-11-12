// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradMatcherTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx942ConvDgradAotModelBinding.cpp
 * @brief Census of the trained kernel ranker hipkernel:Gfx942ConvDgradAot binds in its UED:
 *        it must survive the loader's pre-flight, and it must score every kernel the packed
 *        catalog ships.
 *
 * Both failures are silent at run time. The loader prunes a model it refuses (a
 * trained_against id that does not match the UED, KMD or UMD) with a log line, and the
 * ranker that cannot encode a kernel's metadata (a categorical value its
 * categorical_encoding lacks, as a family added after training gives) throws inside the
 * ranking, which UhdKernelHeuristic catches and logs before ranking every candidate by
 * priority. tflops is the default ranking metric, so either way the model is not used and
 * no other test notices. This suite scores every shipped kernel with the bound model on a
 * dense and a depthwise graph of its dtype and fails the build instead.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942ConvDgradAotPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt once the bundle ships a trained ranker (heuristics/);
 * until then the UED binds only the native fallback score and the call is skipped.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::buildGraph;
using hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::DgradShape;
using hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::graphSpec;
namespace ingestor = hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* ENGINE_NAME = "hipkernel:Gfx942ConvDgradAot";
/// The metric the trained ranker declares, and the arch key the UED binds it under.
constexpr const char* METRIC = "tflops";
constexpr const char* ARCH_KEY = "default";
constexpr const char* ROLE = "L2 sort_kernel_catalog";

/// What each dtype's kernels are scored on: a dense 3x3 and a depthwise 3x3 backward-data
/// convolution. Every shipped kernel is scored on both, whether the graph admits it or not.
std::array<DgradShape, 2> scoredShapes()
{
    DgradShape depthwise;
    depthwise.c = 64;
    depthwise.k = 64;
    depthwise.groups = 64;
    depthwise.h = 28;
    depthwise.w = 28;
    return {DgradShape{}, depthwise};
}

ingestor::DeviceProperties gfx942()
{
    ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    return properties;
}

/// Every selectable kernel the packed catalog ships, as the engine's catalog holds it (a
/// field the descriptor omits takes the KMD default), by dtype.
std::map<std::string, std::vector<ingestor::KernelDefinition>>
    shippedKernelsByDtype(const ingestor::DescriptorSet& set)
{
    std::map<std::string, std::vector<ingestor::KernelDefinition>> byDtype;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ingestor::KernelDefinition definition;
            definition.kernelId = kernel.id;
            definition.packId = pack.id;
            definition.dispatchId = pack.dispatchId;
            definition.metadata = kernel.metadata;
            for(const auto& field : set.schema.fields)
            {
                if(field.defaultValue.has_value())
                {
                    definition.metadata.emplace(field.name, *field.defaultValue);
                }
            }
            definition.priority = kernel.priority;
            definition.name = kernel.name;
            // The MFMA pipelines' weight-transform helpers (kernel_family 2) are loaded beside
            // their instance and never matched, so no catalog the ranker sees holds one.
            if(definition.getIntMetadata("kernel_family") == 2)
            {
                continue;
            }
            byDtype[definition.getStringMetadata("dtype")].push_back(std::move(definition));
        }
    }
    return byDtype;
}

TEST(TestGfx942ConvDgradAotModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(loadedSet(ENGINE_NAME), METRIC, ARCH_KEY);
}

TEST(TestGfx942ConvDgradAotModelBinding, TheBoundL2KernelRankerScoresEveryShippedKernel)
{
    const auto& set = loadedSet(ENGINE_NAME);
    if(!packedTreeObligesLoad(boundIds(set.engine.sortKernelCatalog, ARCH_KEY), ROLE))
    {
        return;
    }
    const auto* model = findModel(set.heuristicsByMetric, METRIC, ARCH_KEY);
    ASSERT_NE(model, nullptr) << "no " << METRIC
                              << " ranker loaded (see TheBoundL2KernelRankerLoads)";

    // The checks makeEngine()'s ranker makes when it first resolves the model.
    const auto ranker = ingestor::UhdKernelHeuristic::tryCreate(
        *model, "gfx942 conv dgrad ranker", set.engine.knobs, set.schema.fieldNames());
    ASSERT_NE(ranker, nullptr) << "the bound " << METRIC
                               << " ranker cannot be brought up; kernels rank by priority";

    registerNativeIngestorSymbols();
    const auto graphMatch
        = ingestor::GraphMatchRegistry::resolve(set.engine.graphMatchNativeSymbol);
    const auto byDtype = shippedKernelsByDtype(set);
    ASSERT_EQ(byDtype.size(), 2U) << "the catalog ships fp16 and bf16 kernels";
    for(const auto& [dtype, kernels] : byDtype)
    {
        const auto graphDtype
            = dtype == "bf16" ? data_objects::DataType::BFLOAT16 : data_objects::DataType::HALF;
        for(const auto& shape : scoredShapes())
        {
            const GraphFixture graph(buildGraph(graphSpec(shape, graphDtype)), gfx942());
            auto context = graph.context();
            context.rankingMetric = METRIC;
            const auto bound = graphMatch(context);
            ASSERT_TRUE(bound.has_value()) << "the engine does not serve the " << dtype
                                           << " graph with " << shape.groups << " groups";

            // One kernel at a time, so a failure names the kernels the model cannot encode.
            std::vector<std::string> unscored;
            for(const auto& kernel : kernels)
            {
                try
                {
                    // score() reports an unusable prediction (NaN, infinite or not
                    // positive) as 0, which ranks the kernel last.
                    const auto score = ranker->score(context, *bound, kernel);
                    if(!std::isfinite(score) || score <= 0.0)
                    {
                        unscored.push_back(kernel.name + " (score " + std::to_string(score) + ")");
                    }
                }
                catch(const std::exception& error)
                {
                    unscored.push_back(kernel.name + " (" + error.what() + ")");
                }
            }
            EXPECT_TRUE(unscored.empty())
                << "the bound " << METRIC << " ranker cannot score " << unscored.size() << " of "
                << kernels.size() << " shipped " << dtype << " kernels, so a graph any of them "
                << "serves ranks its whole catalog by priority; first: " << unscored.front();

            // The whole catalog as one ranking: a ranking that fails falls back to declared
            // order, whose every score is 0.
            ingestor::Catalog catalog;
            catalog.bound = *bound;
            for(const auto& kernel : kernels)
            {
                catalog.entries.push_back(&kernel);
            }
            const auto ranking = ranker->rankScored(catalog, context);
            ASSERT_EQ(ranking.size(), kernels.size());
            EXPECT_TRUE(std::isfinite(ranking.front().score) && ranking.front().score > 0.0)
                << "the " << dtype << " catalog's ranking fell back to declared order";
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
