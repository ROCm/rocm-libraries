// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
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
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestGfx1250ConvFwdAotModelBinding.cpp
 * @brief Census of the trained kernel ranker hipkernel:Gfx1250ConvFwdAot binds in its UED:
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
 * TestGfx1250ConvFwdAotPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The metric the trained ranker declares, and the arch key the UED binds it under.
constexpr const char* METRIC = "tflops";
constexpr const char* ARCH_KEY = "default";
constexpr const char* ROLE = "L2 sort_kernel_catalog";

using gfx1250_conv_fwd_aot::ConvShape;
using gfx1250_conv_fwd_aot::graphOf;

/// What each dtype's kernels are scored on: a dense 3x3 and a depthwise 11x11 pad-5
/// convolution (the shape of the direct_depthwise_spatial instances). Every shipped kernel
/// is scored on both, whether the graph admits it or not.
std::array<ConvShape, 2> scoredShapes()
{
    ConvShape depthwise;
    depthwise.n = 1;
    depthwise.h = 28;
    depthwise.w = 28;
    depthwise.y = 11;
    depthwise.x = 11;
    depthwise.groups = 64;
    depthwise.prePadding = {5, 5};
    depthwise.postPadding = {5, 5};
    return {ConvShape{}, depthwise};
}

/// Every kernel the packed catalog ships, as the engine's catalog holds it (a field the
/// descriptor omits takes the KMD default), by dtype.
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
            byDtype[definition.getStringMetadata("dtype")].push_back(std::move(definition));
        }
    }
    return byDtype;
}

TEST(TestGfx1250ConvFwdAotModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(loadedSet(GFX1250_CONV_FWD_AOT_ENGINE_NAME), METRIC, ARCH_KEY);
}

TEST(TestGfx1250ConvFwdAotModelBinding, TheBoundL2KernelRankerScoresEveryShippedKernel)
{
    const auto& set = loadedSet(GFX1250_CONV_FWD_AOT_ENGINE_NAME);
    if(!packedTreeObligesLoad(boundIds(set.engine.sortKernelCatalog, ARCH_KEY), ROLE))
    {
        return;
    }
    const auto* model = findModel(set.heuristicsByMetric, METRIC, ARCH_KEY);
    ASSERT_NE(model, nullptr) << "no " << METRIC
                              << " ranker loaded (see TheBoundL2KernelRankerLoads)";

    // The checks makeEngine()'s ranker makes when it first resolves the model.
    const auto ranker = ingestor::UhdKernelHeuristic::tryCreate(
        *model, "gfx1250 conv fwd ranker", set.engine.knobs, set.schema.fieldNames());
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
        for(auto shape : scoredShapes())
        {
            shape.dataType = graphDtype;
            const auto graph = graphOf(shape);
            auto context = graph.context();
            context.rankingMetric = METRIC;
            const auto bound = graphMatch(context);
            ASSERT_TRUE(bound.has_value())
                << "the engine does not serve the " << dtype << " graph with " << shape.y << "x"
                << shape.x << " filters";

            // One kernel at a time, so a failure names the kernels the model cannot encode.
            std::vector<std::string> unscored;
            for(const auto& kernel : kernels)
            {
                try
                {
                    const auto score = ranker->score(context, *bound, kernel);
                    if(!std::isfinite(score))
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
