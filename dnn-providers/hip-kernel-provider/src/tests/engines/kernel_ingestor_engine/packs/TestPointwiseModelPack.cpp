// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseModelPack.cpp
 * @brief hipkernel:PointwiseModel, the model-backed pack built by uhd_model_gen
 *        (src/integration_tests/kernel_ingestor_engine/fixtures/pointwise_model): its
 *        generated tree_data ranker loads and decides the ranking.
 *
 * The pack is staged into this binary's unit root only (its kernel is embedded_source, so
 * the dlopened plugin cannot serve it). Its two kernels tie on priority and the 256 kernel
 * has the lower id, so declared order picks 256; the model prefers 64. A ranking with 64
 * first is therefore the model's, not a fallback that agrees by luck.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr const char* ENGINE_NAME = "hipkernel:PointwiseModel";

/// The pack's kernels as the engine's catalog holds them.
std::vector<ingestor::KernelDefinition> shippedKernels(const ingestor::DescriptorSet& set)
{
    std::vector<ingestor::KernelDefinition> kernels;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ingestor::KernelDefinition definition;
            definition.kernelId = kernel.id;
            definition.packId = pack.id;
            definition.dispatchId = pack.dispatchId;
            definition.metadata = kernel.metadata;
            definition.priority = kernel.priority;
            definition.name = kernel.name;
            kernels.push_back(std::move(definition));
        }
    }
    return kernels;
}

TEST(TestPointwiseModelPack, TheGeneratedModelRanksTheKernelDeclaredOrderWouldNot)
{
    const auto& set = loadedSet(ENGINE_NAME);
    ASSERT_TRUE(set.heuristic.has_value())
        << "the generated sort_kernel_catalog model did not load; the engine ranks by "
           "declared order";

    const auto ranker = ingestor::UhdKernelHeuristic::tryCreate(
        *set.heuristic, "pointwise_model ranker", set.engine.knobs, set.schema.fieldNames());
    ASSERT_NE(ranker, nullptr) << "the generated model cannot be brought up";

    registerNativeIngestorSymbols();
    const auto graphMatch
        = ingestor::GraphMatchRegistry::resolve(set.engine.graphMatchNativeSymbol);
    const GraphFixture graph(buildPointwiseGraph());
    const auto context = graph.context();
    const auto bound = graphMatch(context);
    ASSERT_TRUE(bound.has_value()) << "the engine does not serve an fp32 ADD graph";

    // An installed tree can carry the pack once per architecture shard, so the catalog may
    // hold several copies of each kernel; what it must hold is a 64 and a 256 kernel.
    const auto kernels = shippedKernels(set);
    std::set<int64_t> blockSizes;
    for(const auto& kernel : kernels)
    {
        blockSizes.insert(kernel.getIntMetadata("block_size"));
    }
    ASSERT_EQ(blockSizes, (std::set<int64_t>{64, 256})) << "the pack ships a 64 and a 256 kernel";
    ingestor::Catalog catalog;
    catalog.bound = *bound;
    for(const auto& kernel : kernels)
    {
        catalog.entries.push_back(&kernel);
    }
    const auto ranking = ranker->rankScored(catalog, context);
    ASSERT_EQ(ranking.size(), kernels.size());
    EXPECT_TRUE(std::isfinite(ranking.front().score) && ranking.front().score > 0.0)
        << "the ranking fell back to declared order, whose every score is 0";
    const auto first = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) {
        return kernel.kernelId == ranking.front().kernelId;
    });
    ASSERT_NE(first, kernels.end());
    EXPECT_EQ(first->getIntMetadata("block_size"), 64)
        << "the model prefers the 64 kernel; 256 first is declared order";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
