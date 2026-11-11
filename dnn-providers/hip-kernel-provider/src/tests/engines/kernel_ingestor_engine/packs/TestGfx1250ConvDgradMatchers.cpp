// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvDgradAotGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradMatcherTestSupport.hpp"
#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradShippedCatalog.hpp"
#endif
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradTestCatalog.hpp"

/**
 * @file TestGfx1250ConvDgradMatchers.cpp
 * @brief hipkernel:Gfx1250ConvDgradAot's native symbols, reached through the registry by
 *        the names its descriptors carry: the graph matcher on gfx1250 (wave32) only, and
 *        the kernel matcher and fallback score over the whole shipped catalog against the
 *        rocKE adapter's served matrix (Gfx1250ConvDgradTestVectors.hpp). The engine
 *        shares its hooks with hipkernel:Gfx950ConvDgradAot (ConvDgradAotNative.cpp);
 *        TestGfx950ConvDgradMatchers.cpp covers the contract they share. When this build
 *        packed the gfx1250 production tree, the shipped catalog also runs through the real
 *        state manager: every graph's candidates have distinct knob tuples, and one shipped
 *        kernel of each variant passes the pack's signature check. Host-only.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test;
namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx1250_dgrad_test_vectors;
using hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols;

constexpr PackSymbols GFX1250_CONV_DGRAD_AOT{"hipkernel:Gfx1250ConvDgradAot",
                                             "hipkernel.gfx1250_conv_dgrad_aot.graph_match",
                                             "",
                                             "hipkernel.gfx1250_conv_dgrad_aot.kernel_match",
                                             "hipkernel.gfx1250_conv_dgrad_aot.score",
                                             "hipkernel.gfx1250_conv_dgrad_aot.dispatch",
                                             "gfx1250_conv_dgrad.dy.uid",
                                             "gfx1250_conv_dgrad.w.uid",
                                             "gfx1250_conv_dgrad.dx.uid"};
constexpr std::string_view PROBLEM_TOKEN = "_gfx1250_conv_dgrad.problem";
constexpr std::string_view GFX950_GRAPH_MATCHER = "hipkernel.gfx950_conv_dgrad_aot.graph_match";

DeviceProperties deviceProperties(std::string arch = "gfx1250", int warpSize = 32)
{
    // By value: matcher tests run on CPU-only and non-gfx1250 hosts.
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = warpSize;
    return properties;
}

GraphFixture fixtureOf(const DgradShape& shape,
                       data_objects::DataType dtype = data_objects::DataType::HALF,
                       DeviceProperties properties = deviceProperties())
{
    return GraphFixture(buildGraph(graphSpec(shape, dtype), "gfx1250_conv_dgrad_test"),
                        std::move(properties));
}

/// The @p dtype catalog of @p target's engine as one pack, prepared by the gfx1250
/// engine's KernelPrepareFn, in catalog order.
std::vector<KernelDefinition>
    catalogPack(const std::string& dtype, uint8_t pack, const dgrad::Target& target)
{
    return hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::catalogPack(
        dtype, pack, target, GFX1250_CONV_DGRAD_AOT.kernelMatcher);
}

TEST(TestGfx1250ConvDgradGraphMatcher, AcceptsBothStorageTypesOnGfx1250AndBindsItsTokens)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        const auto fixture = fixtureOf(DgradShape{}, dtype);
        const auto bound = matchesGraph(GFX1250_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX1250_CONV_DGRAD_AOT.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX1250_CONV_DGRAD_AOT.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX1250_CONV_DGRAD_AOT.outputToken), 3);
        ASSERT_TRUE(bound->count(std::string(PROBLEM_TOKEN)));
        // The problem token is private: only the tensor bindings are features.
        EXPECT_TRUE(detail::isPrivateToken(std::string(PROBLEM_TOKEN)));
        EXPECT_EQ(bound->size(), 4U);
    }
}

TEST(TestGfx1250ConvDgradGraphMatcher, EachEngineMatchesItsOwnArchitectureOnly)
{
    registerNativeIngestorSymbols();
    const auto gfx950Matcher = GraphMatchRegistry::tryResolve(std::string(GFX950_GRAPH_MATCHER));
    ASSERT_NE(gfx950Matcher, nullptr);
    const auto onGfx1250 = fixtureOf(DgradShape{});
    EXPECT_FALSE(gfx950Matcher(onGfx1250.context()));
    for(const auto& properties : {deviceProperties("gfx950", 64),
                                  deviceProperties("gfx1250", 64),
                                  deviceProperties("gfx942", 32),
                                  deviceProperties("", 32)})
    {
        const auto fixture = fixtureOf(DgradShape{}, data_objects::DataType::HALF, properties);
        EXPECT_FALSE(matchesGraph(GFX1250_CONV_DGRAD_AOT, fixture.context()))
            << properties.gcnArchName << " wave" << properties.warpSize;
    }
    // A gfx1250 variant name still prefix-matches.
    const auto variant
        = fixtureOf(DgradShape{}, data_objects::DataType::HALF, deviceProperties("gfx1250:xnack-"));
    EXPECT_TRUE(matchesGraph(GFX1250_CONV_DGRAD_AOT, variant.context()));
}

TEST(TestGfx1250ConvDgradKernelMatcher, ServesExactlyWhatTheAdapterServesOverTheCatalog)
{
    const std::map<std::string, std::vector<KernelDefinition>> packs{
        {"fp16", catalogPack("fp16", 1, dgrad::GFX1250)},
        {"bf16", catalogPack("bf16", 2, dgrad::GFX1250)}};
    // gfx950's instances in the gfx1250 engine's hands: none parses for it.
    const auto gfx950Pack = catalogPack("fp16", 3, dgrad::GFX950);
    size_t rows = 0;
    size_t served = 0;
    std::map<std::string, size_t> servedByVariant;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem);
        const auto geometry = dgrad::deriveGeometry(problem);
        // No graph states a problem outside the geometry; graphs past 2^31 elements take
        // seconds to build here, and the geometry test covers both.
        if(!geometry || geometry->tensorBytes[2] > dgrad::I32_MAX)
        {
            continue;
        }
        ++rows;
        const std::string dtypeName = row.dtype;
        const auto dtype
            = dtypeName == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
        const auto shape = shapeOf(row.problem);
        const auto fixture = fixtureOf(shape, dtype);
        const auto bound = matchesGraph(GFX1250_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        const auto& own = packs.at(dtypeName);
        for(size_t i = 0; i < own.size(); ++i)
        {
            const bool expected = row.served[i] == '1';
            const bool actual
                = matchesKernel(GFX1250_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(actual, expected) << own[i].name << " on " << shape;
            // A served kernel scores, a refused one ranks last.
            const auto score
                = scoreKernel(GFX1250_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(score >= 0.0, expected) << own[i].name << " score " << score;
            if(actual)
            {
                ++served;
                ++servedByVariant[std::get<std::string>(own[i].metadata.at("direct_variant"))];
            }
        }
        for(const auto& kernel : packs.at(dtypeName == "fp16" ? "bf16" : "fp16"))
        {
            EXPECT_FALSE(matchesKernel(GFX1250_CONV_DGRAD_AOT, fixture.context(), kernel, *bound))
                << kernel.name;
        }
        if(dtypeName == "fp16")
        {
            for(const auto& kernel : gfx950Pack)
            {
                EXPECT_FALSE(
                    matchesKernel(GFX1250_CONV_DGRAD_AOT, fixture.context(), kernel, *bound))
                    << kernel.name;
            }
        }
    }
    EXPECT_GT(rows, 400U);
    EXPECT_GT(served, 6000U);
    for(const auto* variant : {"none",
                               "depthwise_dgrad",
                               "depthwise_dgrad_win",
                               "dgrad_wmma",
                               "dgrad_smallc_wmma",
                               "dgrad_patch_wmma"})
    {
        EXPECT_GT(servedByVariant[variant], 0U) << variant;
    }
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
namespace shipped = hip_kernel_provider::kernel_ingestor_engine::dgrad_shipped_catalog;

/// The shipped bundle's state manager, built once, or null when it is not staged.
const shipped::StateManager* shippedManager(std::vector<std::string>* knobs = nullptr)
{
    const auto& loaded = shipped::shippedCatalog("gfx1250", GFX1250_CONV_DGRAD_AOT.engineName);
    if(knobs != nullptr)
    {
        *knobs = loaded.knobs;
    }
    return loaded.manager.get();
}

const std::string NOT_STAGED = shipped::notStaged("gfx1250", GFX1250_CONV_DGRAD_AOT.engineName);

/// The served-matrix rows a graph can state (within the geometry, dX under 2^31 bytes).
template <typename Visit>
void forEachStatableRow(Visit&& visit)
{
    for(const auto& row : vectors::servedMatrix())
    {
        const auto geometry = dgrad::deriveGeometry(catalog::toProblem(row.problem));
        if(geometry && geometry->tensorBytes[2] <= dgrad::I32_MAX)
        {
            visit(row);
        }
    }
}

TEST(TestGfx1250ConvDgradCatalog, EveryGraphsCandidatesHaveDistinctKnobTuples)
{
    std::vector<std::string> knobs;
    const auto* manager = shippedManager(&knobs);
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    ASSERT_FALSE(knobs.empty());

    // The candidate set is what the real matchers and the shipped metadata (with its KMD
    // defaults) admit, and its size is the adapter's count.
    size_t graphs = 0;
    size_t candidates = 0;
    forEachStatableRow([&](const auto& row) {
        const auto shape = shapeOf(row.problem);
        const auto expected = static_cast<size_t>(
            std::count(row.served, row.served + std::string_view(row.served).size(), '1'));
        const std::string dtypeName = row.dtype;
        const auto dtype
            = dtypeName == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
        const auto fixture = fixtureOf(shape, dtype);
        const auto shipped = manager->unsortedCatalog(fixture.context());
        EXPECT_EQ(shipped.entries.size(), expected) << shape;
        std::set<std::vector<int64_t>> tuples;
        for(const auto& kernel : shipped.entries)
        {
            EXPECT_EQ(std::get<std::string>(kernel->metadata.at("dtype")), dtypeName);
            std::vector<int64_t> tuple;
            for(const auto& knob : knobs)
            {
                const auto ordinal = manager->knobOrdinal(knob, kernel->metadata.at(knob));
                ASSERT_TRUE(ordinal) << knob;
                tuple.push_back(*ordinal);
            }
            EXPECT_TRUE(tuples.insert(tuple).second)
                << kernel->name << " repeats another candidate's knob tuple";
        }
        ++graphs;
        candidates += shipped.entries.size();
    });
    EXPECT_GT(graphs, 400U);
    EXPECT_GT(candidates, 6000U);
}

TEST(TestGfx1250ConvDgradCatalog, ShippedKernelsMatchThePacksSignaturesAndLoad)
{
    // One shipped kernel of each variant, pipeline and dtype, on the first served-matrix
    // problem the adapter serves it on, through prepare(): requireSignatureMatch compares
    // the ABI this pack marshals (names, kinds, sizes) with what hkp_pack recorded, before
    // any device is needed; with a gfx1250 device the code object then loads.
    const auto* manager = shippedManager();
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    const auto device = currentDeviceProperties();
    const bool loadable = device.gcnArchName.rfind("gfx1250", 0) == 0;
    const auto& handler = dispatchHandler(GFX1250_CONV_DGRAD_AOT);
    // Each dtype's kernels sit in their own shards, so both are checked.
    for(const std::string dtypeName : {"fp16", "bf16"})
    {
        const auto kernels = catalogPack(dtypeName, 1, dgrad::GFX1250);
        // By (direct_variant, pipeline): implicit GEMM ships a mem and a wavelet shard.
        std::map<std::pair<std::string, std::string>, std::array<int64_t, 14>> problemOf;
        forEachStatableRow([&](const auto& row) {
            if(std::string_view(row.dtype) != dtypeName)
            {
                return;
            }
            for(size_t i = 0; i < kernels.size(); ++i)
            {
                if(row.served[i] == '1')
                {
                    problemOf.emplace(
                        std::make_pair(
                            std::get<std::string>(kernels[i].metadata.at("direct_variant")),
                            std::get<std::string>(kernels[i].metadata.at("pipeline"))),
                        row.problem);
                }
            }
        });
        ASSERT_EQ(problemOf.size(), 7U)
            << dtypeName << ": every variant and pipeline is served somewhere";
        const auto dtype
            = dtypeName == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
        for(const auto& entry : problemOf)
        {
            const auto& variant = entry.first.first;
            const auto& pipeline = entry.first.second;
            SCOPED_TRACE(testing::Message() << dtypeName << " " << variant << " " << pipeline);
            const auto fixture
                = fixtureOf(shapeOf(entry.second), dtype, loadable ? device : deviceProperties());
            const auto catalog = manager->unsortedCatalog(fixture.context());
            const auto kernel = std::find_if(
                catalog.entries.begin(), catalog.entries.end(), [&](const auto& candidate) {
                    return std::get<std::string>(candidate->metadata.at("direct_variant"))
                               == variant
                           && std::get<std::string>(candidate->metadata.at("pipeline")) == pipeline;
                });
            ASSERT_NE(kernel, catalog.entries.end());
            const auto bound = matchesGraph(GFX1250_CONV_DGRAD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            shipped::expectPreparesOnItsDeviceOnly(
                handler,
                fixture.context(),
                *bound,
                **kernel,
                loadable,
                {"is packaged with arguments", "convolution contract"});
        }
    }
}
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
