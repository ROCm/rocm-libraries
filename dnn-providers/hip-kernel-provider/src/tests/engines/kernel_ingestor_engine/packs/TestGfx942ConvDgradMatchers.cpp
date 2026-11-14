// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
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
 * @file TestGfx942ConvDgradMatchers.cpp
 * @brief hipkernel:Gfx942ConvDgradAot's native symbols, reached through the registry by
 *        the names its descriptors carry: the graph matcher on gfx942 (wave64) only, the
 *        kernel matcher and fallback score over the whole shipped catalog against the rocKE
 *        adapter's served matrix (Gfx942ConvDgradTestVectors.hpp), the MFMA helper lookup,
 *        and the dispatch handler's workspace and preparation, which refuse together what
 *        the plan refuses (gfx942's 64 KiB MFMA staging, a depthwise cpl that does not
 *        divide the channels). The engine shares its hooks with hipkernel:Gfx950ConvDgradAot
 *        (ConvDgradAotNative.cpp); TestGfx950ConvDgradMatchers.cpp covers the graph contract
 *        they share. When this build packed the gfx942 production tree, the shipped catalog
 *        also runs through the real state manager: every graph's candidates have distinct
 *        knob tuples, and one shipped kernel of each variant passes the pack's signature
 *        check (and loads, on a gfx942 device). Host-only otherwise.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test;
namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::conv_dgrad_aot;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx942_dgrad_test_vectors;
using hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols;

constexpr PackSymbols GFX942_CONV_DGRAD_AOT{"hipkernel:Gfx942ConvDgradAot",
                                            "hipkernel.gfx942_conv_dgrad_aot.graph_match",
                                            "",
                                            "hipkernel.gfx942_conv_dgrad_aot.kernel_match",
                                            "hipkernel.gfx942_conv_dgrad_aot.score",
                                            "hipkernel.gfx942_conv_dgrad_aot.dispatch",
                                            "gfx942_conv_dgrad.dy.uid",
                                            "gfx942_conv_dgrad.w.uid",
                                            "gfx942_conv_dgrad.dx.uid"};
constexpr std::string_view PROBLEM_TOKEN = "_gfx942_conv_dgrad.problem";
constexpr std::string_view GFX950_GRAPH_MATCHER = "hipkernel.gfx950_conv_dgrad_aot.graph_match";
constexpr std::string_view GFX1250_GRAPH_MATCHER = "hipkernel.gfx1250_conv_dgrad_aot.graph_match";

DeviceProperties deviceProperties(std::string arch = "gfx942", int warpSize = 64)
{
    // By value: matcher tests run on CPU-only and non-gfx942 hosts.
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = warpSize;
    return properties;
}

GraphFixture fixtureOf(const DgradShape& shape,
                       data_objects::DataType dtype = data_objects::DataType::HALF,
                       DeviceProperties properties = deviceProperties())
{
    return GraphFixture(buildGraph(graphSpec(shape, dtype), "gfx942_conv_dgrad_test"),
                        std::move(properties));
}

data_objects::DataType dtypeOf(const std::string& name)
{
    return name == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
}

/// A kpack kernel of @p entry in pack @p pack, named as the gfx942 bundle names it. The
/// MFMA helpers are filed in one registry the dgrad engines share in this process, keyed by
/// archive and pack, so this suite's packs are numbered from 101: a pack id another dgrad
/// suite also files helpers under would make each lookup ambiguous.
KernelDefinition kernelOf(const catalog::Entry& entry, uint8_t pack = 101)
{
    return hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::kernelOf(
        entry, pack, "gfx942_conv_dgrad_aot_");
}

void prepare(std::vector<KernelDefinition>& kernels)
{
    prepareKernels(kernels, GFX942_CONV_DGRAD_AOT.kernelMatcher);
}

/// The @p dtype catalog of @p target's engine as one pack, prepared by the gfx942 engine's
/// KernelPrepareFn, in catalog order.
std::vector<KernelDefinition>
    catalogPack(const std::string& dtype, uint8_t pack, const dgrad::Target& target = dgrad::GFX942)
{
    return hip_kernel_provider::kernel_ingestor_engine::dgrad_matcher_test::catalogPack(
        dtype, pack, target, GFX942_CONV_DGRAD_AOT.kernelMatcher);
}

const catalog::Entry& variantEntry(
    const std::string& variant,
    const std::function<bool(const catalog::Entry&)>& pick
    = [](const catalog::Entry&) { return true; },
    const std::string& dtype = "fp16")
{
    return catalog::firstEntry(
        dtype,
        [&](const catalog::Entry& e) { return e.text("direct_variant") == variant && pick(e); },
        dgrad::GFX942);
}

std::vector<const catalog::Entry*> helperEntries(const std::string& dtype = "fp16")
{
    std::vector<const catalog::Entry*> helpers;
    for(const auto& entry : catalog::catalog(dtype, dgrad::GFX942))
    {
        if(entry.isHelper())
        {
            helpers.push_back(&entry);
        }
    }
    return helpers;
}

/// @p entry with every helper of its dtype, in pack @p pack, prepared.
std::vector<KernelDefinition> withHelpers(const catalog::Entry& entry, uint8_t pack)
{
    std::vector<KernelDefinition> kernels{kernelOf(entry, pack)};
    for(const auto* helper : helperEntries(entry.dtype))
    {
        kernels.push_back(kernelOf(*helper, pack));
    }
    prepare(kernels);
    return kernels;
}

DgradShape depthwiseShape(int64_t channels = 96)
{
    DgradShape shape;
    shape.c = shape.k = shape.groups = channels;
    return shape;
}

/// A grouped problem tall enough for every fused row: an xcd_tiles row needs eight image
/// row bands.
DgradShape groupedShape(int64_t cpg, int64_t groups)
{
    DgradShape shape;
    shape.n = 16;
    shape.h = shape.w = 28;
    shape.c = shape.k = cpg * groups;
    shape.groups = groups;
    return shape;
}

/// The dense c4k320 problem the MFMA rows serve.
DgradShape mfmaShape()
{
    DgradShape shape;
    shape.n = 1;
    shape.c = 4;
    shape.k = 320;
    shape.h = shape.w = 32;
    return shape;
}

/// The shipped c4k320 MFMA row rebuilt as a single-wave binary baking K/groups = @p kpg,
/// with the transpose helper of that binding, as kernels of pack @p pack: at 1024 its
/// forward pass stages 72 KiB, past gfx942's 64 KiB (TestGfx942ConvDgradGeometry.cpp's
/// TheMfmaForwardPassStagesWithinGfx942sLds), at 768 it stages 54 KiB. No shipped instance
/// reaches the budget.
std::vector<KernelDefinition> wideMfmaPack(int64_t kpg, uint8_t pack)
{
    auto instance = variantEntry("grouped_dgrad_mfma", [](const catalog::Entry& e) {
        return e.integer("cpg") == 4 && e.integer("kpg") == 320;
    });
    instance.fields["kpg"] = kpg;
    instance.fields["waves_k"] = int64_t{1};
    instance.fields["waves_q"] = int64_t{1};
    instance.fields["block_q"] = int64_t{16};
    instance.fields["block_groups"] = int64_t{1};
    instance.fields["instance_id"] = "wide_mfma_k" + std::to_string(kpg);
    auto transpose = variantEntry("transpose", [](const catalog::Entry& e) {
        return e.integer("cpg") == 4 && e.integer("kpg") == 320;
    });
    transpose.fields["kpg"] = kpg;
    transpose.fields["instance_id"] = "wide_transpose_k" + std::to_string(kpg);
    std::vector<KernelDefinition> kernels{kernelOf(instance, pack), kernelOf(transpose, pack)};
    prepare(kernels);
    return kernels;
}

TEST(TestGfx942ConvDgradGraphMatcher, AcceptsBothStorageTypesOnGfx942AndBindsItsTokens)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        const auto fixture = fixtureOf(DgradShape{}, dtype);
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_DGRAD_AOT.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_DGRAD_AOT.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_DGRAD_AOT.outputToken), 3);
        const auto problem = bound->find(std::string(PROBLEM_TOKEN));
        ASSERT_NE(problem, bound->end());
        const auto* values = std::get_if<std::vector<int64_t>>(&problem->second);
        ASSERT_NE(values, nullptr);
        // dY, W, dX, then N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups, dtype.
        EXPECT_EQ(*values,
                  (std::vector<int64_t>{1,
                                        2,
                                        3,
                                        2,
                                        32,
                                        32,
                                        14,
                                        14,
                                        3,
                                        3,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        static_cast<int64_t>(dtype)}));
        // The problem token is private: only the tensor bindings are features.
        EXPECT_TRUE(detail::isPrivateToken(std::string(PROBLEM_TOKEN)));
        EXPECT_EQ(bound->size(), 4U);
    }
}

TEST(TestGfx942ConvDgradGraphMatcher, EachEngineMatchesItsOwnArchitectureOnly)
{
    registerNativeIngestorSymbols();
    // The gfx950 and gfx1250 engines decline a gfx942 device...
    const auto onGfx942 = fixtureOf(DgradShape{});
    for(const auto symbol : {GFX950_GRAPH_MATCHER, GFX1250_GRAPH_MATCHER})
    {
        const auto matcher = GraphMatchRegistry::tryResolve(std::string(symbol));
        ASSERT_NE(matcher, nullptr) << symbol;
        EXPECT_FALSE(matcher(onGfx942.context())) << symbol;
    }
    // ... and the gfx942 engine declines every other device, and a gfx942 one at wave32.
    for(const auto& properties : {deviceProperties("gfx950", 64),
                                  deviceProperties("gfx1250", 32),
                                  deviceProperties("gfx1250", 64),
                                  deviceProperties("gfx942", 32),
                                  deviceProperties("gfx940", 64),
                                  deviceProperties("gfx9420", 64),
                                  deviceProperties("gfx90a", 64),
                                  deviceProperties("", 64)})
    {
        const auto fixture = fixtureOf(DgradShape{}, data_objects::DataType::HALF, properties);
        EXPECT_FALSE(matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context()))
            << properties.gcnArchName << " wave" << properties.warpSize;
    }
    // A gfx942 variant name (MI300X reports its target features) still prefix-matches.
    const auto variant = fixtureOf(
        DgradShape{}, data_objects::DataType::HALF, deviceProperties("gfx942:sramecc+:xnack-"));
    EXPECT_TRUE(matchesGraph(GFX942_CONV_DGRAD_AOT, variant.context()));
}

TEST(TestGfx942ConvDgradGraphMatcher, RefusesWhatTheSharedGraphContractRefuses)
{
    // A sample of the contract TestGfx950ConvDgradMatchers.cpp pins case by case: the
    // gfx942 engine runs the same matcher.
    struct Case
    {
        const char* name;
        std::function<void(GraphSpec&)> change;
    };
    const std::vector<Case> cases{
        {"forward convolution node", [](auto& s) { s.forwardNode = true; }},
        {"half accumulation", [](auto& s) { s.computeType = data_objects::DataType::HALF; }},
        {"float tensors",
         [](auto& s) {
             for(auto& tensor : s.tensors)
             {
                 tensor.dtype = data_objects::DataType::FLOAT;
             }
         }},
        {"NCHW dX", [](auto& s) { s.tensors[2].strides = {6272, 196, 14, 1}; }},
        {"8-byte aligned dY", [](auto& s) { s.tensors[0].alignment = 8; }},
        {"asymmetric padding", [](auto& s) { s.postPadding = {1, 2}; }},
    };
    for(const auto& [name, change] : cases)
    {
        SCOPED_TRACE(name);
        auto spec = graphSpec(DgradShape{});
        change(spec);
        const GraphFixture fixture(buildGraph(spec), deviceProperties());
        EXPECT_FALSE(matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context()));
    }
}

TEST(TestGfx942ConvDgradKernelMatcher, EachFamilyServesItsOwnProblemsInItsOwnDtype)
{
    const auto& implicit = variantEntry("none", [](const catalog::Entry& e) {
        return e.text("pipeline") == "mem" && e.integer("dy_halo") == 0 && e.integer("patch") == 0
               && e.integer("filter_h") == 0 && e.integer("vector_size_a") == 8
               && e.integer("vector_size_b") == 8;
    });
    const auto& depthwise = variantEntry("depthwise_dgrad", [](const catalog::Entry& e) {
        return e.integer("filter_h") == 3 && e.integer("stride_h") == 1;
    });
    const auto& fused = variantEntry(
        "grouped_dgrad_fused", [](const catalog::Entry& e) { return e.integer("cpg") == 16; });
    const auto& mfma = variantEntry("grouped_dgrad_mfma", [](const catalog::Entry& e) {
        return e.integer("cpg") == 4 && e.integer("kpg") == 320;
    });
    auto kernels = std::vector<KernelDefinition>{
        kernelOf(implicit), kernelOf(depthwise), kernelOf(fused), kernelOf(mfma)};
    for(const auto* helper : helperEntries())
    {
        kernels.push_back(kernelOf(*helper));
    }
    prepare(kernels);
    const auto matches = [&](const DgradShape& shape,
                             const KernelDefinition& kernel,
                             data_objects::DataType dtype = data_objects::DataType::HALF) {
        const auto fixture = fixtureOf(shape, dtype);
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        EXPECT_TRUE(bound);
        return bound && matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernel, *bound);
    };
    const DgradShape dense;
    const auto grouped = groupedShape(16, 8);
    EXPECT_TRUE(matches(dense, kernels[0]));
    EXPECT_FALSE(matches(dense, kernels[1]));
    EXPECT_FALSE(matches(dense, kernels[2]));
    EXPECT_FALSE(matches(depthwiseShape(), kernels[0]));
    EXPECT_TRUE(matches(depthwiseShape(), kernels[1]));
    EXPECT_TRUE(matches(grouped, kernels[2]));
    EXPECT_FALSE(matches(grouped, kernels[1]));
    EXPECT_TRUE(matches(mfmaShape(), kernels[3]));
    EXPECT_FALSE(matches(grouped, kernels[3]));
    // The dtype is pinned: an fp16 kernel never serves a bf16 graph.
    EXPECT_FALSE(matches(dense, kernels[0], data_objects::DataType::BFLOAT16));
    EXPECT_FALSE(matches(grouped, kernels[2], data_objects::DataType::BFLOAT16));
    // The helpers are never admitted, whatever the graph.
    for(size_t i = 4; i < kernels.size(); ++i)
    {
        EXPECT_FALSE(matches(mfmaShape(), kernels[i])) << kernels[i].name;
        EXPECT_FALSE(matches(dense, kernels[i])) << kernels[i].name;
    }
}

TEST(TestGfx942ConvDgradKernelMatcher, AnMfmaInstanceNeedsItsHelpersInItsOwnPack)
{
    const auto fixture = fixtureOf(mfmaShape());
    const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& coalesced = variantEntry("grouped_dgrad_mfma", [](const catalog::Entry& e) {
        return e.integer("cpg") == 4 && e.integer("kpg") == 320 && e.integer("waves_k") > 1;
    });
    std::vector<KernelDefinition> lonely{kernelOf(coalesced, 107)};
    prepare(lonely);
    EXPECT_FALSE(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), lonely[0], *bound));
    // The other dtype's helpers in the same pack are not this instance's.
    std::vector<KernelDefinition> foreign{kernelOf(coalesced, 108)};
    for(const auto* helper : helperEntries("bf16"))
    {
        foreign.push_back(kernelOf(*helper, 108));
    }
    prepare(foreign);
    EXPECT_FALSE(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), foreign[0], *bound));
    const auto full = withHelpers(coalesced, 109);
    EXPECT_TRUE(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), full[0], *bound));
}

/// Every served-matrix row a graph can state, in its own dtype: the matcher admits exactly
/// the adapter's instances, scores exactly those, and neither the other dtype's catalog nor
/// gfx950's or gfx1250's (whose metadata does not parse for gfx942) admits any.
TEST(TestGfx942ConvDgradKernelMatcher, ServesExactlyWhatTheAdapterServesOverTheCatalog)
{
    const std::map<std::string, std::vector<KernelDefinition>> packs{
        {"fp16", catalogPack("fp16", 101)}, {"bf16", catalogPack("bf16", 102)}};
    const auto gfx950Pack = catalogPack("fp16", 103, dgrad::GFX950);
    const auto gfx1250Pack = catalogPack("fp16", 104, dgrad::GFX1250);
    size_t rows = 0;
    size_t served = 0;
    size_t expectedServed = 0;
    std::map<std::string, size_t> servedByVariant;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem);
        const auto geometry = dgrad::deriveGeometry(problem);
        if(!geometry)
        {
            // No graph states this problem; the graph matcher tests cover its refusal.
            EXPECT_EQ(std::string_view(row.served).find('1'), std::string_view::npos);
            continue;
        }
        // Graphs past 2^31 elements take seconds to build here; the geometry test covers them.
        if(geometry->tensorBytes[2] > dgrad::I32_MAX)
        {
            continue;
        }
        ++rows;
        expectedServed += static_cast<size_t>(
            std::count(row.served, row.served + std::string_view(row.served).size(), '1'));
        const std::string dtypeName = row.dtype;
        const auto shape = shapeOf(row.problem);
        const auto fixture = fixtureOf(shape, dtypeOf(dtypeName));
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound) << shape;
        const auto& own = packs.at(dtypeName);
        for(size_t i = 0; i < own.size(); ++i)
        {
            const bool expected = row.served[i] == '1';
            const bool actual
                = matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(actual, expected) << own[i].name << " on " << shape;
            // A served kernel scores, a refused one ranks last.
            const auto score
                = scoreKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(score >= 0.0, expected) << own[i].name << " score " << score;
            if(actual)
            {
                ++served;
                ++servedByVariant[std::get<std::string>(own[i].metadata.at("direct_variant"))];
            }
        }
        for(const auto& kernel : packs.at(dtypeName == "fp16" ? "bf16" : "fp16"))
        {
            EXPECT_FALSE(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernel, *bound))
                << kernel.name;
        }
        if(dtypeName == "fp16")
        {
            for(const auto* foreign : {&gfx950Pack, &gfx1250Pack})
            {
                for(const auto& kernel : *foreign)
                {
                    EXPECT_FALSE(
                        matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernel, *bound))
                        << kernel.name;
                }
            }
        }
    }
    // The served count is the adapter's over the same rows, so no floor needs pinning.
    EXPECT_GT(rows, 0U);
    EXPECT_GT(expectedServed, 0U);
    EXPECT_EQ(served, expectedServed);
    std::set<std::string> catalogVariants;
    for(const auto& entry : catalog::catalog("fp16", dgrad::GFX942))
    {
        if(!entry.isHelper())
        {
            catalogVariants.insert(entry.text("direct_variant"));
        }
    }
    for(const auto& variant : catalogVariants)
    {
        EXPECT_GT(servedByVariant[variant], 0U) << variant;
    }
}

TEST(TestGfx942ConvDgradKernelMatcher, ADepthwiseCplRowNeedsChannelsItDivides)
{
    const auto& entry = variantEntry("depthwise_dgrad", [](const catalog::Entry& e) {
        return e.integer("cpl") > 1 && e.integer("stride_h") == 1 && e.integer("filter_h") == 3;
    });
    auto kernels = std::vector<KernelDefinition>{kernelOf(entry, 115)};
    prepare(kernels);
    const auto cpl = entry.integer("cpl");
    for(const auto channels : {int64_t{96}, int64_t{97}})
    {
        const auto fixture = fixtureOf(depthwiseShape(channels));
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernels[0], *bound),
                  channels % cpl == 0)
            << channels << " channels at cpl " << cpl;
    }
}

TEST(TestGfx942ConvDgradScore, DirectPipelinesThenMfmaThenImplicitGemmAndRefusalsLast)
{
    auto kernels = catalogPack("fp16", 130);
    const auto& entries = catalog::catalog("fp16", dgrad::GFX942);
    // Per family (0 implicit GEMM, 1 a single-kernel direct pipeline, 2 an MFMA pipeline),
    // the (min, max) score of the kernels serving the shape.
    const auto scores = [&](const DgradShape& shape) {
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        EXPECT_TRUE(bound);
        std::map<int64_t, std::pair<double, double>> range;
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const auto score
                = scoreKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound);
            if(!matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound))
            {
                EXPECT_EQ(score, -1.0) << kernels[i].name;
                continue;
            }
            const auto& variant = entries[i].text("direct_variant");
            int64_t family = 1;
            if(variant == "none")
            {
                family = 0;
            }
            else if(variant.find("mfma") != std::string::npos)
            {
                family = 2;
            }
            auto [it, inserted] = range.emplace(family, std::make_pair(score, score));
            it->second.first = std::min(it->second.first, score);
            it->second.second = std::max(it->second.second, score);
        }
        return range;
    };
    const auto depthwise = scores(depthwiseShape());
    ASSERT_TRUE(depthwise.count(1));
    EXPECT_GE(depthwise.at(1).first, 3.0);
    const auto grouped = scores(groupedShape(16, 8));
    ASSERT_TRUE(grouped.count(0));
    ASSERT_TRUE(grouped.count(1));
    EXPECT_GT(grouped.at(1).first, grouped.at(0).second);
    EXPECT_GE(grouped.at(0).first, 0.0);
    EXPECT_LT(grouped.at(0).second, 1.01);
    const auto mfma = scores(mfmaShape());
    ASSERT_TRUE(mfma.count(2));
    ASSERT_TRUE(mfma.count(0));
    EXPECT_GE(mfma.at(2).first, 2.0);
    EXPECT_LT(mfma.at(2).second, 3.0);
    EXPECT_GT(mfma.at(2).first, mfma.at(0).second);
}

TEST(TestGfx942ConvDgradDispatch, WorkspaceAndPrepareAgreeWithThePlan)
{
    const auto& handler = dispatchHandler(GFX942_CONV_DGRAD_AOT);
    const auto reachesLoading =
        [&](const GraphFixture& fixture, const BoundTokens& bound, const KernelDefinition& kernel) {
            try
            {
                // The descriptor names no archive, so loading is the first step that can fail.
                handler.prepare(fixture.context(), bound, kernel);
                ADD_FAILURE() << "preparation loaded a kernel from an empty archive path";
            }
            catch(const std::exception& error)
            {
                const std::string message = error.what();
                EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
                EXPECT_EQ(message.find("weight-transform helper"), std::string::npos) << message;
            }
            // The expected load failure leaves its HIP error set on a device host.
            (void)hipGetLastError();
            (void)hipExtGetLastError();
        };
    const auto refusedTogether =
        [&](const GraphFixture& fixture, const BoundTokens& bound, const KernelDefinition& kernel) {
            EXPECT_FALSE(matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), kernel, bound));
            EXPECT_EQ(handler.workspaceBytes(fixture.context(), bound, kernel), 0U);
            EXPECT_THROW(handler.prepare(fixture.context(), bound, kernel),
                         hipdnn_plugin_sdk::HipdnnPluginException);
        };

    // An MFMA binary whose staging passes gfx942's 64 KiB, its helper present: its
    // workspace is 0 and preparation refuses it on the plan, before any code loads, exactly
    // as the matcher refuses it. The same binary at K/groups = 768 is served.
    {
        DgradShape wide = mfmaShape();
        wide.k = 1024;
        const auto kernels = wideMfmaPack(1024, 143);
        const auto fixture = fixtureOf(wide);
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        refusedTogether(fixture, *bound, kernels[0]);
        try
        {
            handler.prepare(fixture.context(), *bound, kernels[0]);
        }
        catch(const std::exception& error)
        {
            EXPECT_NE(std::string(error.what()).find("convolution contract"), std::string::npos)
                << error.what();
        }
        DgradShape fits = mfmaShape();
        fits.k = 768;
        const auto fitting = wideMfmaPack(768, 144);
        const auto fitsFixture = fixtureOf(fits);
        const auto fitsBound = matchesGraph(GFX942_CONV_DGRAD_AOT, fitsFixture.context());
        ASSERT_TRUE(fitsBound);
        EXPECT_TRUE(
            matchesKernel(GFX942_CONV_DGRAD_AOT, fitsFixture.context(), fitting[0], *fitsBound));
        EXPECT_GT(handler.workspaceBytes(fitsFixture.context(), *fitsBound, fitting[0]), 0U);
        reachesLoading(fitsFixture, *fitsBound, fitting[0]);
    }

    // A depthwise cpl row on channels it does not divide: refused together; on channels it
    // divides, no workspace and preparation reaches code loading.
    const auto& depthwise = variantEntry("depthwise_dgrad", [](const catalog::Entry& e) {
        return e.integer("cpl") > 1 && e.integer("stride_h") == 1 && e.integer("filter_h") == 3;
    });
    std::vector<KernelDefinition> depthwiseKernels{kernelOf(depthwise, 141)};
    prepare(depthwiseKernels);
    {
        const auto fixture = fixtureOf(depthwiseShape(97));
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        refusedTogether(fixture, *bound, depthwiseKernels[0]);
    }
    {
        const auto fixture = fixtureOf(depthwiseShape(96));
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        ASSERT_TRUE(
            matchesKernel(GFX942_CONV_DGRAD_AOT, fixture.context(), depthwiseKernels[0], *bound));
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, depthwiseKernels[0]), 0U);
        reachesLoading(fixture, *bound, depthwiseKernels[0]);
    }

    // An MFMA instance it serves: the workspace is the plan's (the transposed and the
    // reorganized weights), and grows with the groups.
    const auto& mfma = variantEntry("grouped_dgrad_mfma", [](const catalog::Entry& e) {
        return e.integer("cpg") == 4 && e.integer("kpg") == 320 && e.integer("waves_k") > 1;
    });
    const auto mfmaKernels = withHelpers(mfma, 142);
    auto grouped = mfmaShape();
    grouped.c *= 2;
    grouped.k *= 2;
    grouped.groups = 2;
    for(const auto& shape : {mfmaShape(), grouped})
    {
        SCOPED_TRACE(testing::Message() << shape);
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        const auto plan = dgrad::instancePlan(dgrad::Problem{shape.n,
                                                             shape.c,
                                                             shape.k,
                                                             shape.h,
                                                             shape.w,
                                                             shape.y,
                                                             shape.x,
                                                             shape.sH,
                                                             shape.sW,
                                                             shape.pH,
                                                             shape.pW,
                                                             shape.dH,
                                                             shape.dW,
                                                             shape.groups},
                                              *mfma.instance);
        ASSERT_TRUE(plan);
        EXPECT_GT(plan->workspaceBytes(), 0);
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, mfmaKernels[0]),
                  static_cast<size_t>(plan->workspaceBytes()));
        reachesLoading(fixture, *bound, mfmaKernels[0]);
    }
    // A helper is never prepared as a candidate.
    {
        const auto fixture = fixtureOf(mfmaShape());
        const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        refusedTogether(fixture, *bound, mfmaKernels[1]);
    }
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
namespace shipped = hip_kernel_provider::kernel_ingestor_engine::dgrad_shipped_catalog;

/// The shipped bundle's state manager, built once, or null when it is not staged.
const shipped::StateManager* shippedManager(std::vector<std::string>* knobs = nullptr)
{
    const auto& loaded = shipped::shippedCatalog("gfx942", GFX942_CONV_DGRAD_AOT.engineName);
    if(knobs != nullptr)
    {
        *knobs = loaded.knobs;
    }
    return loaded.manager.get();
}

const std::string NOT_STAGED = shipped::notStaged("gfx942", GFX942_CONV_DGRAD_AOT.engineName);

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

TEST(TestGfx942ConvDgradCatalog, EveryGraphsCandidatesHaveDistinctKnobTuples)
{
    std::vector<std::string> knobs;
    const auto* manager = shippedManager(&knobs);
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    ASSERT_FALSE(knobs.empty());
    // The depthwise lane ownership is a knob of the gfx942 engine.
    EXPECT_NE(std::find(knobs.begin(), knobs.end(), "cpl"), knobs.end());

    // The candidate set is what the real matchers, the shipped metadata (with its KMD
    // defaults) and the helper lookup admit, and its size is the adapter's count.
    size_t graphs = 0;
    forEachStatableRow([&](const auto& row) {
        const auto shape = shapeOf(row.problem);
        const auto expected = static_cast<size_t>(
            std::count(row.served, row.served + std::string_view(row.served).size(), '1'));
        const std::string dtypeName = row.dtype;
        const auto fixture = fixtureOf(shape, dtypeOf(dtypeName));
        const auto shipped = manager->unsortedCatalog(fixture.context());
        EXPECT_EQ(shipped.entries.size(), expected) << shape;
        std::set<std::vector<int64_t>> tuples;
        for(const auto& kernel : shipped.entries)
        {
            EXPECT_EQ(std::get<std::string>(kernel->metadata.at("dtype")), dtypeName);
            EXPECT_NE(std::get<int64_t>(kernel->metadata.at("kernel_family")), 2) << kernel->name;
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
    });
    EXPECT_GT(graphs, 0U);
}

TEST(TestGfx942ConvDgradCatalog, ShippedKernelsMatchThePacksSignaturesAndLoad)
{
    // One shipped kernel of each variant, pipeline and dtype, on the first served-matrix
    // problem the adapter serves it on, through prepare(): requireSignatureMatch compares
    // the ABI this pack marshals (names, kinds, sizes) with what hkp_pack recorded, before
    // any device is needed; with a gfx942 device the code objects then load, an MFMA
    // instance's transpose and reorganize helpers too.
    const auto* manager = shippedManager();
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    const auto device = currentDeviceProperties();
    const bool loadable = device.gcnArchName.rfind("gfx942", 0) == 0 && device.warpSize == 64;
    const auto& handler = dispatchHandler(GFX942_CONV_DGRAD_AOT);
    // Each dtype's kernels sit in their own shards, so both are checked.
    for(const std::string dtypeName : {"fp16", "bf16"})
    {
        const auto kernels = catalogPack(dtypeName, 101);
        std::set<std::pair<std::string, std::string>> selectable;
        for(const auto& kernel : kernels)
        {
            if(std::get<int64_t>(kernel.metadata.at("kernel_family")) != 2)
            {
                selectable.emplace(std::get<std::string>(kernel.metadata.at("direct_variant")),
                                   std::get<std::string>(kernel.metadata.at("pipeline")));
            }
        }
        // By (direct_variant, pipeline): implicit GEMM ships a mem, a compv3 and a compv4
        // shard.
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
        ASSERT_EQ(problemOf.size(), selectable.size())
            << dtypeName << ": every variant and pipeline is served somewhere";
        for(const auto& entry : problemOf)
        {
            const auto& variant = entry.first.first;
            const auto& pipeline = entry.first.second;
            SCOPED_TRACE(testing::Message() << dtypeName << " " << variant << " " << pipeline);
            const auto fixture = fixtureOf(
                shapeOf(entry.second), dtypeOf(dtypeName), loadable ? device : deviceProperties());
            const auto catalog = manager->unsortedCatalog(fixture.context());
            const auto kernel = std::find_if(
                catalog.entries.begin(), catalog.entries.end(), [&](const auto& candidate) {
                    return std::get<std::string>(candidate->metadata.at("direct_variant"))
                               == variant
                           && std::get<std::string>(candidate->metadata.at("pipeline")) == pipeline;
                });
            ASSERT_NE(kernel, catalog.entries.end());
            const auto bound = matchesGraph(GFX942_CONV_DGRAD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            shipped::expectPreparesOnItsDeviceOnly(
                handler,
                fixture.context(),
                *bound,
                **kernel,
                loadable,
                {"is packaged with arguments", "convolution contract", "weight-transform helper"});
        }
    }
}
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
