// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
#include "tests/engines/kernel_ingestor_engine/packs/ConvAotShippedCatalog.hpp"
#endif
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdAotTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestCatalog.hpp"

/**
 * @file TestGfx942ConvFwdMatchers.cpp
 * @brief hipkernel:Gfx942ConvFwdAot's native symbols, reached through the registry by the
 *        names its descriptors carry: the graph matcher on gfx942 (wave64) only, binding
 *        its own tokens; the kernel matcher and fallback score over the whole shipped
 *        catalog (Gfx942ConvFwdTestVectors.hpp), each instance serving graphs of its own
 *        dtype only; the group-merged depthwise instances; and the dispatch handler, whose
 *        workspace and preparation refuse together what the kernel matcher refuses. The
 *        engine shares its hooks with hipkernel:Gfx950ConvFwdAot (Gfx950ConvFwdNative.cpp);
 *        TestGfx950ConvFwdMatchers.cpp covers the graph and metadata contract they share.
 *        When this build packed the gfx942 production tree, the shipped catalog also runs
 *        through the real state manager: every graph's candidates have distinct knob
 *        tuples, the split-C packs alone name the split-C dispatch, and one shipped kernel
 *        of each variant, pipeline and launch count passes the pack's signature check (and
 *        loads, on a gfx942 device). Host-only otherwise.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::gfx942_fwd_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::conv_fwd_test_catalog;
using hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols;

constexpr const PackSymbols& GFX942_CONV_FWD_AOT = GFX942_FWD.pack;
/// Registered by the shared native code for split-C packs: the gfx942 direct_split packs
/// name it.
constexpr std::string_view SPLIT_C_DISPATCH = "hipkernel.gfx942_conv_fwd_aot.split_c_dispatch";

bool isSplitC(const catalog::Entry& entry)
{
    return entry.integer("split_c") > 1;
}

/// The handler @p entry's pack names: the split-C one (two launches) for a split-C kernel,
/// the single-launch one otherwise.
const IKernelDispatchHandler<Handle>& handlerOf(const catalog::Entry& entry)
{
    registerNativeIngestorSymbols();
    if(isSplitC(entry))
    {
        return *DispatchRegistry<Handle>::resolve(std::string(SPLIT_C_DISPATCH));
    }
    return dispatchHandler(GFX942_CONV_FWD_AOT);
}

GraphFixture fixtureOf(const ConvShape& shape,
                       data_objects::DataType dtype = data_objects::DataType::HALF,
                       DeviceProperties properties = deviceProperties("gfx942"))
{
    return GraphFixture(buildGraph(convSpec(shape, dtype), "gfx942_conv_fwd_test"),
                        std::move(properties));
}

GraphFixture rowFixture(const vectors::ServedRow& row)
{
    return GraphFixture(convGraph(row.problem, dataTypeOf(row.dtype), "gfx942_conv_fwd_test"),
                        deviceProperties("gfx942"));
}

/// The shipped catalog of @p dtype as the engine loads it: each instance's kernel
/// definition, prepared by the kernel matcher's KernelPrepareFn, in catalog order.
const std::vector<KernelDefinition>& catalogKernels(const std::string& dtype)
{
    static const auto s_kernels = [] {
        registerNativeIngestorSymbols();
        const auto prepare
            = KernelPrepareRegistry::resolve(std::string(GFX942_CONV_FWD_AOT.kernelMatcher));
        std::map<std::string, std::vector<KernelDefinition>> byDtype;
        for(const auto* name : {"fp16", "bf16"})
        {
            for(const auto& entry : catalog::gfx942Catalog(name))
            {
                auto kernel = catalog::kernelOf(entry);
                kernel.prepared = prepare(kernel);
                kernel.preparedBy = prepare;
                byDtype[name].push_back(std::move(kernel));
            }
        }
        return byDtype;
    }();
    return s_kernels.at(dtype);
}

/// An entry's kind (implicit GEMM or its direct variant, atom, merge degree, split-C)
/// without its id.
std::string kindOf(const catalog::Entry& entry)
{
    const auto kind = entry.kind();
    return kind.substr(0, kind.rfind(' ')) + (isSplitC(entry) ? " split" : "");
}

/// The first catalog instance of @p dtype that @p pick accepts, as its kernel.
KernelDefinition catalogKernel(const std::function<bool(const catalog::Entry&)>& pick,
                               const std::string& dtype = "fp16")
{
    const auto& entries = catalog::gfx942Catalog(dtype);
    const auto found = std::find_if(entries.begin(), entries.end(), pick);
    if(found == entries.end())
    {
        ADD_FAILURE() << "the gfx942 " << dtype << " catalog holds no such instance";
        return {};
    }
    return catalogKernels(dtype)[static_cast<size_t>(found - entries.begin())];
}

KernelDefinition mergedKernel(int64_t groupMerge, const std::string& dtype = "fp16")
{
    return catalogKernel(
        [&](const catalog::Entry& e) {
            return e.integer("kernel_family") == 0 && e.integer("group_merge") == groupMerge;
        },
        dtype);
}

const std::vector<int64_t>* problemToken(const BoundTokens& bound, std::string_view token)
{
    const auto it = bound.find(std::string(token));
    return it == bound.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
}

/// Prepares @p kernel and expects the step after the contract checks, code loading, to be
/// what fails: the catalog's kernels name no archive here.
void expectReachesCodeLoading(const IKernelDispatchHandler<Handle>& handler,
                              const MatchContext& context,
                              const BoundTokens& bound,
                              const KernelDefinition& kernel)
{
    try
    {
        handler.prepare(context, bound, kernel);
        ADD_FAILURE() << "preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("tensor bindings"), std::string::npos) << message;
        EXPECT_EQ(message.find("packed NHWK output"), std::string::npos) << message;
    }
    // The expected load failure leaves its HIP error set on a device host.
    (void)hipGetLastError();
    (void)hipExtGetLastError();
}

/// Expects @p handler to refuse @p kernel on the graph in both of its queries: no
/// workspace, and preparation throwing BAD_PARAM on the convolution contract before any
/// code loads.
void expectRefusedTogether(const IKernelDispatchHandler<Handle>& handler,
                           const MatchContext& context,
                           const BoundTokens& bound,
                           const KernelDefinition& kernel)
{
    EXPECT_EQ(handler.workspaceBytes(context, bound, kernel), 0U);
    try
    {
        handler.prepare(context, bound, kernel);
        ADD_FAILURE() << "preparation accepted what the plan refuses";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        EXPECT_NE(std::string(error.what())
                      .find("graph or packaged kernel does not satisfy the convolution contract"),
                  std::string::npos)
            << error.what();
    }
    catch(const std::exception& error)
    {
        ADD_FAILURE() << "refused with something other than the contract: " << error.what();
    }
}

TEST(TestGfx942ConvFwdGraphMatcher, BindsItsOwnTokensOnlyOnAGfx942Device)
{
    registerNativeIngestorSymbols();
    const auto depthwise = depthwiseShape(64, 14, 14, 3, 1, 1);
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        for(const auto* arch : {"gfx942", "gfx942:sramecc+:xnack-"})
        {
            SCOPED_TRACE(arch);
            const auto fixture = fixtureOf(depthwise, dtype, deviceProperties(arch));
            const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            // Its own tokens and only those: the public ones are UHD query features.
            EXPECT_EQ(bound->size(), 4U);
            EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.inputAToken), 1);
            EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.inputBToken), 2);
            EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.outputToken), 3);
            EXPECT_EQ(bound->count(std::string(GFX950_FWD.pack.inputAToken)), 0U);
            EXPECT_EQ(bound->count(std::string(GFX950_FWD.problemToken)), 0U);
            const auto* problem = problemToken(*bound, GFX942_FWD.problemToken);
            ASSERT_NE(problem, nullptr);
            // x, w, y, then N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups, dtype.
            EXPECT_EQ(*problem,
                      (std::vector<int64_t>{1,
                                            2,
                                            3,
                                            2,
                                            64,
                                            64,
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
                                            64,
                                            static_cast<int64_t>(dtype)}));
            // The problem token is private: only the tensor bindings are features.
            EXPECT_TRUE(detail::isPrivateToken(std::string(GFX942_FWD.problemToken)));

            // The gfx950 engine binds the same problem under its own token on its device.
            const auto onGfx950 = fixtureOf(depthwise, dtype, deviceProperties("gfx950"));
            const auto gfx950Bound = matchesGraph(GFX950_FWD.pack, onGfx950.context());
            ASSERT_TRUE(gfx950Bound);
            const auto* gfx950Problem = problemToken(*gfx950Bound, GFX950_FWD.problemToken);
            ASSERT_NE(gfx950Problem, nullptr);
            EXPECT_EQ(*problem, *gfx950Problem);
        }
    }
}

TEST(TestGfx942ConvFwdGraphMatcher, EachEngineMatchesItsOwnArchitectureOnly)
{
    registerNativeIngestorSymbols();
    const ConvShape dense = denseShape(32, 32, 14, 14);
    // The gfx950 engine declines a gfx942 device...
    const auto onGfx942 = fixtureOf(dense);
    EXPECT_FALSE(matchesGraph(GFX950_FWD.pack, onGfx942.context()));
    ASSERT_TRUE(matchesGraph(GFX942_CONV_FWD_AOT, onGfx942.context()));
    // ... and the gfx942 engine every other device, and a gfx942 one at wave32.
    for(const auto& properties : {deviceProperties("gfx950"),
                                  deviceProperties("gfx9420"),
                                  deviceProperties("gfx94"),
                                  deviceProperties("gfx940"),
                                  deviceProperties("gfx90a"),
                                  deviceProperties("gfx1250", 32),
                                  deviceProperties(""),
                                  deviceProperties("prefix-gfx942"),
                                  deviceProperties("gfx942", 32)})
    {
        const auto fixture = fixtureOf(dense, data_objects::DataType::HALF, properties);
        EXPECT_FALSE(matchesGraph(GFX942_CONV_FWD_AOT, fixture.context()))
            << "'" << properties.gcnArchName << "' wave" << properties.warpSize;
    }
}

TEST(TestGfx942ConvFwdKernelMatcher, EachEngineServesItsOwnDeviceOnly)
{
    // An instance both engines' metadata describe serves the graph through either engine's
    // kernel matcher on that engine's device, and through neither on the other's: the
    // kernel matcher reads the graph again when the bound tokens are another engine's.
    const auto kernel = aotKernel({1, 1, 1});
    const ConvShape dense = denseShape(32, 32, 14, 14);
    for(const auto& [own, other] :
        {std::pair{GFX942_FWD, GFX950_FWD}, std::pair{GFX950_FWD, GFX942_FWD}})
    {
        SCOPED_TRACE(std::string(own.arch));
        const auto onOwn = fixtureOf(
            dense, data_objects::DataType::HALF, deviceProperties(std::string(own.arch)));
        const auto ownBound = matchesGraph(own.pack, onOwn.context());
        ASSERT_TRUE(ownBound);
        EXPECT_TRUE(matchesKernel(own.pack, onOwn.context(), kernel, *ownBound));
        EXPECT_GE(scoreKernel(own.pack, onOwn.context(), kernel, *ownBound), 0.0);

        const auto onOther = fixtureOf(
            dense, data_objects::DataType::HALF, deviceProperties(std::string(other.arch)));
        const auto otherBound = matchesGraph(other.pack, onOther.context());
        ASSERT_TRUE(otherBound);
        EXPECT_FALSE(matchesKernel(own.pack, onOther.context(), kernel, *otherBound));
        EXPECT_FALSE(matchesKernel(own.pack, onOther.context(), kernel));
        EXPECT_EQ(scoreKernel(own.pack, onOther.context(), kernel, *otherBound), -1.0);
        // Its dispatch refuses the other device too.
        expectRefusedTogether(dispatchHandler(own.pack), onOther.context(), *otherBound, kernel);
    }
}

TEST(TestGfx942ConvFwdKernelMatcher, RefusesAWindowedDepthwiseKernelTheGfx950EngineServes)
{
    // direct_depthwise_win needs gfx950's Toeplitz MFMA form: the gfx942 catalog ships
    // none, and a win kernel in a gfx942 pack is refused by the matcher, the score and
    // the dispatch alike, while the gfx950 engine serves the same kernel.
    auto win = sampleKernel({"direct", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "direct", {0, 0, 0, 0}});
    win.metadata["grouped"] = int64_t{0};
    win.metadata["kernel_family"] = int64_t{1};
    win.metadata["direct_variant"] = std::string("win");
    win.metadata["filter_h"] = win.metadata["filter_w"] = int64_t{3};
    win.metadata["stride_h"] = win.metadata["stride_w"] = int64_t{1};
    win.metadata["pad_h"] = win.metadata["pad_w"] = int64_t{1};
    win.metadata["dilation_h"] = win.metadata["dilation_w"] = int64_t{1};
    win.metadata["block_h"] = int64_t{12};
    win.metadata["block_waves"] = int64_t{8};
    win.metadata["cpg"] = int64_t{1};
    win.metadata["kpg"] = int64_t{1};
    win.metadata["prefetch_rows"] = int64_t{2};
    win.metadata["w_fold"] = int64_t{2};
    const ConvShape depthwise = depthwiseShape(32, 14, 14, 3, 1, 1);

    const auto onGfx950
        = fixtureOf(depthwise, data_objects::DataType::HALF, deviceProperties("gfx950"));
    const auto gfx950Bound = matchesGraph(GFX950_FWD.pack, onGfx950.context());
    ASSERT_TRUE(gfx950Bound);
    EXPECT_TRUE(matchesKernel(GFX950_FWD.pack, onGfx950.context(), win, *gfx950Bound));

    const auto onGfx942
        = fixtureOf(depthwise, data_objects::DataType::HALF, deviceProperties("gfx942"));
    const auto gfx942Bound = matchesGraph(GFX942_FWD.pack, onGfx942.context());
    ASSERT_TRUE(gfx942Bound);
    EXPECT_FALSE(matchesKernel(GFX942_FWD.pack, onGfx942.context(), win, *gfx942Bound));
    EXPECT_FALSE(matchesKernel(GFX942_FWD.pack, onGfx942.context(), win));
    EXPECT_EQ(scoreKernel(GFX942_FWD.pack, onGfx942.context(), win, *gfx942Bound), -1.0);
    expectRefusedTogether(dispatchHandler(GFX942_FWD.pack), onGfx942.context(), *gfx942Bound, win);
}

TEST(TestGfx942ConvFwdKernelMatcher, EveryShippedInstanceServesGraphsOfItsOwnDtypeOnly)
{
    // Over every golden problem, the matcher already admits exactly the adapter's
    // instances of the graph's dtype (TestGfx942ConvFwdGeometry): here no instance of the
    // other dtype is admitted, and the fallback score ranks exactly the admitted ones.
    std::map<std::string, std::map<std::string, size_t>> servedByKind;
    size_t rows = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const std::string dtype = row.dtype;
        const std::string other = dtype == "fp16" ? "bf16" : "fp16";
        const auto fixture = rowFixture(row);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        if(!bound)
        {
            // No graph states this problem: the adapter serves it nothing either.
            EXPECT_EQ(std::string_view(row.served).find_first_not_of('0'), std::string_view::npos)
                << catalog::describe(catalog::toProblem(row.problem));
            continue;
        }
        ++rows;
        const auto& own = catalogKernels(dtype);
        for(size_t i = 0; i < own.size(); ++i)
        {
            const bool served = catalog::bitAt(row.served, i);
            const auto score = scoreKernel(GFX942_CONV_FWD_AOT, fixture.context(), own[i], *bound);
            // A refusal ranks last (-1); the grouped and non-grouped direct kernels take a
            // placeholder (-0.5) above it.
            EXPECT_EQ(score > -1.0, served) << own[i].name << " score " << score;
            if(served)
            {
                ++servedByKind[dtype][kindOf(catalog::gfx942Catalog(dtype)[i])];
            }
        }
        for(const auto& kernel : catalogKernels(other))
        {
            EXPECT_FALSE(matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel, *bound))
                << kernel.name << " (" << other << ") on "
                << catalog::describe(catalog::toProblem(row.problem)) << " " << dtype;
        }
    }
    EXPECT_GT(rows, 0U);
    // Each dtype serves every kind of its catalog somewhere.
    for(const auto* dtype : {"fp16", "bf16"})
    {
        std::set<std::string> kinds;
        for(const auto& entry : catalog::gfx942Catalog(dtype))
        {
            kinds.insert(kindOf(entry));
        }
        ASSERT_FALSE(kinds.empty());
        for(const auto& kind : kinds)
        {
            EXPECT_GT(servedByKind[dtype][kind], 0U) << dtype << " " << kind;
        }
    }
}

TEST(TestGfx942ConvFwdKernelMatcher, MergedKernelServesOnlyDepthwiseGraphsItsMergeDegreeDivides)
{
    const auto serves = [](const ConvShape& shape, const KernelDefinition& kernel) {
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        const bool served = matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel);
        // The bound problem decides as the graph does.
        EXPECT_EQ(bound && matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel, *bound),
                  served);
        return served;
    };
    const auto kpg2 = [] {
        auto shape = groupedShape(64, 1);
        shape.k = 128;
        return shape;
    }();
    const std::vector<std::pair<std::string, ConvShape>> graphs{
        {"depthwise G64", depthwiseShape(64, 14, 14, 3, 1, 1)},
        {"depthwise G48 strided 5x5", depthwiseShape(48, 9, 9, 5, 2, 2)},
        {"depthwise G32", depthwiseShape(32, 14, 14, 3, 1, 1)},
        {"depthwise G24", depthwiseShape(24, 14, 14, 3, 1, 1)},
        {"depthwise G20", depthwiseShape(20, 14, 14, 3, 1, 1)},
        {"depthwise G97", depthwiseShape(97, 14, 14, 3, 1, 1)},
        {"two channels per group", groupedShape(64, 2)},
        {"two outputs per group", kpg2},
        {"dense", denseShape(64, 64, 14, 14)}};
    // Every merge degree the catalog ships.
    std::set<int64_t> degrees;
    for(const auto& entry : catalog::gfx942Catalog("fp16"))
    {
        if(entry.integer("kernel_family") == 0 && entry.integer("group_merge") > 1)
        {
            degrees.insert(entry.integer("group_merge"));
        }
    }
    ASSERT_FALSE(degrees.empty());
    for(const auto degree : degrees)
    {
        const auto merged = mergedKernel(degree);
        for(const auto& [name, shape] : graphs)
        {
            SCOPED_TRACE(name + ", merge " + std::to_string(degree));
            const bool depthwise = shape.c == shape.groups && shape.k == shape.groups;
            EXPECT_EQ(serves(shape, merged), depthwise && shape.groups % degree == 0);
            // Not by the gfx942 engine on another device.
            const auto onGfx950
                = fixtureOf(shape, data_objects::DataType::HALF, deviceProperties("gfx950"));
            EXPECT_FALSE(matchesKernel(GFX942_CONV_FWD_AOT, onGfx950.context(), merged));
        }
    }
}

TEST(TestGfx942ConvFwdKernelMatcher, GroupMergeMetadataDecodesAsTheCatalogDeclaresIt)
{
    const auto fixture = fixtureOf(depthwiseShape(32, 14, 14, 3, 1, 1));
    const auto serves = [&](const KernelDefinition& kernel) {
        return matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel);
    };
    // Copies of a catalog kernel carry its prepared payload, which would answer for the
    // changed metadata: decode them afresh.
    const auto plain = [](KernelDefinition kernel) {
        kernel.prepared = nullptr;
        kernel.preparedBy = nullptr;
        return kernel;
    };
    const auto merged = plain(mergedKernel(16));
    ASSERT_TRUE(serves(merged));
    // A merged identity records its derived widths as 0, never a width.
    for(const auto* field : {"vector_size_a", "vector_size_b", "vector_size_c"})
    {
        SCOPED_TRACE(field);
        auto changed = merged;
        changed.metadata[field] = int64_t{1};
        EXPECT_FALSE(serves(changed));
    }
    // A merge degree is positive and stated.
    for(const auto groupMerge : {int64_t{0}, int64_t{-16}, int64_t{1}})
    {
        SCOPED_TRACE(groupMerge);
        auto changed = merged;
        changed.metadata["group_merge"] = groupMerge;
        EXPECT_FALSE(serves(changed));
    }
    auto absent = merged;
    absent.metadata.erase("group_merge");
    EXPECT_FALSE(serves(absent));

    // An unmerged identity serves with group_merge 1, and its widths refuse a merge
    // degree. The catalog has no scalar-output unmerged instance to serve the depthwise
    // graph, so a vector one on a dense graph.
    const auto dense = fixtureOf(denseShape(64, 64, 14, 14));
    auto unmerged = plain(catalogKernel([](const catalog::Entry& e) {
        return e.integer("kernel_family") == 0 && e.integer("group_merge") == 1
               && e.integer("vector_size_a") == 8 && e.integer("vector_size_c") == 8;
    }));
    ASSERT_TRUE(matchesKernel(GFX942_CONV_FWD_AOT, dense.context(), unmerged));
    unmerged.metadata["group_merge"] = int64_t{16};
    EXPECT_FALSE(matchesKernel(GFX942_CONV_FWD_AOT, dense.context(), unmerged));

    // A direct depthwise identity has no group merge.
    auto direct = plain(catalogKernel([](const catalog::Entry& e) {
        return e.integer("kernel_family") == 1 && e.text("direct_variant") == "std"
               && e.integer("filter_h") == 3 && e.integer("stride_h") == 1;
    }));
    ASSERT_TRUE(serves(direct));
    direct.metadata["group_merge"] = int64_t{16};
    EXPECT_FALSE(serves(direct));
}

TEST(TestGfx942ConvFwdDispatch, WorkspaceAndPrepareRefuseTogether)
{
    const auto& handler = dispatchHandler(GFX942_CONV_FWD_AOT);
    const auto check = [&](const ConvShape& shape,
                           const KernelDefinition& kernel,
                           bool served,
                           data_objects::DataType dtype = data_objects::DataType::HALF) {
        const auto fixture = fixtureOf(shape, dtype);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        ASSERT_EQ(matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel, *bound), served);
        if(served)
        {
            // None of these kernels is split-C, so none needs scratch.
            EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernel), 0U);
            expectReachesCodeLoading(handler, fixture.context(), *bound, kernel);
        }
        else
        {
            expectRefusedTogether(handler, fixture.context(), *bound, kernel);
        }
    };
    const auto merged = mergedKernel(16);
    const auto wide = catalogKernel([](const catalog::Entry& e) {
        return e.integer("kernel_family") == 0 && e.integer("group_merge") == 1
               && e.integer("vector_size_a") == 8 && e.integer("vector_size_c") == 8;
    });
    const auto nongrouped = catalogKernel([](const catalog::Entry& e) {
        return e.text("direct_variant") == "nongrouped" && e.integer("stride_h") == 1
               && e.integer("dilation_h") == 1 && !isSplitC(e);
    });
    {
        SCOPED_TRACE("merged");
        check(depthwiseShape(32, 14, 14, 3, 1, 1), merged, true);
        // A graph that is not depthwise, and a group count the degree does not divide.
        check(groupedShape(32, 2), merged, false);
        check(depthwiseShape(24, 14, 14, 3, 1, 1), merged, false);
    }
    {
        SCOPED_TRACE("vector widths");
        check(denseShape(64, 64, 14, 14), wide, true);
        // Widths that do not divide the per-group channel runs.
        check(denseShape(12, 64, 14, 14), wide, false);
        check(denseShape(64, 20, 14, 14), wide, false);
    }
    {
        SCOPED_TRACE("dtype");
        // An fp16 kernel on the same graph in bf16.
        check(denseShape(64, 64, 14, 14), wide, false, data_objects::DataType::BFLOAT16);
        check(denseShape(64, 64, 14, 14),
              catalogKernel(
                  [](const catalog::Entry& e) {
                      return e.integer("kernel_family") == 0 && e.integer("group_merge") == 1
                             && e.integer("vector_size_a") == 8 && e.integer("vector_size_c") == 8;
                  },
                  "bf16"),
              true,
              data_objects::DataType::BFLOAT16);
    }
    {
        SCOPED_TRACE("nongrouped");
        // A dense graph it tiles, and one with a filter it was not compiled for.
        const auto filter = std::get<int64_t>(nongrouped.metadata.at("filter_h"));
        auto shape = denseShape(64, 64, 8, 64);
        shape.y = shape.x = filter;
        shape.pH = shape.pW = std::get<int64_t>(nongrouped.metadata.at("pad_h"));
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        if(matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), nongrouped, *bound))
        {
            check(shape, nongrouped, true);
        }
        auto other = shape;
        other.y = other.x = filter + 2;
        check(other, nongrouped, false);
    }
}

TEST(TestGfx942ConvFwdDispatch, WorkspaceAndPrepareRefuseTogetherOverTheCatalog)
{
    // On every golden problem, one instance of each kind the adapter refuses and one it
    // serves, through the handler its pack names: what the plan refuses, workspaceBytes()
    // and prepare() refuse together (workspace 0, BAD_PARAM before any code loads); what
    // it serves needs workspace only when split-C, and reaches code loading.
    size_t refused = 0;
    size_t served = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto fixture = rowFixture(row);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        if(!bound)
        {
            continue; // EveryShippedInstanceServesGraphsOfItsOwnDtypeOnly checks these rows.
        }
        SCOPED_TRACE(catalog::describe(catalog::toProblem(row.problem)) + " "
                     + std::string(row.dtype));
        const auto& entries = catalog::gfx942Catalog(row.dtype);
        const auto& kernels = catalogKernels(row.dtype);
        std::set<std::pair<std::string, bool>> checked;
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const bool isServed = catalog::bitAt(row.served, i);
            if(!checked.emplace(kindOf(entries[i]), isServed).second)
            {
                continue;
            }
            SCOPED_TRACE(entries[i].kind());
            const auto& handler = handlerOf(entries[i]);
            if(isServed)
            {
                ++served;
                EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernels[i]) > 0,
                          isSplitC(entries[i]));
                expectReachesCodeLoading(handler, fixture.context(), *bound, kernels[i]);
            }
            else
            {
                ++refused;
                expectRefusedTogether(handler, fixture.context(), *bound, kernels[i]);
            }
        }
    }
    EXPECT_GT(refused, 0U);
    EXPECT_GT(served, 0U);
}

TEST(TestGfx942ConvFwdDispatch, ChecksItsOwnBindingsAndReachesCodeLoading)
{
    const auto fixture = fixtureOf(depthwiseShape(64, 14, 14, 3, 1, 1));
    const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX942_CONV_FWD_AOT);
    const auto merged = mergedKernel(16);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, merged), 0U);

    // The gfx950 engine's tokens are not this engine's bindings.
    const BoundTokens gfx950Bound{{std::string(GFX950_FWD.pack.inputAToken), int64_t{1}},
                                  {std::string(GFX950_FWD.pack.inputBToken), int64_t{2}},
                                  {std::string(GFX950_FWD.pack.outputToken), int64_t{3}}};
    for(const auto& wrong : {BoundTokens{}, gfx950Bound})
    {
        try
        {
            handler.prepare(fixture.context(), wrong, merged);
            ADD_FAILURE() << "preparation accepted bindings another graph matcher made";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_NE(std::string(error.what())
                          .find("gfx942_conv_fwd dispatch requires this graph matcher's tensor "
                                "bindings"),
                      std::string::npos)
                << error.what();
        }
    }
    expectReachesCodeLoading(handler, fixture.context(), *bound, merged);
}

TEST(TestGfx942ConvFwdDispatch, EachHandlerServesOnlyItsOwnPacksKernels)
{
    // The shared native code registers a split-C dispatch for every forward engine, which
    // the gfx942 direct_split packs name. On a served pair, the handler the kernel's pack
    // does not name refuses it together: the split-C one a single-launch kernel, the
    // single-launch one a split-C kernel.
    registerNativeIngestorSymbols();
    const auto* splitC = DispatchRegistry<Handle>::resolve(std::string(SPLIT_C_DISPATCH));
    ASSERT_NE(splitC, nullptr);
    const auto& single = dispatchHandler(GFX942_CONV_FWD_AOT);
    EXPECT_NE(splitC, &single);
    EXPECT_EQ(splitC->launchNames(), (std::set<std::string>{"conv", "reduce"}));
    EXPECT_TRUE(single.launchNames().empty());
    std::map<bool, size_t> refused;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto fixture = rowFixture(row);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        const auto& entries = catalog::gfx942Catalog(row.dtype);
        const auto& kernels = catalogKernels(row.dtype);
        std::set<bool> checked;
        for(size_t i = 0; bound && i < kernels.size(); ++i)
        {
            const bool split = isSplitC(entries[i]);
            if(catalog::bitAt(row.served, i) && checked.insert(split).second)
            {
                SCOPED_TRACE(entries[i].kind());
                expectRefusedTogether(
                    split ? single : *splitC, fixture.context(), *bound, kernels[i]);
                ++refused[split];
            }
        }
    }
    EXPECT_GT(refused[false], 0U);
    EXPECT_GT(refused[true], 0U);
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
namespace shipped = hip_kernel_provider::kernel_ingestor_engine::conv_aot_shipped_catalog;

/// The shipped bundle, loaded once (a null manager when it is not staged).
const shipped::ShippedCatalog& shippedBundle()
{
    return shipped::shippedCatalog("gfx942", GFX942_CONV_FWD_AOT.engineName, "gfx942_conv_fwd_aot");
}

const std::string NOT_STAGED = shipped::notStaged("gfx942", GFX942_CONV_FWD_AOT.engineName);

TEST(TestGfx942ConvFwdCatalog, EveryGraphsCandidatesHaveDistinctKnobTuples)
{
    const auto& bundle = shippedBundle();
    const auto* manager = bundle.manager.get();
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    const auto& knobs = bundle.knobs;
    ASSERT_FALSE(knobs.empty());
    // The catalog has instances that differ only by these, so a ranker needs them as knobs.
    for(const auto* knob : {"waves_per_eu", "group_merge", "atom", "kernel_family"})
    {
        EXPECT_NE(std::find(knobs.begin(), knobs.end(), knob), knobs.end()) << knob;
    }
    // The candidate set is what the real matchers, the shipped metadata (with its KMD
    // defaults) and the candidate filter keep: instance by instance the adapter's selection,
    // which ties the vectors' catalog (from the instance lists) to the shipped bundle.
    size_t graphs = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto& dtypeEntries = catalog::gfx942Catalog(row.dtype);
        std::set<std::string> selected;
        for(size_t i = 0; i < dtypeEntries.size(); ++i)
        {
            if(catalog::bitAt(row.selected, i))
            {
                selected.insert(dtypeEntries[i].text("instance_id"));
            }
        }
        const auto fixture = rowFixture(row);
        const auto candidates = manager->unsortedCatalog(fixture.context());
        const auto where = catalog::describe(catalog::toProblem(row.problem)) + " " + row.dtype;
        EXPECT_EQ(candidates.entries.size(), selected.size()) << where;
        std::set<std::string> names;
        std::set<std::vector<int64_t>> tuples;
        for(const auto& kernel : candidates.entries)
        {
            EXPECT_EQ(std::get<std::string>(kernel->metadata.at("dtype")), row.dtype);
            names.insert(std::get<std::string>(kernel->metadata.at("instance_id")));
            std::vector<int64_t> tuple;
            for(const auto& knob : knobs)
            {
                const auto ordinal = manager->knobOrdinal(knob, kernel->metadata.at(knob));
                ASSERT_TRUE(ordinal) << knob;
                tuple.push_back(*ordinal);
            }
            EXPECT_TRUE(tuples.insert(tuple).second)
                << kernel->name << " repeats another candidate's knob tuple on " << where;
        }
        EXPECT_EQ(names, selected) << where;
        graphs += candidates.entries.empty() ? 0 : 1;
    }
    EXPECT_GT(graphs, 0U);
}

TEST(TestGfx942ConvFwdCatalog, EveryPackDispatchesThroughOneOfTheEnginesHandlers)
{
    const auto& bundle = shippedBundle();
    if(bundle.manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    // The single-launch handler, and the split-C one the direct_split packs name.
    const std::set<std::string> symbols(bundle.dispatchSymbols.begin(),
                                        bundle.dispatchSymbols.end());
    EXPECT_EQ(symbols,
              (std::set<std::string>{std::string(GFX942_CONV_FWD_AOT.dispatch),
                                     std::string(SPLIT_C_DISPATCH)}));
}

TEST(TestGfx942ConvFwdCatalog, ShippedKernelsMatchThePacksSignaturesAndLoad)
{
    // One shipped kernel of each variant, pipeline and launch count per dtype, on a golden
    // problem the engine selects it for, through the prepare() of the handler its pack
    // names: requireSignatureMatch compares the ABI this
    // pack marshals (names, kinds, sizes) with what hkp_pack recorded, before any device
    // is needed; with a wave64 gfx942 device the code object then loads.
    const auto* manager = shippedBundle().manager.get();
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    const auto device = currentDeviceProperties();
    const bool loadable = device.gcnArchName.rfind("gfx942", 0) == 0 && device.warpSize == 64;
    registerNativeIngestorSymbols();
    const auto& single = dispatchHandler(GFX942_CONV_FWD_AOT);
    const auto* splitC = DispatchRegistry<Handle>::resolve(std::string(SPLIT_C_DISPATCH));
    ASSERT_NE(splitC, nullptr);
    const auto isSplit = [](const KernelDefinition& kernel) {
        return std::get<int64_t>(kernel.metadata.at("split_c")) > 1;
    };
    const auto keyOf = [&](const KernelDefinition& kernel) {
        return std::make_tuple(std::get<std::string>(kernel.metadata.at("direct_variant")),
                               std::get<std::string>(kernel.metadata.at("pipeline")),
                               isSplit(kernel));
    };
    using Key = decltype(keyOf(std::declval<const KernelDefinition&>()));
    for(const std::string dtype : {"fp16", "bf16"})
    {
        // Every (variant, pipeline, split-C) the adapter selects somewhere.
        std::set<Key> expected;
        for(const auto& row : vectors::servedMatrix())
        {
            for(size_t i = 0; row.dtype == dtype && i < catalogKernels(dtype).size(); ++i)
            {
                if(catalog::bitAt(row.selected, i))
                {
                    expected.insert(keyOf(catalogKernels(dtype)[i]));
                }
            }
        }
        ASSERT_FALSE(expected.empty()) << dtype;
        std::set<Key> prepared;
        for(const auto& row : vectors::servedMatrix())
        {
            if(row.dtype != dtype)
            {
                continue;
            }
            const auto fixture
                = GraphFixture(convGraph(row.problem, dataTypeOf(dtype), "gfx942_conv_fwd_test"),
                               loadable ? device : deviceProperties("gfx942"));
            const auto candidates = manager->unsortedCatalog(fixture.context());
            const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
            for(const auto* kernel : candidates.entries)
            {
                if(!bound || !prepared.insert(keyOf(*kernel)).second)
                {
                    continue;
                }
                SCOPED_TRACE(dtype + " " + std::get<0>(keyOf(*kernel)) + " "
                             + std::get<1>(keyOf(*kernel)) + (isSplit(*kernel) ? " split" : ""));
                shipped::expectPreparesOnItsDeviceOnly(
                    isSplit(*kernel) ? *splitC : single,
                    fixture.context(),
                    *bound,
                    *kernel,
                    loadable,
                    {"is packaged with arguments", "convolution contract", "tensor bindings"});
            }
        }
        EXPECT_EQ(prepared, expected) << dtype;
    }
}
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
