// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified3DDispatch.cpp
 * @brief hipkernel:Gfx950AttentionUnified3D at the bind and prepare layer: what the graph
 *        match binds, how much workspace the two kernels ask for, and what prepare() refuses
 *        before it reaches a loader.
 *
 * A workspace too small for the scratch lets the segment kernel write past it, and the reduce
 * read whatever follows; neither faults. The launch itself needs a real archive and a gfx950;
 * the grids and the bytes launch() copies are pinned by TestGfx950AttentionUnified3DGeometry.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified3d;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

const IKernelDispatchHandler<Handle>* dispatchHandler()
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
}

BoundTokens bindingsFor(const Graph& graph)
{
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value());
    return bound.value_or(BoundTokens{});
}

std::string prepareFailureFor(const KernelSpec& spec, const GraphSpec& graphSpec = GraphSpec{})
{
    const Graph graph(graphSpec);
    return prepareFailure(graph, bindingsFor(graph), makeKpackKernel(spec, pythonSegmentAbi()));
}

constexpr const char* GEOMETRY_MARKER = "declares no supported decode geometry";
constexpr const char* LAYOUT_MARKER = "cannot lay out its workspace";
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";

} // namespace

TEST(TestGfx950AttentionUnified3DDispatch, DispatchSymbolResolves)
{
    EXPECT_NE(dispatchHandler(), nullptr);
}

/// The default graph (B2, Skv 256, Hq 8, D128) under the 16-segment kernel: 768 bytes of
/// identity tables and 133120 of scratch (TestGfx950AttentionUnified3DGeometry works it out).
TEST(TestGfx950AttentionUnified3DDispatch, WorkspaceIsTheIdentityTablesThenTheScratch)
{
    const Graph graph;
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(
        handler->workspaceBytes(graph.context(), bindingsFor(graph), makeKernel(KernelSpec{})),
        133888U);
}

/// The scratch scales with the candidate's segment count; the identity tables do not.
TEST(TestGfx950AttentionUnified3DDispatch, WorkspaceFollowsTheSegmentCount)
{
    const Graph graph;
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);
    KernelSpec thirtyTwo;
    thirtyTwo.numSegments = 32;
    EXPECT_EQ(handler->workspaceBytes(graph.context(), bound, makeKernel(thirtyTwo)),
              768U + 2U * 133120U);
}

TEST(TestGfx950AttentionUnified3DDispatch, WorkspaceQueryRefusesACandidateItCannotLaunch)
{
    const Graph graph;
    const auto bound = bindingsFor(graph);
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);
    KernelSpec noReduce;
    noReduce.launches = {"segment"};
    EXPECT_THROW(handler->workspaceBytes(graph.context(), bound, makeKernel(noReduce)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// The launch set the state manager holds every kernel of this pack to: a descriptor whose
/// `sources` names any other set is dropped before it reaches the handler.
TEST(TestGfx950AttentionUnified3DDispatch, LaunchesTheSegmentAndReduceKernels)
{
    const auto* handler = dispatchHandler();
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->launchNames(), (std::set<std::string>{"segment", "reduce"}));
}

TEST(TestGfx950AttentionUnified3DDispatch, BindsEachOperandUidToItsOwnToken)
{
    const auto bound = bindingsFor(Graph{});
    EXPECT_EQ(tryGetBoundInt(bound, Q_TOKEN), Q_UID);
    EXPECT_EQ(tryGetBoundInt(bound, K_TOKEN), K_UID);
    EXPECT_EQ(tryGetBoundInt(bound, V_TOKEN), V_UID);
    EXPECT_EQ(tryGetBoundInt(bound, O_TOKEN), O_UID);
    EXPECT_EQ(tryGetBoundInt(bound, SLIDING_WINDOW_TOKEN), 0);
}

/// The positive control: every check before the loader passes for both catalog sections,
/// and the failure that remains is the missing archive.
TEST(TestGfx950AttentionUnified3DDispatch, ReachesTheLoaderOnceEverythingBeforeItIsGood)
{
    GraphSpec d256Graph;
    d256Graph.headSize = 256;
    KernelSpec d256;
    d256.headSize = 256;
    d256.numSegments = 128;
    for(const auto& [graph, kernel] :
        {std::pair{GraphSpec{}, KernelSpec{}}, std::pair{d256Graph, d256}})
    {
        const std::string failure = prepareFailureFor(kernel, graph);
        ASSERT_FALSE(failure.empty());
        for(const char* marker : {GEOMETRY_MARKER, LAYOUT_MARKER, OUTPUT_LAYOUT_MARKER})
        {
            EXPECT_EQ(failure.find(marker), std::string::npos) << failure;
        }
        EXPECT_NE(failure.find("does not exist"), std::string::npos) << failure;
    }
}

/// A record without its reduce launch is refused by name before any archive is touched.
TEST(TestGfx950AttentionUnified3DDispatch, PrepareRefusesARecordWithoutItsReduceLaunch)
{
    KernelSpec noReduce;
    noReduce.launches = {"segment"};
    EXPECT_NE(prepareFailureFor(noReduce).find(GEOMETRY_MARKER), std::string::npos);
}

/// prepare() is reachable without kernel_match: it re-checks the decode rule, since the
/// decode grid would leave a longer query's rows unwritten.
TEST(TestGfx950AttentionUnified3DDispatch, PrepareRefusesAPrefillGraph)
{
    GraphSpec prefill;
    prefill.seqLenQ = 4;
    const Graph graph(prefill);
    const BoundTokens bound = bindingsFor(Graph{});
    const std::string failure
        = prepareFailure(graph, bound, makeKpackKernel(KernelSpec{}, pythonSegmentAbi()));
    EXPECT_NE(failure.find("decode-grid kernel"), std::string::npos) << failure;
}

TEST(TestGfx950AttentionUnified3DDispatch, PrepareRefusesABlockThatDoesNotDivideTheKeys)
{
    GraphSpec graph;
    graph.seqLenKv = 272;
    KernelSpec block32;
    block32.blockSize = 32;
    EXPECT_NE(prepareFailureFor(block32, graph).find(LAYOUT_MARKER), std::string::npos);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
