// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// SDPA fwd/bwd node coverage for the cuDNN-shaped graph wrapper. Construction,
// validation, and the cuDNN->hipDNN setter divergence mappings are host-only
// (hipDNN graph validate() needs no backend). Build/execute are driven against
// the in-tree mock backend, following the TestGraph.cpp fixture pattern. Gated
// behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY && HIPDNN_ENABLE_SDPA in the frontend
// tests CMakeLists.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "CudnnShimTestSupport.hpp"
#include "fake_backend/MockBackendFixture.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
using hipdnn_shim_test::addForwardInputs;
using hipdnn_shim_test::makeTensor;
using ::testing::_;

TEST(TestCudnnShimGraphSDPA, ForwardConstructReturnsOutputNoStatsByDefault)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);

    auto [o, stats] = graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    ASSERT_NE(o, nullptr);
    EXPECT_EQ(stats, nullptr);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraphSDPA, ForwardGenerateStatsProducesStatsOutput)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);

    auto [o, stats] = graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_generate_stats(true));

    ASSERT_NE(o, nullptr);
    ASSERT_NE(stats, nullptr);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraphSDPA, DeprecatedIsInferenceMapsToGenerateStats)
{
    // SHIM-DIVERGENCE(SEMANTIC): set_is_inference(b) == set_generate_stats(!b).
    fe::graph::Graph inferGraph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(inferGraph, q, k, v);

    fe::graph::SDPA_attributes inferAttrs;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    inferAttrs.set_is_inference(true);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    auto [inferO, inferStats] = inferGraph.sdpa(q, k, v, inferAttrs);
    EXPECT_NE(inferO, nullptr);
    EXPECT_EQ(inferStats, nullptr); // inference == no stats
}

TEST(TestCudnnShimGraphSDPA, ForwardAttnScaleOverloadsConfigure)
{
    fe::graph::Graph scalarGraph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(scalarGraph, q, k, v);
    auto [o1, s1] = scalarGraph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_attn_scale(0.125F));
    EXPECT_NE(o1, nullptr);
    EXPECT_TRUE(scalarGraph.validate().is_good());
}

TEST(TestCudnnShimGraphSDPA, ForwardUnsupportedSetterSurfacesRecordedError)
{
    // SHIM-DIVERGENCE(MISSING): set_score_mod has no hipDNN equivalent; the
    // recorded error must drain into the graph and fail validate().
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);

    fe::graph::SDPA_attributes attrs;
    attrs.set_score_mod([](const std::shared_ptr<fe::graph::Graph>&,
                           std::shared_ptr<fe::graph::Tensor_attributes> t) { return t; });
    graph.sdpa(q, k, v, attrs);

    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

TEST(TestCudnnShimGraphSDPA, BackwardConstructAndValidate)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    auto o = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 4);
    auto dO = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 5);
    auto stats = makeTensor(graph, {2, 8, 16, 1}, {128, 16, 1, 1}, 6);

    auto [dq, dk, dv]
        = graph.sdpa_backward(q, k, v, o, dO, stats, fe::graph::SDPA_backward_attributes{});

    ASSERT_NE(dq, nullptr);
    ASSERT_NE(dk, nullptr);
    ASSERT_NE(dv, nullptr);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraphSDPA, BackwardDeterministicRequestSurfacesRecordedError)
{
    // SHIM-DIVERGENCE(MISSING): determinism is correctness-critical; requesting
    // it must fail loudly rather than silently run non-deterministically.
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    auto o = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 4);
    auto dO = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 5);
    auto stats = makeTensor(graph, {2, 8, 16, 1}, {128, 16, 1, 1}, 6);

    graph.sdpa_backward(q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        fe::graph::SDPA_backward_attributes{}.set_deterministic_algorithm(true));

    EXPECT_TRUE(graph.validate().is_bad());
}

TEST(TestCudnnShimGraphSDPA, BackwardDeterministicFalseIsIgnored)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    auto o = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 4);
    auto dO = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 5);
    auto stats = makeTensor(graph, {2, 8, 16, 1}, {128, 16, 1, 1}, 6);

    graph.sdpa_backward(q,
                        k,
                        v,
                        o,
                        dO,
                        stats,
                        fe::graph::SDPA_backward_attributes{}.set_deterministic_algorithm(false));

    EXPECT_TRUE(graph.validate().is_good());
}

// Mock-backed: proves an SDPA node graph routes through the native operation-graph
// path (reaches the backend). create_execution_plans/build_plans/execute and
// native serialize are 1:1 forwards to hipdnn_frontend::graph::Graph, exhaustively
// covered by TestGraph.cpp; not duplicated here.
using TestCudnnShimGraphSDPABackend = hipdnn_shim_test::ShimMockBackendFixture;

TEST_F(TestCudnnShimGraphSDPABackend, BuildOperationGraphReachesBackend)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(::testing::AtLeast(1));

    ASSERT_TRUE(graph.validate().is_good());
    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());
}

// ErrorRecorder first-error-wins: two unsupported backward setters are chained;
// the FIRST recorded message is what surfaces at validate(). Asserting the
// message (not just is_bad) is what distinguishes first-wins from last-wins.
TEST(TestCudnnShimGraphSDPA, DeferredErrorFirstWinsScoreModBeforeSeqLen)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    auto o = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 4);
    auto dO = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 5);
    auto stats = makeTensor(graph, {2, 8, 16, 1}, {128, 16, 1, 1}, 6);

    fe::graph::SDPA_backward_attributes attrs;
    attrs
        .set_score_mod([](const std::shared_ptr<fe::graph::Graph>&,
                          std::shared_ptr<fe::graph::Tensor_attributes> t) { return t; })
        .set_max_total_seq_len_q(128);
    graph.sdpa_backward(q, k, v, o, dO, stats, attrs);

    auto error = graph.validate();
    ASSERT_TRUE(error.is_bad());
    EXPECT_NE(error.get_message().find("score modifier"), std::string::npos);
    EXPECT_EQ(error.get_message().find("max_total_seq_len_q"), std::string::npos);
}

// Same two setters in the opposite order: now the seq-len message wins, proving
// the latch keys on call order rather than on which setter is "more severe".
TEST(TestCudnnShimGraphSDPA, DeferredErrorFirstWinsSeqLenBeforeScoreMod)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    auto o = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 4);
    auto dO = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 5);
    auto stats = makeTensor(graph, {2, 8, 16, 1}, {128, 16, 1, 1}, 6);

    fe::graph::SDPA_backward_attributes attrs;
    attrs.set_max_total_seq_len_q(128).set_score_mod(
        [](const std::shared_ptr<fe::graph::Graph>&,
           std::shared_ptr<fe::graph::Tensor_attributes> t) { return t; });
    graph.sdpa_backward(q, k, v, o, dO, stats, attrs);

    auto error = graph.validate();
    ASSERT_TRUE(error.is_bad());
    EXPECT_NE(error.get_message().find("max_total_seq_len_q"), std::string::npos);
    EXPECT_EQ(error.get_message().find("score modifier"), std::string::npos);
}

// set_causal_mask(true) forwards a bare bool to hipDNN (the cuDNN compound
// alignment/right-bound side effects are intentionally NOT replicated). A causal
// graph must still validate — the mask is a first-class hipDNN attribute.
TEST(TestCudnnShimGraphSDPA, CausalMaskGraphStillValidates)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);

    auto [o, stats] = graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_causal_mask(true));
    ASSERT_NE(o, nullptr);
    EXPECT_TRUE(graph.validate().is_good());
}

// State machine, Mode::Native branch of build_plan_at_index: a graph that HAS an
// operation graph (an sdpa node makes it Native) rejects a non-zero index with
// the "index is invalid" message — distinct from the no-op-graph message covered
// host-only in TestCudnnShimGraph. The index!=0 guard runs before any backend
// call, so this is reachable without a device.
TEST(TestCudnnShimGraphSDPA, BuildPlanAtInvalidIndexOnNativeGraphReportsInvalidIndex)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    auto error = graph.build_plan_at_index(5);
    ASSERT_TRUE(error.is_bad());
    EXPECT_NE(error.get_message().find("index is invalid"), std::string::npos);
    EXPECT_EQ(error.get_message().find("no compiled execution plan"), std::string::npos);
}

// A native (sdpa) graph reports get_execution_plan_count() == 0 until plans are
// created; merely having an operation graph does not manufacture a plan count.
// Reaching count 1 needs create_execution_plans against a real heuristics backend
// (exercised through the mock-backed lowering in TestGraph.cpp); asserting it here
// would duplicate that layer, so the count==1 transition is intentionally left to
// the backend-level suite.
TEST(TestCudnnShimGraphSDPA, NativeGraphPlanCountZeroBeforePlansCreated)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
    addForwardInputs(graph, q, k, v);
    graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    EXPECT_EQ(graph.get_execution_plan_count(), 0);
}

} // namespace
