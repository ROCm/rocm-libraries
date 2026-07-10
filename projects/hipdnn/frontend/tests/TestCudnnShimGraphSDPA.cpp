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

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
using ::testing::_;
using ::testing::Return;

std::shared_ptr<fe::graph::Tensor_attributes> makeTensor(fe::graph::Graph& graph,
                                                         const std::vector<int64_t>& dim,
                                                         const std::vector<int64_t>& stride,
                                                         int64_t uid)
{
    return graph.tensor(fe::graph::Tensor_attributes{}
                            .set_dim(dim)
                            .set_stride(stride)
                            .set_data_type(fe::DataType_t::FLOAT)
                            .set_uid(uid));
}

// Fills q/k/v (BHSD) with the canonical shapes used by the native SDPA node
// tests: batch 2, 8 heads, S_q 16, S_kv 32, head-dim 64.
void addForwardInputs(fe::graph::Graph& graph,
                      std::shared_ptr<fe::graph::Tensor_attributes>& q,
                      std::shared_ptr<fe::graph::Tensor_attributes>& k,
                      std::shared_ptr<fe::graph::Tensor_attributes>& v)
{
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT);
    q = makeTensor(graph, {2, 8, 16, 64}, {8192, 1024, 64, 1}, 1);
    k = makeTensor(graph, {2, 8, 32, 64}, {16384, 2048, 64, 1}, 2);
    v = makeTensor(graph, {2, 8, 32, 64}, {16384, 2048, 64, 1}, 3);
}

TEST(TestCudnnShimGraphSDPA, ForwardConstructReturnsOutputNoStatsByDefault)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
    addForwardInputs(graph, q, k, v);

    auto [o, stats] = graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    ASSERT_NE(o, nullptr);
    EXPECT_EQ(stats, nullptr);
    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraphSDPA, ForwardGenerateStatsProducesStatsOutput)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
    // SHIM-DIVERGENCE(RENAME): float overload maps to set_attn_scale_value; the
    // shared_ptr overload forwards. Both must compile and validate.
    fe::graph::Graph scalarGraph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
    addForwardInputs(graph, q, k, v);

    fe::graph::SDPA_attributes attrs;
    attrs.set_score_mod([](std::shared_ptr<fe::graph::Graph>,
                           std::shared_ptr<fe::graph::Tensor_attributes> t) { return t; });
    graph.sdpa(q, k, v, attrs);

    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

TEST(TestCudnnShimGraphSDPA, BackwardConstructAndValidate)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
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
class TestCudnnShimGraphSDPABackend : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    cudnnHandle_t _handle = nullptr;
    std::array<char, 256> _fakeDescs{};
    size_t _nextFakeDescIdx = 0;

    void SetUp() override
    {
        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
        _handle = reinterpret_cast<cudnnHandle_t>(0x12345678);

        _nextFakeDescIdx = 0;
        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestCudnnShimGraphSDPABackend, BuildOperationGraphReachesBackend)
{
    fe::graph::Graph graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v;
    addForwardInputs(graph, q, k, v);
    graph.sdpa(q, k, v, fe::graph::SDPA_attributes{}.set_name("Sdpa"));

    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(::testing::AtLeast(1));

    ASSERT_TRUE(graph.validate().is_good());
    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());
}

} // namespace
