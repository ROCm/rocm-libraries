// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Coverage for the shim's is_supported_ext() probe: a hipDNN extension with no
// upstream cuDNN frontend equivalent, so the contract asserted here is hipDNN's.
#include "CudnnShimTestSupport.hpp"
#include "fake_backend/MockBackendFixture.hpp"

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <cstdint>
#include <gtest/gtest.h>

#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using ::testing::_;

void addPointwiseGraph(fe::graph::Graph& graph)
{
    const int64_t n = 4;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 1);
    auto b = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 2);
    auto c = graph.pointwise(
        a, b, fe::graph::Pointwise_attributes{}.set_mode(fe::PointwiseMode_t::ADD));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);
}

class TestCudnnShimGraphSupportCheck : public hipdnn_shim_test::ShimMockBackendFixture
{
protected:
    int64_t _availableEngineCount = 1;

    // The probe's only backend query beyond the operation-graph lowering the base
    // fixture already answers: how many engine configs the heuristic returned.
    void installEngineHeuristicResults()
    {
        ON_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINEHEUR_RESULTS, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t* elementCount,
                                  void*) {
                if(elementCount != nullptr)
                {
                    *elementCount = _availableEngineCount;
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }
};

TEST_F(TestCudnnShimGraphSupportCheck, ReportsSupportedWhenEnginesAvailable)
{
    installEngineHeuristicResults();

    fe::graph::Graph graph;
    addPointwiseGraph(graph);

    auto error = graph.is_supported_ext(_handle);
    EXPECT_TRUE(error.is_good()) << error.get_message();
}

TEST_F(TestCudnnShimGraphSupportCheck, ReportsNotSupportedWhenNoEngineIsApplicable)
{
    _availableEngineCount = 0;
    installEngineHeuristicResults();

    fe::graph::Graph graph;
    addPointwiseGraph(graph);

    auto error = graph.is_supported_ext(_handle);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimGraphSupportCheck, AcceptsNonFallbackHeuristicModes)
{
    installEngineHeuristicResults();

    fe::graph::Graph graph;
    addPointwiseGraph(graph);

    auto error = graph.is_supported_ext(_handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    EXPECT_TRUE(error.is_good()) << error.get_message();
}

TEST_F(TestCudnnShimGraphSupportCheck, InvalidOwnedTensorFailsBeforeBackendCall)
{
    fe::graph::Graph graph;
    graph.tensor(fe::graph::Tensor_attributes{}
                     .set_dim({1})
                     .set_data_type(fe::DataType_t::FLOAT)
                     .set_uid(1));

    auto error = graph.is_supported_ext(_handle);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

TEST_F(TestCudnnShimGraphSupportCheck, RecordedSetterErrorSurfaces)
{
    fe::graph::Graph graph;
    graph.set_sm_count(1);

    auto error = graph.is_supported_ext(_handle);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

// A node-less graph has nothing to probe, so it is trivially supported, and the
// probe leaves it built the way the native call would (workspace query answers 0
// only once the operation graph is built).
TEST_F(TestCudnnShimGraphSupportCheck, EmptyGraphIsSupportedAndBuilt)
{
    fe::graph::Graph graph;

    int64_t workspaceSize = -1;
    ASSERT_TRUE(graph.get_workspace_size(workspaceSize).is_bad());

    EXPECT_TRUE(graph.is_supported_ext(nullptr).is_good());

    ASSERT_TRUE(graph.get_workspace_size(workspaceSize).is_good());
    EXPECT_EQ(workspaceSize, 0);
}

} // namespace
