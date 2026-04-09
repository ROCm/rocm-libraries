// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/LayernormBackwardNode.hpp>
#include <hipdnn_test_sdk/constants/LayernormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationLayernormBackwardDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard LayernormBackward graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("LayernormBackwardLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        LayernormBackwardAttributes attrs;
        attrs.set_name("test_op");

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_LAYERNORMBACKWARD_TENSOR_DY_UID)
            .set_name("dy")
            .set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_LAYERNORMBACKWARD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID)
            .set_name("scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID)
            .set_name("mean")
            .set_data_type(DataType::FLOAT);
        mean->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
        attrs.set_mean(std::move(mean));

        auto invVariance = std::make_shared<TensorAttributes>();
        invVariance->set_uid(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID)
            .set_name("inv_variance")
            .set_data_type(DataType::FLOAT);
        invVariance->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
        attrs.set_inv_variance(std::move(invVariance));

        auto epsilon = std::make_shared<TensorAttributes>();
        epsilon->set_uid(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID)
            .set_name("epsilon")
            .set_data_type(DataType::FLOAT);
        epsilon->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
        attrs.set_epsilon(std::move(epsilon));

        auto results = graph->layernorm_backward(dy, x, scale, attrs);
        results[0]->set_uid(K_LAYERNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("dx");
        results[1]
            ->set_uid(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID)
            .set_output(true)
            .set_name("dscale");
        results[2]
            ->set_uid(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID)
            .set_output(true)
            .set_name("dbias");

        return graph;
    }
};

// Builds a standard LayernormBackward graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationLayernormBackwardDescriptorLifting, BasicLayernormBackwardRoundTrip)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify scale tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify dscale tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify dbias tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_uid(),
              K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<LayernormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a LayernormBackwardNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationLayernormBackwardDescriptorLifting, LayernormBackwardTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<LayernormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID].get(),
              opNode->attributes.get_dy().get());
    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID].get(),
              opNode->attributes.get_scale().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID].get(),
              opNode->attributes.get_dx().get());
    // Verify dscale tensor sharing
    EXPECT_EQ(opNode->attributes.get_dscale()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID].get(),
              opNode->attributes.get_dscale().get());
    // Verify dbias tensor sharing
    EXPECT_EQ(opNode->attributes.get_dbias()->get_uid(), K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID].get(),
              opNode->attributes.get_dbias().get());
}

// Builds a LayernormBackward graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationLayernormBackwardDescriptorLifting, LayernormBackwardLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<LayernormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);

    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
}

// Builds a LayernormBackward graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationLayernormBackwardDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    LayernormBackwardAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("LayernormBackwardAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_name("mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    attrs.set_mean(std::move(mean));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_name("inv_variance").set_data_type(DataType::FLOAT);
    invVariance->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    attrs.set_inv_variance(std::move(invVariance));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("epsilon").set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    attrs.set_epsilon(std::move(epsilon));

    auto results = graph->layernorm_backward(dy, x, scale, attrs);
    results[0]->set_output(true).set_name("dx");
    results[1]->set_output(true).set_name("dscale");
    results[2]->set_output(true).set_name("dbias");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);

    // Verify all UIDs are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<LayernormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_dy(), nullptr);
    nodeUids.insert(opNode->attributes.get_dy()->get_uid());
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_scale(), nullptr);
    nodeUids.insert(opNode->attributes.get_scale()->get_uid());
    ASSERT_NE(opNode->attributes.get_dx(), nullptr);
    nodeUids.insert(opNode->attributes.get_dx()->get_uid());
    ASSERT_NE(opNode->attributes.get_dscale(), nullptr);
    nodeUids.insert(opNode->attributes.get_dscale()->get_uid());
    ASSERT_NE(opNode->attributes.get_dbias(), nullptr);
    nodeUids.insert(opNode->attributes.get_dbias()->get_uid());
    ASSERT_EQ(nodeUids.size(), 6u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_dy()->get_dim(), toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(opNode->attributes.get_dy()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_scale()->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(opNode->attributes.get_scale()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dx()->get_dim(), toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(opNode->attributes.get_dx()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dscale()->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(opNode->attributes.get_dscale()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dbias()->get_dim(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(opNode->attributes.get_dbias()->get_stride(),
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
}

} // namespace
