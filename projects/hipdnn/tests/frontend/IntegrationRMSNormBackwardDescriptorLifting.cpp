// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
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
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationRMSNormBackwardDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard RMSNormBackward graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph(bool computeDBias)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("RMSNormBackwardLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_RMSNORMBACKWARD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_RMSNORMBACKWARD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)
            .set_name("scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

        auto invRms = std::make_shared<TensorAttributes>();
        invRms->set_uid(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID)
            .set_name("inv_rms")
            .set_data_type(DataType::FLOAT);
        invRms->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));

        RMSNormBackwardAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_compute_dbias(computeDBias);

        auto results = graph->rmsnorm_backward(dy, x, scale, invRms, attrs);
        results[0]->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("dx");
        results[1]
            ->set_uid(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID)
            .set_output(true)
            .set_name("dscale");
        if(computeDBias)
        {
            results[2]
                ->set_uid(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID)
                .set_output(true)
                .set_name("dbias");
        }

        return graph;
    }
};

// Builds a standard RMSNormBackward graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, BasicRMSNormBackwardRoundTrip)
{
    auto graph = buildGraph(true);
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_name(), "dy");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_name(), "x");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_uid(), K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_name(), "scale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify inv_rms tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_name(), "inv_rms");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_name(), "dx");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify dscale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_name(), "dscale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify dbias tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_name(), "dbias");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a RMSNormBackwardNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

TEST_F(IntegrationRMSNormBackwardDescriptorLifting, BasicRMSNormBackwardNoDBiasRoundTrip)
{
    auto graph = buildGraph(false);
    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_name(), "dy");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_name(), "x");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_uid(), K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_name(), "scale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify inv_rms tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_name(), "inv_rms");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_name(), "dx");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify dscale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_name(), "dscale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify dbias tensor
    ASSERT_EQ(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID), 0u);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a RMSNormBackwardNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, RMSNormBackwardTensorSharingPreserved)
{
    auto originalGraph = buildGraph(true);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID].get(),
              opNode->attributes.get_scale().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_inv_rms()->get_uid(), K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID].get(),
              opNode->attributes.get_inv_rms().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID].get(), opNode->attributes.get_dx().get());
    // Verify dscale tensor sharing
    EXPECT_EQ(opNode->attributes.get_dscale()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID].get(),
              opNode->attributes.get_dscale().get());
    // Verify dbias tensor sharing
    EXPECT_EQ(opNode->attributes.get_dbias()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID].get(),
              opNode->attributes.get_dbias().get());
}

// Builds a RMSNormBackward graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, RMSNormBackwardLiftWithoutFinalization)
{
    auto originalGraph = buildGraph(true);
    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u);

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_name(), "dy");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_name(), "x");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_name(), "scale");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID]->get_name(), "inv_rms");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_name(), "dx");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_name(), "dscale");

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->get_name(), "dbias");
}

// Builds a RMSNormBackward graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("RMSNormBackwardAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_name("inv_rms").set_data_type(DataType::FLOAT);
    invRms->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));

    RMSNormBackwardAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_compute_dbias(true);

    auto results = graph->rmsnorm_backward(dy, x, scale, invRms, attrs);
    results[0]->set_output(true).set_name("dx");
    results[1]->set_output(true).set_name("dscale");
    results[2]->set_output(true).set_name("dbias");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u);

    // Verify all UIDs are non-negative and distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        EXPECT_GE(uid, 0)
            << "Auto-assigned UID should be non-negative"; // NOLINT(readability-implicit-bool-conversion)
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_dy(), nullptr);
    nodeUids.insert(opNode->attributes.get_dy()->get_uid());
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_scale(), nullptr);
    nodeUids.insert(opNode->attributes.get_scale()->get_uid());
    ASSERT_NE(opNode->attributes.get_inv_rms(), nullptr);
    nodeUids.insert(opNode->attributes.get_inv_rms()->get_uid());
    ASSERT_NE(opNode->attributes.get_dx(), nullptr);
    nodeUids.insert(opNode->attributes.get_dx()->get_uid());
    ASSERT_NE(opNode->attributes.get_dscale(), nullptr);
    nodeUids.insert(opNode->attributes.get_dscale()->get_uid());
    ASSERT_NE(opNode->attributes.get_dbias(), nullptr);
    nodeUids.insert(opNode->attributes.get_dbias()->get_uid());
    ASSERT_EQ(nodeUids.size(), 7u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_dy()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(opNode->attributes.get_dy()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_scale()->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(opNode->attributes.get_scale()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(opNode->attributes.get_inv_rms()->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(opNode->attributes.get_inv_rms()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dx()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(opNode->attributes.get_dx()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dscale()->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(opNode->attributes.get_dscale()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dbias()->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(opNode->attributes.get_dbias()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a batchnorm inference graph.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildGraph(true);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to JSON (auto-lowers internally)
    std::string jsonData;
    result = originalGraph->serialize(jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(jsonData.empty());

    // Deserialize from JSON with handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(_handle, jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level attributes
    EXPECT_EQ(liftedGraph->get_name(), "RMSNormBackwardLiftingTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensor count
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u) << "Expected 6 tensors in lifted batchnorm inference graph";

    // Verify tensors
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                      "dy",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_X_UID,
                                      "x",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                      "scale",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                      "inv_rms",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES),
                                      DataType::FLOAT);

    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                      "dx",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                      "dscale",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                      "dbias",
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                      toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES),
                                      DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a RMSNormBackwardNode";

    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

} // namespace
