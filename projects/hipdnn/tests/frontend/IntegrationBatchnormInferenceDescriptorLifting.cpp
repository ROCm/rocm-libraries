// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_test_sdk/constants/BatchnormInferenceConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;

namespace
{
// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationBatchnormInferenceDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard BatchnormInference graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("BatchnormInferenceLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BN_INF_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BN_INF_SPATIAL_DIMS)).set_stride(toVec(K_BN_INF_SPATIAL_STRIDES));

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(K_BN_INF_TENSOR_MEAN_UID).set_name("mean").set_data_type(DataType::FLOAT);
        mean->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

        auto invVariance = std::make_shared<TensorAttributes>();
        invVariance->set_uid(K_BN_INF_TENSOR_INV_VARIANCE_UID)
            .set_name("inv_variance")
            .set_data_type(DataType::FLOAT);
        invVariance->set_dim(toVec(K_BN_INF_CHANNEL_DIMS))
            .set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BN_INF_TENSOR_SCALE_UID).set_name("scale").set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(K_BN_INF_TENSOR_BIAS_UID).set_name("bias").set_data_type(DataType::FLOAT);
        bias->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

        BatchnormInferenceAttributes attrs;
        attrs.set_name("test_op");

        auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, attrs);
        y->set_uid(K_BN_INF_TENSOR_Y_UID).set_output(true).set_name("y");

        return graph;
    }
};

// Builds a standard BatchnormInference graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BasicBatchnormInferenceRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 6u);

    // Verify x tensor (spatial dims)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_uid(), K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_stride(), toVec(K_BN_INF_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_name(), "x");

    // Verify mean tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_uid(), K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_name(), "mean");

    // Verify inv_variance tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_uid(),
              K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_stride(),
              toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_name(), "inv_variance");

    // Verify scale tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_uid(), K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_name(), "scale");

    // Verify bias tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_uid(), K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_name(), "bias");

    // Verify y tensor (spatial dims, inferred from x)
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_uid(), K_BN_INF_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_stride(), toVec(K_BN_INF_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_name(), "y");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a BatchnormInferenceNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BatchnormInferenceTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify mean tensor sharing
    EXPECT_EQ(opNode->attributes.get_mean()->get_uid(), K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID].get(), opNode->attributes.get_mean().get());
    // Verify inv_variance tensor sharing
    EXPECT_EQ(opNode->attributes.get_inv_variance()->get_uid(), K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID].get(),
              opNode->attributes.get_inv_variance().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID].get(), opNode->attributes.get_scale().get());
    // Verify bias tensor sharing
    EXPECT_EQ(opNode->attributes.get_bias()->get_uid(), K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID].get(), opNode->attributes.get_bias().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_BN_INF_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a BatchnormInference graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BatchnormInferenceLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u);

    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_stride(), toVec(K_BN_INF_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_X_UID]->get_name(), "x");
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_MEAN_UID]->get_name(), "mean");
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_stride(),
              toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]->get_name(), "inv_variance");
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_SCALE_UID]->get_name(), "scale");
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_stride(), toVec(K_BN_INF_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_BIAS_UID]->get_name(), "bias");
    ASSERT_NE(tensorMap.count(K_BN_INF_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_stride(), toVec(K_BN_INF_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_INF_TENSOR_Y_UID]->get_name(), "y");
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the lifting round trip and are all distinct.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("AutoUidBatchnormInferenceLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BN_INF_SPATIAL_DIMS)).set_stride(toVec(K_BN_INF_SPATIAL_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_name("mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_name("inv_variance").set_data_type(DataType::FLOAT);
    invVariance->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BN_INF_CHANNEL_DIMS)).set_stride(toVec(K_BN_INF_CHANNEL_STRIDES));

    BatchnormInferenceAttributes attrs;
    attrs.set_name("auto_uid_bn_inf_op");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, attrs);
    y->set_output(true).set_name("y");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u)
        << "Expected 6 tensors in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "All auto-assigned UIDs must be distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify the node references tensors with auto-assigned UIDs
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify all tensor UIDs are pairwise distinct
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto meanUid = opNode->attributes.get_mean()->get_uid();
    auto invVarianceUid = opNode->attributes.get_inv_variance()->get_uid();
    auto scaleUid = opNode->attributes.get_scale()->get_uid();
    auto biasUid = opNode->attributes.get_bias()->get_uid();
    auto yUid = opNode->attributes.get_y()->get_uid();

    const std::set<int64_t> uidSet = {xUid, meanUid, invVarianceUid, scaleUid, biasUid, yUid};
    EXPECT_EQ(uidSet.size(), 6u)
        << "All 6 tensor UIDs must be distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[meanUid]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[invVarianceUid]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[scaleUid]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[biasUid]->get_dim(), toVec(K_BN_INF_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[yUid]->get_dim(), toVec(K_BN_INF_SPATIAL_DIMS));
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a batchnorm inference graph.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildGraph();

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
    EXPECT_EQ(liftedGraph->get_name(), "BatchnormInferenceLiftingTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensor count (x, mean, invVariance, scale, bias, y = 6)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u) << "Expected 6 tensors in lifted batchnorm inference graph";

    // Verify tensors
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_X_UID,
                                      "x",
                                      toVec(K_BN_INF_SPATIAL_DIMS),
                                      toVec(K_BN_INF_SPATIAL_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_MEAN_UID,
                                      "mean",
                                      toVec(K_BN_INF_CHANNEL_DIMS),
                                      toVec(K_BN_INF_CHANNEL_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                      "inv_variance",
                                      toVec(K_BN_INF_CHANNEL_DIMS),
                                      toVec(K_BN_INF_CHANNEL_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_SCALE_UID,
                                      "scale",
                                      toVec(K_BN_INF_CHANNEL_DIMS),
                                      toVec(K_BN_INF_CHANNEL_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_BIAS_UID,
                                      "bias",
                                      toVec(K_BN_INF_CHANNEL_DIMS),
                                      toVec(K_BN_INF_CHANNEL_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BN_INF_TENSOR_Y_UID,
                                      "y",
                                      toVec(K_BN_INF_SPATIAL_DIMS),
                                      toVec(K_BN_INF_SPATIAL_STRIDES),
                                      DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a BatchnormInferenceNode";

    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

} // namespace
