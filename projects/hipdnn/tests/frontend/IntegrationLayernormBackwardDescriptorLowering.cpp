// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/LayernormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationLayernormBackwardDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT
        buildAndDeserialize(LayernormBackwardAttributes& attrs, bool withOptionals, bool manualUids)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("LayernormBackwardIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        if(manualUids)
        {
            dy->set_uid(K_LAYERNORMBACKWARD_TENSOR_DY_UID);
        }
        dy->set_name("dy")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        if(manualUids)
        {
            x->set_uid(K_LAYERNORMBACKWARD_TENSOR_X_UID);
        }
        x->set_name("x")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        if(manualUids)
        {
            scale->set_uid(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
        }
        scale->set_name("scale")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));

        if(withOptionals)
        {
            auto mean = std::make_shared<TensorAttributes>();
            if(manualUids)
            {
                mean->set_uid(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
            }
            mean->set_name("mean")
                .set_data_type(DataType::FLOAT)
                .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS))
                .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
            attrs.set_mean(std::move(mean));

            auto invVariance = std::make_shared<TensorAttributes>();
            if(manualUids)
            {
                invVariance->set_uid(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
            }
            invVariance->set_name("inv_variance")
                .set_data_type(DataType::FLOAT)
                .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS))
                .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
            attrs.set_inv_variance(std::move(invVariance));

            auto epsilon = std::make_shared<TensorAttributes>();
            if(manualUids)
            {
                epsilon->set_uid(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
            }
            epsilon->set_name("epsilon")
                .set_data_type(DataType::FLOAT)
                .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS))
                .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
            attrs.set_epsilon(std::move(epsilon));
        }

        auto [dx, dscale, dbias] = graph->layernorm_backward(dy, x, scale, attrs);
        if(manualUids)
        {
            dx->set_uid(K_LAYERNORMBACKWARD_TENSOR_DX_UID);
        }
        dx->set_output(true)
            .set_name("dx")
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
        if(manualUids)
        {
            dscale->set_uid(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
        }
        dscale->set_output(true)
            .set_name("dscale")
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
        if(manualUids)
        {
            dbias->set_uid(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
        }
        dbias->set_output(true)
            .set_name("dbias")
            .set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationLayernormBackwardDescriptorLowering,
       LayernormBackwardLoweringRoundTripWithOptionals)
{
    LayernormBackwardAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs, true, true);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 9u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_MEAN_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_MEAN_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_MEAN_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID]->data_type,
              DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);

    auto* opNode = node->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(opNode->x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(opNode->scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(opNode->mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(opNode->inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(opNode->dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(opNode->dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(opNode->epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationLayernormBackwardDescriptorLowering,
       LayernormBackwardLoweringRoundTripWithoutOptionals)
{
    LayernormBackwardAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs, false, true);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 6u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);

    auto* opNode = node->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(opNode->x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(opNode->scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(opNode->dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(opNode->dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationLayernormBackwardDescriptorLowering,
       LayernormBackwardLoweringRoundTripAutoAssignedUids)
{
    LayernormBackwardAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs, true, false);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 9u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 9u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);

    auto* opNode = node->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(opNode->dy_tensor_uid) > 0)
        << "DY tensor UID " << opNode->dy_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0)
        << "X tensor UID " << opNode->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(opNode->scale_tensor_uid) > 0)
        << "Scale tensor UID " << opNode->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    ASSERT_TRUE(opNode->mean_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(opNode->mean_tensor_uid.value()) > 0)
        << "Mean tensor UID " << opNode->mean_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    ASSERT_TRUE(opNode->inv_variance_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(opNode->inv_variance_tensor_uid.value()) > 0)
        << "Inverse variance tensor UID " << opNode->inv_variance_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(opNode->dx_tensor_uid) > 0)
        << "DX tensor UID " << opNode->dx_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(opNode->dscale_tensor_uid) > 0)
        << "DScale tensor UID " << opNode->dscale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(opNode->dbias_tensor_uid) > 0)
        << "DBias tensor UID " << opNode->dbias_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    ASSERT_TRUE(opNode->epsilon_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(opNode->epsilon_tensor_uid.value()) > 0)
        << "Epsilon tensor UID " << opNode->epsilon_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All nine tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {opNode->dy_tensor_uid,
                                                  opNode->x_tensor_uid,
                                                  opNode->scale_tensor_uid,
                                                  opNode->mean_tensor_uid.value(),
                                                  opNode->inv_variance_tensor_uid.value(),
                                                  opNode->dx_tensor_uid,
                                                  opNode->dscale_tensor_uid,
                                                  opNode->dbias_tensor_uid,
                                                  opNode->epsilon_tensor_uid.value()};
    EXPECT_EQ(nodeUids.size(), 9u)
        << "Node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(opNode->dy_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->dy_tensor_uid]->dims, toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[opNode->dy_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[opNode->dy_tensor_uid]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->x_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->x_tensor_uid]->dims, toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[opNode->x_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[opNode->x_tensor_uid]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->scale_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->scale_tensor_uid]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[opNode->scale_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[opNode->scale_tensor_uid]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->mean_tensor_uid.value()), 0u);
    EXPECT_EQ(tensorMap[opNode->mean_tensor_uid.value()]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS));
    EXPECT_EQ(tensorMap[opNode->mean_tensor_uid.value()]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    EXPECT_EQ(tensorMap[opNode->mean_tensor_uid.value()]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->inv_variance_tensor_uid.value()), 0u);
    EXPECT_EQ(tensorMap[opNode->inv_variance_tensor_uid.value()]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS));
    EXPECT_EQ(tensorMap[opNode->inv_variance_tensor_uid.value()]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    EXPECT_EQ(tensorMap[opNode->inv_variance_tensor_uid.value()]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->dx_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->dx_tensor_uid]->dims, toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[opNode->dx_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[opNode->dx_tensor_uid]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->dscale_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->dscale_tensor_uid]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[opNode->dscale_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[opNode->dscale_tensor_uid]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(opNode->dbias_tensor_uid), 0u);
    EXPECT_EQ(tensorMap[opNode->dbias_tensor_uid]->dims,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[opNode->dbias_tensor_uid]->strides,
              toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[opNode->dbias_tensor_uid]->data_type, DataTypeSdk::FLOAT);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

} // namespace
