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
        buildAndDeserialize(LayernormBackwardAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("LayernormBackwardIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

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

        auto [dx, dscale, dbias] = graph->layernorm_backward(dy, x, scale, attrs);
        dx->set_uid(K_LAYERNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("dx");
        dx->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
        dscale->set_uid(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID).set_output(true).set_name("dscale");
        dscale->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
        dbias->set_uid(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID).set_output(true).set_name("dbias");
        dbias->set_dim(toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS))
            .set_stride(toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationLayernormBackwardDescriptorLowering, LayernormBackwardLoweringRoundTrip)
{
    LayernormBackwardAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs);

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

} // namespace
