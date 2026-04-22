// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBatchnormDescriptorLowering : public IntegrationTestFixture
{
};

TEST_F(IntegrationBatchnormDescriptorLowering, BatchnormGraphRoundTripAllOptionals)
{
    // Full round-trip with all optional tensors including running stats.
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBnFwdGraphAllOptionals")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BATCHNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BATCHNORM_TENSOR_X_DIMS)).set_stride(toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BATCHNORM_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_BATCHNORM_TENSOR_BIAS_UID).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_BATCHNORM_TENSOR_EPSILON_UID).set_name("Epsilon");

    auto prevRunMean = std::make_shared<TensorAttributes>();
    prevRunMean->set_uid(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID)
        .set_name("PrevRunMean")
        .set_data_type(DataType::FLOAT);
    prevRunMean->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto prevRunVar = std::make_shared<TensorAttributes>();
    prevRunVar->set_uid(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID)
        .set_name("PrevRunVar")
        .set_data_type(DataType::FLOAT);
    prevRunVar->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto momentum = std::make_shared<TensorAttributes>(0.1f);
    momentum->set_uid(K_BATCHNORM_TENSOR_MOMENTUM_UID).set_name("Momentum");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("bn_fwd_op");
    bnAttrs.set_epsilon(epsilon);
    bnAttrs.set_previous_running_stats(prevRunMean, prevRunVar, momentum);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_uid(K_BATCHNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    meanOut->set_uid(K_BATCHNORM_TENSOR_MEAN_UID).set_output(true).set_name("Mean");
    invVarOut->set_uid(K_BATCHNORM_TENSOR_INV_VARIANCE_UID)
        .set_output(true)
        .set_name("InvVariance");
    nextRunMean->set_uid(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID)
        .set_output(true)
        .set_name("NextRunMean");
    nextRunVar->set_uid(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID)
        .set_output(true)
        .set_name("NextRunVar");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    // x, scale, bias, epsilon, y, mean, invVariance,
    // prevRunMean, prevRunVar, momentum, nextRunMean, nextRunVar = 12 tensors
    ASSERT_EQ(graphT.tensors.size(), 12u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_BATCHNORM_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_BATCHNORM_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(scaleT->dims, toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_BATCHNORM_TENSOR_BIAS_UID];
    EXPECT_EQ(biasT->name, "Bias");
    EXPECT_EQ(biasT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(biasT->dims, toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(biasT->strides, toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_BATCHNORM_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(yT->dims, toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(yT->strides, toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    // Verify Mean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MEAN_UID), 0u);
    auto* meanT = tensorMap[K_BATCHNORM_TENSOR_MEAN_UID];
    EXPECT_EQ(meanT->name, "Mean");
    EXPECT_EQ(meanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(meanT->dims.empty());
    EXPECT_FALSE(meanT->strides.empty());

    // Verify InvVariance tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_INV_VARIANCE_UID), 0u);
    auto* invVarT = tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(invVarT->name, "InvVariance");
    EXPECT_EQ(invVarT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(invVarT->dims.empty());
    EXPECT_FALSE(invVarT->strides.empty());

    // Verify PrevRunMean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    auto* prevRunMeanT = tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID];
    EXPECT_EQ(prevRunMeanT->name, "PrevRunMean");
    EXPECT_EQ(prevRunMeanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(prevRunMeanT->dims, toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(prevRunMeanT->strides, toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify PrevRunVar tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    auto* prevRunVarT = tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID];
    EXPECT_EQ(prevRunVarT->name, "PrevRunVar");
    EXPECT_EQ(prevRunVarT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(prevRunVarT->dims, toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(prevRunVarT->strides, toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify NextRunMean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    auto* nextRunMeanT = tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID];
    EXPECT_EQ(nextRunMeanT->name, "NextRunMean");
    EXPECT_EQ(nextRunMeanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(nextRunMeanT->dims.empty());
    EXPECT_FALSE(nextRunMeanT->strides.empty());

    // Verify NextRunVar tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    auto* nextRunVarT = tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID];
    EXPECT_EQ(nextRunVarT->name, "NextRunVar");
    EXPECT_EQ(nextRunVarT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(nextRunVarT->dims.empty());
    EXPECT_FALSE(nextRunVarT->strides.empty());

    // -- Verify batchnorm forward operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BatchnormAttributes);

    auto* bnFwd = node->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    EXPECT_EQ(bnFwd->x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(bnFwd->scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(bnFwd->bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(bnFwd->epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(bnFwd->y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);

    // Verify mean and inv_variance are set
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->mean_tensor_uid.value(), K_BATCHNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->inv_variance_tensor_uid.value(), K_BATCHNORM_TENSOR_INV_VARIANCE_UID);

    // Verify running stats are set
    ASSERT_TRUE(bnFwd->prev_running_mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->prev_running_mean_tensor_uid.value(),
              K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    ASSERT_TRUE(bnFwd->prev_running_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->prev_running_variance_tensor_uid.value(),
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    ASSERT_TRUE(bnFwd->momentum_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->momentum_tensor_uid.value(), K_BATCHNORM_TENSOR_MOMENTUM_UID);
    ASSERT_TRUE(bnFwd->next_running_mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->next_running_mean_tensor_uid.value(),
              K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    ASSERT_TRUE(bnFwd->next_running_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->next_running_variance_tensor_uid.value(),
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);

    // No peer stats
    EXPECT_EQ(bnFwd->peer_stats_tensor_uid.size(), 0u);
}

TEST_F(IntegrationBatchnormDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidBnFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BATCHNORM_TENSOR_X_DIMS)).set_stride(toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_name("Epsilon");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_output(true);
    meanOut->set_output(true);
    invVarOut->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // x, scale, bias, epsilon, y, mean, invVariance = 7 tensors
    ASSERT_EQ(graphT.tensors.size(), 7u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 7u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The batchnorm forward operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnFwd = graphT.nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bnFwd->x_tensor_uid) > 0)
        << "X tensor UID " << bnFwd->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnFwd->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bnFwd->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnFwd->bias_tensor_uid) > 0)
        << "Bias tensor UID " << bnFwd->bias_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnFwd->epsilon_tensor_uid) > 0)
        << "Epsilon tensor UID " << bnFwd->epsilon_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnFwd->y_tensor_uid) > 0)
        << "Y tensor UID " << bnFwd->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // Mean and inv_variance should be set with valid UIDs
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(bnFwd->mean_tensor_uid.value()) > 0)
        << "Mean tensor UID " << bnFwd->mean_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(bnFwd->inv_variance_tensor_uid.value()) > 0)
        << "InvVariance tensor UID "
        << bnFwd->inv_variance_tensor_uid.value() // NOLINT(readability-implicit-bool-conversion)
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All seven tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {bnFwd->x_tensor_uid,
                                                  bnFwd->scale_tensor_uid,
                                                  bnFwd->bias_tensor_uid,
                                                  bnFwd->epsilon_tensor_uid,
                                                  bnFwd->y_tensor_uid,
                                                  bnFwd->mean_tensor_uid.value(),
                                                  bnFwd->inv_variance_tensor_uid.value()};
    EXPECT_EQ(nodeUids.size(), 7u)
        << "Batchnorm forward node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

TEST_F(IntegrationBatchnormDescriptorLowering, MinimalRequiredOnlyRoundTrip)
{
    // Tests the absolute minimum: x, scale, bias, epsilon as inputs -> y, mean, invVar as outputs
    // No running stats, no peer stats
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("MinimalBnFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BATCHNORM_MINIMAL_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BATCHNORM_TENSOR_X_DIMS)).set_stride(toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BATCHNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID).set_name("Epsilon");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("minimal_bn").set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_uid(K_BATCHNORM_MINIMAL_TENSOR_Y_UID).set_output(true).set_name("Y");
    meanOut->set_uid(K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID).set_output(true).set_name("Mean");
    invVarOut->set_uid(K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID)
        .set_output(true)
        .set_name("InvVariance");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnFwd = graphT.nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    // Required tensors
    EXPECT_EQ(bnFwd->x_tensor_uid, K_BATCHNORM_MINIMAL_TENSOR_X_UID);
    EXPECT_EQ(bnFwd->scale_tensor_uid, K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID);
    EXPECT_EQ(bnFwd->bias_tensor_uid, K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID);
    EXPECT_EQ(bnFwd->epsilon_tensor_uid, K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID);
    EXPECT_EQ(bnFwd->y_tensor_uid, K_BATCHNORM_MINIMAL_TENSOR_Y_UID);

    // Optional outputs set by the graph's batchnorm() method
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->mean_tensor_uid.value(), K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID);
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->inv_variance_tensor_uid.value(), K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID);

    // No running stats
    EXPECT_FALSE(bnFwd->prev_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->prev_running_variance_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->momentum_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_variance_tensor_uid.has_value());

    // No peer stats
    EXPECT_EQ(bnFwd->peer_stats_tensor_uid.size(), 0u);
}

} // namespace
