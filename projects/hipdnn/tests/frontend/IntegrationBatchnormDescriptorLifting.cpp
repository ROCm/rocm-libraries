// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
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
class IntegrationBatchnormDescriptorLifting : public IntegrationTestFixture
{
protected:
    // Builds a full batchnorm graph with all optional tensors for round-trip testing
    static std::shared_ptr<TestableGraphLifting> buildBatchnormGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("BatchnormLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BATCHNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BATCHNORM_TENSOR_X_DIMS))
            .set_stride(toVec(K_BATCHNORM_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BATCHNORM_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
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

        return graph;
    }

    // Builds a minimal batchnorm graph with only required tensors (no running stats)
    static std::shared_ptr<TestableGraphLifting> buildMinimalBatchnormGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("MinimalBatchnormLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BATCHNORM_MINIMAL_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BATCHNORM_TENSOR_X_DIMS))
            .set_stride(toVec(K_BATCHNORM_TENSOR_X_STRIDES));

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
        bnAttrs.set_name("minimal_bn_op").set_epsilon(epsilon);

        auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
            = graph->batchnorm(x, scale, bias, bnAttrs);
        y->set_uid(K_BATCHNORM_MINIMAL_TENSOR_Y_UID).set_output(true).set_name("Y");
        meanOut->set_uid(K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID).set_output(true).set_name("Mean");
        invVarOut->set_uid(K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID)
            .set_output(true)
            .set_name("InvVariance");
        // nextRunMean and nextRunVar are nullptr when no running stats

        return graph;
    }
};

// Builds a full batchnorm graph with all optional tensors, lowers via
// build_operation_graph(handle), lifts back with fromBackendDescriptor(),
// and performs comprehensive field-by-field validation.
TEST_F(IntegrationBatchnormDescriptorLifting, BasicBatchnormRoundTrip)
{
    auto originalGraph = buildBatchnormGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u) << "Expected 12 tensors in lifted graph (x, scale, bias, "
                                        "epsilon, " // NOLINT(readability-implicit-bool-conversion)
                                        "prevRunMean, prevRunVar, momentum, y, mean, invVariance, "
                                        "nextRunMean, nextRunVar)";

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_BATCHNORM_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_uid(), K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(liftedX->get_name(), "X");
    EXPECT_EQ(liftedX->get_dim(), toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_BATCHNORM_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_uid(), K_BATCHNORM_TENSOR_Y_UID);
    EXPECT_EQ(liftedY->get_name(), "Y");
    EXPECT_EQ(liftedY->get_dim(), toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_SCALE_UID), 0u);
    auto liftedScale = tensorMap[K_BATCHNORM_TENSOR_SCALE_UID];
    EXPECT_EQ(liftedScale->get_uid(), K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(liftedScale->get_name(), "Scale");
    EXPECT_EQ(liftedScale->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(liftedScale->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(liftedScale->get_data_type(), DataType::FLOAT);

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_BIAS_UID), 0u);
    auto liftedBias = tensorMap[K_BATCHNORM_TENSOR_BIAS_UID];
    EXPECT_EQ(liftedBias->get_uid(), K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(liftedBias->get_name(), "Bias");
    EXPECT_EQ(liftedBias->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(liftedBias->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(liftedBias->get_data_type(), DataType::FLOAT);

    // Verify PrevRunMean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    auto liftedPrevRunMean = tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID];
    EXPECT_EQ(liftedPrevRunMean->get_uid(), K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(liftedPrevRunMean->get_name(), "PrevRunMean");
    EXPECT_EQ(liftedPrevRunMean->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(liftedPrevRunMean->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(liftedPrevRunMean->get_data_type(), DataType::FLOAT);

    // Verify PrevRunVar tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    auto liftedPrevRunVar = tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID];
    EXPECT_EQ(liftedPrevRunVar->get_uid(), K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(liftedPrevRunVar->get_name(), "PrevRunVar");
    EXPECT_EQ(liftedPrevRunVar->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(liftedPrevRunVar->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(liftedPrevRunVar->get_data_type(), DataType::FLOAT);

    // Verify Epsilon tensor (scalar): pass-by-value with actual value preserved
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_EPSILON_UID), 0u);
    auto liftedEpsilon = tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID];
    EXPECT_EQ(liftedEpsilon->get_uid(), K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(liftedEpsilon->get_name(), "Epsilon");
    EXPECT_EQ(liftedEpsilon->get_dim(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedEpsilon->get_stride(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedEpsilon->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(liftedEpsilon->get_pass_by_value());
    ASSERT_TRUE(liftedEpsilon->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedEpsilon->get_pass_by_value<float>().value(), 1e-5f);

    // Verify Momentum tensor (scalar): pass-by-value with actual value preserved
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MOMENTUM_UID), 0u);
    auto liftedMomentum = tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID];
    EXPECT_EQ(liftedMomentum->get_uid(), K_BATCHNORM_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(liftedMomentum->get_name(), "Momentum");
    EXPECT_EQ(liftedMomentum->get_dim(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedMomentum->get_stride(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedMomentum->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(liftedMomentum->get_pass_by_value());
    ASSERT_TRUE(liftedMomentum->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedMomentum->get_pass_by_value<float>().value(), 0.1f);

    // Verify Mean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MEAN_UID), 0u);
    auto liftedMean = tensorMap[K_BATCHNORM_TENSOR_MEAN_UID];
    EXPECT_EQ(liftedMean->get_uid(), K_BATCHNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(liftedMean->get_name(), "Mean");
    EXPECT_EQ(liftedMean->get_dim(), toVec(K_BATCHNORM_TENSOR_MEAN_DIMS));
    EXPECT_EQ(liftedMean->get_stride(), toVec(K_BATCHNORM_TENSOR_MEAN_STRIDES));
    EXPECT_EQ(liftedMean->get_data_type(), DataType::FLOAT);

    // Verify InvVariance tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_INV_VARIANCE_UID), 0u);
    auto liftedInvVar = tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(liftedInvVar->get_uid(), K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(liftedInvVar->get_name(), "InvVariance");
    EXPECT_EQ(liftedInvVar->get_dim(), toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS));
    EXPECT_EQ(liftedInvVar->get_stride(), toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES));
    EXPECT_EQ(liftedInvVar->get_data_type(), DataType::FLOAT);

    // Verify NextRunMean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    auto liftedNextRunMean = tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID];
    EXPECT_EQ(liftedNextRunMean->get_uid(), K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(liftedNextRunMean->get_name(), "NextRunMean");
    EXPECT_EQ(liftedNextRunMean->get_dim(), toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS));
    EXPECT_EQ(liftedNextRunMean->get_stride(), toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES));
    EXPECT_EQ(liftedNextRunMean->get_data_type(), DataType::FLOAT);

    // Verify NextRunVar tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    auto liftedNextRunVar = tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID];
    EXPECT_EQ(liftedNextRunVar->get_uid(), K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(liftedNextRunVar->get_name(), "NextRunVar");
    EXPECT_EQ(liftedNextRunVar->get_dim(), toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS));
    EXPECT_EQ(liftedNextRunVar->get_stride(),
              toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES));
    EXPECT_EQ(liftedNextRunVar->get_data_type(), DataType::FLOAT);

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr)
        << "Expected a BatchnormNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name and compute data type
    EXPECT_EQ(bnNode->attributes.get_name(), "bn_fwd_op");
    EXPECT_EQ(bnNode->attributes.compute_data_type, DataType::FLOAT);
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormTensorSharingPreserved)
{
    auto originalGraph = buildBatchnormGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_X_UID].get(), bnNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_SCALE_UID].get(), bnNode->attributes.get_scale().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_BIAS_UID].get(), bnNode->attributes.get_bias().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID].get(),
              bnNode->attributes.get_epsilon().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_Y_UID].get(), bnNode->attributes.get_y().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_MEAN_UID].get(), bnNode->attributes.get_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID].get(),
              bnNode->attributes.get_inv_variance().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID].get(),
              bnNode->attributes.get_prev_running_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID].get(),
              bnNode->attributes.get_prev_running_variance().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID].get(),
              bnNode->attributes.get_momentum().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID].get(),
              bnNode->attributes.get_next_running_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID].get(),
              bnNode->attributes.get_next_running_variance().get());
}

// Builds a full batchnorm graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all batchnorm fields survive the FlatBuffer-direct path.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormLiftWithoutFinalization)
{
    auto originalGraph = buildBatchnormGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensor count
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u)
        << "Expected 12 tensors in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    // Verify operation name and compute data type
    EXPECT_EQ(bnNode->attributes.get_name(), "bn_fwd_op");
    EXPECT_EQ(bnNode->attributes.compute_data_type, DataType::FLOAT);

    // Verify key tensor dims and names
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_X_UID]->get_dim(), toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_X_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_X_UID]->get_name(), "X");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_Y_UID]->get_dim(), toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_Y_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_Y_UID]->get_name(), "Y");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_SCALE_UID]->get_name(), "Scale");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_BIAS_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_BIAS_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_BIAS_UID]->get_name(), "Bias");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID]->get_name(), "Epsilon");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MOMENTUM_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID]->get_name(), "Momentum");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID]->get_name(), "PrevRunMean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID]->get_name(), "PrevRunVar");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_MEAN_UID]->get_name(), "Mean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID]->get_name(), "InvVariance");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID]->get_name(), "NextRunMean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID]->get_name(), "NextRunVar");
}

// Builds a minimal batchnorm graph (required tensors only + mean/invVariance,
// no running stats), lowers, lifts, and verifies optional running stats are absent.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormMinimalRequiredTensorsRoundTrip)
{
    auto originalGraph = buildMinimalBatchnormGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u)
        << "Expected 7 tensors (x, scale, bias, epsilon, y, mean, "
           "invVariance)"; // NOLINT(readability-implicit-bool-conversion)

    // Verify required tensor UIDs, names, dims, strides
    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_uid(),
              K_BATCHNORM_MINIMAL_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_name(), "X");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_name(), "Scale");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_name(), "Bias");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID]->get_name(), "Epsilon");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_name(), "Y");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_dim(),
              toVec(K_BATCHNORM_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_stride(),
              toVec(K_BATCHNORM_TENSOR_X_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID]->get_name(), "Mean");
    EXPECT_FALSE(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID]->get_dim().empty());

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID]->get_name(), "InvVariance");
    EXPECT_FALSE(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID]->get_dim().empty());

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr)
        << "Expected a BatchnormNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(bnNode->attributes.get_name(), "minimal_bn_op");

    // Verify optional running stats tensors are absent
    EXPECT_EQ(bnNode->attributes.get_prev_running_mean(), nullptr);
    EXPECT_EQ(bnNode->attributes.get_prev_running_variance(), nullptr);
    EXPECT_EQ(bnNode->attributes.get_momentum(), nullptr);
    EXPECT_EQ(bnNode->attributes.get_next_running_mean(), nullptr);
    EXPECT_EQ(bnNode->attributes.get_next_running_variance(), nullptr);
}

// Full round-trip, then verifies all 5 running stats tensors survive lifting
// with correct UIDs, names, and dimensions.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormRunningStatsPreserved)
{
    auto originalGraph = buildBatchnormGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr)
        << "Expected a BatchnormNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify prev_running_mean
    auto prevRunMean = bnNode->attributes.get_prev_running_mean();
    ASSERT_NE(prevRunMean, nullptr) << "prev_running_mean should not be null";
    EXPECT_EQ(prevRunMean->get_uid(), K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(prevRunMean->get_name(), "PrevRunMean");
    EXPECT_EQ(prevRunMean->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(prevRunMean->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify prev_running_variance
    auto prevRunVar = bnNode->attributes.get_prev_running_variance();
    ASSERT_NE(prevRunVar, nullptr) << "prev_running_variance should not be null";
    EXPECT_EQ(prevRunVar->get_uid(), K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(prevRunVar->get_name(), "PrevRunVar");
    EXPECT_EQ(prevRunVar->get_dim(), toVec(K_BATCHNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(prevRunVar->get_stride(), toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));

    // Verify momentum (scalar)
    auto momentumTensor = bnNode->attributes.get_momentum();
    ASSERT_NE(momentumTensor, nullptr) << "momentum should not be null";
    EXPECT_EQ(momentumTensor->get_uid(), K_BATCHNORM_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(momentumTensor->get_name(), "Momentum");

    // Verify next_running_mean (inferred dims)
    auto nextRunMean = bnNode->attributes.get_next_running_mean();
    ASSERT_NE(nextRunMean, nullptr) << "next_running_mean should not be null";
    EXPECT_EQ(nextRunMean->get_uid(), K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(nextRunMean->get_name(), "NextRunMean");
    EXPECT_FALSE(nextRunMean->get_dim().empty());
    EXPECT_FALSE(nextRunMean->get_stride().empty());

    // Verify next_running_variance (inferred dims)
    auto nextRunVar = bnNode->attributes.get_next_running_variance();
    ASSERT_NE(nextRunVar, nullptr) << "next_running_variance should not be null";
    EXPECT_EQ(nextRunVar->get_uid(), K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(nextRunVar->get_name(), "NextRunVar");
    EXPECT_FALSE(nextRunVar->get_dim().empty());
    EXPECT_FALSE(nextRunVar->get_stride().empty());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormAutoAssignedUidsPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("AutoUidBatchnormLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Build tensors with distinct dims but NO set_uid() calls
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BATCHNORM_AUTO_DATA_DIMS)).set_stride(toVec(K_BATCHNORM_AUTO_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BATCHNORM_AUTO_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_AUTO_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BATCHNORM_AUTO_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_AUTO_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_name("Epsilon");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("auto_uid_bn_op").set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_output(true).set_name("Y");
    meanOut->set_output(true).set_name("Mean");
    invVarOut->set_output(true).set_name("InvVariance");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u)
        << "Expected 7 tensors (x, scale, bias, epsilon, y, mean, "
           "invVariance)"; // NOLINT(readability-implicit-bool-conversion)

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

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    // Verify tensor dims survived the round trip by identifying tensors via dims
    auto xUid = bnNode->attributes.get_x()->get_uid();
    auto scaleUid = bnNode->attributes.get_scale()->get_uid();
    auto yUid = bnNode->attributes.get_y()->get_uid();

    EXPECT_NE(xUid, scaleUid);
    EXPECT_NE(xUid, yUid);

    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_BATCHNORM_AUTO_DATA_DIMS));
    EXPECT_EQ(tensorMap[xUid]->get_stride(), toVec(K_BATCHNORM_AUTO_DATA_STRIDES));
    EXPECT_EQ(tensorMap[scaleUid]->get_dim(), toVec(K_BATCHNORM_AUTO_PARAM_DIMS));
    EXPECT_EQ(tensorMap[scaleUid]->get_stride(), toVec(K_BATCHNORM_AUTO_PARAM_STRIDES));
    EXPECT_EQ(tensorMap[yUid]->get_dim(), toVec(K_BATCHNORM_AUTO_DATA_DIMS));
    EXPECT_EQ(tensorMap[yUid]->get_stride(), toVec(K_BATCHNORM_AUTO_DATA_STRIDES));

    // Verify epsilon pass-by-value scalar survived the round trip
    auto epsilonTensor = bnNode->attributes.get_epsilon();
    ASSERT_NE(epsilonTensor, nullptr);
    EXPECT_TRUE(epsilonTensor->get_pass_by_value());
    ASSERT_TRUE(epsilonTensor->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(epsilonTensor->get_pass_by_value<float>().value(), 1e-5f);
}

// Builds a batchnorm graph with peer_stats tensors, performs a full round-trip,
// and verifies peer_stats appear in the lifted tensor map and node attributes.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormPeerStatsPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PeerStatsBatchnormLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

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

    // Create peer_stats tensors
    auto peerStat0 = std::make_shared<TensorAttributes>();
    peerStat0->set_uid(K_BATCHNORM_TENSOR_PEER_STAT_0_UID)
        .set_name("PeerStat0")
        .set_data_type(DataType::FLOAT);
    peerStat0->set_dim(toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(K_BATCHNORM_TENSOR_PEER_STAT_1_UID)
        .set_name("PeerStat1")
        .set_data_type(DataType::FLOAT);
    peerStat1->set_dim(toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("peer_stats_bn_op");
    bnAttrs.set_epsilon(epsilon);
    bnAttrs.set_previous_running_stats(prevRunMean, prevRunVar, momentum);
    bnAttrs.set_peer_stats({peerStat0, peerStat1});

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

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify tensor map includes the 2 peer_stats tensors (12 base + 2 peer = 14)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 14u)
        << "Expected 14 tensors (12 base + 2 peer_stats)"; // NOLINT(readability-implicit-bool-conversion)

    // Verify peer_stats tensors appear in the tensor map with correct UIDs and names
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PEER_STAT_0_UID), 0u);
    auto liftedPeerStat0 = tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_0_UID];
    EXPECT_EQ(liftedPeerStat0->get_uid(), K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStat0->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStat0->get_dim(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS));
    EXPECT_EQ(liftedPeerStat0->get_stride(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
    EXPECT_EQ(liftedPeerStat0->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PEER_STAT_1_UID), 0u);
    auto liftedPeerStat1 = tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_1_UID];
    EXPECT_EQ(liftedPeerStat1->get_uid(), K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
    EXPECT_EQ(liftedPeerStat1->get_name(), "PeerStat1");
    EXPECT_EQ(liftedPeerStat1->get_dim(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS));
    EXPECT_EQ(liftedPeerStat1->get_stride(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
    EXPECT_EQ(liftedPeerStat1->get_data_type(), DataType::FLOAT);

    // Verify the lifted node's attributes reference peer_stats correctly
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    const auto& liftedPeerStats = bnNode->attributes.get_peer_stats();
    ASSERT_EQ(liftedPeerStats.size(), 2u)
        << "Expected 2 peer_stats tensors in lifted node"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(liftedPeerStats[0]->get_uid(), K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStats[0]->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStats[1]->get_uid(), K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
    EXPECT_EQ(liftedPeerStats[1]->get_name(), "PeerStat1");

    // Verify pointer equality: peer_stats in node attributes share objects with tensor map
    EXPECT_EQ(liftedPeerStats[0].get(), liftedPeerStat0.get());
    EXPECT_EQ(liftedPeerStats[1].get(), liftedPeerStat1.get());
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a full batchnorm training graph with running stats.
TEST_F(IntegrationBatchnormDescriptorLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildBatchnormGraph();

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
    EXPECT_EQ(liftedGraph->get_name(), "BatchnormLiftingTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensor count (x, scale, bias, epsilon, prevRunMean, prevRunVar, momentum,
    //                       y, mean, invVariance, nextRunMean, nextRunVar = 12)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u) << "Expected 12 tensors in lifted batchnorm graph";

    // Verify key tensors
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_X_UID,
                                      "X",
                                      toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_X_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_SCALE_UID,
                                      "Scale",
                                      toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_BIAS_UID,
                                      "Bias",
                                      toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_Y_UID,
                                      "Y",
                                      toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_X_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_MEAN_UID,
                                      "Mean",
                                      toVec(K_BATCHNORM_TENSOR_MEAN_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_MEAN_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_BATCHNORM_TENSOR_INV_VARIANCE_UID,
                                      "InvVariance",
                                      toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS),
                                      toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES),
                                      DataType::FLOAT);

    // Verify running stats tensors are present
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID]->get_name(), "PrevRunMean");
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID]->get_name(), "PrevRunVar");
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID]->get_name(), "NextRunMean");
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID]->get_name(), "NextRunVar");

    // Verify scalar tensors (epsilon, momentum)
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID]->get_name(), "Epsilon");
    ASSERT_NE(tensorMap.count(K_BATCHNORM_TENSOR_MOMENTUM_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID]->get_name(), "Momentum");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr) << "Expected a BatchnormNode";

    EXPECT_EQ(bnNode->attributes.get_name(), "bn_fwd_op");
    EXPECT_EQ(bnNode->attributes.compute_data_type, DataType::FLOAT);
}

} // namespace
