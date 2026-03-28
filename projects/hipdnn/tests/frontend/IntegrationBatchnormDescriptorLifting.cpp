// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::fromBackendDescriptor;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

class IntegrationBatchnormDescriptorLifting : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    // Builds a full batchnorm graph with all optional tensors for round-trip testing
    static std::shared_ptr<TestableGraph> buildBatchnormGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("BatchnormLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BATCHNORM_INTEG_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BATCHNORM_INTEG_DATA_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_DATA_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BATCHNORM_INTEG_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(K_BATCHNORM_INTEG_TENSOR_BIAS_UID)
            .set_name("Bias")
            .set_data_type(DataType::FLOAT);
        bias->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

        auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
        epsilon->set_uid(K_BATCHNORM_INTEG_TENSOR_EPSILON_UID).set_name("Epsilon");

        auto prevRunMean = std::make_shared<TensorAttributes>();
        prevRunMean->set_uid(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID)
            .set_name("PrevRunMean")
            .set_data_type(DataType::FLOAT);
        prevRunMean->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

        auto prevRunVar = std::make_shared<TensorAttributes>();
        prevRunVar->set_uid(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID)
            .set_name("PrevRunVar")
            .set_data_type(DataType::FLOAT);
        prevRunVar->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

        auto momentum = std::make_shared<TensorAttributes>(0.1f);
        momentum->set_uid(K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID).set_name("Momentum");

        BatchnormAttributes bnAttrs;
        bnAttrs.set_name("bn_fwd_op");
        bnAttrs.set_epsilon(epsilon);
        bnAttrs.set_previous_running_stats(prevRunMean, prevRunVar, momentum);

        auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
            = graph->batchnorm(x, scale, bias, bnAttrs);
        y->set_uid(K_BATCHNORM_INTEG_TENSOR_Y_UID).set_output(true).set_name("Y");
        meanOut->set_uid(K_BATCHNORM_INTEG_TENSOR_MEAN_UID).set_output(true).set_name("Mean");
        invVarOut->set_uid(K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID)
            .set_output(true)
            .set_name("InvVariance");
        nextRunMean->set_uid(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID)
            .set_output(true)
            .set_name("NextRunMean");
        nextRunVar->set_uid(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID)
            .set_output(true)
            .set_name("NextRunVar");

        return graph;
    }

    // Builds a minimal batchnorm graph with only required tensors (no running stats)
    static std::shared_ptr<TestableGraph> buildMinimalBatchnormGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("MinimalBatchnormLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BATCHNORM_MINIMAL_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BATCHNORM_INTEG_DATA_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_DATA_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID)
            .set_name("Bias")
            .set_data_type(DataType::FLOAT);
        bias->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

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

    hipdnnHandle_t _handle = nullptr;
};

// Builds a full batchnorm graph with all optional tensors, lowers via
// build_operation_graph(handle), lifts back with fromBackendDescriptor(),
// and performs comprehensive field-by-field validation.
TEST_F(IntegrationBatchnormDescriptorLifting, BasicBatchnormRoundTrip)
{
    auto originalGraph = buildBatchnormGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u)
        << "Expected 12 tensors in lifted graph (x, scale, bias, epsilon, "
           "prevRunMean, prevRunVar, momentum, y, mean, invVariance, "
           "nextRunMean, nextRunVar)";

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_BATCHNORM_INTEG_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_uid(), K_BATCHNORM_INTEG_TENSOR_X_UID);
    EXPECT_EQ(liftedX->get_name(), "X");
    EXPECT_EQ(liftedX->get_dim(), toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_BATCHNORM_INTEG_DATA_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_BATCHNORM_INTEG_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_uid(), K_BATCHNORM_INTEG_TENSOR_Y_UID);
    EXPECT_EQ(liftedY->get_name(), "Y");
    EXPECT_EQ(liftedY->get_dim(), toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_BATCHNORM_INTEG_DATA_STRIDES));
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_SCALE_UID), 0u);
    auto liftedScale = tensorMap[K_BATCHNORM_INTEG_TENSOR_SCALE_UID];
    EXPECT_EQ(liftedScale->get_uid(), K_BATCHNORM_INTEG_TENSOR_SCALE_UID);
    EXPECT_EQ(liftedScale->get_name(), "Scale");
    EXPECT_EQ(liftedScale->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedScale->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedScale->get_data_type(), DataType::FLOAT);

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_BIAS_UID), 0u);
    auto liftedBias = tensorMap[K_BATCHNORM_INTEG_TENSOR_BIAS_UID];
    EXPECT_EQ(liftedBias->get_uid(), K_BATCHNORM_INTEG_TENSOR_BIAS_UID);
    EXPECT_EQ(liftedBias->get_name(), "Bias");
    EXPECT_EQ(liftedBias->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedBias->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedBias->get_data_type(), DataType::FLOAT);

    // Verify PrevRunMean tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    auto liftedPrevRunMean = tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID];
    EXPECT_EQ(liftedPrevRunMean->get_uid(), K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(liftedPrevRunMean->get_name(), "PrevRunMean");
    EXPECT_EQ(liftedPrevRunMean->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedPrevRunMean->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedPrevRunMean->get_data_type(), DataType::FLOAT);

    // Verify PrevRunVar tensor
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    auto liftedPrevRunVar = tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID];
    EXPECT_EQ(liftedPrevRunVar->get_uid(), K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(liftedPrevRunVar->get_name(), "PrevRunVar");
    EXPECT_EQ(liftedPrevRunVar->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedPrevRunVar->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedPrevRunVar->get_data_type(), DataType::FLOAT);

    // Verify Epsilon tensor (scalar): pass-by-value with actual value preserved
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_EPSILON_UID), 0u);
    auto liftedEpsilon = tensorMap[K_BATCHNORM_INTEG_TENSOR_EPSILON_UID];
    EXPECT_EQ(liftedEpsilon->get_uid(), K_BATCHNORM_INTEG_TENSOR_EPSILON_UID);
    EXPECT_EQ(liftedEpsilon->get_name(), "Epsilon");
    EXPECT_EQ(liftedEpsilon->get_dim(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedEpsilon->get_stride(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedEpsilon->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(liftedEpsilon->get_pass_by_value());
    ASSERT_TRUE(liftedEpsilon->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedEpsilon->get_pass_by_value<float>().value(), 1e-5f);

    // Verify Momentum tensor (scalar): pass-by-value with actual value preserved
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID), 0u);
    auto liftedMomentum = tensorMap[K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID];
    EXPECT_EQ(liftedMomentum->get_uid(), K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(liftedMomentum->get_name(), "Momentum");
    EXPECT_EQ(liftedMomentum->get_dim(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedMomentum->get_stride(), std::vector<int64_t>{1});
    EXPECT_EQ(liftedMomentum->get_data_type(), DataType::FLOAT);
    EXPECT_TRUE(liftedMomentum->get_pass_by_value());
    ASSERT_TRUE(liftedMomentum->get_pass_by_value<float>().has_value());
    EXPECT_FLOAT_EQ(liftedMomentum->get_pass_by_value<float>().value(), 0.1f);

    // Verify Mean tensor (inferred dims)
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_MEAN_UID), 0u);
    auto liftedMean = tensorMap[K_BATCHNORM_INTEG_TENSOR_MEAN_UID];
    EXPECT_EQ(liftedMean->get_uid(), K_BATCHNORM_INTEG_TENSOR_MEAN_UID);
    EXPECT_EQ(liftedMean->get_name(), "Mean");
    EXPECT_FALSE(liftedMean->get_dim().empty());
    EXPECT_FALSE(liftedMean->get_stride().empty());

    // Verify InvVariance tensor (inferred dims)
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID), 0u);
    auto liftedInvVar = tensorMap[K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(liftedInvVar->get_uid(), K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(liftedInvVar->get_name(), "InvVariance");
    EXPECT_FALSE(liftedInvVar->get_dim().empty());
    EXPECT_FALSE(liftedInvVar->get_stride().empty());

    // Verify NextRunMean tensor (inferred dims)
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    auto liftedNextRunMean = tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID];
    EXPECT_EQ(liftedNextRunMean->get_uid(), K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(liftedNextRunMean->get_name(), "NextRunMean");
    EXPECT_FALSE(liftedNextRunMean->get_dim().empty());
    EXPECT_FALSE(liftedNextRunMean->get_stride().empty());

    // Verify NextRunVar tensor (inferred dims)
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    auto liftedNextRunVar = tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID];
    EXPECT_EQ(liftedNextRunVar->get_uid(), K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(liftedNextRunVar->get_name(), "NextRunVar");
    EXPECT_FALSE(liftedNextRunVar->get_dim().empty());
    EXPECT_FALSE(liftedNextRunVar->get_stride().empty());

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr) << "Expected a BatchnormNode";

    // Verify operation name and compute data type
    EXPECT_EQ(bnNode->attributes.get_name(), "bn_fwd_op");
    EXPECT_EQ(bnNode->attributes.compute_data_type, DataType::FLOAT);
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormTensorSharingPreserved)
{
    auto originalGraph = buildBatchnormGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_X_UID].get(), bnNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_SCALE_UID].get(),
              bnNode->attributes.get_scale().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_BIAS_UID].get(),
              bnNode->attributes.get_bias().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_EPSILON_UID].get(),
              bnNode->attributes.get_epsilon().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_Y_UID].get(), bnNode->attributes.get_y().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_MEAN_UID].get(),
              bnNode->attributes.get_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID].get(),
              bnNode->attributes.get_inv_variance().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID].get(),
              bnNode->attributes.get_prev_running_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID].get(),
              bnNode->attributes.get_prev_running_variance().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID].get(),
              bnNode->attributes.get_momentum().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID].get(),
              bnNode->attributes.get_next_running_mean().get());
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID].get(),
              bnNode->attributes.get_next_running_variance().get());
}

// Builds a full batchnorm graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all batchnorm fields survive the FlatBuffer-direct path.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormLiftWithoutFinalization)
{
    auto originalGraph = buildBatchnormGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph via fromBackendDescriptor
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensor count
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u) << "Expected 12 tensors in lifted graph";

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr);

    // Verify operation name and compute data type
    EXPECT_EQ(bnNode->attributes.get_name(), "bn_fwd_op");
    EXPECT_EQ(bnNode->attributes.compute_data_type, DataType::FLOAT);

    // Verify key tensor dims and names
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_X_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_X_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_DATA_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_X_UID]->get_name(), "X");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_Y_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_Y_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_DATA_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_Y_UID]->get_name(), "Y");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_SCALE_UID]->get_name(), "Scale");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_BIAS_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_BIAS_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_BIAS_UID]->get_name(), "Bias");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_EPSILON_UID]->get_name(), "Epsilon");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID]->get_name(), "Momentum");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID]->get_name(), "PrevRunMean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID]->get_name(),
              "PrevRunVar");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_MEAN_UID]->get_name(), "Mean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID]->get_name(), "InvVariance");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID]->get_name(), "NextRunMean");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID]->get_name(),
              "NextRunVar");
}

// Builds a minimal batchnorm graph (required tensors only + mean/invVariance,
// no running stats), lowers, lifts, and verifies optional running stats are absent.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormMinimalRequiredTensorsRoundTrip)
{
    auto originalGraph = buildMinimalBatchnormGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u)
        << "Expected 7 tensors (x, scale, bias, epsilon, y, mean, invVariance)";

    // Verify required tensor UIDs, names, dims, strides
    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_uid(),
              K_BATCHNORM_MINIMAL_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_name(), "X");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_X_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_DATA_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_name(), "Scale");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_name(), "Bias");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID]->get_name(), "Epsilon");

    ASSERT_NE(tensorMap.count(K_BATCHNORM_MINIMAL_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_name(), "Y");
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_dim(),
              toVec(K_BATCHNORM_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BATCHNORM_MINIMAL_TENSOR_Y_UID]->get_stride(),
              toVec(K_BATCHNORM_INTEG_DATA_STRIDES));

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
    ASSERT_NE(bnNode, nullptr) << "Expected a BatchnormNode";

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

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnNode = dynamic_cast<BatchnormNode*>(subNodes[0].get());
    ASSERT_NE(bnNode, nullptr) << "Expected a BatchnormNode";

    // Verify prev_running_mean
    auto prevRunMean = bnNode->attributes.get_prev_running_mean();
    ASSERT_NE(prevRunMean, nullptr) << "prev_running_mean should not be null";
    EXPECT_EQ(prevRunMean->get_uid(), K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(prevRunMean->get_name(), "PrevRunMean");
    EXPECT_EQ(prevRunMean->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(prevRunMean->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    // Verify prev_running_variance
    auto prevRunVar = bnNode->attributes.get_prev_running_variance();
    ASSERT_NE(prevRunVar, nullptr) << "prev_running_variance should not be null";
    EXPECT_EQ(prevRunVar->get_uid(), K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(prevRunVar->get_name(), "PrevRunVar");
    EXPECT_EQ(prevRunVar->get_dim(), toVec(K_BATCHNORM_INTEG_PARAM_DIMS));
    EXPECT_EQ(prevRunVar->get_stride(), toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    // Verify momentum (scalar)
    auto momentumTensor = bnNode->attributes.get_momentum();
    ASSERT_NE(momentumTensor, nullptr) << "momentum should not be null";
    EXPECT_EQ(momentumTensor->get_uid(), K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(momentumTensor->get_name(), "Momentum");

    // Verify next_running_mean (inferred dims)
    auto nextRunMean = bnNode->attributes.get_next_running_mean();
    ASSERT_NE(nextRunMean, nullptr) << "next_running_mean should not be null";
    EXPECT_EQ(nextRunMean->get_uid(), K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(nextRunMean->get_name(), "NextRunMean");
    EXPECT_FALSE(nextRunMean->get_dim().empty());
    EXPECT_FALSE(nextRunMean->get_stride().empty());

    // Verify next_running_variance (inferred dims)
    auto nextRunVar = bnNode->attributes.get_next_running_variance();
    ASSERT_NE(nextRunVar, nullptr) << "next_running_variance should not be null";
    EXPECT_EQ(nextRunVar->get_uid(), K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(nextRunVar->get_name(), "NextRunVar");
    EXPECT_FALSE(nextRunVar->get_dim().empty());
    EXPECT_FALSE(nextRunVar->get_stride().empty());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationBatchnormDescriptorLifting, BatchnormAutoAssignedUidsPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
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

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 7u)
        << "Expected 7 tensors (x, scale, bias, epsilon, y, mean, invVariance)";

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "All auto-assigned UIDs must be distinct";

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
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PeerStatsBatchnormLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BATCHNORM_INTEG_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BATCHNORM_INTEG_DATA_DIMS))
        .set_stride(toVec(K_BATCHNORM_INTEG_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BATCHNORM_INTEG_TENSOR_SCALE_UID)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_BATCHNORM_INTEG_TENSOR_BIAS_UID)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_BATCHNORM_INTEG_TENSOR_EPSILON_UID).set_name("Epsilon");

    auto prevRunMean = std::make_shared<TensorAttributes>();
    prevRunMean->set_uid(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_MEAN_UID)
        .set_name("PrevRunMean")
        .set_data_type(DataType::FLOAT);
    prevRunMean->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    auto prevRunVar = std::make_shared<TensorAttributes>();
    prevRunVar->set_uid(K_BATCHNORM_INTEG_TENSOR_PREV_RUNNING_VARIANCE_UID)
        .set_name("PrevRunVar")
        .set_data_type(DataType::FLOAT);
    prevRunVar->set_dim(toVec(K_BATCHNORM_INTEG_PARAM_DIMS))
        .set_stride(toVec(K_BATCHNORM_INTEG_PARAM_STRIDES));

    auto momentum = std::make_shared<TensorAttributes>(0.1f);
    momentum->set_uid(K_BATCHNORM_INTEG_TENSOR_MOMENTUM_UID).set_name("Momentum");

    // Create peer_stats tensors
    auto peerStat0 = std::make_shared<TensorAttributes>();
    peerStat0->set_uid(K_BATCHNORM_INTEG_TENSOR_PEER_STAT_0_UID)
        .set_name("PeerStat0")
        .set_data_type(DataType::FLOAT);
    peerStat0->set_dim(toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS))
        .set_stride(toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(K_BATCHNORM_INTEG_TENSOR_PEER_STAT_1_UID)
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
    y->set_uid(K_BATCHNORM_INTEG_TENSOR_Y_UID).set_output(true).set_name("Y");
    meanOut->set_uid(K_BATCHNORM_INTEG_TENSOR_MEAN_UID).set_output(true).set_name("Mean");
    invVarOut->set_uid(K_BATCHNORM_INTEG_TENSOR_INV_VARIANCE_UID)
        .set_output(true)
        .set_name("InvVariance");
    nextRunMean->set_uid(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_MEAN_UID)
        .set_output(true)
        .set_name("NextRunMean");
    nextRunVar->set_uid(K_BATCHNORM_INTEG_TENSOR_NEXT_RUNNING_VARIANCE_UID)
        .set_output(true)
        .set_name("NextRunVar");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify tensor map includes the 2 peer_stats tensors (12 base + 2 peer = 14)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 14u) << "Expected 14 tensors (12 base + 2 peer_stats)";

    // Verify peer_stats tensors appear in the tensor map with correct UIDs and names
    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PEER_STAT_0_UID), 0u);
    auto liftedPeerStat0 = tensorMap[K_BATCHNORM_INTEG_TENSOR_PEER_STAT_0_UID];
    EXPECT_EQ(liftedPeerStat0->get_uid(), K_BATCHNORM_INTEG_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStat0->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStat0->get_dim(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS));
    EXPECT_EQ(liftedPeerStat0->get_stride(), toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
    EXPECT_EQ(liftedPeerStat0->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_BATCHNORM_INTEG_TENSOR_PEER_STAT_1_UID), 0u);
    auto liftedPeerStat1 = tensorMap[K_BATCHNORM_INTEG_TENSOR_PEER_STAT_1_UID];
    EXPECT_EQ(liftedPeerStat1->get_uid(), K_BATCHNORM_INTEG_TENSOR_PEER_STAT_1_UID);
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
    ASSERT_EQ(liftedPeerStats.size(), 2u) << "Expected 2 peer_stats tensors in lifted node";

    EXPECT_EQ(liftedPeerStats[0]->get_uid(), K_BATCHNORM_INTEG_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStats[0]->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStats[1]->get_uid(), K_BATCHNORM_INTEG_TENSOR_PEER_STAT_1_UID);
    EXPECT_EQ(liftedPeerStats[1]->get_name(), "PeerStat1");

    // Verify pointer equality: peer_stats in node attributes share objects with tensor map
    EXPECT_EQ(liftedPeerStats[0].get(), liftedPeerStat0.get());
    EXPECT_EQ(liftedPeerStats[1].get(), liftedPeerStat1.get());
}

} // namespace
