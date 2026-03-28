// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_test_sdk/constants/BatchnormBackwardConstants.hpp>
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
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

class IntegrationBatchnormBackwardDescriptorLifting : public ::testing::Test
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

    // Builds a batchnorm backward graph with optional mean/invVariance for round-trip testing
    static std::shared_ptr<TestableGraph> buildBatchnormBackwardGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("BnBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_BN_BWD_INTEG_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BN_BWD_INTEG_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BN_BWD_INTEG_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(K_BN_BWD_INTEG_TENSOR_MEAN_UID)
            .set_name("Mean")
            .set_data_type(DataType::FLOAT);
        mean->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto invVar = std::make_shared<TensorAttributes>();
        invVar->set_uid(K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID)
            .set_name("InvVariance")
            .set_data_type(DataType::FLOAT);
        invVar->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("bn_bwd_op");
        bnBwdAttrs.set_saved_mean_and_inv_variance(mean, invVar);

        auto [dxOut, dscaleOut, dbiasOut] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
        dxOut->set_uid(K_BN_BWD_INTEG_TENSOR_DX_UID).set_output(true).set_name("DX");
        dscaleOut->set_uid(K_BN_BWD_INTEG_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
        dbiasOut->set_uid(K_BN_BWD_INTEG_TENSOR_DBIAS_UID).set_output(true).set_name("DBias");

        return graph;
    }

    // Builds a batchnorm backward graph with peer_stats tensors for distributed testing
    static std::shared_ptr<TestableGraph> buildBatchnormBackwardGraphWithPeerStats()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("BnBwdPeerStatsLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_BN_BWD_INTEG_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BN_BWD_INTEG_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BN_BWD_INTEG_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(K_BN_BWD_INTEG_TENSOR_MEAN_UID)
            .set_name("Mean")
            .set_data_type(DataType::FLOAT);
        mean->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto invVar = std::make_shared<TensorAttributes>();
        invVar->set_uid(K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID)
            .set_name("InvVariance")
            .set_data_type(DataType::FLOAT);
        invVar->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto peerStat0 = std::make_shared<TensorAttributes>();
        peerStat0->set_uid(K_BN_BWD_INTEG_TENSOR_PEER_STAT_0_UID)
            .set_name("PeerStat0")
            .set_data_type(DataType::FLOAT);
        peerStat0->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        auto peerStat1 = std::make_shared<TensorAttributes>();
        peerStat1->set_uid(K_BN_BWD_INTEG_TENSOR_PEER_STAT_1_UID)
            .set_name("PeerStat1")
            .set_data_type(DataType::FLOAT);
        peerStat1->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("bn_bwd_peer_stats_op");
        bnBwdAttrs.set_saved_mean_and_inv_variance(mean, invVar);
        bnBwdAttrs.set_peer_stats({peerStat0, peerStat1});

        auto [dxOut, dscaleOut, dbiasOut] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
        dxOut->set_uid(K_BN_BWD_INTEG_TENSOR_DX_UID).set_output(true).set_name("DX");
        dscaleOut->set_uid(K_BN_BWD_INTEG_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
        dbiasOut->set_uid(K_BN_BWD_INTEG_TENSOR_DBIAS_UID).set_output(true).set_name("DBias");

        return graph;
    }

    // Builds a minimal batchnorm backward graph (no optional mean/invVariance)
    static std::shared_ptr<TestableGraph> buildMinimalBatchnormBackwardGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("MinimalBnBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_BN_BWD_MINIMAL_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BN_BWD_MINIMAL_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BN_BWD_INTEG_DATA_DIMS)).set_stride(toVec(K_BN_BWD_INTEG_DATA_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BN_BWD_MINIMAL_TENSOR_SCALE_UID)
            .set_name("Scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BN_BWD_INTEG_PARAM_DIMS))
            .set_stride(toVec(K_BN_BWD_INTEG_PARAM_STRIDES));

        BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_name("minimal_bn_bwd_op");

        auto [dxOut, dscaleOut, dbiasOut] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
        dxOut->set_uid(K_BN_BWD_MINIMAL_TENSOR_DX_UID).set_output(true).set_name("DX");
        dscaleOut->set_uid(K_BN_BWD_MINIMAL_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
        dbiasOut->set_uid(K_BN_BWD_MINIMAL_TENSOR_DBIAS_UID).set_output(true).set_name("DBias");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a batchnorm backward graph with mean/invVariance, lowers via
// build_operation_graph(handle), lifts back with fromBackendDescriptor(),
// and validates all tensors and operation attributes.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting, BasicBatchnormBackwardRoundTrip)
{
    auto originalGraph = buildBatchnormBackwardGraph();

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
    // dy, x, scale, mean, invVar, dx, dscale, dbias = 8
    ASSERT_EQ(tensorMap.size(), 8u) << "Expected 8 tensors in lifted graph";

    // DY tensor
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DY_UID), 0u);
    auto liftedDy = tensorMap[K_BN_BWD_INTEG_TENSOR_DY_UID];
    EXPECT_EQ(liftedDy->get_uid(), K_BN_BWD_INTEG_TENSOR_DY_UID);
    EXPECT_EQ(liftedDy->get_name(), "DY");
    EXPECT_EQ(liftedDy->get_dim(), toVec(K_BN_BWD_INTEG_DATA_DIMS));
    EXPECT_EQ(liftedDy->get_stride(), toVec(K_BN_BWD_INTEG_DATA_STRIDES));
    EXPECT_EQ(liftedDy->get_data_type(), DataType::FLOAT);

    // X tensor
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_BN_BWD_INTEG_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_uid(), K_BN_BWD_INTEG_TENSOR_X_UID);
    EXPECT_EQ(liftedX->get_name(), "X");
    EXPECT_EQ(liftedX->get_dim(), toVec(K_BN_BWD_INTEG_DATA_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_BN_BWD_INTEG_DATA_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Scale tensor
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_SCALE_UID), 0u);
    auto liftedScale = tensorMap[K_BN_BWD_INTEG_TENSOR_SCALE_UID];
    EXPECT_EQ(liftedScale->get_uid(), K_BN_BWD_INTEG_TENSOR_SCALE_UID);
    EXPECT_EQ(liftedScale->get_name(), "Scale");
    EXPECT_EQ(liftedScale->get_dim(), toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedScale->get_stride(), toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedScale->get_data_type(), DataType::FLOAT);

    // Mean tensor (optional, set in this graph)
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_MEAN_UID), 0u);
    auto liftedMean = tensorMap[K_BN_BWD_INTEG_TENSOR_MEAN_UID];
    EXPECT_EQ(liftedMean->get_uid(), K_BN_BWD_INTEG_TENSOR_MEAN_UID);
    EXPECT_EQ(liftedMean->get_name(), "Mean");
    EXPECT_EQ(liftedMean->get_dim(), toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedMean->get_stride(), toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedMean->get_data_type(), DataType::FLOAT);

    // InvVariance tensor (optional, set in this graph)
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID), 0u);
    auto liftedInvVar = tensorMap[K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(liftedInvVar->get_uid(), K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(liftedInvVar->get_name(), "InvVariance");
    EXPECT_EQ(liftedInvVar->get_dim(), toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedInvVar->get_stride(), toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedInvVar->get_data_type(), DataType::FLOAT);

    // DX tensor (output)
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DX_UID), 0u);
    auto liftedDx = tensorMap[K_BN_BWD_INTEG_TENSOR_DX_UID];
    EXPECT_EQ(liftedDx->get_uid(), K_BN_BWD_INTEG_TENSOR_DX_UID);
    EXPECT_EQ(liftedDx->get_name(), "DX");
    EXPECT_FALSE(liftedDx->get_dim().empty());
    EXPECT_FALSE(liftedDx->get_stride().empty());

    // DScale tensor (output)
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DSCALE_UID), 0u);
    auto liftedDscale = tensorMap[K_BN_BWD_INTEG_TENSOR_DSCALE_UID];
    EXPECT_EQ(liftedDscale->get_uid(), K_BN_BWD_INTEG_TENSOR_DSCALE_UID);
    EXPECT_EQ(liftedDscale->get_name(), "DScale");
    EXPECT_FALSE(liftedDscale->get_dim().empty());
    EXPECT_FALSE(liftedDscale->get_stride().empty());

    // DBias tensor (output)
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DBIAS_UID), 0u);
    auto liftedDbias = tensorMap[K_BN_BWD_INTEG_TENSOR_DBIAS_UID];
    EXPECT_EQ(liftedDbias->get_uid(), K_BN_BWD_INTEG_TENSOR_DBIAS_UID);
    EXPECT_EQ(liftedDbias->get_name(), "DBias");
    EXPECT_FALSE(liftedDbias->get_dim().empty());
    EXPECT_FALSE(liftedDbias->get_stride().empty());

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr) << "Expected a BatchnormBackwardNode";

    // Verify operation name and compute data type
    EXPECT_EQ(bnBwdNode->attributes.get_name(), "bn_bwd_op");
    EXPECT_EQ(bnBwdNode->attributes.compute_data_type, DataType::FLOAT);
}

// Verifies tensor pointer sharing is preserved after lifting.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting, BatchnormBackwardTensorSharingPreserved)
{
    auto originalGraph = buildBatchnormBackwardGraph();

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

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr);

    // Verify pointer equality between tensor map and node attributes
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DY_UID].get(), bnBwdNode->attributes.get_dy().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_X_UID].get(), bnBwdNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_SCALE_UID].get(),
              bnBwdNode->attributes.get_scale().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_MEAN_UID].get(),
              bnBwdNode->attributes.get_mean().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID].get(),
              bnBwdNode->attributes.get_inv_variance().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DX_UID].get(), bnBwdNode->attributes.get_dx().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DSCALE_UID].get(),
              bnBwdNode->attributes.get_dscale().get());
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DBIAS_UID].get(),
              bnBwdNode->attributes.get_dbias().get());
}

// Serializes to binary (FlatBuffer path), lifts without finalization, and verifies all fields.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting, BatchnormBackwardLiftWithoutFinalization)
{
    auto originalGraph = buildBatchnormBackwardGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 8u) << "Expected 8 tensors in lifted graph";

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr);

    EXPECT_EQ(bnBwdNode->attributes.get_name(), "bn_bwd_op");
    EXPECT_EQ(bnBwdNode->attributes.compute_data_type, DataType::FLOAT);

    // Verify key tensor dims and names
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DY_UID]->get_dim(), toVec(K_BN_BWD_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DY_UID]->get_stride(),
              toVec(K_BN_BWD_INTEG_DATA_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DY_UID]->get_name(), "DY");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_X_UID]->get_dim(), toVec(K_BN_BWD_INTEG_DATA_DIMS));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_X_UID]->get_stride(),
              toVec(K_BN_BWD_INTEG_DATA_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_X_UID]->get_name(), "X");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_SCALE_UID]->get_name(), "Scale");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_MEAN_UID]->get_name(), "Mean");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_INV_VARIANCE_UID]->get_name(), "InvVariance");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DX_UID]->get_name(), "DX");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DSCALE_UID]->get_name(), "DScale");

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_BN_BWD_INTEG_TENSOR_DBIAS_UID]->get_name(), "DBias");
}

// Builds a minimal graph (no optional mean/invVariance), verifies optional tensors
// are absent after lifting.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting,
       BatchnormBackwardMinimalRequiredTensorsRoundTrip)
{
    auto originalGraph = buildMinimalBatchnormBackwardGraph();

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
    // dy, x, scale, dx, dscale, dbias = 6 (no mean/invVariance)
    ASSERT_EQ(tensorMap.size(), 6u)
        << "Expected 6 tensors in minimal lifted graph (no mean/invVariance)";

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr);

    EXPECT_EQ(bnBwdNode->attributes.get_name(), "minimal_bn_bwd_op");
    EXPECT_NE(bnBwdNode->attributes.get_dy(), nullptr);
    EXPECT_NE(bnBwdNode->attributes.get_x(), nullptr);
    EXPECT_NE(bnBwdNode->attributes.get_scale(), nullptr);
    EXPECT_NE(bnBwdNode->attributes.get_dx(), nullptr);
    EXPECT_NE(bnBwdNode->attributes.get_dscale(), nullptr);
    EXPECT_NE(bnBwdNode->attributes.get_dbias(), nullptr);
    EXPECT_EQ(bnBwdNode->attributes.get_mean(), nullptr);
    EXPECT_EQ(bnBwdNode->attributes.get_inv_variance(), nullptr);
}

// Builds a batchnorm backward graph with peer_stats tensors, performs a full round-trip,
// and verifies peer_stats appear in the lifted tensor map and node attributes.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting, BatchnormBackwardPeerStatsPreserved)
{
    auto originalGraph = buildBatchnormBackwardGraphWithPeerStats();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify tensor map includes the 2 peer_stats tensors (8 base + 2 peer = 10)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 10u) << "Expected 10 tensors (8 base + 2 peer_stats)";

    // Verify peer_stats tensors appear in the tensor map with correct UIDs, names, dims, strides
    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_PEER_STAT_0_UID), 0u);
    auto liftedPeerStat0 = tensorMap[K_BN_BWD_INTEG_TENSOR_PEER_STAT_0_UID];
    EXPECT_EQ(liftedPeerStat0->get_uid(), K_BN_BWD_INTEG_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStat0->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStat0->get_dim(), toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedPeerStat0->get_stride(), toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedPeerStat0->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_BN_BWD_INTEG_TENSOR_PEER_STAT_1_UID), 0u);
    auto liftedPeerStat1 = tensorMap[K_BN_BWD_INTEG_TENSOR_PEER_STAT_1_UID];
    EXPECT_EQ(liftedPeerStat1->get_uid(), K_BN_BWD_INTEG_TENSOR_PEER_STAT_1_UID);
    EXPECT_EQ(liftedPeerStat1->get_name(), "PeerStat1");
    EXPECT_EQ(liftedPeerStat1->get_dim(), toVec(K_BN_BWD_INTEG_PARAM_DIMS));
    EXPECT_EQ(liftedPeerStat1->get_stride(), toVec(K_BN_BWD_INTEG_PARAM_STRIDES));
    EXPECT_EQ(liftedPeerStat1->get_data_type(), DataType::FLOAT);

    // Verify the lifted node's attributes reference peer_stats correctly
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr);

    const auto& liftedPeerStats = bnBwdNode->attributes.get_peer_stats();
    ASSERT_EQ(liftedPeerStats.size(), 2u) << "Expected 2 peer_stats tensors in lifted node";

    EXPECT_EQ(liftedPeerStats[0]->get_uid(), K_BN_BWD_INTEG_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(liftedPeerStats[0]->get_name(), "PeerStat0");
    EXPECT_EQ(liftedPeerStats[1]->get_uid(), K_BN_BWD_INTEG_TENSOR_PEER_STAT_1_UID);
    EXPECT_EQ(liftedPeerStats[1]->get_name(), "PeerStat1");

    // Verify pointer equality: peer_stats in node attributes share objects with tensor map
    EXPECT_EQ(liftedPeerStats[0].get(), liftedPeerStat0.get());
    EXPECT_EQ(liftedPeerStats[1].get(), liftedPeerStat1.get());
}

// Builds a batchnorm backward graph without explicit set_uid() calls, performs a round-trip,
// and verifies all auto-assigned UIDs are distinct and tensor dims survive.
TEST_F(IntegrationBatchnormBackwardDescriptorLifting, BatchnormBackwardAutoAssignedUidsPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidBnBwdLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Build tensors with distinct dims but NO set_uid() calls
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_BN_BWD_AUTO_DATA_DIMS)).set_stride(toVec(K_BN_BWD_AUTO_DATA_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BN_BWD_AUTO_DATA_DIMS)).set_stride(toVec(K_BN_BWD_AUTO_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BN_BWD_AUTO_PARAM_DIMS)).set_stride(toVec(K_BN_BWD_AUTO_PARAM_STRIDES));

    BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_name("auto_uid_bn_bwd_op");

    auto [dxOut, dscaleOut, dbiasOut] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
    dxOut->set_output(true).set_name("DX");
    dscaleOut->set_output(true).set_name("DScale");
    dbiasOut->set_output(true).set_name("DBias");

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
    // dy, x, scale, dx, dscale, dbias = 6 tensors (no optional mean/invVariance)
    ASSERT_EQ(tensorMap.size(), 6u) << "Expected 6 tensors (dy, x, scale, dx, dscale, dbias)";

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

    auto* bnBwdNode = dynamic_cast<BatchnormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(bnBwdNode, nullptr);

    // Verify tensor dims survived the round trip by identifying tensors via dims
    auto dyUid = bnBwdNode->attributes.get_dy()->get_uid();
    auto scaleUid = bnBwdNode->attributes.get_scale()->get_uid();
    auto dxUid = bnBwdNode->attributes.get_dx()->get_uid();

    EXPECT_NE(dyUid, scaleUid);
    EXPECT_NE(dyUid, dxUid);

    EXPECT_EQ(tensorMap[dyUid]->get_dim(), toVec(K_BN_BWD_AUTO_DATA_DIMS));
    EXPECT_EQ(tensorMap[dyUid]->get_stride(), toVec(K_BN_BWD_AUTO_DATA_STRIDES));
    EXPECT_EQ(tensorMap[scaleUid]->get_dim(), toVec(K_BN_BWD_AUTO_PARAM_DIMS));
    EXPECT_EQ(tensorMap[scaleUid]->get_stride(), toVec(K_BN_BWD_AUTO_PARAM_STRIDES));
    EXPECT_EQ(tensorMap[dxUid]->get_dim(), toVec(K_BN_BWD_AUTO_DATA_DIMS));
    EXPECT_EQ(tensorMap[dxUid]->get_stride(), toVec(K_BN_BWD_AUTO_DATA_STRIDES));
}

} // namespace
