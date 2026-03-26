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

} // namespace
