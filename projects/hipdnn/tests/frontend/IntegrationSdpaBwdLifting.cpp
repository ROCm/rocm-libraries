// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_test_sdk/constants/SdpaBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

using TestableGraph = hipdnn_tests::TestableGraphLifting;

// Builds an SDPA backward graph via the frontend, lowers it through the backend C-API
// via build_operation_graph(), then lifts it back with fromBackendDescriptor()
// and verifies the reconstructed graph matches the original.
class IntegrationSdpaBwdLifting : public ::testing::Test
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

    // Builds a standard SDPA backward graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildSdpaBwdGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("LiftingSdpaBwdGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_uid(K_SDPA_BWD_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
        q->set_dim(toVec(K_SDPA_BWD_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_Q_STRIDES));

        auto k = std::make_shared<TensorAttributes>();
        k->set_uid(K_SDPA_BWD_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
        k->set_dim(toVec(K_SDPA_BWD_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_K_STRIDES));

        auto v = std::make_shared<TensorAttributes>();
        v->set_uid(K_SDPA_BWD_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
        v->set_dim(toVec(K_SDPA_BWD_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_V_STRIDES));

        auto o = std::make_shared<TensorAttributes>();
        o->set_uid(K_SDPA_BWD_TENSOR_O_UID).set_name("O").set_data_type(DataType::FLOAT);
        o->set_dim(toVec(K_SDPA_BWD_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_O_STRIDES));

        auto dO = std::make_shared<TensorAttributes>();
        dO->set_uid(K_SDPA_BWD_TENSOR_DO_UID).set_name("dO").set_data_type(DataType::FLOAT);
        dO->set_dim(toVec(K_SDPA_BWD_TENSOR_DO_DIMS))
            .set_stride(toVec(K_SDPA_BWD_TENSOR_DO_STRIDES));

        auto stats = std::make_shared<TensorAttributes>();
        stats->set_uid(K_SDPA_BWD_TENSOR_STATS_UID)
            .set_name("Stats")
            .set_data_type(DataType::FLOAT);
        stats->set_dim(toVec(K_SDPA_BWD_TENSOR_STATS_DIMS))
            .set_stride(toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES));

        SdpaBackwardAttributes sdpaAttrs;
        sdpaAttrs.set_name("sdpa_bwd_op");

        auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, sdpaAttrs);
        dq->set_uid(K_SDPA_BWD_TENSOR_DQ_UID).set_output(true).set_name("dQ");
        dk->set_uid(K_SDPA_BWD_TENSOR_DK_UID).set_output(true).set_name("dK");
        dv->set_uid(K_SDPA_BWD_TENSOR_DV_UID).set_output(true).set_name("dV");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds an SDPA backward graph with required tensors only, lowers via
// build_operation_graph(handle), lifts back with fromBackendDescriptor(), and verifies
// tensor dimensions, data types, graph-level types, graph name, and operation name.
TEST_F(IntegrationSdpaBwdLifting, SdpaBwdRoundTripViaCApi)
{
    auto originalGraph = buildSdpaBwdGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types and name
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_name(), "LiftingSdpaBwdGraph");

    // Verify tensors by UID — 9 required tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u) << "Expected 9 tensors in lifted SDPA backward graph";

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_stride(), toVec(K_SDPA_BWD_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_K_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_K_UID]->get_stride(), toVec(K_SDPA_BWD_TENSOR_K_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_V_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_V_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_V_UID]->get_stride(), toVec(K_SDPA_BWD_TENSOR_V_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_O_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_O_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_O_UID]->get_stride(), toVec(K_SDPA_BWD_TENSOR_O_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DO_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DO_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DO_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DO_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DO_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_STATS_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_STATS_UID]->get_dim(),
              toVec(K_SDPA_BWD_TENSOR_STATS_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_STATS_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DQ_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DQ_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DQ_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DQ_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DQ_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DK_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DK_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DK_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DK_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DV_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DV_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DV_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DV_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DV_STRIDES));

    // Verify the lifted graph has 1 SDPA backward operation node with the correct name
    auto& subNodes = liftedGraph->getSubNodes();
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    ASSERT_NE(sdpaNode, nullptr) << "Expected a SdpaBwdNode";
    EXPECT_EQ(sdpaNode->attributes.get_name(), "sdpa_bwd_op");
}

// Verifies that tensors are accessible by UID on the reconstructed graph and that the
// tensor objects are shared between the tensor map and the node attributes.
TEST_F(IntegrationSdpaBwdLifting, SdpaBwdTensorSharingPreserved)
{
    auto originalGraph = buildSdpaBwdGraph();

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

    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    // Verify UIDs on node attributes
    EXPECT_EQ(sdpaNode->attributes.get_q()->get_uid(), K_SDPA_BWD_TENSOR_Q_UID);
    EXPECT_EQ(sdpaNode->attributes.get_k()->get_uid(), K_SDPA_BWD_TENSOR_K_UID);
    EXPECT_EQ(sdpaNode->attributes.get_v()->get_uid(), K_SDPA_BWD_TENSOR_V_UID);
    EXPECT_EQ(sdpaNode->attributes.get_o()->get_uid(), K_SDPA_BWD_TENSOR_O_UID);
    EXPECT_EQ(sdpaNode->attributes.get_do()->get_uid(), K_SDPA_BWD_TENSOR_DO_UID);
    EXPECT_EQ(sdpaNode->attributes.get_stats()->get_uid(), K_SDPA_BWD_TENSOR_STATS_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dq()->get_uid(), K_SDPA_BWD_TENSOR_DQ_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dk()->get_uid(), K_SDPA_BWD_TENSOR_DK_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dv()->get_uid(), K_SDPA_BWD_TENSOR_DV_UID);

    // Verify the node references the same tensor objects as the tensor map
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID].get(), sdpaNode->attributes.get_q().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_K_UID].get(), sdpaNode->attributes.get_k().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_V_UID].get(), sdpaNode->attributes.get_v().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_O_UID].get(), sdpaNode->attributes.get_o().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DO_UID].get(), sdpaNode->attributes.get_do().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_STATS_UID].get(), sdpaNode->attributes.get_stats().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DQ_UID].get(), sdpaNode->attributes.get_dq().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DK_UID].get(), sdpaNode->attributes.get_dk().get());
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DV_UID].get(), sdpaNode->attributes.get_dv().get());
}

// Builds an SDPA backward graph with all optional tensors, boolean flags, and scalar
// parameters set, lowers via the C-API, lifts back, and verifies every attribute survives.
TEST_F(IntegrationSdpaBwdLifting, SdpaBwdWithAllOptionalAttributesViaCApi)
{
    auto originalGraph = buildSdpaBwdGraph();

    auto& subNodes = originalGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    // Optional input tensors
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_SDPA_BWD_TENSOR_SCALE_UID).set_name("SCALE");
    scale->set_value(0.125f); // pass-by-value scalar

    auto attnMask = std::make_shared<TensorAttributes>();
    attnMask->set_uid(K_SDPA_BWD_TENSOR_ATTN_MASK_UID)
        .set_name("ATTN_MASK")
        .set_data_type(DataType::FLOAT);
    attnMask->set_dim(toVec(K_SDPA_BWD_TENSOR_ATTN_MASK_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_ATTN_MASK_STRIDES));

    auto seqLenQ = std::make_shared<TensorAttributes>();
    seqLenQ->set_uid(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID)
        .set_name("SEQ_LEN_Q")
        .set_data_type(DataType::INT32);
    seqLenQ->set_dim(toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_STRIDES));

    auto seqLenKv = std::make_shared<TensorAttributes>();
    seqLenKv->set_uid(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID)
        .set_name("SEQ_LEN_KV")
        .set_data_type(DataType::INT32);
    seqLenKv->set_dim(toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_STRIDES));

    auto seed = std::make_shared<TensorAttributes>();
    seed->set_uid(K_SDPA_BWD_TENSOR_SEED_UID).set_name("SEED");
    seed->set_value(int32_t{42});

    auto offset = std::make_shared<TensorAttributes>();
    offset->set_uid(K_SDPA_BWD_TENSOR_OFFSET_UID).set_name("OFFSET");
    offset->set_value(int32_t{0});

    auto dropoutMask = std::make_shared<TensorAttributes>();
    dropoutMask->set_uid(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID)
        .set_name("DROPOUT_MASK")
        .set_data_type(DataType::UINT8);
    dropoutMask->set_dim(toVec(K_SDPA_BWD_TENSOR_DROPOUT_MASK_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_DROPOUT_MASK_STRIDES));

    auto dropoutScale = std::make_shared<TensorAttributes>();
    dropoutScale->set_uid(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID).set_name("DROPOUT_SCALE");
    dropoutScale->set_value(1.0f / (1.0f - 0.1f));

    auto dropoutScaleInv = std::make_shared<TensorAttributes>();
    dropoutScaleInv->set_uid(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID).set_name("DROPOUT_SCALE_INV");
    dropoutScaleInv->set_value(1.0f - 0.1f);

    // Optional output tensor
    auto dBias = std::make_shared<TensorAttributes>();
    dBias->set_uid(K_SDPA_BWD_TENSOR_DBIAS_UID)
        .set_name("DBIAS")
        .set_data_type(DataType::FLOAT)
        .set_output(true);
    dBias->set_dim(toVec(K_SDPA_BWD_TENSOR_DBIAS_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_DBIAS_STRIDES));

    sdpaNode->attributes.set_attn_scale(scale)
        .set_bias(attnMask)
        .set_seq_len_q(seqLenQ)
        .set_seq_len_kv(seqLenKv)
        .set_dropout(0.1f, seed, offset)
        .set_dropout_mask(dropoutMask)
        .set_dropout_scale(dropoutScale)
        .set_dropout_scale_inv(dropoutScaleInv)
        .set_dbias(dBias)
        .set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true)
        .set_attn_scale_value(0.125f)
        .set_diagonal_band_left_bound(-1)
        .set_diagonal_band_right_bound(1)
        .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // 9 required + 10 optional (all serialized as tensors through the C-API)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 19u) << "Expected 9 required + 10 optional tensors";

    // Verify all optional tensor UIDs are present and their properties match.
    const auto scalarDims = toVec(K_SDPA_BWD_TENSOR_SCALAR_DIMS);
    const auto scalarStrides = toVec(K_SDPA_BWD_TENSOR_SCALAR_STRIDES);
    const auto seqLenDims = toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_DIMS);
    const auto seqLenStrides = toVec(K_SDPA_BWD_TENSOR_SEQ_LEN_STRIDES);

    // scale — scalar tensor
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SCALE_UID]->get_dim(), scalarDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SCALE_UID]->get_stride(), scalarStrides);

    // attn_mask — [batch, num_heads, seq_q, seq_kv]
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_ATTN_MASK_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_ATTN_MASK_UID]->get_dim(),
              toVec(K_SDPA_BWD_TENSOR_ATTN_MASK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_ATTN_MASK_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_ATTN_MASK_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_ATTN_MASK_UID]->get_data_type(), DataType::FLOAT);

    // seq_len_q — [batch, 1, 1, 1] INT32
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID]->get_dim(), seqLenDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID]->get_stride(), seqLenStrides);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID]->get_data_type(), DataType::INT32);

    // seq_len_kv — [batch, 1, 1, 1] INT32
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID]->get_dim(), seqLenDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID]->get_stride(), seqLenStrides);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID]->get_data_type(), DataType::INT32);

    // seed — scalar tensor
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_SEED_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEED_UID]->get_dim(), scalarDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_SEED_UID]->get_stride(), scalarStrides);

    // offset — scalar tensor
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_OFFSET_UID]->get_dim(), scalarDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_OFFSET_UID]->get_stride(), scalarStrides);

    // dropout_mask — [batch, num_heads, seq_q, seq_kv]
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID]->get_dim(),
              toVec(K_SDPA_BWD_TENSOR_DROPOUT_MASK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DROPOUT_MASK_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID]->get_data_type(), DataType::UINT8);

    // dropout_scale — scalar tensor
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID]->get_dim(), scalarDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID]->get_stride(), scalarStrides);

    // dropout_scale_inv — scalar tensor
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID]->get_dim(), scalarDims);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID]->get_stride(), scalarStrides);

    // dBias — [batch, num_heads, seq_q, seq_kv]
    ASSERT_NE(tensorMap.count(K_SDPA_BWD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DBIAS_UID]->get_dim(),
              toVec(K_SDPA_BWD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DBIAS_UID]->get_stride(),
              toVec(K_SDPA_BWD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DBIAS_UID]->get_data_type(), DataType::FLOAT);

    // Verify node attributes on the lifted SdpaBwdNode
    auto& liftedNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(liftedNodes.size(), 1u);
    auto* liftedNode = dynamic_cast<SdpaBwdNode*>(liftedNodes[0].get());
    ASSERT_NE(liftedNode, nullptr);

    const auto& attrs = liftedNode->attributes;

    // Optional tensor UIDs on the node
    ASSERT_NE(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale()->get_uid(), K_SDPA_BWD_TENSOR_SCALE_UID);
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), K_SDPA_BWD_TENSOR_ATTN_MASK_UID);
    ASSERT_NE(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q()->get_uid(), K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_NE(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv()->get_uid(), K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_NE(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_seed()->get_uid(), K_SDPA_BWD_TENSOR_SEED_UID);
    ASSERT_NE(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_offset()->get_uid(), K_SDPA_BWD_TENSOR_OFFSET_UID);
    ASSERT_NE(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask()->get_uid(), K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);
    ASSERT_NE(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale()->get_uid(), K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_NE(attrs.get_dropout_scale_inv(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale_inv()->get_uid(), K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);
    ASSERT_NE(attrs.get_dbias(), nullptr);
    EXPECT_EQ(attrs.get_dbias()->get_uid(), K_SDPA_BWD_TENSOR_DBIAS_UID);

    // Boolean flags
    EXPECT_TRUE(attrs.alibi_mask);
    EXPECT_TRUE(attrs.padding_mask);
    EXPECT_TRUE(attrs.causal_mask);
    EXPECT_TRUE(attrs.causal_mask_bottom_right);

    // Scalar values
    ASSERT_TRUE(attrs.dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs.dropout_probability.value(), 0.1f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(attrs.attn_scale_value.value(), 0.125f);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(attrs.left_bound.value(), -1);
    ASSERT_TRUE(attrs.right_bound.has_value());
    EXPECT_EQ(attrs.right_bound.value(), 1);

    // Diagonal alignment
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

// Builds an SDPA backward graph without explicit tensor UIDs, then verifies
// auto-assigned UIDs are unique and correctly referenced after a round-trip.
TEST_F(IntegrationSdpaBwdLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidSdpaBwdGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Create tensors WITHOUT explicit UIDs
    auto q = std::make_shared<TensorAttributes>();
    q->set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_BWD_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_BWD_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_BWD_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_V_STRIDES));

    auto o = std::make_shared<TensorAttributes>();
    o->set_name("O").set_data_type(DataType::FLOAT);
    o->set_dim(toVec(K_SDPA_BWD_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_O_STRIDES));

    auto dO = std::make_shared<TensorAttributes>();
    dO->set_name("dO").set_data_type(DataType::FLOAT);
    dO->set_dim(toVec(K_SDPA_BWD_TENSOR_DO_DIMS)).set_stride(toVec(K_SDPA_BWD_TENSOR_DO_STRIDES));

    auto stats = std::make_shared<TensorAttributes>();
    stats->set_name("Stats").set_data_type(DataType::FLOAT);
    stats->set_dim(toVec(K_SDPA_BWD_TENSOR_STATS_DIMS))
        .set_stride(toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES));

    SdpaBackwardAttributes sdpaAttrs;

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(sdpaAttrs));
    dq->set_output(true);
    dk->set_output(true);
    dv->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // All 9 auto-assigned UIDs should be unique
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u) << "Expected 9 tensors with auto-assigned UIDs";

    std::unordered_set<int64_t> uids;
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.insert(uid);
    }
    EXPECT_EQ(uids.size(), 9u) << "Auto-assigned tensor UIDs are not unique";

    // Verify the node references resolve to tensors in the map
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    EXPECT_NE(uids.count(sdpaNode->attributes.get_q()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_k()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_v()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_o()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_do()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_stats()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_dq()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_dk()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_dv()->get_uid()), 0u);
}

// Builds an SDPA backward graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// the reconstructed graph matches the original.
TEST_F(IntegrationSdpaBwdLifting, SdpaBwdLiftWithoutFinalization)
{
    auto originalGraph = buildSdpaBwdGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
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

    // Verify the lifted graph has 1 SDPA backward operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr) << "Expected a SdpaBwdNode";

    // Verify tensor dims survive the serialization round-trip
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_stride(), toVec(K_SDPA_BWD_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_K_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DQ_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DQ_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DK_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DV_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DV_DIMS));
}

// Serializes an SDPA backward graph to binary, then deserializes via the backend
// using a handle (full finalization). Verifies the reconstructed graph matches the original.
TEST_F(IntegrationSdpaBwdLifting, SdpaBwdDeserializeViaBackendWithHandle)
{
    auto originalGraph = buildSdpaBwdGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    ASSERT_FALSE(data.empty());

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize(_handle, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types and name
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_name(), "LiftingSdpaBwdGraph");

    // Verify 9 required tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);

    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_Q_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_K_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DQ_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DQ_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DK_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BWD_TENSOR_DV_UID]->get_dim(), toVec(K_SDPA_BWD_TENSOR_DV_DIMS));

    // Verify the node is an SdpaBwdNode with the correct operation name
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);
    EXPECT_EQ(sdpaNode->attributes.get_name(), "sdpa_bwd_op");
    EXPECT_EQ(sdpaNode->attributes.get_q()->get_uid(), K_SDPA_BWD_TENSOR_Q_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dv()->get_uid(), K_SDPA_BWD_TENSOR_DV_UID);
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for an SDPA backward graph.
TEST_F(IntegrationSdpaBwdLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildSdpaBwdGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to JSON (auto-lowers internally)
    std::string jsonData;
    result = originalGraph->serialize(jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(jsonData.empty());

    // Deserialize from JSON with handle
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize(_handle, jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types and name
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_name(), "LiftingSdpaBwdGraph");

    // Verify tensors by UID — 9 required tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u) << "Expected 9 tensors in lifted SDPA backward graph";

    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_Q_UID,
                                      "Q",
                                      toVec(K_SDPA_BWD_TENSOR_Q_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_Q_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_K_UID,
                                      "K",
                                      toVec(K_SDPA_BWD_TENSOR_K_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_K_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_V_UID,
                                      "V",
                                      toVec(K_SDPA_BWD_TENSOR_V_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_V_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_O_UID,
                                      "O",
                                      toVec(K_SDPA_BWD_TENSOR_O_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_O_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_DO_UID,
                                      "dO",
                                      toVec(K_SDPA_BWD_TENSOR_DO_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_DO_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_STATS_UID,
                                      "Stats",
                                      toVec(K_SDPA_BWD_TENSOR_STATS_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_DQ_UID,
                                      "dQ",
                                      toVec(K_SDPA_BWD_TENSOR_DQ_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_DQ_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_DK_UID,
                                      "dK",
                                      toVec(K_SDPA_BWD_TENSOR_DK_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_DK_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_SDPA_BWD_TENSOR_DV_UID,
                                      "dV",
                                      toVec(K_SDPA_BWD_TENSOR_DV_DIMS),
                                      toVec(K_SDPA_BWD_TENSOR_DV_STRIDES),
                                      DataType::FLOAT);

    // Verify the lifted graph has 1 SDPA backward operation node with the correct name
    auto& subNodes = liftedGraph->getSubNodes();
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* sdpaNode = dynamic_cast<SdpaBwdNode*>(subNodes[0].get());
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    ASSERT_NE(sdpaNode, nullptr) << "Expected a SdpaBwdNode";
    EXPECT_EQ(sdpaNode->attributes.get_name(), "sdpa_bwd_op");
}

} // namespace
