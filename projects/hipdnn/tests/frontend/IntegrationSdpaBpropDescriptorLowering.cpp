// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/SdpaBpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;
using DiagonalAlignmentSdk = hipdnn_data_sdk::data_objects::DiagonalAlignment;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationSdpaBpropDescriptorLowering : public ::testing::Test
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

    hipdnnHandle_t _handle = nullptr;
};

// Builds an SDPA backward graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies ALL tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationSdpaBpropDescriptorLowering, SdpaBpropGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaBpropGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_BPROP_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_BPROP_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_BPROP_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_BPROP_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_BPROP_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_BPROP_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));

    auto o = std::make_shared<TensorAttributes>();
    o->set_uid(K_SDPA_BPROP_TENSOR_O_UID).set_name("O").set_data_type(DataType::FLOAT);
    o->set_dim(toVec(K_SDPA_BPROP_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

    auto dO = std::make_shared<TensorAttributes>();
    dO->set_uid(K_SDPA_BPROP_TENSOR_DO_UID).set_name("dO").set_data_type(DataType::FLOAT);
    dO->set_dim(toVec(K_SDPA_BPROP_TENSOR_DO_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

    auto stats = std::make_shared<TensorAttributes>();
    stats->set_uid(K_SDPA_BPROP_TENSOR_STATS_UID).set_name("Stats").set_data_type(DataType::FLOAT);
    stats->set_dim(toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

    SdpaBackwardAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_bprop_op");

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(sdpaAttrs));
    dq->set_uid(K_SDPA_BPROP_TENSOR_DQ_UID).set_output(true).set_name("dQ");
    dk->set_uid(K_SDPA_BPROP_TENSOR_DK_UID).set_output(true).set_name("dK");
    dv->set_uid(K_SDPA_BPROP_TENSOR_DV_UID).set_output(true).set_name("dV");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors (Q, K, V, O, dO, Stats, dQ, dK, dV = 9 tensors) --
    ASSERT_EQ(graphT.tensors.size(), 9u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify Q tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_Q_UID), 0u);
    auto* qT = tensorMap[K_SDPA_BPROP_TENSOR_Q_UID];
    EXPECT_EQ(qT->name, "Q");
    EXPECT_EQ(qT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(qT->dims, toVec(K_SDPA_BPROP_TENSOR_Q_DIMS));
    EXPECT_EQ(qT->strides, toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    EXPECT_FALSE(qT->virtual_);

    // Verify K tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_K_UID), 0u);
    auto* kT = tensorMap[K_SDPA_BPROP_TENSOR_K_UID];
    EXPECT_EQ(kT->name, "K");
    EXPECT_EQ(kT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(kT->dims, toVec(K_SDPA_BPROP_TENSOR_K_DIMS));
    EXPECT_EQ(kT->strides, toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    EXPECT_FALSE(kT->virtual_);

    // Verify V tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_V_UID), 0u);
    auto* vT = tensorMap[K_SDPA_BPROP_TENSOR_V_UID];
    EXPECT_EQ(vT->name, "V");
    EXPECT_EQ(vT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(vT->dims, toVec(K_SDPA_BPROP_TENSOR_V_DIMS));
    EXPECT_EQ(vT->strides, toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    EXPECT_FALSE(vT->virtual_);

    // Verify O tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_O_UID), 0u);
    auto* oT = tensorMap[K_SDPA_BPROP_TENSOR_O_UID];
    EXPECT_EQ(oT->name, "O");
    EXPECT_EQ(oT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(oT->dims, toVec(K_SDPA_BPROP_TENSOR_O_DIMS));
    EXPECT_EQ(oT->strides, toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    EXPECT_FALSE(oT->virtual_);

    // Verify dO tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DO_UID), 0u);
    auto* doT = tensorMap[K_SDPA_BPROP_TENSOR_DO_UID];
    EXPECT_EQ(doT->name, "dO");
    EXPECT_EQ(doT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(doT->dims, toVec(K_SDPA_BPROP_TENSOR_DO_DIMS));
    EXPECT_EQ(doT->strides, toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    EXPECT_FALSE(doT->virtual_);

    // Verify Stats tensor - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_STATS_UID), 0u);
    auto* statsT = tensorMap[K_SDPA_BPROP_TENSOR_STATS_UID];
    EXPECT_EQ(statsT->name, "Stats");
    EXPECT_EQ(statsT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(statsT->dims, toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS));
    EXPECT_EQ(statsT->strides, toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    EXPECT_FALSE(statsT->virtual_);

    // Verify dQ tensor (output) - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DQ_UID), 0u);
    auto* dqT = tensorMap[K_SDPA_BPROP_TENSOR_DQ_UID];
    EXPECT_EQ(dqT->name, "dQ");
    EXPECT_EQ(dqT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(dqT->dims, toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS));
    EXPECT_EQ(dqT->strides, toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    EXPECT_FALSE(dqT->virtual_);

    // Verify dK tensor (output) - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DK_UID), 0u);
    auto* dkT = tensorMap[K_SDPA_BPROP_TENSOR_DK_UID];
    EXPECT_EQ(dkT->name, "dK");
    EXPECT_EQ(dkT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(dkT->dims, toVec(K_SDPA_BPROP_TENSOR_DK_DIMS));
    EXPECT_EQ(dkT->strides, toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    EXPECT_FALSE(dkT->virtual_);

    // Verify dV tensor (output) - UID, name, data type, dims, strides
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DV_UID), 0u);
    auto* dvT = tensorMap[K_SDPA_BPROP_TENSOR_DV_UID];
    EXPECT_EQ(dvT->name, "dV");
    EXPECT_EQ(dvT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(dvT->dims, toVec(K_SDPA_BPROP_TENSOR_DV_DIMS));
    EXPECT_EQ(dvT->strides, toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
    EXPECT_FALSE(dvT->virtual_);

    // -- Verify SDPA bprop operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->name, "sdpa_bprop_op");
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::SdpaBackwardAttributes);

    auto* sdpa = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(sdpa, nullptr);

    // Verify required tensor UID references in operation node
    EXPECT_EQ(sdpa->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(sdpa->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(sdpa->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(sdpa->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(sdpa->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(sdpa->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(sdpa->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(sdpa->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(sdpa->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

    // Verify optional tensor UIDs are absent when not set
    EXPECT_FALSE(sdpa->scale_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->attn_mask_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->seq_len_q_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->seq_len_kv_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->seed_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->offset_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_mask_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_scale_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_scale_inv_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dbias_tensor_uid.has_value());

    // Verify all default scalar/boolean/enum values
    EXPECT_EQ(sdpa->diagonal_alignment, DiagonalAlignmentSdk::TOP_LEFT);
    EXPECT_FALSE(sdpa->alibi_mask);
    EXPECT_FALSE(sdpa->padding_mask);
    EXPECT_FALSE(sdpa->causal_mask);
    EXPECT_FALSE(sdpa->causal_mask_bottom_right);
    EXPECT_FALSE(sdpa->dropout_probability.has_value());
    EXPECT_FALSE(sdpa->attn_scale_value.has_value());
    EXPECT_FALSE(sdpa->left_bound.has_value());
    EXPECT_FALSE(sdpa->right_bound.has_value());
}

// Exercises packer code paths for optional input tensors (attn_mask, seed,
// offset), non-default booleans, optional float and int64 scalars,
// and non-default enum values. Verifies ALL tensor properties and ALL
// scalar/data fields.
TEST_F(IntegrationSdpaBpropDescriptorLowering, SdpaBpropWithOptionalTensorsAndScalars)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaBpropOptionals")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_BPROP_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_BPROP_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_BPROP_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_BPROP_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_BPROP_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_BPROP_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));

    auto o = std::make_shared<TensorAttributes>();
    o->set_uid(K_SDPA_BPROP_TENSOR_O_UID).set_name("O").set_data_type(DataType::FLOAT);
    o->set_dim(toVec(K_SDPA_BPROP_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

    auto dO = std::make_shared<TensorAttributes>();
    dO->set_uid(K_SDPA_BPROP_TENSOR_DO_UID).set_name("dO").set_data_type(DataType::FLOAT);
    dO->set_dim(toVec(K_SDPA_BPROP_TENSOR_DO_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

    auto statsTensor = std::make_shared<TensorAttributes>();
    statsTensor->set_uid(K_SDPA_BPROP_TENSOR_STATS_UID)
        .set_name("Stats")
        .set_data_type(DataType::FLOAT);
    statsTensor->set_dim(toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

    // Optional tensors
    auto attnMask = std::make_shared<TensorAttributes>();
    attnMask->set_uid(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID).set_name("ATTN_MASK");
    attnMask->set_data_type(DataType::FLOAT);
    attnMask->set_dim(toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS));
    attnMask->set_stride(toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES));

    auto seed = std::make_shared<TensorAttributes>();
    seed->set_uid(K_SDPA_BPROP_TENSOR_SEED_UID).set_name("SEED").set_data_type(DataType::FLOAT);
    seed->set_dim(toVec(K_SDPA_BPROP_TENSOR_SCALAR_DIMS));
    seed->set_stride(toVec(K_SDPA_BPROP_TENSOR_SCALAR_STRIDES));

    auto offset = std::make_shared<TensorAttributes>();
    offset->set_uid(K_SDPA_BPROP_TENSOR_OFFSET_UID)
        .set_name("OFFSET")
        .set_data_type(DataType::FLOAT);
    offset->set_dim(toVec(K_SDPA_BPROP_TENSOR_SCALAR_DIMS));
    offset->set_stride(toVec(K_SDPA_BPROP_TENSOR_SCALAR_STRIDES));

    SdpaBackwardAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_bprop_optionals");
    sdpaAttrs.set_bias(attnMask);
    sdpaAttrs.set_dropout(0.1f, seed, offset);
    sdpaAttrs.set_causal_mask(true);
    sdpaAttrs.set_causal_mask_bottom_right(true);
    // Note: alibi_mask and padding_mask are left at false because
    // padding_mask=true requires seq_len_q/seq_len_kv tensors.
    sdpaAttrs.set_attn_scale_value(0.125f);
    sdpaAttrs.set_diagonal_band_left_bound(0);
    sdpaAttrs.set_diagonal_band_right_bound(128);
    sdpaAttrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, statsTensor, std::move(sdpaAttrs));
    dq->set_uid(K_SDPA_BPROP_TENSOR_DQ_UID).set_output(true).set_name("dQ");
    dk->set_uid(K_SDPA_BPROP_TENSOR_DK_UID).set_output(true).set_name("dK");
    dv->set_uid(K_SDPA_BPROP_TENSOR_DV_UID).set_output(true).set_name("dV");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // Q + K + V + O + dO + Stats + dQ + dK + dV + attn_mask + seed + offset = 12
    ASSERT_EQ(graphT.tensors.size(), 12u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify ALL tensor UIDs are present
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_Q_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_K_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_V_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_O_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DO_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_STATS_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DQ_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DK_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DV_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_SEED_UID), 0u);
    EXPECT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_OFFSET_UID), 0u);

    // Verify optional tensor properties: attn_mask
    auto* attnMaskT = tensorMap[K_SDPA_BPROP_TENSOR_ATTN_MASK_UID];
    EXPECT_EQ(attnMaskT->name, "ATTN_MASK");
    EXPECT_EQ(attnMaskT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(attnMaskT->dims, toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS));
    EXPECT_EQ(attnMaskT->strides, toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES));

    // Verify optional tensor properties: seed
    auto* seedT = tensorMap[K_SDPA_BPROP_TENSOR_SEED_UID];
    EXPECT_EQ(seedT->name, "SEED");
    EXPECT_EQ(seedT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(seedT->dims, toVec(K_SDPA_BPROP_TENSOR_SCALAR_DIMS));
    EXPECT_EQ(seedT->strides, toVec(K_SDPA_BPROP_TENSOR_SCALAR_STRIDES));

    // Verify optional tensor properties: offset
    auto* offsetT = tensorMap[K_SDPA_BPROP_TENSOR_OFFSET_UID];
    EXPECT_EQ(offsetT->name, "OFFSET");
    EXPECT_EQ(offsetT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(offsetT->dims, toVec(K_SDPA_BPROP_TENSOR_SCALAR_DIMS));
    EXPECT_EQ(offsetT->strides, toVec(K_SDPA_BPROP_TENSOR_SCALAR_STRIDES));

    // -- Verify SDPA bprop node attributes --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->name, "sdpa_bprop_optionals");
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::SdpaBackwardAttributes);

    auto* sdpa = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(sdpa, nullptr);

    // Required tensor UIDs
    EXPECT_EQ(sdpa->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(sdpa->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(sdpa->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(sdpa->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(sdpa->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(sdpa->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(sdpa->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(sdpa->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(sdpa->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

    // Optional tensor UIDs (present)
    ASSERT_TRUE(sdpa->attn_mask_tensor_uid.has_value());
    EXPECT_EQ(sdpa->attn_mask_tensor_uid.value(), K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);

    ASSERT_TRUE(sdpa->seed_tensor_uid.has_value());
    EXPECT_EQ(sdpa->seed_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SEED_UID);

    ASSERT_TRUE(sdpa->offset_tensor_uid.has_value());
    EXPECT_EQ(sdpa->offset_tensor_uid.value(), K_SDPA_BPROP_TENSOR_OFFSET_UID);

    // Optional tensor UIDs that were NOT set should be absent
    EXPECT_FALSE(sdpa->scale_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->seq_len_q_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->seq_len_kv_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_mask_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_scale_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dropout_scale_inv_tensor_uid.has_value());
    EXPECT_FALSE(sdpa->dbias_tensor_uid.has_value());

    // Boolean attributes
    EXPECT_TRUE(sdpa->causal_mask);
    EXPECT_TRUE(sdpa->causal_mask_bottom_right);
    EXPECT_FALSE(sdpa->alibi_mask);
    EXPECT_FALSE(sdpa->padding_mask);

    // Float scalars
    ASSERT_TRUE(sdpa->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(sdpa->dropout_probability.value(), 0.1f);

    ASSERT_TRUE(sdpa->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(sdpa->attn_scale_value.value(), 0.125f);

    // Integer scalars
    ASSERT_TRUE(sdpa->left_bound.has_value());
    EXPECT_EQ(sdpa->left_bound.value(), 0);

    ASSERT_TRUE(sdpa->right_bound.has_value());
    EXPECT_EQ(sdpa->right_bound.value(), 128);

    // Enum values
    EXPECT_EQ(sdpa->diagonal_alignment, DiagonalAlignmentSdk::BOTTOM_RIGHT);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip. Tensors are created WITHOUT explicit UIDs.
TEST_F(IntegrationSdpaBpropDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidSdpaBpropGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Create tensors WITHOUT explicit UIDs
    auto q = std::make_shared<TensorAttributes>();
    q->set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_BPROP_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_BPROP_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_BPROP_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));

    auto o = std::make_shared<TensorAttributes>();
    o->set_name("O").set_data_type(DataType::FLOAT);
    o->set_dim(toVec(K_SDPA_BPROP_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

    auto dO = std::make_shared<TensorAttributes>();
    dO->set_name("dO").set_data_type(DataType::FLOAT);
    dO->set_dim(toVec(K_SDPA_BPROP_TENSOR_DO_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

    auto stats = std::make_shared<TensorAttributes>();
    stats->set_name("Stats").set_data_type(DataType::FLOAT);
    stats->set_dim(toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

    SdpaBackwardAttributes sdpaAttrs;

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(sdpaAttrs));
    dq->set_output(true);
    dk->set_output(true);
    dv->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // Q + K + V + O + dO + Stats + dQ + dK + dV = 9 tensors
    ASSERT_EQ(graphT.tensors.size(), 9u);

    // All auto-assigned UIDs should be unique
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 9u) << "Tensor UIDs are not unique";

    // The SDPA bprop operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* sdpa = graphT.nodes[0]->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(sdpa, nullptr);

    // All tensor UIDs referenced by the node should exist in the graph tensors
    EXPECT_TRUE(uids.count(sdpa->q_tensor_uid) > 0)
        << "Q tensor UID " << sdpa->q_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->k_tensor_uid) > 0)
        << "K tensor UID " << sdpa->k_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->v_tensor_uid) > 0)
        << "V tensor UID " << sdpa->v_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->o_tensor_uid) > 0)
        << "O tensor UID " << sdpa->o_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->do_tensor_uid) > 0)
        << "dO tensor UID " << sdpa->do_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->stats_tensor_uid) > 0)
        << "Stats tensor UID " << sdpa->stats_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->dq_tensor_uid) > 0)
        << "dQ tensor UID " << sdpa->dq_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->dk_tensor_uid) > 0)
        << "dK tensor UID " << sdpa->dk_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->dv_tensor_uid) > 0)
        << "dV tensor UID " << sdpa->dv_tensor_uid << " not found in graph tensors";

    // All nine required tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {sdpa->q_tensor_uid,
                                                  sdpa->k_tensor_uid,
                                                  sdpa->v_tensor_uid,
                                                  sdpa->o_tensor_uid,
                                                  sdpa->do_tensor_uid,
                                                  sdpa->stats_tensor_uid,
                                                  sdpa->dq_tensor_uid,
                                                  sdpa->dk_tensor_uid,
                                                  sdpa->dv_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 9u) << "SDPA bprop node tensor UIDs are not distinct";
}

// Verifies that the operation name survives the round-trip through
// pack -> backend descriptor -> serialize -> deserialize.
TEST_F(IntegrationSdpaBpropDescriptorLowering, OperationNameRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestOpNameGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_BPROP_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_BPROP_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_BPROP_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_BPROP_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_BPROP_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_BPROP_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));

    auto o = std::make_shared<TensorAttributes>();
    o->set_uid(K_SDPA_BPROP_TENSOR_O_UID).set_name("O").set_data_type(DataType::FLOAT);
    o->set_dim(toVec(K_SDPA_BPROP_TENSOR_O_DIMS)).set_stride(toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

    auto dO = std::make_shared<TensorAttributes>();
    dO->set_uid(K_SDPA_BPROP_TENSOR_DO_UID).set_name("dO").set_data_type(DataType::FLOAT);
    dO->set_dim(toVec(K_SDPA_BPROP_TENSOR_DO_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

    auto stats = std::make_shared<TensorAttributes>();
    stats->set_uid(K_SDPA_BPROP_TENSOR_STATS_UID).set_name("Stats").set_data_type(DataType::FLOAT);
    stats->set_dim(toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

    const std::string expectedName = "my_sdpa_bprop_layer_42";
    SdpaBackwardAttributes sdpaAttrs;
    sdpaAttrs.set_name(expectedName);

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(sdpaAttrs));
    dq->set_uid(K_SDPA_BPROP_TENSOR_DQ_UID).set_output(true).set_name("dQ");
    dk->set_uid(K_SDPA_BPROP_TENSOR_DK_UID).set_output(true).set_name("dK");
    dv->set_uid(K_SDPA_BPROP_TENSOR_DV_UID).set_output(true).set_name("dV");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // Verify operation name survived the round-trip
    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->name, expectedName);
    EXPECT_EQ(graphT.nodes[0]->attributes.type, NodeAttrType::SdpaBackwardAttributes);
}

} // namespace
