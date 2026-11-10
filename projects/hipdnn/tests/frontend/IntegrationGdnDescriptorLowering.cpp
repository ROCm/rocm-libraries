// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/gdn_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/GdnConstants.hpp>
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
using GateDomainSdk = hipdnn_flatbuffers_sdk::data_objects::GateDomain;

namespace
{

// Which optional operands the test graph carries. state_indices cannot combine with
// checkpoints, so the dense-state and pooled-state variants are separate graphs.
enum class GdnVariant
{
    MINIMAL, // required operands only
    DENSE_STATE, // initial_state, safe-gate a_log / dt_bias, final_state, state_checkpoints
    POOLED_STATE, // initial_state pool + state_indices, final_state written in place
};

template <typename DimsT, typename StridesT>
std::shared_ptr<TensorAttributes> makeTensor(int64_t uid,
                                             bool assignUids,
                                             const std::string& name,
                                             DataType dataType,
                                             const DimsT& dims,
                                             const StridesT& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    if(assignUids)
    {
        tensor->set_uid(uid);
    }
    tensor->set_name(name).set_data_type(dataType);
    tensor->set_dim(toVec(dims)).set_stride(toVec(strides));
    return tensor;
}

void markOutput(const std::shared_ptr<TensorAttributes>& tensor,
                int64_t uid,
                bool assignUids,
                const std::string& name)
{
    if(!tensor)
    {
        return;
    }
    if(assignUids)
    {
        tensor->set_uid(uid);
    }
    tensor->set_output(true).set_name(name);
}

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationGdnDescriptorLowering : public IntegrationTestFixture
{
protected:
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(GdnAttributes& attrs,
                                                                     GdnVariant variant
                                                                     = GdnVariant::MINIMAL,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("GdnIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = makeTensor(K_GDN_TENSOR_Q_UID,
                            assignUids,
                            "q",
                            DataType::FLOAT,
                            K_GDN_TENSOR_Q_DIMS,
                            K_GDN_TENSOR_Q_STRIDES);
        auto k = makeTensor(K_GDN_TENSOR_K_UID,
                            assignUids,
                            "k",
                            DataType::FLOAT,
                            K_GDN_TENSOR_K_DIMS,
                            K_GDN_TENSOR_K_STRIDES);
        auto v = makeTensor(K_GDN_TENSOR_V_UID,
                            assignUids,
                            "v",
                            DataType::FLOAT,
                            K_GDN_TENSOR_V_DIMS,
                            K_GDN_TENSOR_V_STRIDES);
        auto g = makeTensor(K_GDN_TENSOR_G_UID,
                            assignUids,
                            "g",
                            DataType::FLOAT,
                            K_GDN_TENSOR_G_DIMS,
                            K_GDN_TENSOR_G_STRIDES);
        auto beta = makeTensor(K_GDN_TENSOR_BETA_UID,
                               assignUids,
                               "beta",
                               DataType::FLOAT,
                               K_GDN_TENSOR_BETA_DIMS,
                               K_GDN_TENSOR_BETA_STRIDES);
        auto cuSeqlens = makeTensor(K_GDN_TENSOR_CU_SEQLENS_UID,
                                    assignUids,
                                    "cu_seqlens",
                                    DataType::INT32,
                                    K_GDN_TENSOR_CU_SEQLENS_DIMS,
                                    K_GDN_TENSOR_CU_SEQLENS_STRIDES);

        if(variant != GdnVariant::MINIMAL)
        {
            attrs.set_initial_state(makeTensor(K_GDN_TENSOR_INITIAL_STATE_UID,
                                               assignUids,
                                               "initial_state",
                                               DataType::FLOAT,
                                               K_GDN_TENSOR_INITIAL_STATE_DIMS,
                                               K_GDN_TENSOR_INITIAL_STATE_STRIDES));
            attrs.set_output_final_state(true);
        }
        if(variant == GdnVariant::DENSE_STATE)
        {
            attrs.set_safe_gate(true);
            attrs.set_a_log(makeTensor(K_GDN_TENSOR_A_LOG_UID,
                                       assignUids,
                                       "a_log",
                                       DataType::FLOAT,
                                       K_GDN_TENSOR_A_LOG_DIMS,
                                       K_GDN_TENSOR_A_LOG_STRIDES));
            attrs.set_dt_bias(makeTensor(K_GDN_TENSOR_DT_BIAS_UID,
                                         assignUids,
                                         "dt_bias",
                                         DataType::FLOAT,
                                         K_GDN_TENSOR_DT_BIAS_DIMS,
                                         K_GDN_TENSOR_DT_BIAS_STRIDES));
            attrs.set_checkpoint_every_n_tokens(2);
        }
        if(variant == GdnVariant::POOLED_STATE)
        {
            attrs.set_state_indices(makeTensor(K_GDN_TENSOR_STATE_INDICES_UID,
                                               assignUids,
                                               "state_indices",
                                               DataType::INT32,
                                               K_GDN_TENSOR_STATE_INDICES_DIMS,
                                               K_GDN_TENSOR_STATE_INDICES_STRIDES));
            attrs.set_overwrite_initial_state(true);
        }

        auto [o, finalState, stateCheckpoints] = graph->gdn(q, k, v, g, beta, cuSeqlens, attrs);
        markOutput(o, K_GDN_TENSOR_O_UID, assignUids, "o");
        markOutput(finalState, K_GDN_TENSOR_FINAL_STATE_UID, assignUids, "final_state");
        markOutput(
            stateCheckpoints, K_GDN_TENSOR_STATE_CHECKPOINTS_UID, assignUids, "state_checkpoints");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip with every dense-state operand: verifies tensors and every attribute.
TEST_F(IntegrationGdnDescriptorLowering, GdnLoweringRoundTrip)
{
    GdnAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_scale(0.25F);
    attrs.set_use_qk_l2norm(true);
    attrs.set_use_beta_sigmoid(true);
    attrs.set_allow_neg_eigval(true);
    attrs.set_batch_invariant(true);

    auto graphT = buildAndDeserialize(attrs, GdnVariant::DENSE_STATE);

    ASSERT_EQ(graphT.tensors.size(), 12u);
    auto tensorMap = buildTensorMap(graphT);
    const std::vector<std::pair<int64_t, std::string>> expectedTensors = {
        {K_GDN_TENSOR_Q_UID, "q"},
        {K_GDN_TENSOR_K_UID, "k"},
        {K_GDN_TENSOR_V_UID, "v"},
        {K_GDN_TENSOR_G_UID, "g"},
        {K_GDN_TENSOR_BETA_UID, "beta"},
        {K_GDN_TENSOR_CU_SEQLENS_UID, "cu_seqlens"},
        {K_GDN_TENSOR_INITIAL_STATE_UID, "initial_state"},
        {K_GDN_TENSOR_A_LOG_UID, "a_log"},
        {K_GDN_TENSOR_DT_BIAS_UID, "dt_bias"},
        {K_GDN_TENSOR_O_UID, "o"},
        {K_GDN_TENSOR_FINAL_STATE_UID, "final_state"},
        {K_GDN_TENSOR_STATE_CHECKPOINTS_UID, "state_checkpoints"},
    };
    for(const auto& [uid, name] : expectedTensors)
    {
        ASSERT_NE(tensorMap.count(uid), 0u) << name;
        EXPECT_EQ(tensorMap[uid]->name, name);
    }
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_Q_UID]->dims, toVec(K_GDN_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_Q_UID]->strides, toVec(K_GDN_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_CU_SEQLENS_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_O_UID]->dims, toVec(K_GDN_TENSOR_O_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_O_UID]->strides, toVec(K_GDN_TENSOR_O_STRIDES));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]->dims, toVec(K_GDN_TENSOR_FINAL_STATE_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_STATE_CHECKPOINTS_UID]->dims,
              toVec(K_GDN_TENSOR_STATE_CHECKPOINTS_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_STATE_CHECKPOINTS_UID]->strides,
              toVec(K_GDN_TENSOR_STATE_CHECKPOINTS_STRIDES));

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::GdnAttributes);
    EXPECT_EQ(node->name, "test_op");

    auto* opNode = node->attributes.AsGdnAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->q_tensor_uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(opNode->k_tensor_uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(opNode->v_tensor_uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(opNode->g_tensor_uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(opNode->beta_tensor_uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(opNode->cu_seqlens_tensor_uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(opNode->o_tensor_uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(opNode->initial_state_tensor_uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(opNode->a_log_tensor_uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(opNode->dt_bias_tensor_uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_FALSE(opNode->state_indices_tensor_uid.has_value());
    EXPECT_EQ(opNode->final_state_tensor_uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(opNode->state_checkpoints_tensor_uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);

    ASSERT_TRUE(opNode->scale.has_value());
    EXPECT_FLOAT_EQ(opNode->scale.value(), 0.25F);
    EXPECT_TRUE(opNode->output_final_state);
    EXPECT_TRUE(opNode->use_qk_l2norm);
    EXPECT_EQ(opNode->checkpoint_every_n_tokens, 2);
    EXPECT_TRUE(opNode->use_beta_sigmoid);
    EXPECT_TRUE(opNode->allow_neg_eigval);
    EXPECT_TRUE(opNode->safe_gate);
    EXPECT_EQ(opNode->gate_domain, GateDomainSdk::LOG);
    EXPECT_TRUE(opNode->batch_invariant);
    EXPECT_FALSE(opNode->overwrite_initial_state);
}

// Minimal graph: optional operands and scalars keep their schema defaults.
TEST_F(IntegrationGdnDescriptorLowering, MinimalGraphKeepsDefaults)
{
    GdnAttributes attrs;
    attrs.set_name("test_minimal");
    attrs.set_gate_domain(GateDomain::LINEAR);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.tensors.size(), 7u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsGdnAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_FALSE(opNode->initial_state_tensor_uid.has_value());
    EXPECT_FALSE(opNode->a_log_tensor_uid.has_value());
    EXPECT_FALSE(opNode->dt_bias_tensor_uid.has_value());
    EXPECT_FALSE(opNode->state_indices_tensor_uid.has_value());
    EXPECT_FALSE(opNode->final_state_tensor_uid.has_value());
    EXPECT_FALSE(opNode->state_checkpoints_tensor_uid.has_value());
    EXPECT_FALSE(opNode->scale.has_value());
    EXPECT_FALSE(opNode->output_final_state);
    EXPECT_EQ(opNode->checkpoint_every_n_tokens, 0);
    EXPECT_EQ(opNode->gate_domain, GateDomainSdk::LINEAR);
}

// Pooled state: state_indices addresses the pool and the final state overwrites it.
TEST_F(IntegrationGdnDescriptorLowering, PooledStateRoundTrip)
{
    GdnAttributes attrs;
    attrs.set_name("test_pool");

    auto graphT = buildAndDeserialize(attrs, GdnVariant::POOLED_STATE);

    ASSERT_EQ(graphT.tensors.size(), 10u);
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_STATE_INDICES_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_STATE_INDICES_UID]->data_type, DataTypeSdk::INT32);

    auto* opNode = graphT.nodes[0]->attributes.AsGdnAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->state_indices_tensor_uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(opNode->initial_state_tensor_uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(opNode->final_state_tensor_uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_TRUE(opNode->overwrite_initial_state);
    EXPECT_FALSE(opNode->state_checkpoints_tensor_uid.has_value());
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationGdnDescriptorLowering, OperationComputeDataTypeOverride)
{
    GdnAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

// Verifies that the optional scale attribute survives lowering round-trip.
TEST_F(IntegrationGdnDescriptorLowering, ScalePreservedInRoundTrip)
{
    GdnAttributes attrs;
    attrs.set_name("test_scale");
    attrs.set_scale(0.5F);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsGdnAttributes();
    ASSERT_NE(opNode, nullptr);
    ASSERT_TRUE(opNode->scale.has_value());
    EXPECT_FLOAT_EQ(opNode->scale.value(), 0.5F);
}

} // namespace
