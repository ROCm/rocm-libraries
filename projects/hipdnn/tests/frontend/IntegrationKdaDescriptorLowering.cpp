// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/kda_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/KdaConstants.hpp>
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

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationKdaDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(KdaAttributes& attrs,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("KdaIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            q->set_uid(K_KDA_TENSOR_Q_UID);
        }
        q->set_name("q").set_data_type(DataType::FLOAT);
        q->set_dim(toVec(K_KDA_TENSOR_Q_DIMS)).set_stride(toVec(K_KDA_TENSOR_Q_STRIDES));

        auto k = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            k->set_uid(K_KDA_TENSOR_K_UID);
        }
        k->set_name("k").set_data_type(DataType::FLOAT);
        k->set_dim(toVec(K_KDA_TENSOR_K_DIMS)).set_stride(toVec(K_KDA_TENSOR_K_STRIDES));

        auto v = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            v->set_uid(K_KDA_TENSOR_V_UID);
        }
        v->set_name("v").set_data_type(DataType::FLOAT);
        v->set_dim(toVec(K_KDA_TENSOR_V_DIMS)).set_stride(toVec(K_KDA_TENSOR_V_STRIDES));

        auto g = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            g->set_uid(K_KDA_TENSOR_G_UID);
        }
        g->set_name("g").set_data_type(DataType::FLOAT);
        g->set_dim(toVec(K_KDA_TENSOR_G_DIMS)).set_stride(toVec(K_KDA_TENSOR_G_STRIDES));

        auto beta = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            beta->set_uid(K_KDA_TENSOR_BETA_UID);
        }
        beta->set_name("beta").set_data_type(DataType::FLOAT);
        beta->set_dim(toVec(K_KDA_TENSOR_BETA_DIMS)).set_stride(toVec(K_KDA_TENSOR_BETA_STRIDES));

        auto cuSeqlens = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            cuSeqlens->set_uid(K_KDA_TENSOR_CU_SEQLENS_UID);
        }
        cuSeqlens->set_name("cu_seqlens").set_data_type(DataType::INT32);
        cuSeqlens->set_dim(toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS))
            .set_stride(toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));

        // Hand-written: the generator emits this call only for single-output methods.
        auto [o, finalState, stateCheckpoints] = graph->kda_ext(q, k, v, g, beta, cuSeqlens, attrs);
        o->set_output(true).set_name("o");
        if(assignUids)
        {
            o->set_uid(K_KDA_TENSOR_O_UID);
        }
        if(finalState)
        {
            finalState->set_output(true).set_name("final_state");
            if(assignUids)
            {
                finalState->set_uid(K_KDA_TENSOR_FINAL_STATE_UID);
            }
        }
        if(stateCheckpoints)
        {
            stateCheckpoints->set_output(true).set_name("state_checkpoints");
            if(assignUids)
            {
                stateCheckpoints->set_uid(K_KDA_TENSOR_STATE_CHECKPOINTS_UID);
            }
        }

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationKdaDescriptorLowering, KdaLoweringRoundTrip)
{
    KdaAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_scale(0.35355339F);
    attrs.set_output_final_state(true);
    attrs.set_use_qk_l2norm(true);
    attrs.set_checkpoint_every_n_tokens(4);
    attrs.set_use_beta_sigmoid(true);
    attrs.set_allow_neg_eigval(true);
    attrs.set_safe_gate(true);
    attrs.set_gate_lower_bound(-5.0F);
    attrs.set_batch_invariant(true);
    attrs.set_overwrite_initial_state(false);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors: q, k, v, g, beta, cu_seqlens, o, plus final_state
    // (output_final_state) and state_checkpoints (checkpoint_every_n_tokens = 4).
    ASSERT_EQ(graphT.tensors.size(), 9u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->dims, toVec(K_KDA_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->strides, toVec(K_KDA_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->name, "q");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->dims, toVec(K_KDA_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->strides, toVec(K_KDA_TENSOR_K_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->name, "k");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->dims, toVec(K_KDA_TENSOR_V_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->strides, toVec(K_KDA_TENSOR_V_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->name, "v");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_G_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->dims, toVec(K_KDA_TENSOR_G_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->strides, toVec(K_KDA_TENSOR_G_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->name, "g");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_BETA_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->dims, toVec(K_KDA_TENSOR_BETA_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->strides, toVec(K_KDA_TENSOR_BETA_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->name, "beta");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->dims, toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->strides,
              toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->name, "cu_seqlens");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->dims, toVec(K_KDA_TENSOR_O_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->strides, toVec(K_KDA_TENSOR_O_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->name, "o");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::KdaAttributes);

    auto* opNode = node->attributes.AsKdaAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->q_tensor_uid, K_KDA_TENSOR_Q_UID);
    EXPECT_EQ(opNode->k_tensor_uid, K_KDA_TENSOR_K_UID);
    EXPECT_EQ(opNode->v_tensor_uid, K_KDA_TENSOR_V_UID);
    EXPECT_EQ(opNode->g_tensor_uid, K_KDA_TENSOR_G_UID);
    EXPECT_EQ(opNode->beta_tensor_uid, K_KDA_TENSOR_BETA_UID);
    EXPECT_EQ(opNode->cu_seqlens_tensor_uid, K_KDA_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(opNode->o_tensor_uid, K_KDA_TENSOR_O_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->gate_domain, GateDomainSdk::LOG);

    EXPECT_EQ(opNode->scale, 0.35355339F);
    EXPECT_EQ(opNode->output_final_state, true);
    EXPECT_EQ(opNode->use_qk_l2norm, true);
    EXPECT_EQ(opNode->checkpoint_every_n_tokens, 4);
    EXPECT_EQ(opNode->use_beta_sigmoid, true);
    EXPECT_EQ(opNode->allow_neg_eigval, true);
    EXPECT_EQ(opNode->safe_gate, true);
    EXPECT_EQ(opNode->gate_lower_bound, -5.0F);
    EXPECT_EQ(opNode->batch_invariant, true);
    EXPECT_EQ(opNode->overwrite_initial_state, false);
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationKdaDescriptorLowering, OperationComputeDataTypeOverride)
{
    KdaAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_scale(0.35355339F);
    attrs.set_output_final_state(true);
    attrs.set_use_qk_l2norm(true);
    attrs.set_checkpoint_every_n_tokens(4);
    attrs.set_use_beta_sigmoid(true);
    attrs.set_allow_neg_eigval(true);
    attrs.set_safe_gate(true);
    attrs.set_gate_lower_bound(-5.0F);
    attrs.set_batch_invariant(true);
    attrs.set_overwrite_initial_state(false);
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

// Verifies that the optional scale attribute survives lowering round-trip.
TEST_F(IntegrationKdaDescriptorLowering, ScalePreservedInRoundTrip)
{
    KdaAttributes attrs;
    attrs.set_name("test_scale");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_scale(0.5F);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsKdaAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->scale.has_value());
    EXPECT_FLOAT_EQ(opNode->scale.value(), 0.5F);
}

// Verifies that the optional gate_lower_bound attribute survives lowering round-trip.
TEST_F(IntegrationKdaDescriptorLowering, GateLowerBoundPreservedInRoundTrip)
{
    KdaAttributes attrs;
    attrs.set_name("test_gate_lower_bound");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_gate_lower_bound(0.5F);
    attrs.set_safe_gate(true); // gate_lower_bound requires safe_gate

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsKdaAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->gate_lower_bound.has_value());
    EXPECT_FLOAT_EQ(opNode->gate_lower_bound.value(), 0.5F);
}

} // namespace
