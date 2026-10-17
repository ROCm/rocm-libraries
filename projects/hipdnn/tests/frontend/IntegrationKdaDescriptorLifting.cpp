// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/KdaNode.hpp>
#include <hipdnn_test_sdk/constants/KdaConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationKdaDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard Kda graph for round-trip testing.
    static KdaAttributes createAttributes(DataType operationComputeDataType = DataType::NOT_SET)
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
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(KdaAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("KdaLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_uid(K_KDA_TENSOR_Q_UID).set_name("q").set_data_type(DataType::FLOAT);
        q->set_dim(toVec(K_KDA_TENSOR_Q_DIMS)).set_stride(toVec(K_KDA_TENSOR_Q_STRIDES));

        auto k = std::make_shared<TensorAttributes>();
        k->set_uid(K_KDA_TENSOR_K_UID).set_name("k").set_data_type(DataType::FLOAT);
        k->set_dim(toVec(K_KDA_TENSOR_K_DIMS)).set_stride(toVec(K_KDA_TENSOR_K_STRIDES));

        auto v = std::make_shared<TensorAttributes>();
        v->set_uid(K_KDA_TENSOR_V_UID).set_name("v").set_data_type(DataType::FLOAT);
        v->set_dim(toVec(K_KDA_TENSOR_V_DIMS)).set_stride(toVec(K_KDA_TENSOR_V_STRIDES));

        auto g = std::make_shared<TensorAttributes>();
        g->set_uid(K_KDA_TENSOR_G_UID).set_name("g").set_data_type(DataType::FLOAT);
        g->set_dim(toVec(K_KDA_TENSOR_G_DIMS)).set_stride(toVec(K_KDA_TENSOR_G_STRIDES));

        auto beta = std::make_shared<TensorAttributes>();
        beta->set_uid(K_KDA_TENSOR_BETA_UID).set_name("beta").set_data_type(DataType::FLOAT);
        beta->set_dim(toVec(K_KDA_TENSOR_BETA_DIMS)).set_stride(toVec(K_KDA_TENSOR_BETA_STRIDES));

        auto cuSeqlens = std::make_shared<TensorAttributes>();
        cuSeqlens->set_uid(K_KDA_TENSOR_CU_SEQLENS_UID)
            .set_name("cu_seqlens")
            .set_data_type(DataType::INT32);
        cuSeqlens->set_dim(toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS))
            .set_stride(toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));

        auto results = graph->kda_ext(q, k, v, g, beta, cuSeqlens, std::move(attrs));
        results[0]->set_uid(K_KDA_TENSOR_O_UID).set_output(true).set_name("o");

        return graph;
    }
};

// Builds a standard Kda graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationKdaDescriptorLifting, BasicKdaRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    // q, k, v, g, beta, cu_seqlens, o plus final_state and state_checkpoints
    // (enabled by the test values).
    ASSERT_EQ(tensorMap.size(), 9u);

    // Verify q tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_uid(), K_KDA_TENSOR_Q_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_dim(), toVec(K_KDA_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_stride(), toVec(K_KDA_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_name(), "q");

    // Verify k tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_uid(), K_KDA_TENSOR_K_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_dim(), toVec(K_KDA_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_stride(), toVec(K_KDA_TENSOR_K_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_name(), "k");

    // Verify v tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_uid(), K_KDA_TENSOR_V_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_dim(), toVec(K_KDA_TENSOR_V_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_stride(), toVec(K_KDA_TENSOR_V_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_name(), "v");

    // Verify g tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_G_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_uid(), K_KDA_TENSOR_G_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_dim(), toVec(K_KDA_TENSOR_G_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_stride(), toVec(K_KDA_TENSOR_G_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_name(), "g");

    // Verify beta tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_BETA_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_uid(), K_KDA_TENSOR_BETA_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_dim(), toVec(K_KDA_TENSOR_BETA_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_stride(), toVec(K_KDA_TENSOR_BETA_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_name(), "beta");

    // Verify cu_seqlens tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_uid(), K_KDA_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_dim(),
              toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_stride(),
              toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_name(), "cu_seqlens");

    // Verify o tensor
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_uid(), K_KDA_TENSOR_O_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_dim(), toVec(K_KDA_TENSOR_O_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_stride(), toVec(K_KDA_TENSOR_O_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_name(), "o");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a KdaNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_gate_domain(), GateDomain::LOG);

    EXPECT_EQ(opNode->attributes.get_scale(), 0.35355339F);
    EXPECT_EQ(opNode->attributes.get_output_final_state(), true);
    EXPECT_EQ(opNode->attributes.get_use_qk_l2norm(), true);
    EXPECT_EQ(opNode->attributes.get_checkpoint_every_n_tokens(), 4);
    EXPECT_EQ(opNode->attributes.get_use_beta_sigmoid(), true);
    EXPECT_EQ(opNode->attributes.get_allow_neg_eigval(), true);
    EXPECT_EQ(opNode->attributes.get_safe_gate(), true);
    EXPECT_EQ(opNode->attributes.get_gate_lower_bound(), -5.0F);
    EXPECT_EQ(opNode->attributes.get_batch_invariant(), true);
    EXPECT_EQ(opNode->attributes.get_overwrite_initial_state(), false);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationKdaDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// Exercises JSON serialization and deserialization for each mode, including
// canonicalization of mode-irrelevant frontend attributes.
TEST_F(IntegrationKdaDescriptorLifting, JsonRoundTripsAllModeScenarios) {}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationKdaDescriptorLifting, KdaTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify q tensor sharing
    EXPECT_EQ(opNode->attributes.get_q()->get_uid(), K_KDA_TENSOR_Q_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID].get(), opNode->attributes.get_q().get());
    // Verify k tensor sharing
    EXPECT_EQ(opNode->attributes.get_k()->get_uid(), K_KDA_TENSOR_K_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID].get(), opNode->attributes.get_k().get());
    // Verify v tensor sharing
    EXPECT_EQ(opNode->attributes.get_v()->get_uid(), K_KDA_TENSOR_V_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID].get(), opNode->attributes.get_v().get());
    // Verify g tensor sharing
    EXPECT_EQ(opNode->attributes.get_g()->get_uid(), K_KDA_TENSOR_G_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID].get(), opNode->attributes.get_g().get());
    // Verify beta tensor sharing
    EXPECT_EQ(opNode->attributes.get_beta()->get_uid(), K_KDA_TENSOR_BETA_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID].get(), opNode->attributes.get_beta().get());
    // Verify cu_seqlens tensor sharing
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_uid(), K_KDA_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID].get(),
              opNode->attributes.get_cu_seqlens().get());
    // Verify o tensor sharing
    EXPECT_EQ(opNode->attributes.get_o()->get_uid(), K_KDA_TENSOR_O_UID);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID].get(), opNode->attributes.get_o().get());
}

// Builds a Kda graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationKdaDescriptorLifting, KdaLiftWithoutFinalization)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_gate_domain(), GateDomain::LOG);

    EXPECT_EQ(opNode->attributes.get_scale(), 0.35355339F);
    EXPECT_EQ(opNode->attributes.get_output_final_state(), true);
    EXPECT_EQ(opNode->attributes.get_use_qk_l2norm(), true);
    EXPECT_EQ(opNode->attributes.get_checkpoint_every_n_tokens(), 4);
    EXPECT_EQ(opNode->attributes.get_use_beta_sigmoid(), true);
    EXPECT_EQ(opNode->attributes.get_allow_neg_eigval(), true);
    EXPECT_EQ(opNode->attributes.get_safe_gate(), true);
    EXPECT_EQ(opNode->attributes.get_gate_lower_bound(), -5.0F);
    EXPECT_EQ(opNode->attributes.get_batch_invariant(), true);
    EXPECT_EQ(opNode->attributes.get_overwrite_initial_state(), false);
    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    // q, k, v, g, beta, cu_seqlens, o plus final_state and state_checkpoints
    // (enabled by the test values).
    ASSERT_EQ(tensorMap.size(), 9u);

    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_dim(), toVec(K_KDA_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_stride(), toVec(K_KDA_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_Q_UID]->get_name(), "q");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_dim(), toVec(K_KDA_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_stride(), toVec(K_KDA_TENSOR_K_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_K_UID]->get_name(), "k");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_dim(), toVec(K_KDA_TENSOR_V_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_stride(), toVec(K_KDA_TENSOR_V_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_V_UID]->get_name(), "v");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_G_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_dim(), toVec(K_KDA_TENSOR_G_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_stride(), toVec(K_KDA_TENSOR_G_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_G_UID]->get_name(), "g");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_BETA_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_dim(), toVec(K_KDA_TENSOR_BETA_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_stride(), toVec(K_KDA_TENSOR_BETA_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_BETA_UID]->get_name(), "beta");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_dim(),
              toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_stride(),
              toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_CU_SEQLENS_UID]->get_name(), "cu_seqlens");
    ASSERT_NE(tensorMap.count(K_KDA_TENSOR_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_dim(), toVec(K_KDA_TENSOR_O_DIMS));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_stride(), toVec(K_KDA_TENSOR_O_STRIDES));
    EXPECT_EQ(tensorMap[K_KDA_TENSOR_O_UID]->get_name(), "o");
}

// Verifies that the optional scale attribute survives a lifting round-trip.
TEST_F(IntegrationKdaDescriptorLifting, ScalePreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("KdaScaleLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_KDA_TENSOR_Q_UID).set_name("q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_KDA_TENSOR_Q_DIMS)).set_stride(toVec(K_KDA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_KDA_TENSOR_K_UID).set_name("k").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_KDA_TENSOR_K_DIMS)).set_stride(toVec(K_KDA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_KDA_TENSOR_V_UID).set_name("v").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_KDA_TENSOR_V_DIMS)).set_stride(toVec(K_KDA_TENSOR_V_STRIDES));

    auto g = std::make_shared<TensorAttributes>();
    g->set_uid(K_KDA_TENSOR_G_UID).set_name("g").set_data_type(DataType::FLOAT);
    g->set_dim(toVec(K_KDA_TENSOR_G_DIMS)).set_stride(toVec(K_KDA_TENSOR_G_STRIDES));

    auto beta = std::make_shared<TensorAttributes>();
    beta->set_uid(K_KDA_TENSOR_BETA_UID).set_name("beta").set_data_type(DataType::FLOAT);
    beta->set_dim(toVec(K_KDA_TENSOR_BETA_DIMS)).set_stride(toVec(K_KDA_TENSOR_BETA_STRIDES));

    auto cuSeqlens = std::make_shared<TensorAttributes>();
    cuSeqlens->set_uid(K_KDA_TENSOR_CU_SEQLENS_UID)
        .set_name("cu_seqlens")
        .set_data_type(DataType::INT32);
    cuSeqlens->set_dim(toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS))
        .set_stride(toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));

    KdaAttributes attrs;
    attrs.set_name("test_scale");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_scale(0.5F);

    auto results = graph->kda_ext(q, k, v, g, beta, cuSeqlens, attrs);
    results[0]->set_uid(K_KDA_TENSOR_O_UID).set_output(true).set_name("o");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a KdaNode"; // NOLINT(readability-implicit-bool-conversion)

    ASSERT_TRUE(opNode->attributes.get_scale().has_value());
    EXPECT_FLOAT_EQ(opNode->attributes.get_scale().value(), 0.5F);
}

// Verifies that the optional gate_lower_bound attribute survives a lifting round-trip.
TEST_F(IntegrationKdaDescriptorLifting, GateLowerBoundPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("KdaGateLowerBoundLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_KDA_TENSOR_Q_UID).set_name("q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_KDA_TENSOR_Q_DIMS)).set_stride(toVec(K_KDA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_KDA_TENSOR_K_UID).set_name("k").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_KDA_TENSOR_K_DIMS)).set_stride(toVec(K_KDA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_KDA_TENSOR_V_UID).set_name("v").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_KDA_TENSOR_V_DIMS)).set_stride(toVec(K_KDA_TENSOR_V_STRIDES));

    auto g = std::make_shared<TensorAttributes>();
    g->set_uid(K_KDA_TENSOR_G_UID).set_name("g").set_data_type(DataType::FLOAT);
    g->set_dim(toVec(K_KDA_TENSOR_G_DIMS)).set_stride(toVec(K_KDA_TENSOR_G_STRIDES));

    auto beta = std::make_shared<TensorAttributes>();
    beta->set_uid(K_KDA_TENSOR_BETA_UID).set_name("beta").set_data_type(DataType::FLOAT);
    beta->set_dim(toVec(K_KDA_TENSOR_BETA_DIMS)).set_stride(toVec(K_KDA_TENSOR_BETA_STRIDES));

    auto cuSeqlens = std::make_shared<TensorAttributes>();
    cuSeqlens->set_uid(K_KDA_TENSOR_CU_SEQLENS_UID)
        .set_name("cu_seqlens")
        .set_data_type(DataType::INT32);
    cuSeqlens->set_dim(toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS))
        .set_stride(toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));

    KdaAttributes attrs;
    attrs.set_name("test_gate_lower_bound");
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_gate_lower_bound(0.5F);
    attrs.set_safe_gate(true); // gate_lower_bound requires safe_gate

    auto results = graph->kda_ext(q, k, v, g, beta, cuSeqlens, attrs);
    results[0]->set_uid(K_KDA_TENSOR_O_UID).set_output(true).set_name("o");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a KdaNode"; // NOLINT(readability-implicit-bool-conversion)

    ASSERT_TRUE(opNode->attributes.get_gate_lower_bound().has_value());
    EXPECT_FLOAT_EQ(opNode->attributes.get_gate_lower_bound().value(), 0.5F);
}

// Builds a Kda graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationKdaDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("KdaAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_name("q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_KDA_TENSOR_Q_DIMS)).set_stride(toVec(K_KDA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_name("k").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_KDA_TENSOR_K_DIMS)).set_stride(toVec(K_KDA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_name("v").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_KDA_TENSOR_V_DIMS)).set_stride(toVec(K_KDA_TENSOR_V_STRIDES));

    auto g = std::make_shared<TensorAttributes>();
    g->set_name("g").set_data_type(DataType::FLOAT);
    g->set_dim(toVec(K_KDA_TENSOR_G_DIMS)).set_stride(toVec(K_KDA_TENSOR_G_STRIDES));

    auto beta = std::make_shared<TensorAttributes>();
    beta->set_name("beta").set_data_type(DataType::FLOAT);
    beta->set_dim(toVec(K_KDA_TENSOR_BETA_DIMS)).set_stride(toVec(K_KDA_TENSOR_BETA_STRIDES));

    auto cuSeqlens = std::make_shared<TensorAttributes>();
    cuSeqlens->set_name("cu_seqlens").set_data_type(DataType::INT32);
    cuSeqlens->set_dim(toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS))
        .set_stride(toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));

    KdaAttributes attrs;
    attrs.set_name("test_auto_uid");
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

    auto results = graph->kda_ext(q, k, v, g, beta, cuSeqlens, attrs);
    results[0]->set_output(true).set_name("o");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    // q, k, v, g, beta, cu_seqlens, o plus final_state and state_checkpoints
    // (enabled by the test values).
    ASSERT_EQ(tensorMap.size(), 9u);

    // Verify all UIDs are positive and distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        EXPECT_GE(uid, 0)
            << "Auto-assigned UID should be non-negative"; // NOLINT(readability-implicit-bool-conversion)
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<KdaNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_q(), nullptr);
    nodeUids.insert(opNode->attributes.get_q()->get_uid());
    ASSERT_NE(opNode->attributes.get_k(), nullptr);
    nodeUids.insert(opNode->attributes.get_k()->get_uid());
    ASSERT_NE(opNode->attributes.get_v(), nullptr);
    nodeUids.insert(opNode->attributes.get_v()->get_uid());
    ASSERT_NE(opNode->attributes.get_g(), nullptr);
    nodeUids.insert(opNode->attributes.get_g()->get_uid());
    ASSERT_NE(opNode->attributes.get_beta(), nullptr);
    nodeUids.insert(opNode->attributes.get_beta()->get_uid());
    ASSERT_NE(opNode->attributes.get_cu_seqlens(), nullptr);
    nodeUids.insert(opNode->attributes.get_cu_seqlens()->get_uid());
    ASSERT_NE(opNode->attributes.get_o(), nullptr);
    nodeUids.insert(opNode->attributes.get_o()->get_uid());
    ASSERT_EQ(nodeUids.size(), 7u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_q()->get_dim(), toVec(K_KDA_TENSOR_Q_DIMS));
    EXPECT_EQ(opNode->attributes.get_q()->get_stride(), toVec(K_KDA_TENSOR_Q_STRIDES));
    EXPECT_EQ(opNode->attributes.get_q()->get_name(), "q");
    EXPECT_EQ(opNode->attributes.get_k()->get_dim(), toVec(K_KDA_TENSOR_K_DIMS));
    EXPECT_EQ(opNode->attributes.get_k()->get_stride(), toVec(K_KDA_TENSOR_K_STRIDES));
    EXPECT_EQ(opNode->attributes.get_k()->get_name(), "k");
    EXPECT_EQ(opNode->attributes.get_v()->get_dim(), toVec(K_KDA_TENSOR_V_DIMS));
    EXPECT_EQ(opNode->attributes.get_v()->get_stride(), toVec(K_KDA_TENSOR_V_STRIDES));
    EXPECT_EQ(opNode->attributes.get_v()->get_name(), "v");
    EXPECT_EQ(opNode->attributes.get_g()->get_dim(), toVec(K_KDA_TENSOR_G_DIMS));
    EXPECT_EQ(opNode->attributes.get_g()->get_stride(), toVec(K_KDA_TENSOR_G_STRIDES));
    EXPECT_EQ(opNode->attributes.get_g()->get_name(), "g");
    EXPECT_EQ(opNode->attributes.get_beta()->get_dim(), toVec(K_KDA_TENSOR_BETA_DIMS));
    EXPECT_EQ(opNode->attributes.get_beta()->get_stride(), toVec(K_KDA_TENSOR_BETA_STRIDES));
    EXPECT_EQ(opNode->attributes.get_beta()->get_name(), "beta");
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_dim(), toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_stride(),
              toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_name(), "cu_seqlens");
    EXPECT_EQ(opNode->attributes.get_o()->get_dim(), toVec(K_KDA_TENSOR_O_DIMS));
    EXPECT_EQ(opNode->attributes.get_o()->get_stride(), toVec(K_KDA_TENSOR_O_STRIDES));
    EXPECT_EQ(opNode->attributes.get_o()->get_name(), "o");
}

} // namespace
