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
#include <hipdnn_frontend/node/CausalConv1dNode.hpp>
#include <hipdnn_test_sdk/constants/CausalConv1dConstants.hpp>
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
class IntegrationCausalConv1dDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard CausalConv1d graph for round-trip testing.
    static CausalConv1dAttributes createAttributes(DataType operationComputeDataType
                                                   = DataType::NOT_SET)
    {
        CausalConv1dAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_activation(CausalConv1dActivation::SILU);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(CausalConv1dAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("CausalConv1dLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_CAUSAL_CONV1D_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS))
            .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES));

        auto weight = std::make_shared<TensorAttributes>();
        weight->set_uid(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID)
            .set_name("weight")
            .set_data_type(DataType::FLOAT);
        weight->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS))
            .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES));

        auto cuSeqlens = std::make_shared<TensorAttributes>();
        cuSeqlens->set_uid(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID)
            .set_name("cu_seqlens")
            .set_data_type(DataType::INT32);
        cuSeqlens->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS))
            .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES));

        auto results = graph->causal_conv1d(x, weight, cuSeqlens, std::move(attrs));
        results[0]->set_uid(K_CAUSAL_CONV1D_TENSOR_Y_UID).set_output(true).set_name("y");

        return graph;
    }
};

// Builds a standard CausalConv1d graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationCausalConv1dDescriptorLifting, BasicCausalConv1dRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 4u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_uid(), K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_name(), "x");

    // Verify weight tensor
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_uid(),
              K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_name(), "weight");

    // Verify cu_seqlens tensor
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_uid(),
              K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_name(), "cu_seqlens");

    // Verify y tensor
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_uid(), K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_name(), "y");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a CausalConv1dNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_activation(), CausalConv1dActivation::SILU);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationCausalConv1dDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// Lifts a graph carrying every optional tensor and verifies each one, plus the
// output_final_state flag derived from the final_state tensor, survives lifting.
TEST_F(IntegrationCausalConv1dDescriptorLifting, CausalConv1dStateTensorsSurviveLifting)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("CausalConv1dStateLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    const auto makeTensor
        = [](int64_t uid, const char* name, DataType dataType, auto dims, auto strides) {
              auto tensor = std::make_shared<TensorAttributes>();
              tensor->set_uid(uid).set_name(name).set_data_type(dataType);
              tensor->set_dim(toVec(dims)).set_stride(toVec(strides));
              return tensor;
          };

    auto x = makeTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                        "x",
                        DataType::FLOAT,
                        K_CAUSAL_CONV1D_TENSOR_X_DIMS,
                        K_CAUSAL_CONV1D_TENSOR_X_STRIDES);
    auto weight = makeTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                             "weight",
                             DataType::FLOAT,
                             K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS,
                             K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES);
    auto cuSeqlens = makeTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                "cu_seqlens",
                                DataType::INT32,
                                K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS,
                                K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES);

    auto attrs = createAttributes();
    attrs
        .set_bias(makeTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                             "bias",
                             DataType::FLOAT,
                             K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS,
                             K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES))
        .set_initial_state(makeTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                      "initial_state",
                                      DataType::FLOAT,
                                      K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS,
                                      K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES))
        .set_state_indices(makeTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                      "state_indices",
                                      DataType::INT32,
                                      K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS,
                                      K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES))
        .set_output_final_state(true);

    auto results = graph->causal_conv1d(x, weight, cuSeqlens, std::move(attrs));
    results[0]->set_uid(K_CAUSAL_CONV1D_TENSOR_Y_UID).set_output(true).set_name("y");
    ASSERT_NE(results[1], nullptr);
    results[1]
        ->set_uid(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID)
        .set_output(true)
        .set_name("final_state");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 8u);
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES));
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID]->get_data_type(),
              DataType::INT32);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(opNode->attributes.get_output_final_state());
    ASSERT_NE(opNode->attributes.get_bias(), nullptr);
    EXPECT_EQ(opNode->attributes.get_bias()->get_uid(), K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_NE(opNode->attributes.get_initial_state(), nullptr);
    EXPECT_EQ(opNode->attributes.get_initial_state()->get_uid(),
              K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_NE(opNode->attributes.get_state_indices(), nullptr);
    EXPECT_EQ(opNode->attributes.get_state_indices()->get_uid(),
              K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_NE(opNode->attributes.get_final_state(), nullptr);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID].get(),
              opNode->attributes.get_final_state().get());
    EXPECT_EQ(opNode->attributes.get_activation(), CausalConv1dActivation::SILU);
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationCausalConv1dDescriptorLifting, CausalConv1dTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify weight tensor sharing
    EXPECT_EQ(opNode->attributes.get_weight()->get_uid(), K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID].get(),
              opNode->attributes.get_weight().get());
    // Verify cu_seqlens tensor sharing
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_uid(),
              K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID].get(),
              opNode->attributes.get_cu_seqlens().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a CausalConv1d graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationCausalConv1dDescriptorLifting, CausalConv1dLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_activation(), CausalConv1dActivation::SILU);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]->get_name(), "x");
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]->get_name(), "weight");
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]->get_name(), "cu_seqlens");
    ASSERT_NE(tensorMap.count(K_CAUSAL_CONV1D_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]->get_name(), "y");
}

// Builds a CausalConv1d graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationCausalConv1dDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("CausalConv1dAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS))
        .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES));

    auto weight = std::make_shared<TensorAttributes>();
    weight->set_name("weight").set_data_type(DataType::FLOAT);
    weight->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS))
        .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES));

    auto cuSeqlens = std::make_shared<TensorAttributes>();
    cuSeqlens->set_name("cu_seqlens").set_data_type(DataType::INT32);
    cuSeqlens->set_dim(toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS))
        .set_stride(toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES));

    CausalConv1dAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_activation(CausalConv1dActivation::SILU);

    auto results = graph->causal_conv1d(x, weight, cuSeqlens, attrs);
    results[0]->set_output(true).set_name("y");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

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

    auto* opNode = dynamic_cast<CausalConv1dNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_weight(), nullptr);
    nodeUids.insert(opNode->attributes.get_weight()->get_uid());
    ASSERT_NE(opNode->attributes.get_cu_seqlens(), nullptr);
    nodeUids.insert(opNode->attributes.get_cu_seqlens()->get_uid());
    ASSERT_NE(opNode->attributes.get_y(), nullptr);
    nodeUids.insert(opNode->attributes.get_y()->get_uid());
    ASSERT_EQ(nodeUids.size(), 4u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(opNode->attributes.get_weight()->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(opNode->attributes.get_weight()->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(opNode->attributes.get_weight()->get_name(), "weight");
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_dim(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS));
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_stride(),
              toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_cu_seqlens()->get_name(), "cu_seqlens");
    EXPECT_EQ(opNode->attributes.get_y()->get_dim(), toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS));
    EXPECT_EQ(opNode->attributes.get_y()->get_stride(), toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES));
    EXPECT_EQ(opNode->attributes.get_y()->get_name(), "y");
}

} // namespace
