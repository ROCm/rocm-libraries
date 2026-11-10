// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/CausalConv1dOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/CausalConv1dConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_backend::test_utilities::verifyTensorDescriptor;
using hipdnn_tests::toVec;

// =============================================================================
// CausalConv1dOperationDescriptor::fromNode() Tests
// =============================================================================

class TestCausalConv1dOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_CAUSAL_CONV1D_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT weightAttrs;
        weightAttrs.uid = K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID;
        weightAttrs.data_type = DataType::FLOAT;
        weightAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS);
        weightAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]
            = TensorDescriptor::fromFlatBuffer(weightAttrs);
        TensorAttributesT cuSeqlensAttrs;
        cuSeqlensAttrs.uid = K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID;
        cuSeqlensAttrs.data_type = DataType::INT32;
        cuSeqlensAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS);
        cuSeqlensAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]
            = TensorDescriptor::fromFlatBuffer(cuSeqlensAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_CAUSAL_CONV1D_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_CAUSAL_CONV1D_TENSOR_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS);
        biasAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT initialStateAttrs;
        initialStateAttrs.uid = K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID;
        initialStateAttrs.data_type = DataType::FLOAT;
        initialStateAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS);
        initialStateAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID]
            = TensorDescriptor::fromFlatBuffer(initialStateAttrs);
        TensorAttributesT stateIndicesAttrs;
        stateIndicesAttrs.uid = K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID;
        stateIndicesAttrs.data_type = DataType::INT32;
        stateIndicesAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS);
        stateIndicesAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID]
            = TensorDescriptor::fromFlatBuffer(stateIndicesAttrs);
        TensorAttributesT finalStateAttrs;
        finalStateAttrs.uid = K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID;
        finalStateAttrs.data_type = DataType::FLOAT;
        finalStateAttrs.dims = toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS);
        finalStateAttrs.strides = toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES);

        _tensorMap[K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID]
            = TensorDescriptor::fromFlatBuffer(finalStateAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::CausalConv1dAttributesT
        createStandardCausalConv1dAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::CausalConv1dAttributesT attrs;
        attrs.x_tensor_uid = K_CAUSAL_CONV1D_TENSOR_X_UID;
        attrs.weight_tensor_uid = K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID;
        attrs.cu_seqlens_tensor_uid = K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID;
        attrs.y_tensor_uid = K_CAUSAL_CONV1D_TENSOR_Y_UID;
        attrs.bias_tensor_uid = K_CAUSAL_CONV1D_TENSOR_BIAS_UID;
        attrs.initial_state_tensor_uid = K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID;
        attrs.state_indices_tensor_uid = K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID;
        attrs.final_state_tensor_uid = K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID;
        attrs.activation = CausalConv1dActivation::SILU;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardCausalConv1dAttrs());
        return node;
    }
};

TEST_F(TestCausalConv1dOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CAUSAL_CONV1D_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
}

TEST_F(TestCausalConv1dOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    // NodeFactory::createOperationFromNode delegates to fromNode internally.
    // Verify the delegation produces a valid, correctly-typed descriptor.
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    const auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::CausalConv1dAttributes);
    auto desc = std::static_pointer_cast<CausalConv1dOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().weight_tensor_uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getData().cu_seqlens_tensor_uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getData().initial_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getData().state_indices_tensor_uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getData().final_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(desc->getData().activation, CausalConv1dActivation::SILU);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
}

TEST_F(TestCausalConv1dOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestCausalConv1dOperationFromNode, PreservesCausalConv1dActivation)
{
    auto node = createStandardNode();
    auto attrs = createStandardCausalConv1dAttrs();
    attrs.activation = CausalConv1dActivation::IDENTITY;
    node.attributes.Set(attrs);
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().activation, CausalConv1dActivation::IDENTITY);
}

TEST_F(TestCausalConv1dOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
}

TEST_F(TestCausalConv1dOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_X_UID]);
    EXPECT_EQ(desc->getWeightDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID]);
    EXPECT_EQ(desc->getCuSeqlensDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_Y_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_BIAS_UID]);
    EXPECT_EQ(desc->getInitialStateDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID]);
    EXPECT_EQ(desc->getStateIndicesDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID]);
    EXPECT_EQ(desc->getFinalStateDesc(), _tensorMap[K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID]);
}

TEST_F(TestCausalConv1dOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{4, 1}));

    ASSERT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getWeightDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getWeightDesc()->getData().dims, (std::vector<int64_t>{4, 4}));
    EXPECT_EQ(desc->getWeightDesc()->getData().strides, (std::vector<int64_t>{4, 1}));

    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().dims, (std::vector<int64_t>{3}));
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{4, 1}));

    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims, (std::vector<int64_t>{4}));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().dims, (std::vector<int64_t>{2, 4, 3}));
    EXPECT_EQ(desc->getInitialStateDesc()->getData().strides, (std::vector<int64_t>{12, 3, 1}));

    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().dims, (std::vector<int64_t>{2}));
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().dims, (std::vector<int64_t>{2, 4, 3}));
    EXPECT_EQ(desc->getFinalStateDesc()->getData().strides, (std::vector<int64_t>{12, 3, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestCausalConv1dOperationFromNodeMissingTensor : public TestCausalConv1dOperationFromNode,
                                                       public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestCausalConv1dOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CausalConv1dOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestCausalConv1dOperationFromNodeMissingTensor,
                         ::testing::Values(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                           K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                           K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                           K_CAUSAL_CONV1D_TENSOR_Y_UID));

TEST_F(TestCausalConv1dOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardCausalConv1dAttrs();
    attrs.bias_tensor_uid = flatbuffers::nullopt;
    attrs.initial_state_tensor_uid = flatbuffers::nullopt;
    attrs.state_indices_tensor_uid = flatbuffers::nullopt;
    attrs.final_state_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_NE(desc->getYDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4u);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getWeightDesc());
    EXPECT_EQ(tensors[2], desc->getCuSeqlensDesc());
    EXPECT_EQ(tensors[3], desc->getYDesc());
}

TEST_F(TestCausalConv1dOperationFromNode, FailsWhenOptionalBiasUidSetButTensorMissing)
{
    _tensorMap.erase(K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CausalConv1dOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCausalConv1dOperationFromNode, FailsWhenOptionalInitialStateUidSetButTensorMissing)
{
    _tensorMap.erase(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CausalConv1dOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCausalConv1dOperationFromNode, FailsWhenOptionalStateIndicesUidSetButTensorMissing)
{
    _tensorMap.erase(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CausalConv1dOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCausalConv1dOperationFromNode, FailsWhenOptionalFinalStateUidSetButTensorMissing)
{
    _tensorMap.erase(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CausalConv1dOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCausalConv1dOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    EXPECT_EQ(tensors[0]->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
}

TEST_F(TestCausalConv1dOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::CausalConv1dAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCausalConv1dAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->weight_tensor_uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(rebuiltAttrs->cu_seqlens_tensor_uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->initial_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(rebuiltAttrs->state_indices_tensor_uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(rebuiltAttrs->final_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(rebuiltAttrs->activation, CausalConv1dActivation::SILU);
}

TEST_F(TestCausalConv1dOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify activation
    hipdnnCausalConv1dActivation_t activation = {};
    int64_t activationCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                       HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                       1,
                       &activationCount,
                       &activation);
    ASSERT_EQ(activation, HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_CAUSAL_CONV1D_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {8, 4}, {4, 1});

    // Verify weight tensor
    hipdnn_backend::ScopedDescriptor weightScoped;
    int64_t weightCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &weightCount,
                       static_cast<void*>(weightScoped.getPtr()));
    ASSERT_EQ(weightCount, 1);
    ASSERT_NE(weightScoped.get(), nullptr);
    verifyTensorDescriptor(
        weightScoped.get(), K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID, HIPDNN_DATA_FLOAT, {4, 4}, {4, 1});

    // Verify cu_seqlens tensor
    hipdnn_backend::ScopedDescriptor cuSeqlensScoped;
    int64_t cuSeqlensCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &cuSeqlensCount,
                       static_cast<void*>(cuSeqlensScoped.getPtr()));
    ASSERT_EQ(cuSeqlensCount, 1);
    ASSERT_NE(cuSeqlensScoped.get(), nullptr);
    verifyTensorDescriptor(
        cuSeqlensScoped.get(), K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID, HIPDNN_DATA_INT32, {3}, {1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(
        yScoped.get(), K_CAUSAL_CONV1D_TENSOR_Y_UID, HIPDNN_DATA_FLOAT, {8, 4}, {4, 1});

    // Verify bias tensor (optional)
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    verifyTensorDescriptor(
        biasScoped.get(), K_CAUSAL_CONV1D_TENSOR_BIAS_UID, HIPDNN_DATA_FLOAT, {4}, {1});

    // Verify initial_state tensor (optional)
    hipdnn_backend::ScopedDescriptor initialStateScoped;
    int64_t initialStateCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &initialStateCount,
                       static_cast<void*>(initialStateScoped.getPtr()));
    ASSERT_EQ(initialStateCount, 1);
    ASSERT_NE(initialStateScoped.get(), nullptr);
    verifyTensorDescriptor(initialStateScoped.get(),
                           K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 4, 3},
                           {12, 3, 1});

    // Verify state_indices tensor (optional)
    hipdnn_backend::ScopedDescriptor stateIndicesScoped;
    int64_t stateIndicesCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &stateIndicesCount,
                       static_cast<void*>(stateIndicesScoped.getPtr()));
    ASSERT_EQ(stateIndicesCount, 1);
    ASSERT_NE(stateIndicesScoped.get(), nullptr);
    verifyTensorDescriptor(stateIndicesScoped.get(),
                           K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                           HIPDNN_DATA_INT32,
                           {2},
                           {1});

    // Verify final_state tensor (optional)
    hipdnn_backend::ScopedDescriptor finalStateScoped;
    int64_t finalStateCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &finalStateCount,
                       static_cast<void*>(finalStateScoped.getPtr()));
    ASSERT_EQ(finalStateCount, 1);
    ASSERT_NE(finalStateScoped.get(), nullptr);
    verifyTensorDescriptor(finalStateScoped.get(),
                           K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 4, 3},
                           {12, 3, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CAUSAL_CONV1D_EXT);
}

TEST_F(TestCausalConv1dOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_causalconv1d_1";

    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_causalconv1d_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_causalconv1d_1");
}

TEST_F(TestCausalConv1dOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestCausalConv1dOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = CausalConv1dOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
