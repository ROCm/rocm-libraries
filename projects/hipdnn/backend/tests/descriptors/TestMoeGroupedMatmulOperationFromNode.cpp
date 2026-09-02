// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/MoeGroupedMatmulOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulConstants.hpp>
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
// MoeGroupedMatmulOperationDescriptor::fromNode() Tests
// =============================================================================

class TestMoeGroupedMatmulOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT tokenAttrs;
        tokenAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID;
        tokenAttrs.data_type = DataType::FLOAT;
        tokenAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS);
        tokenAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]
            = TensorDescriptor::fromFlatBuffer(tokenAttrs);
        TensorAttributesT weightAttrs;
        weightAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID;
        weightAttrs.data_type = DataType::FLOAT;
        weightAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS);
        weightAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]
            = TensorDescriptor::fromFlatBuffer(weightAttrs);
        TensorAttributesT firstTokenOffsetAttrs;
        firstTokenOffsetAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID;
        firstTokenOffsetAttrs.data_type = DataType::INT32;
        firstTokenOffsetAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS);
        firstTokenOffsetAttrs.strides
            = toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]
            = TensorDescriptor::fromFlatBuffer(firstTokenOffsetAttrs);
        TensorAttributesT tokenIndexAttrs;
        tokenIndexAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID;
        tokenIndexAttrs.data_type = DataType::INT32;
        tokenIndexAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS);
        tokenIndexAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID]
            = TensorDescriptor::fromFlatBuffer(tokenIndexAttrs);
        TensorAttributesT tokenKsAttrs;
        tokenKsAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID;
        tokenKsAttrs.data_type = DataType::INT32;
        tokenKsAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS);
        tokenKsAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID]
            = TensorDescriptor::fromFlatBuffer(tokenKsAttrs);
        TensorAttributesT outputAttrs;
        outputAttrs.uid = K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID;
        outputAttrs.data_type = DataType::FLOAT;
        outputAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS);
        outputAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]
            = TensorDescriptor::fromFlatBuffer(outputAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributesT
        createStandardMoeGroupedMatmulAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributesT attrs;
        attrs.token_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID;
        attrs.weight_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID;
        attrs.first_token_offset_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID;
        attrs.token_index_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID;
        attrs.token_ks_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID;
        attrs.output_tensor_uid = K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID;
        attrs.mode = MoeGroupedMatmulMode::SCATTER;
        attrs.top_k = 2;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardMoeGroupedMatmulAttrs());
        return node;
    }
};

TEST_F(TestMoeGroupedMatmulOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR);
    EXPECT_EQ(desc->getData().token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeGroupedMatmulAttributes);
    auto desc = std::static_pointer_cast<MoeGroupedMatmulOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getData().weight_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getData().first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getData().token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getData().token_ks_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(desc->getData().output_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(desc->getData().mode, MoeGroupedMatmulMode::SCATTER);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getOutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, PreservesNONEMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::NONE;
    attrs.token_index_tensor_uid = flatbuffers::nullopt;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().mode, MoeGroupedMatmulMode::NONE);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsTokenIndexInNONEMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::NONE;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsTokenKsInNONEMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::NONE;
    attrs.token_index_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsNoncanonicalTopKInNONEMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::NONE;
    attrs.token_index_tensor_uid = flatbuffers::nullopt;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 1;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, PreservesGATHERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::GATHER;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().mode, MoeGroupedMatmulMode::GATHER);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsMissingTokenIndexInGATHERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::GATHER;
    attrs.token_index_tensor_uid = flatbuffers::nullopt;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsTokenKsInGATHERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::GATHER;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsNoncanonicalTopKInGATHERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::GATHER;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.top_k = 1;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, PreservesSCATTERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::SCATTER;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().mode, MoeGroupedMatmulMode::SCATTER);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsMissingTokenIndexInSCATTERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::SCATTER;
    attrs.token_index_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsMissingTokenKsInSCATTERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::SCATTER;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsBelowMinimumTopKInSCATTERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::SCATTER;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, RejectsAboveMaximumTopKInSCATTERMode)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.mode = MoeGroupedMatmulMode::SCATTER;
    attrs.top_k = 3;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getTokenDesc(), nullptr);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_NE(desc->getOutputDesc(), nullptr);
    EXPECT_EQ(desc->getOutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getTokenDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]);
    EXPECT_EQ(desc->getWeightDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc(),
              _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]);
    EXPECT_EQ(desc->getOutputDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]);
    EXPECT_EQ(desc->getTokenIndexDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID]);
    EXPECT_EQ(desc->getTokenKsDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID]);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getTokenDesc(), nullptr);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getTokenDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getTokenDesc()->getData().dims, (std::vector<int64_t>{1, 8, 16}));
    EXPECT_EQ(desc->getTokenDesc()->getData().strides, (std::vector<int64_t>{128, 16, 1}));

    ASSERT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_EQ(desc->getWeightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(desc->getWeightDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getWeightDesc()->getData().dims, (std::vector<int64_t>{2, 16, 32}));
    EXPECT_EQ(desc->getWeightDesc()->getData().strides, (std::vector<int64_t>{512, 32, 1}));

    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().dims, (std::vector<int64_t>{2, 1, 1}));
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1}));

    ASSERT_NE(desc->getOutputDesc(), nullptr);
    EXPECT_EQ(desc->getOutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(desc->getOutputDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getOutputDesc()->getData().dims, (std::vector<int64_t>{1, 8, 32}));
    EXPECT_EQ(desc->getOutputDesc()->getData().strides, (std::vector<int64_t>{256, 32, 1}));

    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().dims, (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().strides, (std::vector<int64_t>{8, 1, 1}));

    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().dims, (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(desc->getTokenKsDesc()->getData().strides, (std::vector<int64_t>{8, 1, 1}));
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWithMissingTokenTensor)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWithMissingWeightTensor)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWithMissingFirstTokenOffsetTensor)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWithMissingOutputTensor)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardMoeGroupedMatmulAttrs();
    attrs.token_index_tensor_uid = flatbuffers::nullopt;
    attrs.token_ks_tensor_uid = flatbuffers::nullopt;
    attrs.mode = MoeGroupedMatmulMode::NONE;
    attrs.top_k = 0;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getTokenDesc(), nullptr);
    EXPECT_NE(desc->getWeightDesc(), nullptr);
    EXPECT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_NE(desc->getOutputDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getTokenIndexDesc(), nullptr);
    EXPECT_EQ(desc->getTokenKsDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4u);
    EXPECT_EQ(tensors[0], desc->getTokenDesc());
    EXPECT_EQ(tensors[1], desc->getWeightDesc());
    EXPECT_EQ(tensors[2], desc->getFirstTokenOffsetDesc());
    EXPECT_EQ(tensors[3], desc->getOutputDesc());
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWhenOptionalTokenIndexUidSetButTensorMissing)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, FailsWhenOptionalTokenKsUidSetButTensorMissing)
{
    _tensorMap.erase(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    EXPECT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeGroupedMatmulAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(rebuiltAttrs->weight_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(rebuiltAttrs->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(rebuiltAttrs->token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(rebuiltAttrs->token_ks_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(rebuiltAttrs->mode, MoeGroupedMatmulMode::SCATTER);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify mode
    hipdnnMoeGroupedMatmulMode_t mode = {};
    int64_t modeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                       HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                       1,
                       &modeCount,
                       &mode);
    ASSERT_EQ(mode, HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER);

    // Verify token tensor
    hipdnn_backend::ScopedDescriptor tokenScoped;
    int64_t tokenCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenCount,
                       static_cast<void*>(tokenScoped.getPtr()));
    ASSERT_EQ(tokenCount, 1);
    ASSERT_NE(tokenScoped.get(), nullptr);
    verifyTensorDescriptor(tokenScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 8, 16},
                           {128, 16, 1});

    // Verify weight tensor
    hipdnn_backend::ScopedDescriptor weightScoped;
    int64_t weightCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &weightCount,
                       static_cast<void*>(weightScoped.getPtr()));
    ASSERT_EQ(weightCount, 1);
    ASSERT_NE(weightScoped.get(), nullptr);
    verifyTensorDescriptor(weightScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 16, 32},
                           {512, 32, 1});

    // Verify first_token_offset tensor
    hipdnn_backend::ScopedDescriptor firstTokenOffsetScoped;
    int64_t firstTokenOffsetCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &firstTokenOffsetCount,
                       static_cast<void*>(firstTokenOffsetScoped.getPtr()));
    ASSERT_EQ(firstTokenOffsetCount, 1);
    ASSERT_NE(firstTokenOffsetScoped.get(), nullptr);
    verifyTensorDescriptor(firstTokenOffsetScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                           HIPDNN_DATA_INT32,
                           {2, 1, 1},
                           {1, 1, 1});

    // Verify output tensor
    hipdnn_backend::ScopedDescriptor outputScoped;
    int64_t outputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &outputCount,
                       static_cast<void*>(outputScoped.getPtr()));
    ASSERT_EQ(outputCount, 1);
    ASSERT_NE(outputScoped.get(), nullptr);
    verifyTensorDescriptor(outputScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 8, 32},
                           {256, 32, 1});

    // Verify token_index tensor (optional)
    hipdnn_backend::ScopedDescriptor tokenIndexScoped;
    int64_t tokenIndexCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenIndexCount,
                       static_cast<void*>(tokenIndexScoped.getPtr()));
    ASSERT_EQ(tokenIndexCount, 1);
    ASSERT_NE(tokenIndexScoped.get(), nullptr);
    verifyTensorDescriptor(tokenIndexScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                           HIPDNN_DATA_INT32,
                           {1, 8, 1},
                           {8, 1, 1});

    // Verify token_ks tensor (optional)
    hipdnn_backend::ScopedDescriptor tokenKsScoped;
    int64_t tokenKsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenKsCount,
                       static_cast<void*>(tokenKsScoped.getPtr()));
    ASSERT_EQ(tokenKsCount, 1);
    ASSERT_NE(tokenKsScoped.get(), nullptr);
    verifyTensorDescriptor(tokenKsScoped.get(),
                           K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                           HIPDNN_DATA_INT32,
                           {1, 8, 1},
                           {8, 1, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_EXT);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_moegroupedmatmul_1";

    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_moegroupedmatmul_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_moegroupedmatmul_1");
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestMoeGroupedMatmulOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = MoeGroupedMatmulOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
