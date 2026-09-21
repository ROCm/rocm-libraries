// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/MoeGroupedMatmulBwdOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulBwdConstants.hpp>
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
// MoeGroupedMatmulBwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestMoeGroupedMatmulBwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT doutputAttrs;
        doutputAttrs.uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID;
        doutputAttrs.data_type = DataType::FLOAT;
        doutputAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS);
        doutputAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]
            = TensorDescriptor::fromFlatBuffer(doutputAttrs);
        TensorAttributesT tokenAttrs;
        tokenAttrs.uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID;
        tokenAttrs.data_type = DataType::FLOAT;
        tokenAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS);
        tokenAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]
            = TensorDescriptor::fromFlatBuffer(tokenAttrs);
        TensorAttributesT firstTokenOffsetAttrs;
        firstTokenOffsetAttrs.uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID;
        firstTokenOffsetAttrs.data_type = DataType::INT32;
        firstTokenOffsetAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS);
        firstTokenOffsetAttrs.strides
            = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]
            = TensorDescriptor::fromFlatBuffer(firstTokenOffsetAttrs);
        TensorAttributesT dweightAttrs;
        dweightAttrs.uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID;
        dweightAttrs.data_type = DataType::FLOAT;
        dweightAttrs.dims = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS);
        dweightAttrs.strides = toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES);

        _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]
            = TensorDescriptor::fromFlatBuffer(dweightAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulBwdAttributesT
        createStandardMoeGroupedMatmulBwdAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulBwdAttributesT attrs;
        attrs.doutput_tensor_uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID;
        attrs.token_tensor_uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID;
        attrs.first_token_offset_tensor_uid
            = K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID;
        attrs.dweight_tensor_uid = K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardMoeGroupedMatmulBwdAttrs());
        return node;
    }
};

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeGroupedMatmulBwdAttributes);
    auto desc = std::static_pointer_cast<MoeGroupedMatmulBwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(desc->getData().token_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getData().first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getData().dweight_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getDoutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getDweightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDoutputDesc(), nullptr);
    EXPECT_EQ(desc->getDoutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    ASSERT_NE(desc->getTokenDesc(), nullptr);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_NE(desc->getDweightDesc(), nullptr);
    EXPECT_EQ(desc->getDweightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getDoutputDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]);
    EXPECT_EQ(desc->getTokenDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc(),
              _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]);
    EXPECT_EQ(desc->getDweightDesc(), _tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDoutputDesc(), nullptr);
    EXPECT_EQ(desc->getDoutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(desc->getDoutputDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDoutputDesc()->getData().dims, (std::vector<int64_t>{1, 8, 32}));
    EXPECT_EQ(desc->getDoutputDesc()->getData().strides, (std::vector<int64_t>{256, 32, 1}));

    ASSERT_NE(desc->getTokenDesc(), nullptr);
    EXPECT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(desc->getTokenDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getTokenDesc()->getData().dims, (std::vector<int64_t>{1, 8, 16}));
    EXPECT_EQ(desc->getTokenDesc()->getData().strides, (std::vector<int64_t>{128, 16, 1}));

    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().dims, (std::vector<int64_t>{2, 1, 1}));
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1}));

    ASSERT_NE(desc->getDweightDesc(), nullptr);
    EXPECT_EQ(desc->getDweightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
    EXPECT_EQ(desc->getDweightDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDweightDesc()->getData().dims, (std::vector<int64_t>{2, 16, 32}));
    // dweight is column-major [K*N, 1, K]; the other tensors are row-major.
    EXPECT_EQ(desc->getDweightDesc()->getData().strides, (std::vector<int64_t>{512, 1, 16}));
}

namespace
{

// Identifies the tensor to drop from the tensor map before calling fromNode().
struct MissingTensorCase
{
    int64_t uid;
    const char* name;
};

} // namespace

class TestMoeGroupedMatmulBwdOperationFromNodeMissingTensor
    : public TestMoeGroupedMatmulBwdOperationFromNode,
      public ::testing::WithParamInterface<MissingTensorCase>
{
};

TEST_P(TestMoeGroupedMatmulBwdOperationFromNodeMissingTensor, FromNodeFails)
{
    _tensorMap.erase(GetParam().uid);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    RequiredTensors,
    TestMoeGroupedMatmulBwdOperationFromNodeMissingTensor,
    ::testing::Values(MissingTensorCase{K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID, "Doutput"},
                      MissingTensorCase{K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID, "Token"},
                      MissingTensorCase{K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                        "FirstTokenOffset"},
                      MissingTensorCase{K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID, "Dweight"}),
    [](const ::testing::TestParamInfo<MissingTensorCase>& info) { return info.param.name; });

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    EXPECT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeGroupedMatmulBwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsMoeGroupedMatmulBwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(rebuiltAttrs->token_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(rebuiltAttrs->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(rebuiltAttrs->dweight_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify doutput tensor
    hipdnn_backend::ScopedDescriptor doutputScoped;
    int64_t doutputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &doutputCount,
                       static_cast<void*>(doutputScoped.getPtr()));
    ASSERT_EQ(doutputCount, 1);
    ASSERT_NE(doutputScoped.get(), nullptr);
    verifyTensorDescriptor(doutputScoped.get(),
                           K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 8, 32},
                           {256, 32, 1});

    // Verify token tensor
    hipdnn_backend::ScopedDescriptor tokenScoped;
    int64_t tokenCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenCount,
                       static_cast<void*>(tokenScoped.getPtr()));
    ASSERT_EQ(tokenCount, 1);
    ASSERT_NE(tokenScoped.get(), nullptr);
    verifyTensorDescriptor(tokenScoped.get(),
                           K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 8, 16},
                           {128, 16, 1});

    // Verify first_token_offset tensor
    hipdnn_backend::ScopedDescriptor firstTokenOffsetScoped;
    int64_t firstTokenOffsetCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &firstTokenOffsetCount,
                       static_cast<void*>(firstTokenOffsetScoped.getPtr()));
    ASSERT_EQ(firstTokenOffsetCount, 1);
    ASSERT_NE(firstTokenOffsetScoped.get(), nullptr);
    verifyTensorDescriptor(firstTokenOffsetScoped.get(),
                           K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                           HIPDNN_DATA_INT32,
                           {2, 1, 1},
                           {1, 1, 1});

    // Verify dweight tensor
    hipdnn_backend::ScopedDescriptor dweightScoped;
    int64_t dweightCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dweightCount,
                       static_cast<void*>(dweightScoped.getPtr()));
    ASSERT_EQ(dweightCount, 1);
    ASSERT_NE(dweightScoped.get(), nullptr);
    // dweight is column-major [K*N, 1, K]; the other tensors are row-major.
    verifyTensorDescriptor(dweightScoped.get(),
                           K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 16, 32},
                           {512, 1, 16});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_BWD_EXT);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_moegroupedmatmulbwd_1";

    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_moegroupedmatmulbwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_moegroupedmatmulbwd_1");
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestMoeGroupedMatmulBwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = MoeGroupedMatmulBwdOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
