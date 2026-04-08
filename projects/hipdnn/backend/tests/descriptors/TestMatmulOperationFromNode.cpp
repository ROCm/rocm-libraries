// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/MatmulOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// MatmulOperationDescriptor::fromNode() Tests
// =============================================================================

class TestMatmulOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT aAttrs;
        aAttrs.uid = K_MATMUL_TENSOR_A_UID;
        aAttrs.data_type = DataType::FLOAT;
        aAttrs.dims = toVec(K_MATMUL_TENSOR_A_DIMS);
        aAttrs.strides = toVec(K_MATMUL_TENSOR_A_STRIDES);
        _tensorMap[K_MATMUL_TENSOR_A_UID] = TensorDescriptor::fromFlatBuffer(aAttrs);

        TensorAttributesT bAttrs;
        bAttrs.uid = K_MATMUL_TENSOR_B_UID;
        bAttrs.data_type = DataType::FLOAT;
        bAttrs.dims = toVec(K_MATMUL_TENSOR_B_DIMS);
        bAttrs.strides = toVec(K_MATMUL_TENSOR_B_STRIDES);
        _tensorMap[K_MATMUL_TENSOR_B_UID] = TensorDescriptor::fromFlatBuffer(bAttrs);

        TensorAttributesT cAttrs;
        cAttrs.uid = K_MATMUL_TENSOR_C_UID;
        cAttrs.data_type = DataType::FLOAT;
        cAttrs.dims = toVec(K_MATMUL_TENSOR_C_DIMS);
        cAttrs.strides = toVec(K_MATMUL_TENSOR_C_STRIDES);
        _tensorMap[K_MATMUL_TENSOR_C_UID] = TensorDescriptor::fromFlatBuffer(cAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::MatmulAttributesT createStandardMatmulAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::MatmulAttributesT attrs;
        attrs.a_tensor_uid = K_MATMUL_TENSOR_A_UID;
        attrs.b_tensor_uid = K_MATMUL_TENSOR_B_UID;
        attrs.c_tensor_uid = K_MATMUL_TENSOR_C_UID;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardMatmulAttrs());
        return node;
    }
};

TEST_F(TestMatmulOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    EXPECT_EQ(desc->getData().a_tensor_uid, K_MATMUL_TENSOR_A_UID);
}

TEST_F(TestMatmulOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MatmulAttributes);
    auto desc = std::static_pointer_cast<MatmulOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().a_tensor_uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(desc->getData().b_tensor_uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(desc->getData().c_tensor_uid, K_MATMUL_TENSOR_C_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getADesc()->getData().uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(desc->getBDesc()->getData().uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(desc->getCDesc()->getData().uid, K_MATMUL_TENSOR_C_UID);
    // Verify pointer equality: descriptor holds the same shared_ptr as the tensor map
    EXPECT_EQ(desc->getADesc(), _tensorMap[K_MATMUL_TENSOR_A_UID]);
    EXPECT_EQ(desc->getBDesc(), _tensorMap[K_MATMUL_TENSOR_B_UID]);
    EXPECT_EQ(desc->getCDesc(), _tensorMap[K_MATMUL_TENSOR_C_UID]);
}

TEST_F(TestMatmulOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestMatmulOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getADesc(), nullptr);
    EXPECT_EQ(desc->getADesc()->getData().uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(desc->getADesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getADesc()->getData().dims, toVec(K_MATMUL_TENSOR_A_DIMS));
    EXPECT_EQ(desc->getADesc()->getData().strides, toVec(K_MATMUL_TENSOR_A_STRIDES));

    ASSERT_NE(desc->getBDesc(), nullptr);
    EXPECT_EQ(desc->getBDesc()->getData().uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(desc->getBDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBDesc()->getData().dims, toVec(K_MATMUL_TENSOR_B_DIMS));
    EXPECT_EQ(desc->getBDesc()->getData().strides, toVec(K_MATMUL_TENSOR_B_STRIDES));

    ASSERT_NE(desc->getCDesc(), nullptr);
    EXPECT_EQ(desc->getCDesc()->getData().uid, K_MATMUL_TENSOR_C_UID);
    EXPECT_EQ(desc->getCDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getCDesc()->getData().dims, toVec(K_MATMUL_TENSOR_C_DIMS));
    EXPECT_EQ(desc->getCDesc()->getData().strides, toVec(K_MATMUL_TENSOR_C_STRIDES));
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingATensor)
{
    _tensorMap.erase(K_MATMUL_TENSOR_A_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingBTensor)
{
    _tensorMap.erase(K_MATMUL_TENSOR_B_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingCTensor)
{
    _tensorMap.erase(K_MATMUL_TENSOR_C_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMatmulOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_MATMUL_TENSOR_C_UID);
}

TEST_F(TestMatmulOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MatmulAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsMatmulAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->a_tensor_uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(rebuiltAttrs->b_tensor_uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(rebuiltAttrs->c_tensor_uid, K_MATMUL_TENSOR_C_UID);
}

TEST_F(TestMatmulOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_MATMUL_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify a tensor
    hipdnn_backend::ScopedDescriptor aScoped;
    int64_t aCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_ADESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &aCount,
                       static_cast<void*>(aScoped.getPtr()));
    ASSERT_EQ(aCount, 1);
    ASSERT_NE(aScoped.get(), nullptr);
    verifyTensorDescriptor(aScoped.get(),
                           K_MATMUL_TENSOR_A_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_MATMUL_TENSOR_A_DIMS),
                           toVec(K_MATMUL_TENSOR_A_STRIDES));

    // Verify b tensor
    hipdnn_backend::ScopedDescriptor bScoped;
    int64_t bCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_BDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &bCount,
                       static_cast<void*>(bScoped.getPtr()));
    ASSERT_EQ(bCount, 1);
    ASSERT_NE(bScoped.get(), nullptr);
    verifyTensorDescriptor(bScoped.get(),
                           K_MATMUL_TENSOR_B_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_MATMUL_TENSOR_B_DIMS),
                           toVec(K_MATMUL_TENSOR_B_STRIDES));

    // Verify c tensor
    hipdnn_backend::ScopedDescriptor cScoped;
    int64_t cCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_CDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &cCount,
                       static_cast<void*>(cScoped.getPtr()));
    ASSERT_EQ(cCount, 1);
    ASSERT_NE(cScoped.get(), nullptr);
    verifyTensorDescriptor(cScoped.get(),
                           K_MATMUL_TENSOR_C_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_MATMUL_TENSOR_C_DIMS),
                           toVec(K_MATMUL_TENSOR_C_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MATMUL_EXT);
}

TEST_F(TestMatmulOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_matmul_1";

    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_matmul_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_matmul_1");
}

TEST_F(TestMatmulOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestMatmulOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
