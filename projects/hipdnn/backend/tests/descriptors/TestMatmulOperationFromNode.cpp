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
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

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
        aAttrs.uid = 30;
        aAttrs.data_type = DataType::FLOAT;
        aAttrs.dims = {1, 1, 64, 128};
        aAttrs.strides = {8192, 8192, 128, 1};

        _tensorMap[30] = TensorDescriptor::fromFlatBuffer(aAttrs);
        TensorAttributesT bAttrs;
        bAttrs.uid = 31;
        bAttrs.data_type = DataType::FLOAT;
        bAttrs.dims = {1, 1, 128, 256};
        bAttrs.strides = {32768, 32768, 256, 1};

        _tensorMap[31] = TensorDescriptor::fromFlatBuffer(bAttrs);
        TensorAttributesT cAttrs;
        cAttrs.uid = 32;
        cAttrs.data_type = DataType::FLOAT;
        cAttrs.dims = {1, 1, 64, 256};
        cAttrs.strides = {16384, 16384, 256, 1};

        _tensorMap[32] = TensorDescriptor::fromFlatBuffer(cAttrs);
    }

    static hipdnn_data_sdk::data_objects::MatmulAttributesT createStandardMatmulAttrs()
    {
        hipdnn_data_sdk::data_objects::MatmulAttributesT attrs;
        attrs.a_tensor_uid = 30;
        attrs.b_tensor_uid = 31;
        attrs.c_tensor_uid = 32;
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
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().a_tensor_uid, 30);
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
    EXPECT_EQ(desc->getData().a_tensor_uid, 30);
    EXPECT_EQ(desc->getData().b_tensor_uid, 31);
    EXPECT_EQ(desc->getData().c_tensor_uid, 32);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getADesc()->getData().uid, 30);
    EXPECT_EQ(desc->getBDesc()->getData().uid, 31);
    EXPECT_EQ(desc->getCDesc()->getData().uid, 32);
}

TEST_F(TestMatmulOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestMatmulOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getADesc(), nullptr);
    EXPECT_EQ(desc->getADesc()->getData().uid, 30);
    ASSERT_NE(desc->getBDesc(), nullptr);
    EXPECT_EQ(desc->getBDesc()->getData().uid, 31);
    ASSERT_NE(desc->getCDesc(), nullptr);
    EXPECT_EQ(desc->getCDesc()->getData().uid, 32);
}

TEST_F(TestMatmulOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getADesc(), _tensorMap[30]);
    EXPECT_EQ(desc->getBDesc(), _tensorMap[31]);
    EXPECT_EQ(desc->getCDesc(), _tensorMap[32]);
}

TEST_F(TestMatmulOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getADesc(), nullptr);
    EXPECT_EQ(desc->getADesc()->getData().uid, 30);
    EXPECT_EQ(desc->getADesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getADesc()->getData().dims, (std::vector<int64_t>{1, 1, 64, 128}));
    EXPECT_EQ(desc->getADesc()->getData().strides, (std::vector<int64_t>{8192, 8192, 128, 1}));

    ASSERT_NE(desc->getBDesc(), nullptr);
    EXPECT_EQ(desc->getBDesc()->getData().uid, 31);
    EXPECT_EQ(desc->getBDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBDesc()->getData().dims, (std::vector<int64_t>{1, 1, 128, 256}));
    EXPECT_EQ(desc->getBDesc()->getData().strides, (std::vector<int64_t>{32768, 32768, 256, 1}));

    ASSERT_NE(desc->getCDesc(), nullptr);
    EXPECT_EQ(desc->getCDesc()->getData().uid, 32);
    EXPECT_EQ(desc->getCDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getCDesc()->getData().dims, (std::vector<int64_t>{1, 1, 64, 256}));
    EXPECT_EQ(desc->getCDesc()->getData().strides, (std::vector<int64_t>{16384, 16384, 256, 1}));
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingATensor)
{
    _tensorMap.erase(30);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingBTensor)
{
    _tensorMap.erase(31);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MatmulOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestMatmulOperationFromNode, FailsWithMissingCTensor)
{
    _tensorMap.erase(32);
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
    EXPECT_EQ(tensors[0]->getData().uid, 30);
    EXPECT_EQ(tensors[1]->getData().uid, 31);
    EXPECT_EQ(tensors[2]->getData().uid, 32);
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
    EXPECT_EQ(rebuiltAttrs->a_tensor_uid, 30);
    EXPECT_EQ(rebuiltAttrs->b_tensor_uid, 31);
    EXPECT_EQ(rebuiltAttrs->c_tensor_uid, 32);
}

TEST_F(TestMatmulOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = MatmulOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_MATMUL_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify a tensor
    hipdnn_backend::ScopedDescriptor aScoped;
    int64_t aCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_A_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &aCount,
                       static_cast<void*>(aScoped.getPtr()));
    ASSERT_EQ(aCount, 1);
    ASSERT_NE(aScoped.get(), nullptr);
    verifyTensorDescriptor(
        aScoped.get(), 30, HIPDNN_DATA_FLOAT, {1, 1, 64, 128}, {8192, 8192, 128, 1});

    // Verify b tensor
    hipdnn_backend::ScopedDescriptor bScoped;
    int64_t bCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_B_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &bCount,
                       static_cast<void*>(bScoped.getPtr()));
    ASSERT_EQ(bCount, 1);
    ASSERT_NE(bScoped.get(), nullptr);
    verifyTensorDescriptor(
        bScoped.get(), 31, HIPDNN_DATA_FLOAT, {1, 1, 128, 256}, {32768, 32768, 256, 1});

    // Verify c tensor
    hipdnn_backend::ScopedDescriptor cScoped;
    int64_t cCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MATMUL_C_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &cCount,
                       static_cast<void*>(cScoped.getPtr()));
    ASSERT_EQ(cCount, 1);
    ASSERT_NE(cScoped.get(), nullptr);
    verifyTensorDescriptor(
        cScoped.get(), 32, HIPDNN_DATA_FLOAT, {1, 1, 64, 256}, {16384, 16384, 256, 1});

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MATMUL);
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
