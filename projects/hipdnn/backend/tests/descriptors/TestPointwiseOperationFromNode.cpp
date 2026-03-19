// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/PointwiseOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;

// =============================================================================
// PointwiseOperationDescriptor::fromNode() Tests
// =============================================================================

class TestPointwiseOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT in0Attrs;
        in0Attrs.uid = 40;
        in0Attrs.data_type = DataType::FLOAT;
        in0Attrs.dims = {1, 64, 32, 32};
        in0Attrs.strides = {65536, 1024, 32, 1};

        _tensorMap[40] = TensorDescriptor::fromFlatBuffer(in0Attrs);
        TensorAttributesT out0Attrs;
        out0Attrs.uid = 41;
        out0Attrs.data_type = DataType::FLOAT;
        out0Attrs.dims = {1, 64, 32, 32};
        out0Attrs.strides = {65536, 1024, 32, 1};

        _tensorMap[41] = TensorDescriptor::fromFlatBuffer(out0Attrs);
        TensorAttributesT in1Attrs;
        in1Attrs.uid = 3;
        in1Attrs.data_type = DataType::FLOAT;
        in1Attrs.dims = {1};
        in1Attrs.strides = {1};

        _tensorMap[3] = TensorDescriptor::fromFlatBuffer(in1Attrs);
        TensorAttributesT in2Attrs;
        in2Attrs.uid = 4;
        in2Attrs.data_type = DataType::FLOAT;
        in2Attrs.dims = {1};
        in2Attrs.strides = {1};

        _tensorMap[4] = TensorDescriptor::fromFlatBuffer(in2Attrs);
    }

    static hipdnn_data_sdk::data_objects::PointwiseAttributesT createStandardPointwiseAttrs()
    {
        hipdnn_data_sdk::data_objects::PointwiseAttributesT attrs;
        attrs.in_0_tensor_uid = 40;
        attrs.out_0_tensor_uid = 41;
        attrs.in_1_tensor_uid = 3;
        attrs.in_2_tensor_uid = 4;
        attrs.operation = PointwiseMode::ADD;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardPointwiseAttrs());
        return node;
    }
};

TEST_F(TestPointwiseOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(desc->getData().in_0_tensor_uid, 40);
}
TEST_F(TestPointwiseOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PointwiseAttributes);
    auto desc = std::static_pointer_cast<PointwiseOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().in_0_tensor_uid, 40);
    EXPECT_EQ(desc->getData().out_0_tensor_uid, 41);
    EXPECT_EQ(desc->getData().in_1_tensor_uid, 3);
    EXPECT_EQ(desc->getData().in_2_tensor_uid, 4);
    EXPECT_EQ(desc->getData().operation, PointwiseMode::ADD);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getIn0Desc()->getData().uid, 40);
    EXPECT_EQ(desc->getOut0Desc()->getData().uid, 41);
}

TEST_F(TestPointwiseOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestPointwiseOperationFromNode, PreservesPointwiseMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardPointwiseAttrs();
    attrs.operation = PointwiseMode::MUL;
    node.attributes.Set(attrs);
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().operation, PointwiseMode::MUL);
}

TEST_F(TestPointwiseOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getIn0Desc(), nullptr);
    EXPECT_EQ(desc->getIn0Desc()->getData().uid, 40);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
    EXPECT_EQ(desc->getOut0Desc()->getData().uid, 41);
}

TEST_F(TestPointwiseOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getIn0Desc(), _tensorMap[40]);
    EXPECT_EQ(desc->getOut0Desc(), _tensorMap[41]);
}

TEST_F(TestPointwiseOperationFromNode, FailsWithMissingIn0Tensor)
{
    _tensorMap.erase(40);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PointwiseOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPointwiseOperationFromNode, FailsWithMissingOut0Tensor)
{
    _tensorMap.erase(41);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PointwiseOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPointwiseOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    EXPECT_EQ(tensors[0]->getData().uid, 40);
    EXPECT_EQ(tensors[1]->getData().uid, 41);
}

TEST_F(TestPointwiseOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PointwiseAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPointwiseAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->in_0_tensor_uid, 40);
    EXPECT_EQ(rebuiltAttrs->out_0_tensor_uid, 41);
    EXPECT_EQ(rebuiltAttrs->in_1_tensor_uid, 3);
    EXPECT_EQ(rebuiltAttrs->in_2_tensor_uid, 4);
    EXPECT_EQ(rebuiltAttrs->operation, PointwiseMode::ADD);
}

TEST_F(TestPointwiseOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify operation
    hipdnnPointwiseMode_t operation = {};
    int64_t operationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operationCount, &operation);
    ASSERT_EQ(operation, HIPDNN_POINTWISE_ADD);

    // Verify in_0 tensor
    HipdnnBackendDescriptor* in0TensorDesc = nullptr;
    int64_t in0Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &in0Count,
                       static_cast<void*>(&in0TensorDesc));
    ASSERT_EQ(in0Count, 1);
    ASSERT_NE(in0TensorDesc, nullptr);
    int64_t in0Uid = 0;
    int64_t in0UidCount = 0;
    in0TensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &in0UidCount, &in0Uid);
    EXPECT_EQ(in0Uid, 40);

    // Verify out_0 tensor
    HipdnnBackendDescriptor* out0TensorDesc = nullptr;
    int64_t out0Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &out0Count,
                       static_cast<void*>(&out0TensorDesc));
    ASSERT_EQ(out0Count, 1);
    ASSERT_NE(out0TensorDesc, nullptr);
    int64_t out0Uid = 0;
    int64_t out0UidCount = 0;
    out0TensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &out0UidCount, &out0Uid);
    EXPECT_EQ(out0Uid, 41);

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_POINTWISE);
}

TEST_F(TestPointwiseOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_pointwise_1";

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_pointwise_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_pointwise_1");
}

TEST_F(TestPointwiseOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestPointwiseOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
