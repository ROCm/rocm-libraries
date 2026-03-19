// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/PointwiseOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <optional>
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
    ASSERT_NE(desc->getIn1Desc(), nullptr);
    EXPECT_EQ(desc->getIn1Desc()->getData().uid, 3);
    ASSERT_NE(desc->getIn2Desc(), nullptr);
    EXPECT_EQ(desc->getIn2Desc()->getData().uid, 4);
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
    ASSERT_NE(desc->getIn1Desc(), nullptr);
    EXPECT_EQ(desc->getIn1Desc()->getData().uid, 3);
    ASSERT_NE(desc->getIn2Desc(), nullptr);
    EXPECT_EQ(desc->getIn2Desc()->getData().uid, 4);
}

TEST_F(TestPointwiseOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getIn0Desc(), _tensorMap[40]);
    EXPECT_EQ(desc->getOut0Desc(), _tensorMap[41]);
    EXPECT_EQ(desc->getIn1Desc(), _tensorMap[3]);
    EXPECT_EQ(desc->getIn2Desc(), _tensorMap[4]);
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
    EXPECT_EQ(tensors[2]->getData().uid, 3);
    EXPECT_EQ(tensors[3]->getData().uid, 4);
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
    hipdnn_backend::ScopedDescriptor in0Scoped;
    int64_t in0Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &in0Count,
                       static_cast<void*>(in0Scoped.getPtr()));
    ASSERT_EQ(in0Count, 1);
    ASSERT_NE(in0Scoped.get(), nullptr);
    int64_t in0Uid = 0;
    int64_t in0UidCount = 0;
    in0Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &in0UidCount, &in0Uid);
    EXPECT_EQ(in0Uid, 40);

    // Verify out_0 tensor
    hipdnn_backend::ScopedDescriptor out0Scoped;
    int64_t out0Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &out0Count,
                       static_cast<void*>(out0Scoped.getPtr()));
    ASSERT_EQ(out0Count, 1);
    ASSERT_NE(out0Scoped.get(), nullptr);
    int64_t out0Uid = 0;
    int64_t out0UidCount = 0;
    out0Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &out0UidCount, &out0Uid);
    EXPECT_EQ(out0Uid, 41);

    // Verify in_1 tensor (optional, set in standard fixture)
    hipdnn_backend::ScopedDescriptor in1Scoped;
    int64_t in1Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &in1Count,
                       static_cast<void*>(in1Scoped.getPtr()));
    ASSERT_EQ(in1Count, 1);
    ASSERT_NE(in1Scoped.get(), nullptr);
    int64_t in1Uid = 0;
    int64_t in1UidCount = 0;
    in1Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &in1UidCount, &in1Uid);
    EXPECT_EQ(in1Uid, 3);

    // Verify in_2 tensor (optional, set in standard fixture)
    hipdnn_backend::ScopedDescriptor in2Scoped;
    int64_t in2Count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &in2Count,
                       static_cast<void*>(in2Scoped.getPtr()));
    ASSERT_EQ(in2Count, 1);
    ASSERT_NE(in2Scoped.get(), nullptr);
    int64_t in2Uid = 0;
    int64_t in2UidCount = 0;
    in2Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &in2UidCount, &in2Uid);
    EXPECT_EQ(in2Uid, 4);

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

TEST_F(TestPointwiseOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardPointwiseAttrs();
    attrs.in_1_tensor_uid = std::nullopt;
    attrs.in_2_tensor_uid = std::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_NE(desc->getIn0Desc(), nullptr);
    EXPECT_EQ(desc->getIn0Desc()->getData().uid, 40);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
    EXPECT_EQ(desc->getOut0Desc()->getData().uid, 41);
    EXPECT_EQ(desc->getIn1Desc(), nullptr);
    EXPECT_EQ(desc->getIn2Desc(), nullptr);
}

TEST_F(TestPointwiseOperationFromNode, FailsWhenOptionalIn1UidSetButTensorMissing)
{
    _tensorMap.erase(3);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PointwiseOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPointwiseOperationFromNode, FailsWhenOptionalIn2UidSetButTensorMissing)
{
    _tensorMap.erase(4);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PointwiseOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPointwiseOperationFromNode, FromNodePreservesReluScalars)
{
    auto attrs = createStandardPointwiseAttrs();
    attrs.operation = PointwiseMode::RELU_FWD;
    attrs.relu_lower_clip = -1.0F;
    attrs.relu_upper_clip = 6.0F;
    attrs.relu_lower_clip_slope = 0.01F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    // Verify via getData()
    EXPECT_TRUE(desc->getData().relu_lower_clip.has_value());
    EXPECT_FLOAT_EQ(desc->getData().relu_lower_clip.value(), -1.0F);
    EXPECT_TRUE(desc->getData().relu_upper_clip.has_value());
    EXPECT_FLOAT_EQ(desc->getData().relu_upper_clip.value(), 6.0F);
    EXPECT_TRUE(desc->getData().relu_lower_clip_slope.has_value());
    EXPECT_FLOAT_EQ(desc->getData().relu_lower_clip_slope.value(), 0.01F);

    // Verify round-trip via buildNode()
    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPointwiseAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->relu_lower_clip.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->relu_lower_clip.value(), -1.0F);
    ASSERT_TRUE(rebuiltAttrs->relu_upper_clip.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->relu_upper_clip.value(), 6.0F);
    ASSERT_TRUE(rebuiltAttrs->relu_lower_clip_slope.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->relu_lower_clip_slope.value(), 0.01F);
}

TEST_F(TestPointwiseOperationFromNode, FromNodePreservesSwishEluSoftplusScalars)
{
    auto attrs = createStandardPointwiseAttrs();
    attrs.operation = PointwiseMode::SWISH_FWD;
    attrs.swish_beta = 1.5F;
    attrs.elu_alpha = 0.25F;
    attrs.softplus_beta = 2.0F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().swish_beta.has_value());
    EXPECT_FLOAT_EQ(desc->getData().swish_beta.value(), 1.5F);
    EXPECT_TRUE(desc->getData().elu_alpha.has_value());
    EXPECT_FLOAT_EQ(desc->getData().elu_alpha.value(), 0.25F);
    EXPECT_TRUE(desc->getData().softplus_beta.has_value());
    EXPECT_FLOAT_EQ(desc->getData().softplus_beta.value(), 2.0F);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPointwiseAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->swish_beta.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->swish_beta.value(), 1.5F);
    ASSERT_TRUE(rebuiltAttrs->elu_alpha.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->elu_alpha.value(), 0.25F);
    ASSERT_TRUE(rebuiltAttrs->softplus_beta.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->softplus_beta.value(), 2.0F);
}

TEST_F(TestPointwiseOperationFromNode, FromNodePreservesAxis)
{
    auto attrs = createStandardPointwiseAttrs();
    attrs.operation = PointwiseMode::GEN_INDEX;
    attrs.axis_tensor_uid = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().axis_tensor_uid.has_value());
    EXPECT_EQ(desc->getData().axis_tensor_uid.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPointwiseAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->axis_tensor_uid.has_value());
    EXPECT_EQ(rebuiltAttrs->axis_tensor_uid.value(), 2);
}

TEST_F(TestPointwiseOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    auto desc = PointwiseOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPointwiseAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->relu_lower_clip.has_value());
    EXPECT_FALSE(rebuiltAttrs->relu_upper_clip.has_value());
    EXPECT_FALSE(rebuiltAttrs->relu_lower_clip_slope.has_value());
    EXPECT_FALSE(rebuiltAttrs->swish_beta.has_value());
    EXPECT_FALSE(rebuiltAttrs->elu_alpha.has_value());
    EXPECT_FALSE(rebuiltAttrs->softplus_beta.has_value());
    EXPECT_FALSE(rebuiltAttrs->axis_tensor_uid.has_value());
}
