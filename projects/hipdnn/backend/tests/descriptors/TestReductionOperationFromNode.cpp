// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ReductionOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
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
// ReductionOperationDescriptor::fromNode() Tests
// =============================================================================

class TestReductionOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_REDUCTION_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_REDUCTION_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_REDUCTION_TENSOR_X_STRIDES);

        _tensorMap[K_REDUCTION_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_REDUCTION_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_REDUCTION_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_REDUCTION_TENSOR_Y_STRIDES);

        _tensorMap[K_REDUCTION_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::ReductionAttributesT createStandardReductionAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::ReductionAttributesT attrs;
        attrs.in_tensor_uid = K_REDUCTION_TENSOR_X_UID;
        attrs.out_tensor_uid = K_REDUCTION_TENSOR_Y_UID;
        attrs.mode = ReductionMode::ADD;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardReductionAttrs());
        return node;
    }
};

TEST_F(TestReductionOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    EXPECT_EQ(desc->getData().in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
}

TEST_F(TestReductionOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ReductionAttributes);
    auto desc = std::static_pointer_cast<ReductionOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().out_tensor_uid, K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().mode, ReductionMode::ADD);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_REDUCTION_TENSOR_Y_UID);
}

TEST_F(TestReductionOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestReductionOperationFromNode, PreservesReductionMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardReductionAttrs();
    attrs.mode = ReductionMode::AMAX;
    node.attributes.Set(attrs);
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().mode, ReductionMode::AMAX);
}

TEST_F(TestReductionOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_REDUCTION_TENSOR_X_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_REDUCTION_TENSOR_Y_UID);
}

TEST_F(TestReductionOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_REDUCTION_TENSOR_X_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_REDUCTION_TENSOR_Y_UID]);
}

TEST_F(TestReductionOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{2, 3, 4, 4}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{48, 16, 4, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{2, 3, 1, 1}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{3, 1, 1, 1}));
}

TEST_F(TestReductionOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_REDUCTION_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ReductionOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestReductionOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_REDUCTION_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ReductionOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestReductionOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0]->getData().uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_REDUCTION_TENSOR_Y_UID);
}

TEST_F(TestReductionOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ReductionAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsReductionAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->out_tensor_uid, K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->mode, ReductionMode::ADD);
}

TEST_F(TestReductionOperationFromNode, FromNodePreservesIsDeterministic)
{
    auto attrs = createStandardReductionAttrs();
    attrs.is_deterministic = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().is_deterministic);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsReductionAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->is_deterministic);
}

TEST_F(TestReductionOperationFromNode, BuildNodePreservesDefaultIsDeterministic)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsReductionAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->is_deterministic);
}

TEST_F(TestReductionOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify mode
    hipdnnReduceTensorOp_t mode = {};
    int64_t modeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &modeCount, &mode);
    ASSERT_EQ(mode, HIPDNN_REDUCE_TENSOR_ADD);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_REDUCTION_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {2, 3, 4, 4}, {48, 16, 4, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(
        yScoped.get(), K_REDUCTION_TENSOR_Y_UID, HIPDNN_DATA_FLOAT, {2, 3, 1, 1}, {3, 1, 1, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_REDUCTION_EXT);
}

TEST_F(TestReductionOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_reduction_1";

    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_reduction_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_reduction_1");
}

TEST_F(TestReductionOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestReductionOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ReductionOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestReductionOperationFromNode, FailsWithWrongAttributesType)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(ConvolutionFwdAttributesT{});

    ASSERT_THROW_HIPDNN_STATUS(ReductionOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}
