// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/RopeOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RopeConstants.hpp>
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
// RopeOperationDescriptor::fromNode() Tests
// =============================================================================

class TestRopeOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_ROPE_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_ROPE_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_ROPE_TENSOR_X_STRIDES);

        _tensorMap[K_ROPE_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT freqsAttrs;
        freqsAttrs.uid = K_ROPE_TENSOR_FREQS_UID;
        freqsAttrs.data_type = DataType::FLOAT;
        freqsAttrs.dims = toVec(K_ROPE_TENSOR_FREQS_DIMS);
        freqsAttrs.strides = toVec(K_ROPE_TENSOR_FREQS_STRIDES);

        _tensorMap[K_ROPE_TENSOR_FREQS_UID] = TensorDescriptor::fromFlatBuffer(freqsAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_ROPE_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_ROPE_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_ROPE_TENSOR_Y_STRIDES);

        _tensorMap[K_ROPE_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::RopeAttributesT createStandardRopeAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::RopeAttributesT attrs;
        attrs.x_tensor_uid = K_ROPE_TENSOR_X_UID;
        attrs.freqs_tensor_uid = K_ROPE_TENSOR_FREQS_UID;
        attrs.y_tensor_uid = K_ROPE_TENSOR_Y_UID;
        attrs.output_scale = 0.5;
        attrs.rope_dim = 4;
        attrs.rope_offset_ext = 0;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardRopeAttrs());
        return node;
    }
};

TEST_F(TestRopeOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_ROPE_TENSOR_X_UID);
}

TEST_F(TestRopeOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::RopeAttributes);
    auto desc = std::static_pointer_cast<RopeOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().freqs_tensor_uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_ROPE_TENSOR_Y_UID);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(desc->getFreqsDesc()->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_ROPE_TENSOR_X_UID);
    ASSERT_NE(desc->getFreqsDesc(), nullptr);
    EXPECT_EQ(desc->getFreqsDesc()->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_ROPE_TENSOR_X_UID]);
    EXPECT_EQ(desc->getFreqsDesc(), _tensorMap[K_ROPE_TENSOR_FREQS_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_ROPE_TENSOR_Y_UID]);
}

TEST_F(TestRopeOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{2, 4, 16, 8}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{512, 128, 8, 1}));

    ASSERT_NE(desc->getFreqsDesc(), nullptr);
    EXPECT_EQ(desc->getFreqsDesc()->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(desc->getFreqsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getFreqsDesc()->getData().dims, (std::vector<int64_t>{1, 1, 16, 4}));
    EXPECT_EQ(desc->getFreqsDesc()->getData().strides, (std::vector<int64_t>{64, 64, 4, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_ROPE_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{2, 4, 16, 8}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{512, 128, 8, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestRopeOperationFromNodeMissingTensor : public TestRopeOperationFromNode,
                                               public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestRopeOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RopeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestRopeOperationFromNodeMissingTensor,
                         ::testing::Values(K_ROPE_TENSOR_X_UID,
                                           K_ROPE_TENSOR_FREQS_UID,
                                           K_ROPE_TENSOR_Y_UID));

TEST_F(TestRopeOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::RopeAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsRopeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->freqs_tensor_uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationFromNode, FromNodePreservesRopeOffset)
{
    auto attrs = createStandardRopeAttrs();
    attrs.rope_offset_ext = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().rope_offset_ext.has_value());
    EXPECT_EQ(desc->getData().rope_offset_ext.value(), 2);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsRopeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->rope_offset_ext.has_value());
    EXPECT_EQ(rebuiltAttrs->rope_offset_ext.value(), 2);
}

TEST_F(TestRopeOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    node.attributes.AsRopeAttributes()->rope_offset_ext.reset();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsRopeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->rope_offset_ext.has_value());
}

TEST_F(TestRopeOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_ROPE_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {2, 4, 16, 8}, {512, 128, 8, 1});

    // Verify freqs tensor
    hipdnn_backend::ScopedDescriptor freqsScoped;
    int64_t freqsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &freqsCount,
                       static_cast<void*>(freqsScoped.getPtr()));
    ASSERT_EQ(freqsCount, 1);
    ASSERT_NE(freqsScoped.get(), nullptr);
    verifyTensorDescriptor(freqsScoped.get(),
                           K_ROPE_TENSOR_FREQS_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 1, 16, 4},
                           {64, 64, 4, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(
        yScoped.get(), K_ROPE_TENSOR_Y_UID, HIPDNN_DATA_FLOAT, {2, 4, 16, 8}, {512, 128, 8, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_ROPE_FWD_EXT);
}

TEST_F(TestRopeOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_rope_1";

    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_rope_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_rope_1");
}

TEST_F(TestRopeOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestRopeOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = RopeOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
