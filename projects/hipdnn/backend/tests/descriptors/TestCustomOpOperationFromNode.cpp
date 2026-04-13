// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/CustomOpOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>

#include <array>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// CustomOpOperationDescriptor::fromNode() Tests
// =============================================================================

class TestCustomOpOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        // Input tensor 0
        TensorAttributesT input0Attrs;
        input0Attrs.uid = K_CUSTOM_OP_INPUT_UID_0;
        input0Attrs.data_type = DataType::FLOAT;
        input0Attrs.dims = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_DIMS);
        input0Attrs.strides = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_STRIDES);
        _tensorMap[K_CUSTOM_OP_INPUT_UID_0] = TensorDescriptor::fromFlatBuffer(input0Attrs);

        // Input tensor 1
        TensorAttributesT input1Attrs;
        input1Attrs.uid = K_CUSTOM_OP_INPUT_UID_1;
        input1Attrs.data_type = DataType::FLOAT;
        input1Attrs.dims = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_DIMS);
        input1Attrs.strides = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_STRIDES);
        _tensorMap[K_CUSTOM_OP_INPUT_UID_1] = TensorDescriptor::fromFlatBuffer(input1Attrs);

        // Output tensor 0
        TensorAttributesT output0Attrs;
        output0Attrs.uid = K_CUSTOM_OP_OUTPUT_UID_0;
        output0Attrs.data_type = DataType::FLOAT;
        output0Attrs.dims = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_DIMS);
        output0Attrs.strides = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_STRIDES);
        _tensorMap[K_CUSTOM_OP_OUTPUT_UID_0] = TensorDescriptor::fromFlatBuffer(output0Attrs);
    }

    static CustomOpAttributesT createStandardCustomOpAttrs()
    {
        CustomOpAttributesT attrs;
        attrs.custom_op_id = K_CUSTOM_OP_ID;
        attrs.input_tensor_uids = {K_CUSTOM_OP_INPUT_UID_0, K_CUSTOM_OP_INPUT_UID_1};
        attrs.output_tensor_uids = {K_CUSTOM_OP_OUTPUT_UID_0};
        attrs.data = {K_CUSTOM_OP_OPAQUE_DATA.begin(), K_CUSTOM_OP_OPAQUE_DATA.end()};
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardCustomOpAttrs());
        return node;
    }
};

TEST_F(TestCustomOpOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().custom_op_id, K_CUSTOM_OP_ID);
}

TEST_F(TestCustomOpOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::CustomOpAttributes);
    auto desc = std::static_pointer_cast<CustomOpOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    EXPECT_EQ(desc->getData().custom_op_id, K_CUSTOM_OP_ID);
    ASSERT_EQ(desc->getData().input_tensor_uids.size(), 2);
    EXPECT_EQ(desc->getData().input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(desc->getData().input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(desc->getData().output_tensor_uids.size(), 1);
    EXPECT_EQ(desc->getData().output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(desc->getData().data, K_CUSTOM_OP_OPAQUE_DATA);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestCustomOpOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestCustomOpOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getInputDescs().size(), 2);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().uid, K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestCustomOpOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    const std::vector<int64_t> expectedDims(K_CUSTOM_OP_TENSOR_DIMS.begin(),
                                            K_CUSTOM_OP_TENSOR_DIMS.end());
    const std::vector<int64_t> expectedStrides(K_CUSTOM_OP_TENSOR_STRIDES.begin(),
                                               K_CUSTOM_OP_TENSOR_STRIDES.end());

    // Verify inputs
    ASSERT_EQ(desc->getInputDescs().size(), 2);
    ASSERT_NE(desc->getInputDescs()[0], nullptr);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().dims, expectedDims);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().strides, expectedStrides);

    ASSERT_NE(desc->getInputDescs()[1], nullptr);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().uid, K_CUSTOM_OP_INPUT_UID_1);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().dims, expectedDims);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().strides, expectedStrides);

    // Verify output
    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    ASSERT_NE(desc->getOutputDescs()[0], nullptr);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().dims, expectedDims);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().strides, expectedStrides);
}

TEST_F(TestCustomOpOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getInputDescs()[0], _tensorMap[K_CUSTOM_OP_INPUT_UID_0]);
    EXPECT_EQ(desc->getInputDescs()[1], _tensorMap[K_CUSTOM_OP_INPUT_UID_1]);
    EXPECT_EQ(desc->getOutputDescs()[0], _tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]);
}

TEST_F(TestCustomOpOperationFromNode, FailsWithMissingInputTensor)
{
    _tensorMap.erase(K_CUSTOM_OP_INPUT_UID_0);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CustomOpOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCustomOpOperationFromNode, FailsWithMissingOutputTensor)
{
    _tensorMap.erase(K_CUSTOM_OP_OUTPUT_UID_0);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(CustomOpOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCustomOpOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 2 inputs + 1 output = 3
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(tensors[1]->getData().uid, K_CUSTOM_OP_INPUT_UID_1);
    EXPECT_EQ(tensors[2]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestCustomOpOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::CustomOpAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCustomOpAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->custom_op_id, K_CUSTOM_OP_ID);
    ASSERT_EQ(rebuiltAttrs->input_tensor_uids.size(), 2);
    EXPECT_EQ(rebuiltAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(rebuiltAttrs->input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(rebuiltAttrs->output_tensor_uids.size(), 1);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(rebuiltAttrs->data, K_CUSTOM_OP_OPAQUE_DATA);
}

TEST_F(TestCustomOpOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify custom_op_id
    int64_t idCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT, HIPDNN_TYPE_CHAR, 0, &idCount, nullptr);
    ASSERT_EQ(idCount, static_cast<int64_t>(K_CUSTOM_OP_ID.size() + 1));

    std::vector<char> idBuffer(static_cast<size_t>(idCount));
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                       HIPDNN_TYPE_CHAR,
                       idCount,
                       &idCount,
                       idBuffer.data());
    EXPECT_STREQ(idBuffer.data(), K_CUSTOM_OP_ID.c_str());

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CUSTOM_OP_EXT);

    // Verify name (empty default from fixture, count==1 for null terminator)
    int64_t nameCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &nameCount, nullptr);
    EXPECT_EQ(nameCount, 1);

    // --- Input tensor descriptors ---

    // Query count
    int64_t inputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       0,
                       &inputCount,
                       nullptr);
    ASSERT_EQ(inputCount, 2);

    // Retrieve all input descriptors
    std::array<hipdnnBackendDescriptor_t, 2> inputDescs = {};
    int64_t retrievedInputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       &retrievedInputCount,
                       static_cast<void*>(inputDescs.data()));
    ASSERT_EQ(retrievedInputCount, 2);

    // Wrap in ScopedDescriptor for RAII
    const hipdnn_backend::ScopedDescriptor input0Scoped(inputDescs[0]);
    const hipdnn_backend::ScopedDescriptor input1Scoped(inputDescs[1]);
    ASSERT_NE(input0Scoped.get(), nullptr);
    ASSERT_NE(input1Scoped.get(), nullptr);

    const std::vector<int64_t> expectedDims(K_CUSTOM_OP_TENSOR_DIMS.begin(),
                                            K_CUSTOM_OP_TENSOR_DIMS.end());
    const std::vector<int64_t> expectedStrides(K_CUSTOM_OP_TENSOR_STRIDES.begin(),
                                               K_CUSTOM_OP_TENSOR_STRIDES.end());

    hipdnn_backend::test_utilities::verifyTensorDescriptor(input0Scoped.get(),
                                                           K_CUSTOM_OP_INPUT_UID_0,
                                                           HIPDNN_DATA_FLOAT,
                                                           expectedDims,
                                                           expectedStrides);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(input1Scoped.get(),
                                                           K_CUSTOM_OP_INPUT_UID_1,
                                                           HIPDNN_DATA_FLOAT,
                                                           expectedDims,
                                                           expectedStrides);

    // --- Output tensor descriptors ---

    // Query count
    int64_t outputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       0,
                       &outputCount,
                       nullptr);
    ASSERT_EQ(outputCount, 1);

    // Retrieve output descriptor
    std::array<hipdnnBackendDescriptor_t, 1> outputDescs = {};
    int64_t retrievedOutputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &retrievedOutputCount,
                       static_cast<void*>(outputDescs.data()));
    ASSERT_EQ(retrievedOutputCount, 1);

    // Wrap in ScopedDescriptor for RAII
    const hipdnn_backend::ScopedDescriptor output0Scoped(outputDescs[0]);
    ASSERT_NE(output0Scoped.get(), nullptr);

    hipdnn_backend::test_utilities::verifyTensorDescriptor(output0Scoped.get(),
                                                           K_CUSTOM_OP_OUTPUT_UID_0,
                                                           HIPDNN_DATA_FLOAT,
                                                           expectedDims,
                                                           expectedStrides);
}

TEST_F(TestCustomOpOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_custom_op_1";

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_custom_op_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_custom_op_1");
}

TEST_F(TestCustomOpOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestCustomOpOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestCustomOpOperationFromNode, FromNodeWithSingleInputAndOutput)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;

    CustomOpAttributesT attrs;
    attrs.custom_op_id = K_CUSTOM_OP_ID;
    attrs.input_tensor_uids = {K_CUSTOM_OP_INPUT_UID_0};
    attrs.output_tensor_uids = {K_CUSTOM_OP_OUTPUT_UID_0};
    attrs.data = {K_CUSTOM_OP_OPAQUE_DATA.begin(), K_CUSTOM_OP_OPAQUE_DATA.end()};
    node.attributes.Set(attrs);

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    ASSERT_EQ(desc->getInputDescs().size(), 1);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCustomOpAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_EQ(rebuiltAttrs->input_tensor_uids.size(), 1);
    EXPECT_EQ(rebuiltAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(rebuiltAttrs->output_tensor_uids.size(), 1);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestCustomOpOperationFromNode, FromNodeWithZeroInputs)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;

    CustomOpAttributesT attrs;
    attrs.custom_op_id = K_CUSTOM_OP_ID;
    attrs.input_tensor_uids = {};
    attrs.output_tensor_uids = {K_CUSTOM_OP_OUTPUT_UID_0};
    attrs.data = {K_CUSTOM_OP_OPAQUE_DATA.begin(), K_CUSTOM_OP_OPAQUE_DATA.end()};
    node.attributes.Set(attrs);

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    ASSERT_EQ(desc->getInputDescs().size(), 0);
    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCustomOpAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->input_tensor_uids.empty());
    ASSERT_EQ(rebuiltAttrs->output_tensor_uids.size(), 1);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestCustomOpOperationFromNode, FromNodeWithZeroOutputs)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;

    CustomOpAttributesT attrs;
    attrs.custom_op_id = K_CUSTOM_OP_ID;
    attrs.input_tensor_uids = {K_CUSTOM_OP_INPUT_UID_0};
    attrs.output_tensor_uids = {};
    attrs.data = {K_CUSTOM_OP_OPAQUE_DATA.begin(), K_CUSTOM_OP_OPAQUE_DATA.end()};
    node.attributes.Set(attrs);

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    ASSERT_EQ(desc->getInputDescs().size(), 1);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(desc->getOutputDescs().size(), 0);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCustomOpAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_EQ(rebuiltAttrs->input_tensor_uids.size(), 1);
    EXPECT_EQ(rebuiltAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_TRUE(rebuiltAttrs->output_tensor_uids.empty());
}

TEST_F(TestCustomOpOperationFromNode, FromNodeWithTwoOutputs)
{
    // Add second output tensor to the map
    TensorAttributesT output1Attrs;
    output1Attrs.uid = K_CUSTOM_OP_OUTPUT_UID_1;
    output1Attrs.data_type = DataType::FLOAT;
    output1Attrs.dims = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_DIMS);
    output1Attrs.strides = hipdnn_tests::toVec(K_CUSTOM_OP_TENSOR_STRIDES);
    _tensorMap[K_CUSTOM_OP_OUTPUT_UID_1] = TensorDescriptor::fromFlatBuffer(output1Attrs);

    NodeT node;
    node.compute_data_type = DataType::FLOAT;

    CustomOpAttributesT attrs;
    attrs.custom_op_id = K_CUSTOM_OP_ID;
    attrs.input_tensor_uids = {K_CUSTOM_OP_INPUT_UID_0, K_CUSTOM_OP_INPUT_UID_1};
    attrs.output_tensor_uids = {K_CUSTOM_OP_OUTPUT_UID_0, K_CUSTOM_OP_OUTPUT_UID_1};
    attrs.data = {K_CUSTOM_OP_OPAQUE_DATA.begin(), K_CUSTOM_OP_OPAQUE_DATA.end()};
    node.attributes.Set(attrs);

    auto desc = CustomOpOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    ASSERT_EQ(desc->getInputDescs().size(), 2);
    EXPECT_EQ(desc->getInputDescs()[0]->getData().uid, K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(desc->getInputDescs()[1]->getData().uid, K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(desc->getOutputDescs().size(), 2);
    EXPECT_EQ(desc->getOutputDescs()[0]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(desc->getOutputDescs()[1]->getData().uid, K_CUSTOM_OP_OUTPUT_UID_1);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsCustomOpAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_EQ(rebuiltAttrs->input_tensor_uids.size(), 2);
    ASSERT_EQ(rebuiltAttrs->output_tensor_uids.size(), 2);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(rebuiltAttrs->output_tensor_uids[1], K_CUSTOM_OP_OUTPUT_UID_1);
}
