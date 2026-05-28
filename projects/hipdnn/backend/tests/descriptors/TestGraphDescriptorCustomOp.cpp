// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "descriptors/CustomOpOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedCustomOp(HipdnnBackendDescriptor* input0,
                            HipdnnBackendDescriptor* input1,
                            HipdnnBackendDescriptor* output0,
                            hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                            const std::string& name = "")
{
    auto wrapper = createDescriptor<CustomOpOperationDescriptor>();
    auto desc = wrapper->asDescriptor<CustomOpOperationDescriptor>();

    std::array<HipdnnBackendDescriptor*, 2> inputs = {input0, input1};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(inputs.data()));

    std::array<HipdnnBackendDescriptor*, 1> outputs = {output0};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(outputs.data()));

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(K_CUSTOM_OP_ID.size()),
                       K_CUSTOM_OP_ID.c_str());

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(K_CUSTOM_OP_OPAQUE_DATA.size()),
                       K_CUSTOM_OP_OPAQUE_DATA.data());

    desc->setAttribute(HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.c_str());
    }

    desc->finalize();
    return wrapper;
}

} // namespace

class TestGraphDescriptorCustomOp : public ::testing::Test
{
public:
    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handle));
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

namespace
{

// Helper: builds and serializes a custom op with one tensor on the given side
// (zero on the other). Returns the unpacked GraphT for assertion.
//
// @param tensorDesc  A finalized tensor descriptor to place on one side.
// @param tensorSideAttr  The attribute to set — either INPUTS or OUTPUTS — controls which side
//                        receives the tensor (the other side remains zero).
// @param fixture  The test fixture (provides getDescriptor() and setHandle()).
std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>
    buildAndSerializeZeroPortCustomOp(HipdnnBackendDescriptor* tensorDesc,
                                      hipdnnBackendAttributeName_t tensorSideAttr,
                                      TestGraphDescriptorCustomOp& fixture)
{
    auto wrapper = createDescriptor<CustomOpOperationDescriptor>();
    auto desc = wrapper->asDescriptor<CustomOpOperationDescriptor>();

    std::array<HipdnnBackendDescriptor*, 1> tensors = {tensorDesc};
    desc->setAttribute(tensorSideAttr,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(tensors.data()));

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(K_CUSTOM_OP_ID.size()),
                       K_CUSTOM_OP_ID.c_str());

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(K_CUSTOM_OP_OPAQUE_DATA.size()),
                       K_CUSTOM_OP_OPAQUE_DATA.data());

    hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    EXPECT_NO_THROW(desc->finalize());

    auto graphDesc = fixture.getDescriptor();
    fixture.setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    EXPECT_NO_THROW(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            static_cast<const void*>(ops.data())));
    EXPECT_NO_THROW(graphDesc->finalize());

    auto serialized = graphDesc->getSerializedGraph();
    EXPECT_NE(serialized.ptr, nullptr);
    EXPECT_GT(serialized.size, 0UL);

    return UnPackGraph(serialized.ptr);
}

} // namespace

TEST_F(TestGraphDescriptorCustomOp, BuildFromSingleCustomOpOperation)
{
    auto input0Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_0, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto input1Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_1, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                             toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                             toVec(K_CUSTOM_OP_TENSOR_STRIDES));

    auto customOp = createFinalizedCustomOp(input0Desc.get(), input1Desc.get(), output0Desc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {customOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node.attributes.type, NodeAttributes::CustomOpAttributes);

    auto* customAttrs = node.attributes.AsCustomOpAttributes();
    ASSERT_NE(customAttrs, nullptr);
    EXPECT_EQ(customAttrs->custom_op_id, K_CUSTOM_OP_ID);
    ASSERT_EQ(customAttrs->input_tensor_uids.size(), 2);
    EXPECT_EQ(customAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(customAttrs->input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(customAttrs->output_tensor_uids.size(), 1);
    EXPECT_EQ(customAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(customAttrs->data, K_CUSTOM_OP_OPAQUE_DATA);
}

TEST_F(TestGraphDescriptorCustomOp, FinalizeSucceedsWithZeroInputTensors)
{
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                             toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                             toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto graphT = buildAndSerializeZeroPortCustomOp(
        output0Desc.get(), HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT, *this);

    ASSERT_EQ(graphT->nodes.size(), 1);
    auto* customAttrs = graphT->nodes[0]->attributes.AsCustomOpAttributes();
    ASSERT_NE(customAttrs, nullptr);
    EXPECT_TRUE(customAttrs->input_tensor_uids.empty());
    ASSERT_EQ(customAttrs->output_tensor_uids.size(), 1);
    EXPECT_EQ(customAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestGraphDescriptorCustomOp, FinalizeSucceedsWithZeroOutputTensors)
{
    auto input0Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_0, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto graphT = buildAndSerializeZeroPortCustomOp(
        input0Desc.get(), HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT, *this);

    ASSERT_EQ(graphT->nodes.size(), 1);
    auto* customAttrs = graphT->nodes[0]->attributes.AsCustomOpAttributes();
    ASSERT_NE(customAttrs, nullptr);
    ASSERT_EQ(customAttrs->input_tensor_uids.size(), 1);
    EXPECT_EQ(customAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_TRUE(customAttrs->output_tensor_uids.empty());
}

TEST_F(TestGraphDescriptorCustomOp, ComputeDataTypePreserved)
{
    auto input0Desc = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_0,
                                            toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                            toVec(K_CUSTOM_OP_TENSOR_STRIDES),
                                            HIPDNN_DATA_HALF);
    auto input1Desc = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_1,
                                            toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                            toVec(K_CUSTOM_OP_TENSOR_STRIDES),
                                            HIPDNN_DATA_HALF);
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                             toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                             toVec(K_CUSTOM_OP_TENSOR_STRIDES),
                                             HIPDNN_DATA_HALF);

    auto customOp = createFinalizedCustomOp(
        input0Desc.get(), input1Desc.get(), output0Desc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {customOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::HALF);
    ASSERT_EQ(node.attributes.type, NodeAttributes::CustomOpAttributes);

    auto* customAttrs = node.attributes.AsCustomOpAttributes();
    ASSERT_NE(customAttrs, nullptr);
    EXPECT_EQ(customAttrs->custom_op_id, K_CUSTOM_OP_ID);
}

TEST_F(TestGraphDescriptorCustomOp, OperationNamePreservedInGraph)
{
    auto input0Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_0, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto input1Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_1, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                             toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                             toVec(K_CUSTOM_OP_TENSOR_STRIDES));

    auto customOp = createFinalizedCustomOp(
        input0Desc.get(), input1Desc.get(), output0Desc.get(), HIPDNN_DATA_FLOAT, "test_custom_op");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {customOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->name, "test_custom_op");
}

TEST_F(TestGraphDescriptorCustomOp, OperationNameRoundTripThroughLifting)
{
    auto input0Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_0, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto input1Desc = createFinalizedTensor(
        K_CUSTOM_OP_INPUT_UID_1, toVec(K_CUSTOM_OP_TENSOR_DIMS), toVec(K_CUSTOM_OP_TENSOR_STRIDES));
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                             toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                             toVec(K_CUSTOM_OP_TENSOR_STRIDES));

    auto customOp = createFinalizedCustomOp(input0Desc.get(),
                                            input1Desc.get(),
                                            output0Desc.get(),
                                            HIPDNN_DATA_FLOAT,
                                            "lift_custom_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {customOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize and deserialize via FlatBuffer
    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    // Rebuild from the deserialized graph using NodeFactory
    auto tensorMap = NodeFactory::buildTensorMap(graphT->tensors);
    ASSERT_EQ(graphT->nodes.size(), 1);

    auto rebuilt = NodeFactory::createOperationFromNode(*graphT->nodes[0], tensorMap);
    ASSERT_NE(rebuilt, nullptr);

    auto* graphOp = rebuilt->asGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto rebuiltNode = graphOp->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "lift_custom_name");
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::CustomOpAttributes);
}
