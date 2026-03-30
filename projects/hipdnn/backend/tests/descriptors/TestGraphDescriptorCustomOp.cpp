// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "descriptors/CustomOpOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;

namespace
{

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedCustomOp(HipdnnBackendDescriptor* input0,
                            HipdnnBackendDescriptor* input1,
                            HipdnnBackendDescriptor* output0,
                            hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
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

TEST_F(TestGraphDescriptorCustomOp, BuildFromSingleCustomOpOperation)
{
    auto input0Desc = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_0, {2, 3}, {3, 1});
    auto input1Desc = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_1, {2, 3}, {3, 1});
    auto output0Desc = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0, {2, 3}, {3, 1});

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

TEST_F(TestGraphDescriptorCustomOp, ComputeDataTypePreserved)
{
    auto input0Desc
        = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_0, {2, 3}, {3, 1}, HIPDNN_DATA_HALF);
    auto input1Desc
        = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_1, {2, 3}, {3, 1}, HIPDNN_DATA_HALF);
    auto output0Desc
        = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0, {2, 3}, {3, 1}, HIPDNN_DATA_HALF);

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
