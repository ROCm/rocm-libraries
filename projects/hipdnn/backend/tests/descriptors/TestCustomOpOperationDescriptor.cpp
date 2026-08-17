// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/CustomOpOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestCustomOpOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<CustomOpOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<CustomOpOperationDescriptor>();
    }

    void setAllAttributesExcept(std::initializer_list<hipdnnBackendAttributeName_t> skip = {}) const
    {
        auto desc = getDescriptor();
        auto shouldSkip = [&](hipdnnBackendAttributeName_t attr) {
            return std::find(skip.begin(), skip.end(), attr) != skip.end();
        };

        if(!shouldSkip(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT))
        {
            std::array<HipdnnBackendDescriptor*, 2> inputs = {_input0.get(), _input1.get()};
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               2,
                               static_cast<const void*>(inputs.data()));
        }

        if(!shouldSkip(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT))
        {
            std::array<HipdnnBackendDescriptor*, 1> outputs = {_output0.get()};
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               static_cast<const void*>(outputs.data()));
        }

        if(!shouldSkip(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT))
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                               HIPDNN_TYPE_CHAR,
                               static_cast<int64_t>(K_CUSTOM_OP_ID.size()),
                               K_CUSTOM_OP_ID.c_str());
        }

        if(!shouldSkip(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT))
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                               HIPDNN_TYPE_CHAR,
                               static_cast<int64_t>(K_CUSTOM_OP_OPAQUE_DATA.size()),
                               K_CUSTOM_OP_OPAQUE_DATA.data());
        }

        if(!shouldSkip(HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT))
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void makeFinalized() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _input0 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _input1 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _output0 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _output1 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<CustomOpOperationDescriptor>();
        _input0 = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_0,
                                        toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                        toVec(K_CUSTOM_OP_TENSOR_STRIDES));
        _input1 = createFinalizedTensor(K_CUSTOM_OP_INPUT_UID_1,
                                        toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                        toVec(K_CUSTOM_OP_TENSOR_STRIDES));
        _output0 = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_0,
                                         toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                         toVec(K_CUSTOM_OP_TENSOR_STRIDES));
        _output1 = createFinalizedTensor(K_CUSTOM_OP_OUTPUT_UID_1,
                                         toVec(K_CUSTOM_OP_TENSOR_DIMS),
                                         toVec(K_CUSTOM_OP_TENSOR_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _input0.reset();
        _input1.reset();
        _output0.reset();
        _output1.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT);
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeSucceedsWithoutInputs)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT});
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeSucceedsWithoutOutputs)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT});
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeFailsWithoutCustomOpId)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeFailsWithEmptyCustomOpId)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT});
    auto desc = getDescriptor();
    const std::string emptyId;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT, HIPDNN_TYPE_CHAR, 0, emptyId.c_str());
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCustomOpOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setAllAttributesExcept({HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, SetInputTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> inputs = {_input0.get(), _input1.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(inputs.data())));

    ASSERT_EQ(desc->getInputDescs().size(), 2);
    ASSERT_EQ(desc->getData().input_tensor_uids.size(), 2);
    ASSERT_EQ(desc->getData().input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(desc->getData().input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
}

TEST_F(TestCustomOpOperationDescriptor, SetOutputTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 1> outputs = {_output0.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(outputs.data())));

    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    ASSERT_EQ(desc->getData().output_tensor_uids.size(), 1);
    ASSERT_EQ(desc->getData().output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(TestCustomOpOperationDescriptor, SetMultipleOutputTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> outputs = {_output0.get(), _output1.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(outputs.data())));

    ASSERT_EQ(desc->getOutputDescs().size(), 2);
    ASSERT_EQ(desc->getData().output_tensor_uids.size(), 2);
    ASSERT_EQ(desc->getData().output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    ASSERT_EQ(desc->getData().output_tensor_uids[1], K_CUSTOM_OP_OUTPUT_UID_1);
}

TEST_F(TestCustomOpOperationDescriptor, SetCustomOpId)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(K_CUSTOM_OP_ID.size()),
                                       K_CUSTOM_OP_ID.c_str()));

    ASSERT_EQ(desc->getData().custom_op_id, K_CUSTOM_OP_ID);
}

TEST_F(TestCustomOpOperationDescriptor, SetOpaqueData)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(K_CUSTOM_OP_OPAQUE_DATA.size()),
                                       K_CUSTOM_OP_OPAQUE_DATA.data()));

    ASSERT_EQ(desc->getData().data, K_CUSTOM_OP_OPAQUE_DATA);
}

TEST_F(TestCustomOpOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestCustomOpOperationDescriptor, SetInputTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 1> inputs = {_unfinalizedTensor.get()};
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  static_cast<const void*>(inputs.data())),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestCustomOpOperationDescriptor, SetInputTensorFailsWrongType)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCustomOpOperationDescriptor, SetInputTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestCustomOpOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCustomOpOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCustomOpOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, GetAttributeInputTensorArray)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 2);

    std::array<HipdnnBackendDescriptor*, 2> rawRetrieved = {nullptr, nullptr};
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &elementCount,
                                       static_cast<void*>(rawRetrieved.data())));

    auto retrieved0 = std::unique_ptr<HipdnnBackendDescriptor>(rawRetrieved[0]);
    auto retrieved1 = std::unique_ptr<HipdnnBackendDescriptor>(rawRetrieved[1]);
    ASSERT_NE(retrieved0, nullptr);
    ASSERT_NE(retrieved1, nullptr);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeOutputTensorArray)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);

    HipdnnBackendDescriptor* rawRetrieved = nullptr;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawRetrieved)));

    auto retrieved = std::unique_ptr<HipdnnBackendDescriptor>(rawRetrieved);
    ASSERT_NE(retrieved, nullptr);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeCustomOpId)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, static_cast<int64_t>(K_CUSTOM_OP_ID.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(elementCount));
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       elementCount,
                                       &elementCount,
                                       buffer.data()));
    ASSERT_EQ(std::string(buffer.data()), K_CUSTOM_OP_ID);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeOpaqueData)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, static_cast<int64_t>(K_CUSTOM_OP_OPAQUE_DATA.size()));

    std::vector<uint8_t> buffer(static_cast<size_t>(elementCount));
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       elementCount,
                                       &elementCount,
                                       buffer.data()));
    ASSERT_EQ(buffer, K_CUSTOM_OP_OPAQUE_DATA);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeComputeType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnDataType_t retrieved = HIPDNN_DATA_DOUBLE;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_FLOAT);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  &elementCount,
                                                  nullptr),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeCustomOpIdQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT, HIPDNN_TYPE_CHAR, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestCustomOpOperationDescriptor, GetAttributeOpaqueDataQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT, HIPDNN_TYPE_CHAR, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0].get(), _input0->asDescriptor<TensorDescriptor>().get());
    ASSERT_EQ(tensors[1].get(), _input1->asDescriptor<TensorDescriptor>().get());
    ASSERT_EQ(tensors[2].get(), _output0->asDescriptor<TensorDescriptor>().get());
}

TEST_F(TestCustomOpOperationDescriptor, GetTensorDescriptorsOrderIsInputsThenOutputs)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);

    // First two should be inputs, last should be output
    EXPECT_EQ(tensors[0].get(), _input0->asDescriptor<TensorDescriptor>().get());
    EXPECT_EQ(tensors[1].get(), _input1->asDescriptor<TensorDescriptor>().get());
    EXPECT_EQ(tensors[2].get(), _output0->asDescriptor<TensorDescriptor>().get());
}

TEST_F(TestCustomOpOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::CustomOpAttributes);

    auto* customAttrs = node->attributes.AsCustomOpAttributes();
    ASSERT_NE(customAttrs, nullptr);
    ASSERT_EQ(customAttrs->custom_op_id, K_CUSTOM_OP_ID);
    ASSERT_EQ(customAttrs->input_tensor_uids.size(), 2);
    ASSERT_EQ(customAttrs->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(customAttrs->input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(customAttrs->output_tensor_uids.size(), 1);
    ASSERT_EQ(customAttrs->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);
    ASSERT_EQ(customAttrs->data, K_CUSTOM_OP_OPAQUE_DATA);
}

TEST_F(TestCustomOpOperationDescriptor, BuildNodeWithHalfComputeType)
{
    auto desc = getDescriptor();

    std::array<HipdnnBackendDescriptor*, 2> inputs = {_input0.get(), _input1.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(inputs.data()));

    std::array<HipdnnBackendDescriptor*, 1> outputs = {_output0.get()};
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

    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestCustomOpOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_EQ(desc->getInputDescs().size(), 2);
    ASSERT_EQ(desc->getOutputDescs().size(), 1);
    ASSERT_EQ(desc->getInputDescs()[0].get(), _input0->asDescriptor<TensorDescriptor>().get());
    ASSERT_EQ(desc->getInputDescs()[1].get(), _input1->asDescriptor<TensorDescriptor>().get());
    ASSERT_EQ(desc->getOutputDescs()[0].get(), _output0->asDescriptor<TensorDescriptor>().get());
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("CustomOpOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("custom_op_id=" + K_CUSTOM_OP_ID), std::string::npos);
    ASSERT_NE(str.find(std::to_string(K_CUSTOM_OP_INPUT_UID_0)), std::string::npos);
    ASSERT_NE(str.find(std::to_string(K_CUSTOM_OP_INPUT_UID_1)), std::string::npos);
    ASSERT_NE(str.find(std::to_string(K_CUSTOM_OP_OUTPUT_UID_0)), std::string::npos);
    ASSERT_NE(str.find("data_size=" + std::to_string(K_CUSTOM_OP_OPAQUE_DATA.size())),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type=FLOAT"), std::string::npos);
}

// =============================================================================
// Interface Casting Tests
// =============================================================================

TEST_F(TestCustomOpOperationDescriptor, TryAsGraphOperationReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
}

TEST_F(TestCustomOpOperationDescriptor, TryAsGraphOperationReturnsNullForWrongType)
{
    auto graphOp = _input0->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
