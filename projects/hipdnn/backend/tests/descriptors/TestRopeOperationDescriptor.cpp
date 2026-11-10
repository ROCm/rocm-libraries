// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/RopeOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RopeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <optional>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestRopeOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<RopeOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<RopeOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_freqsDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
        }
    }

    void setRopeParams() const
    {
        auto desc = getDescriptor();

        auto outputScale = static_cast<float>(0.5);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE, HIPDNN_TYPE_FLOAT, 1, &outputScale);
        auto ropeDim = static_cast<int64_t>(4);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM, HIPDNN_TYPE_INT64, 1, &ropeDim);
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        setRopeParams();
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _freqsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<RopeOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_ROPE_TENSOR_X_UID,
                                       toVec(K_ROPE_TENSOR_X_DIMS),
                                       toVec(K_ROPE_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _freqsDesc = createFinalizedTensor(K_ROPE_TENSOR_FREQS_UID,
                                           toVec(K_ROPE_TENSOR_FREQS_DIMS),
                                           toVec(K_ROPE_TENSOR_FREQS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
        _yDesc = createFinalizedTensor(K_ROPE_TENSOR_Y_UID,
                                       toVec(K_ROPE_TENSOR_Y_DIMS),
                                       toVec(K_ROPE_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _freqsDesc.reset();
        _yDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR);
}

TEST_F(TestRopeOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestRopeOperationDescriptorFinalizeFailsWithout
    : public TestRopeOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestRopeOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestRopeOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                           HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                                           HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestRopeOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_ROPE_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestRopeOperationDescriptor, SetTensorDescriptorFreqs)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_freqsDesc));

    ASSERT_EQ(desc->getData().freqs_tensor_uid, K_ROPE_TENSOR_FREQS_UID);
    ASSERT_NE(desc->getFreqsDesc(), nullptr);
}

TEST_F(TestRopeOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_ROPE_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestRopeOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestRopeOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRopeOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRopeOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestRopeOperationDescriptor, SetOutputScale)
{
    auto desc = getDescriptor();
    auto outputScale = static_cast<float>(0.5);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE, HIPDNN_TYPE_FLOAT, 1, &outputScale));
    EXPECT_EQ(desc->getData().output_scale, outputScale);
}

TEST_F(TestRopeOperationDescriptor, SetRopeDim)
{
    auto desc = getDescriptor();
    auto ropeDim = static_cast<int64_t>(4);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM, HIPDNN_TYPE_INT64, 1, &ropeDim));
    EXPECT_EQ(desc->getData().rope_dim, ropeDim);
}

TEST_F(TestRopeOperationDescriptor, SetRopeOffset)
{
    auto desc = getDescriptor();
    auto ropeOffset = static_cast<int64_t>(0);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT, HIPDNN_TYPE_INT64, 1, &ropeOffset));
    EXPECT_EQ(desc->getData().rope_offset_ext, ropeOffset);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestRopeOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRopeOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardX(retrievedX);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetAttributeRopeParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // output scale
    float outputScale = 0;
    int64_t outputScaleCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE,
                                       HIPDNN_TYPE_FLOAT,
                                       1,
                                       &outputScaleCount,
                                       &outputScale));
    ASSERT_EQ(outputScaleCount, 1);
    EXPECT_EQ(outputScale, static_cast<float>(0.5));

    // rope dim
    int64_t ropeDim = 0;
    int64_t ropeDimCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM, HIPDNN_TYPE_INT64, 1, &ropeDimCount, &ropeDim));
    ASSERT_EQ(ropeDimCount, 1);
    EXPECT_EQ(ropeDim, static_cast<int64_t>(4));

    // rope offset: unset, so no element is returned
    int64_t ropeOffset = 0;
    int64_t ropeOffsetCount = -1;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &ropeOffsetCount,
                                       &ropeOffset));
    EXPECT_EQ(ropeOffsetCount, 0);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeRopeOffsetWhenSet)
{
    setRequiredAttributes();
    auto desc = getDescriptor();
    auto offset = static_cast<int64_t>(0);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT, HIPDNN_TYPE_INT64, 1, &offset);
    desc->finalize();

    int64_t ropeOffset = -1;
    int64_t ropeOffsetCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &ropeOffsetCount,
                                       &ropeOffset));
    ASSERT_EQ(ropeOffsetCount, 1);
    EXPECT_EQ(ropeOffset, static_cast<int64_t>(0));
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeTensorFreqsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getFreqsDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_ROPE_TENSOR_X_UID);
    ASSERT_EQ(desc->getFreqsDesc()->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_ROPE_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestRopeOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("RopeOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_ROPE_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("freqs_uid=" + std::to_string(K_ROPE_TENSOR_FREQS_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_ROPE_TENSOR_Y_UID)), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_ROPE_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_ROPE_TENSOR_FREQS_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->attributes.type, NodeAttributes::RopeAttributes);

    auto* attrs = node->attributes.AsRopeAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_ROPE_TENSOR_X_UID);
    ASSERT_EQ(attrs->freqs_tensor_uid, K_ROPE_TENSOR_FREQS_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_ROPE_TENSOR_Y_UID);
}

TEST_F(TestRopeOperationDescriptor, GetTensorDescriptorsOrderIsXFreqsY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [XDESC, FREQSDESC, YDESC] matches UIDs [9100, 9101, 9102]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getFreqsDesc());
    EXPECT_EQ(tensors[2], desc->getYDesc());
}

TEST_F(TestRopeOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_ROPE_TENSOR_X_UID);
}

TEST_F(TestRopeOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_rope_op";

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(name.size()),
                                       name.c_str()));

    // Finalize and verify name round-trips
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_rope_op");
}

TEST_F(TestRopeOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
{
    auto desc = getDescriptor();
    const std::string name = "my_op";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, static_cast<int64_t>(name.size() + 1));
}

// =============================================================================
// Operation Type Tests
// =============================================================================

TEST_F(TestRopeOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_ROPE_FWD_EXT);
}

TEST_F(TestRopeOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRopeOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_rope";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_rope");
}
