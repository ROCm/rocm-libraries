// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/ReductionOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestReductionOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ReductionOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ReductionOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_REDUCTION_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    }

    void setRequiredAttributes() const
    {
        setTensors();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto mode = HIPDNN_REDUCE_TENSOR_ADD;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ReductionOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
        _yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _yDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestReductionOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
}

TEST_F(TestReductionOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestReductionOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    auto mode = HIPDNN_REDUCE_TENSOR_ADD;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, FinalizeFailsWithoutReductionMode)
{
    setTensors();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestReductionOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestReductionOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().out_tensor_uid, K_REDUCTION_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestReductionOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestReductionOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestReductionOperationDescriptor, SetReductionMode)
{
    auto desc = getDescriptor();
    auto mode = HIPDNN_REDUCE_TENSOR_ADD;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode));

    ASSERT_EQ(desc->getData().mode, ReductionMode::ADD);
}

TEST_F(TestReductionOperationDescriptor, SetReductionModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto mode = HIPDNN_REDUCE_TENSOR_ADD;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 2, &mode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, SetReductionModeWrongType)
{
    auto desc = getDescriptor();
    int64_t wrongValue = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_INT64, 1, &wrongValue),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestReductionOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestReductionOperationDescriptor, SetComputeDataTypeWrongType)
{
    auto desc = getDescriptor();
    int64_t wrongValue = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_INT64, 1, &wrongValue),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Boolean Fields
// =============================================================================

TEST_F(TestReductionOperationDescriptor, SetAttributeIsDeterministic)
{
    auto desc = getDescriptor();

    // Verify default is false
    setRequiredAttributes();
    desc->finalize();
    bool retrieved = true;
    int64_t elementCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    EXPECT_FALSE(retrieved);

    // Now test with explicit true
    auto wrapper2 = createDescriptor<ReductionOperationDescriptor>();
    auto desc2 = wrapper2->asDescriptor<ReductionOperationDescriptor>();
    desc2->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc2->setAttribute(
        HIPDNN_ATTR_OPERATION_REDUCTION_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc2->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto mode = HIPDNN_REDUCE_TENSOR_ADD;
    desc2->setAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);
    bool trueVal = true;
    desc2->setAttribute(HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc2->finalize();

    retrieved = false;
    elementCount = 0;
    desc2->getAttribute(
        HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    ASSERT_EQ(elementCount, 1);
    EXPECT_TRUE(retrieved);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeIsDeterministicQueryMode)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC, HIPDNN_TYPE_BOOLEAN, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestReductionOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestReductionOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestReductionOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
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

TEST_F(TestReductionOperationDescriptor, GetAttributeReductionParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // mode
    hipdnnReduceTensorOp_t mode = HIPDNN_REDUCE_TENSOR_ADD;
    int64_t modeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &modeCount, &mode));
    ASSERT_EQ(modeCount, 1);
    EXPECT_EQ(mode, HIPDNN_REDUCE_TENSOR_ADD);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestReductionOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestReductionOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeReductionModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_REDUCTION_OPERATOR,
                                       HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeReductionModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_REDUCTION_OPERATOR,
                                                  HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestReductionOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_REDUCTION_TENSOR_X_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_REDUCTION_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestReductionOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("ReductionOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_REDUCTION_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_REDUCTION_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestReductionOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    ASSERT_EQ(tensors[0]->getData().uid, K_REDUCTION_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_REDUCTION_TENSOR_Y_UID);
}

TEST_F(TestReductionOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ReductionAttributes);

    auto* attrs = node->attributes.AsReductionAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
    ASSERT_EQ(attrs->out_tensor_uid, K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(attrs->mode, ReductionMode::ADD);
}

TEST_F(TestReductionOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestReductionOperationDescriptor, GetTensorDescriptorsOrderIsXY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    // Verify ordering: [XDESC, YDESC] matches UIDs [90, 91]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getYDesc());
}

TEST_F(TestReductionOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    ASSERT_EQ(tensors[0]->getData().uid, K_REDUCTION_TENSOR_X_UID);
}

TEST_F(TestReductionOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestReductionOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_reduction_op";

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
    EXPECT_STREQ(buffer.data(), "test_reduction_op");
}

TEST_F(TestReductionOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestReductionOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_REDUCTION_EXT);
}

TEST_F(TestReductionOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestReductionOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_reduction";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_reduction");
}
