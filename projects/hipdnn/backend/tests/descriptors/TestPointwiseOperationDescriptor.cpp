// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/PointwiseOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestPointwiseOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<PointwiseOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<PointwiseOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_out0Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);
    }

    void makeUnaryFinalized(hipdnnPointwiseMode_t mode = HIPDNN_POINTWISE_RELU_FWD) const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_out0Desc);
        auto operation = mode;
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
        auto computeType = HIPDNN_DATA_FLOAT;
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        desc->finalize();
    }

    void setRequiredAttributes() const
    {
        setTensors();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto operation = HIPDNN_POINTWISE_ADD;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in0Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _out0Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in1Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in2Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<PointwiseOperationDescriptor>();
        _in0Desc = createFinalizedTensor(
            K_PW_TENSOR_IN0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
        _out0Desc = createFinalizedTensor(
            K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
        _in1Desc = createFinalizedTensor(K_PW_TENSOR_IN1_UID);
        _in2Desc = createFinalizedTensor(K_PW_TENSOR_IN2_UID);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _in0Desc.reset();
        _out0Desc.reset();
        _in1Desc.reset();
        _in2Desc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutIn0Tensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutOut0Tensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();

    auto operation = HIPDNN_POINTWISE_ADD;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutPointwiseMode)
{
    setTensors();

    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn0)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    ASSERT_NE(desc->getIn0Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorOut0)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc));

    ASSERT_EQ(desc->getData().out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn1)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc));

    ASSERT_EQ(desc->getData().in_1_tensor_uid, K_PW_TENSOR_IN1_UID);
    ASSERT_NE(desc->getIn1Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn2)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc));

    ASSERT_EQ(desc->getData().in_2_tensor_uid, K_PW_TENSOR_IN2_UID);
    ASSERT_NE(desc->getIn2Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetAxisAsInt64)
{
    auto desc = getDescriptor();
    int64_t axisValue = 2;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 1, &axisValue));

    ASSERT_EQ(desc->getData().axis_tensor_uid, 2);
}

TEST_F(TestPointwiseOperationDescriptor, GetAxisAsInt64)
{
    setRequiredAttributes();
    auto desc = getDescriptor();
    int64_t axisValue = 3;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 1, &axisValue);
    desc->finalize();

    int64_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 1, &elementCount, &retrieved));
    ASSERT_EQ(elementCount, 1);
    ASSERT_EQ(retrieved, 3);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_INT64, 1, &_in0Desc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_in0Desc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetPointwiseMode)
{
    auto desc = getDescriptor();
    auto operation = HIPDNN_POINTWISE_ADD;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation));

    ASSERT_EQ(desc->getData().operation, PointwiseMode::ADD);
}

TEST_F(TestPointwiseOperationDescriptor, SetPointwiseModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto operation = HIPDNN_POINTWISE_ADD;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 2, &operation),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestPointwiseOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawIn0 = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawIn0)));
    const std::unique_ptr<HipdnnBackendDescriptor> retrievedIn0(rawIn0);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedIn0, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributePointwiseParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // operation
    hipdnnPointwiseMode_t operation = HIPDNN_POINTWISE_ABS;
    int64_t operationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operationCount, &operation));
    ASSERT_EQ(operationCount, 1);
    EXPECT_EQ(operation, HIPDNN_POINTWISE_ADD);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests (Parameterized)
// =============================================================================

struct QueryReturnsOneParam
{
    hipdnnBackendAttributeName_t attribute;
    hipdnnBackendAttributeType_t type;
    const char* name;
};

class TestPointwiseOperationDescriptorQueryReturnsOne
    : public TestPointwiseOperationDescriptor,
      public ::testing::WithParamInterface<QueryReturnsOneParam>
{
};

TEST_P(TestPointwiseOperationDescriptorQueryReturnsOne, QueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(GetParam().attribute, GetParam().type, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllAttributes,
    TestPointwiseOperationDescriptorQueryReturnsOne,
    ::testing::Values(
        QueryReturnsOneParam{
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, "TensorIn0"},
        QueryReturnsOneParam{HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                             "TensorOut0"},
        QueryReturnsOneParam{
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, "TensorIn1"},
        QueryReturnsOneParam{
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, "TensorIn2"},
        QueryReturnsOneParam{
            HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, "PointwiseMode"},
        QueryReturnsOneParam{
            HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, "ComputeType"}),
    [](const ::testing::TestParamInfo<QueryReturnsOneParam>& info) {
        return std::string(info.param.name);
    });

TEST_F(TestPointwiseOperationDescriptor, GetAttributeAxisQueryReturnsZeroWhenUnset)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 0);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeAxisQueryReturnsOneWhenSet)
{
    int64_t axisValue = 3;
    getDescriptor()->setAttribute(HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 1, &axisValue);
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributePointwiseModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getIn0Desc(), nullptr);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
    ASSERT_NE(desc->getIn1Desc(), nullptr);
    ASSERT_NE(desc->getIn2Desc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getIn0Desc()->getData().uid, K_PW_TENSOR_IN0_UID);
    ASSERT_EQ(desc->getOut0Desc()->getData().uid, K_PW_TENSOR_OUT0_UID);
    ASSERT_EQ(desc->getIn1Desc()->getData().uid, K_PW_TENSOR_IN1_UID);
    ASSERT_EQ(desc->getIn2Desc()->getData().uid, K_PW_TENSOR_IN2_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("PointwiseOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("in_0_uid=1300"), std::string::npos);
    ASSERT_NE(str.find("out_0_uid=1301"), std::string::npos);
    ASSERT_NE(str.find("in_1_uid=1302"), std::string::npos);
    ASSERT_NE(str.find("in_2_uid=1303"), std::string::npos);
    ASSERT_NE(str.find("axis=nullopt"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetTensorDescriptorsReturnsBinaryOpTensors)
{
    // Set up a binary op (IN_0, IN_1, OUT_0 only)
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto operation = HIPDNN_POINTWISE_ADD;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    desc->finalize();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_PW_TENSOR_IN0_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_PW_TENSOR_OUT0_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_PW_TENSOR_IN1_UID);
}

TEST_F(TestPointwiseOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::PointwiseAttributes);

    auto* attrs = node->attributes.AsPointwiseAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    ASSERT_EQ(attrs->out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    ASSERT_EQ(attrs->in_1_tensor_uid, K_PW_TENSOR_IN1_UID);
    ASSERT_EQ(attrs->in_2_tensor_uid, K_PW_TENSOR_IN2_UID);
    ASSERT_FALSE(attrs->axis_tensor_uid.has_value());
}

TEST_F(TestPointwiseOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestPointwiseOperationDescriptor, GetTensorDescriptorsOrderIsIn0Out0In1In2)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    // Verify ordering: [IN_0, OUT_0, IN_1, IN_2]
    EXPECT_EQ(tensors[0], desc->getIn0Desc());
    EXPECT_EQ(tensors[1], desc->getOut0Desc());
    EXPECT_EQ(tensors[2], desc->getIn1Desc());
    EXPECT_EQ(tensors[3], desc->getIn2Desc());
}

TEST_F(TestPointwiseOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_PW_TENSOR_IN0_UID);
}

TEST_F(TestPointwiseOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _in0Desc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Scalar Attribute Set/Get Tests (Parameterized)
// =============================================================================

struct OptionalFloatParam
{
    hipdnnBackendAttributeName_t attribute;
    hipdnnPointwiseMode_t pointwiseMode;
    float value;
    const char* name;
};

class TestPointwiseOperationDescriptorOptionalFloat
    : public ::testing::TestWithParam<OptionalFloatParam>
{
public:
    std::shared_ptr<PointwiseOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<PointwiseOperationDescriptor>();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in0Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _out0Desc = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<PointwiseOperationDescriptor>();
        _in0Desc = createFinalizedTensor(
            K_PW_TENSOR_IN0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
        _out0Desc = createFinalizedTensor(
            K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    }

    void TearDown() override
    {
        _wrapper.reset();
        _in0Desc.reset();
        _out0Desc.reset();
    }
};

TEST_P(TestPointwiseOperationDescriptorOptionalFloat, SetGetValue)
{
    auto param = GetParam();
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    auto operation = param.pointwiseMode;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    auto value = param.value;
    ASSERT_NO_THROW(desc->setAttribute(param.attribute, HIPDNN_TYPE_FLOAT, 1, &value));
    desc->finalize();

    float retrieved = 0.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(param.attribute, HIPDNN_TYPE_FLOAT, 1, &elementCount, &retrieved));
    ASSERT_EQ(elementCount, 1);
    ASSERT_FLOAT_EQ(retrieved, param.value);
}

INSTANTIATE_TEST_SUITE_P(
    ScalarAttributes,
    TestPointwiseOperationDescriptorOptionalFloat,
    ::testing::Values(
        OptionalFloatParam{HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                           HIPDNN_POINTWISE_RELU_FWD,
                           0.5f,
                           "ReluLowerClip"},
        OptionalFloatParam{HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                           HIPDNN_POINTWISE_RELU_FWD,
                           6.0f,
                           "ReluUpperClip"},
        OptionalFloatParam{HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                           HIPDNN_POINTWISE_RELU_FWD,
                           0.01f,
                           "ReluLowerClipSlope"},
        OptionalFloatParam{
            HIPDNN_ATTR_POINTWISE_SWISH_BETA, HIPDNN_POINTWISE_SWISH_FWD, 1.5f, "SwishBeta"},
        OptionalFloatParam{
            HIPDNN_ATTR_POINTWISE_ELU_ALPHA, HIPDNN_POINTWISE_ELU_FWD, 1.0f, "EluAlpha"},
        OptionalFloatParam{HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                           HIPDNN_POINTWISE_SOFTPLUS_FWD,
                           2.0f,
                           "SoftplusBeta"}),
    [](const ::testing::TestParamInfo<OptionalFloatParam>& info) {
        return std::string(info.param.name);
    });

// =============================================================================
// Optional Float Null elementCount Safety Tests (Issue 1)
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetOptionalFloatUnsetWithNullElementCount)
{
    makeUnaryFinalized();
    auto desc = getDescriptor();

    float retrieved = 0.0f;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_FLOAT, 1, nullptr, &retrieved));
}

TEST_F(TestPointwiseOperationDescriptor, GetOptionalFloatSetWithNullElementCount)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    auto operation = HIPDNN_POINTWISE_RELU_FWD;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto value = 0.5f;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_FLOAT, 1, &value);
    desc->finalize();

    float retrieved = 0.0f;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_FLOAT, 1, nullptr, &retrieved));
    ASSERT_FLOAT_EQ(retrieved, 0.5f);
}

// =============================================================================
// Attribute Type Validation Tests (Issue 2)
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetOptionalFloatRejectsWrongAttributeType)
{
    auto desc = getDescriptor();
    auto value = 1.0f;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_INT64, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, GetOptionalFloatRejectsWrongAttributeType)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    auto operation = HIPDNN_POINTWISE_RELU_FWD;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto value = 0.5f;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_FLOAT, 1, &value);
    desc->finalize();

    float retrieved = 0.0f;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_INT64, 1, &elementCount, &retrieved),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetOptionalFloatRejectsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, HIPDNN_TYPE_FLOAT, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Null Optional Tensor Descriptor Tests (Issue 3)
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributeIn1ReturnsZeroCountForUnaryOp)
{
    makeUnaryFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    HipdnnBackendDescriptor* retrieved = nullptr;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrieved)));
    ASSERT_EQ(elementCount, 0);
    ASSERT_EQ(retrieved, nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeIn2ReturnsZeroCountForUnaryOp)
{
    makeUnaryFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    HipdnnBackendDescriptor* retrieved = nullptr;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrieved)));
    ASSERT_EQ(elementCount, 0);
    ASSERT_EQ(retrieved, nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeIn1QueryReturnsZeroForUnaryOp)
{
    makeUnaryFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 0);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeNullTensorRejectsWrongType)
{
    makeUnaryFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, HIPDNN_TYPE_INT64, 1, &elementCount, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}
