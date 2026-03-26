// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/PoolingFwdOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestPoolingFwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<PoolingFwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<PoolingFwdOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_yDesc);
    }

    void setPoolingParams() const
    {
        auto desc = getDescriptor();
        auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
        auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
        auto stride = toVec(K_POOL_FWD_STRIDE);
        auto windowSize = toVec(K_POOL_FWD_WINDOW_SIZE);

        desc->setAttribute(
            HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
        desc->setAttribute(
            HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data());
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setPoolingParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto poolingMode = HIPDNN_POOLING_MODE_MAX;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode);
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
        _wrapper = createDescriptor<PoolingFwdOperationDescriptor>();
        _xDesc = createFinalizedTensor(
            K_POOL_FWD_TENSOR_X_UID, toVec(K_POOL_FWD_TENSOR_X_DIMS), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
        _yDesc = createFinalizedTensor(
            K_POOL_FWD_TENSOR_Y_UID, toVec(K_POOL_FWD_TENSOR_Y_DIMS), toVec(K_POOL_FWD_TENSOR_Y_STRIDES));
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

TEST_F(TestPoolingFwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setPoolingParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    setPoolingParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutPrePadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
    auto stride = toVec(K_POOL_FWD_STRIDE);
    auto windowSize = toVec(K_POOL_FWD_WINDOW_SIZE);

    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutPostPadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
    auto stride = toVec(K_POOL_FWD_STRIDE);
    auto windowSize = toVec(K_POOL_FWD_WINDOW_SIZE);

    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutStride)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
    auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
    auto windowSize = toVec(K_POOL_FWD_WINDOW_SIZE);

    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutWindowSize)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
    auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
    auto stride = toVec(K_POOL_FWD_STRIDE);

    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setPoolingParams();
    auto poolingMode = HIPDNN_POOLING_MODE_MAX;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, FinalizeFailsWithoutPoolingMode)
{
    setTensors();
    setPoolingParams();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_POOL_FWD_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_POOL_FWD_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_xDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Pooling Parameters
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingPrePadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> prePadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.pre_padding.size(), 2);
    ASSERT_EQ(data.pre_padding[0], 1);
    ASSERT_EQ(data.pre_padding[1], 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingPostPadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> postPadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.post_padding.size(), 2);
    ASSERT_EQ(data.post_padding[0], 1);
    ASSERT_EQ(data.post_padding[1], 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingStride)
{
    auto desc = getDescriptor();
    std::vector<int64_t> stride = {2, 2};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.stride.size(), 2);
    ASSERT_EQ(data.stride[0], 2);
    ASSERT_EQ(data.stride[1], 2);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingWindowSize)
{
    auto desc = getDescriptor();
    std::vector<int64_t> windowSize = {3, 3};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.window_size.size(), 2);
    ASSERT_EQ(data.window_size[0], 3);
    ASSERT_EQ(data.window_size[1], 3);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingMode)
{
    auto desc = getDescriptor();
    auto poolingMode = HIPDNN_POOLING_MODE_MAX;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode));

    ASSERT_EQ(desc->getData().pooling_mode, PoolingMode::MAX);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto poolingMode = HIPDNN_POOLING_MODE_MAX;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 2, &poolingMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetPoolingParamsWrongType)
{
    auto desc = getDescriptor();
    auto padding = toVec(K_POOL_FWD_PRE_PADDING);

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_CHAR, 2, padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_xDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPoolingFwdOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardX(retrievedX);
}

// =============================================================================
// GetAttribute Tests - Pooling Parameters
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributePoolingParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, &prePaddingCount, prePadding.data()));

    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, toVec(K_POOL_FWD_PRE_PADDING));


    // post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, &postPaddingCount, postPadding.data()));
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_POOL_FWD_POST_PADDING));


    // stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data()));
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_POOL_FWD_STRIDE));


    // window_size
    std::vector<int64_t> windowSize(2);
    int64_t windowSizeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, &windowSizeCount, windowSize.data()));
    ASSERT_EQ(windowSizeCount, 2);
    EXPECT_EQ(windowSize, toVec(K_POOL_FWD_WINDOW_SIZE));

    // pooling mode
    hipdnnPoolingMode_t poolingMode = HIPDNN_POOLING_MODE_AVERAGE;
    int64_t poolingModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingModeCount, &poolingMode));
    ASSERT_EQ(poolingModeCount, 1);
    EXPECT_EQ(poolingMode, HIPDNN_POOLING_MODE_MAX);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributePoolingModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributePrePaddingQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributePrePaddingQueryThenRetrieve)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Query: get the element count
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);

    // Retrieve: use the queried count to allocate and fetch
    std::vector<int64_t> prePadding(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       elementCount,
                                       &retrievedCount,
                                       prePadding.data()));
    ASSERT_EQ(retrievedCount, 2);
    EXPECT_EQ(prePadding, toVec(K_POOL_FWD_PRE_PADDING));
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributePoolingModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_POOL_FWD_TENSOR_X_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_POOL_FWD_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("PoolingFwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_POOL_FWD_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_POOL_FWD_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    ASSERT_EQ(tensors[0]->getData().uid, K_POOL_FWD_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_POOL_FWD_TENSOR_Y_UID);
}

TEST_F(TestPoolingFwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::PoolingFwdAttributes);

    auto* poolAttrs = node->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(poolAttrs, nullptr);
    ASSERT_EQ(poolAttrs->x_tensor_uid, K_POOL_FWD_TENSOR_X_UID);
    ASSERT_EQ(poolAttrs->y_tensor_uid, K_POOL_FWD_TENSOR_Y_UID);
    EXPECT_EQ(poolAttrs->pre_padding, toVec(K_POOL_FWD_PRE_PADDING));
    EXPECT_EQ(poolAttrs->stride, toVec(K_POOL_FWD_STRIDE));
    EXPECT_EQ(poolAttrs->window_size, toVec(K_POOL_FWD_WINDOW_SIZE));
    EXPECT_EQ(poolAttrs->pooling_mode, PoolingMode::MAX);
}

TEST_F(TestPoolingFwdOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetTensorDescriptorsOrderIsXY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    // Verify ordering: [X, Y] matches UIDs [40, 41]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getYDesc());
}

TEST_F(TestPoolingFwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    ASSERT_EQ(tensors[0]->getData().uid, K_POOL_FWD_TENSOR_X_UID);
}

TEST_F(TestPoolingFwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestPoolingFwdOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_poolingfwd_op";

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
    EXPECT_STREQ(buffer.data(), "test_poolingfwd_op");
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_POOLING_FORWARD);
}

TEST_F(TestPoolingFwdOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPoolingFwdOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_poolingfwd";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_poolingfwd");
}
