// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/ResampleFwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ResampleFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestResampleFwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ResampleFwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ResampleFwdOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_indexDesc);
    }

    void setResampleParams() const
    {
        auto desc = getDescriptor();
        auto prePadding = toVec(K_RESAMPLE_FWD_PRE_PADDING);
        auto postPadding = toVec(K_RESAMPLE_FWD_POST_PADDING);
        auto stride = toVec(K_RESAMPLE_FWD_STRIDE);
        auto window = toVec(K_RESAMPLE_FWD_WINDOW);

        desc->setAttribute(
            HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setResampleParams();
        auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
        auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
        auto compType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType);
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
    std::unique_ptr<HipdnnBackendDescriptor> _indexDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ResampleFwdOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_RESAMPLE_FWD_TENSOR_X_UID,
                                       toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS),
                                       toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));
        _yDesc = createFinalizedTensor(K_RESAMPLE_FWD_TENSOR_Y_UID,
                                       toVec(K_RESAMPLE_FWD_TENSOR_Y_DIMS),
                                       toVec(K_RESAMPLE_FWD_TENSOR_Y_STRIDES));
        _indexDesc = createFinalizedTensor(K_RESAMPLE_FWD_TENSOR_INDEX_UID,
                                           toVec(K_RESAMPLE_FWD_TENSOR_INDEX_DIMS),
                                           toVec(K_RESAMPLE_FWD_TENSOR_INDEX_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _yDesc.reset();
        _indexDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeSucceedsWithoutOptionalTensors)
{
    auto desc = getDescriptor();
    // Set only X and Y (no Index)
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setResampleParams();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    auto compType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType);
    ASSERT_NO_THROW(desc->finalize());
}

TEST_F(TestResampleFwdOperationDescriptor, GetTensorDescriptorsWithoutOptional)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setResampleParams();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    auto compType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType);
    desc->finalize();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0]->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
}

TEST_F(TestResampleFwdOperationDescriptor, BuildNodeWithoutOptionalTensors)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    setResampleParams();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    auto compType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);

    auto* attrs = node->attributes.AsResampleFwdAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_FALSE(attrs->index_tensor_uid.has_value());
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_indexDesc);
    setResampleParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_indexDesc);
    setResampleParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutPrePadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto postPadding = toVec(K_RESAMPLE_FWD_POST_PADDING);
    auto stride = toVec(K_RESAMPLE_FWD_STRIDE);
    auto window = toVec(K_RESAMPLE_FWD_WINDOW);

    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutPostPadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_RESAMPLE_FWD_PRE_PADDING);
    auto stride = toVec(K_RESAMPLE_FWD_STRIDE);
    auto window = toVec(K_RESAMPLE_FWD_WINDOW);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutStride)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_RESAMPLE_FWD_PRE_PADDING);
    auto postPadding = toVec(K_RESAMPLE_FWD_POST_PADDING);
    auto window = toVec(K_RESAMPLE_FWD_WINDOW);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutWindow)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_RESAMPLE_FWD_PRE_PADDING);
    auto postPadding = toVec(K_RESAMPLE_FWD_POST_PADDING);
    auto stride = toVec(K_RESAMPLE_FWD_STRIDE);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutResampleMode)
{
    setTensors();
    setResampleParams();
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutPaddingMode)
{
    setTensors();
    setResampleParams();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorDescriptorIndex)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_indexDesc));

    ASSERT_EQ(desc->getData().index_tensor_uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    ASSERT_NE(desc->getIndexDesc(), nullptr);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Resample Parameters
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, SetResamplePrePadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> prePadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.pre_padding.size(), 2);
    ASSERT_EQ(data.pre_padding[0], 1);
    ASSERT_EQ(data.pre_padding[1], 1);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResamplePostPadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> postPadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.post_padding.size(), 2);
    ASSERT_EQ(data.post_padding[0], 1);
    ASSERT_EQ(data.post_padding[1], 1);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResampleStride)
{
    auto desc = getDescriptor();
    std::vector<int64_t> stride = {2, 2};

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.stride.size(), 2);
    ASSERT_EQ(data.stride[0], 2);
    ASSERT_EQ(data.stride[1], 2);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResampleWindow)
{
    auto desc = getDescriptor();
    std::vector<int64_t> window = {3, 3};

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.window.size(), 2);
    ASSERT_EQ(data.window[0], 3);
    ASSERT_EQ(data.window[1], 3);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResampleMode)
{
    auto desc = getDescriptor();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode));

    ASSERT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResampleModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 2, &resampleMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, SetPaddingMode)
{
    auto desc = getDescriptor();
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode));

    ASSERT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleFwdOperationDescriptor, SetPaddingModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 2, &paddingMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, SetResampleParamsWrongType)
{
    auto desc = getDescriptor();
    auto padding = toVec(K_RESAMPLE_FWD_PRE_PADDING);

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_CHAR, 2, padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestResampleFwdOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardX(retrievedX);
}

// =============================================================================
// GetAttribute Tests - Resample Parameters
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeResampleParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &prePaddingCount,
                                       prePadding.data()));

    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, toVec(K_RESAMPLE_FWD_PRE_PADDING));

    // post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &postPaddingCount,
                                       postPadding.data()));
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_RESAMPLE_FWD_POST_PADDING));

    // stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data()));
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_RESAMPLE_FWD_STRIDE));

    // window
    std::vector<int64_t> window(2);
    int64_t windowCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, &windowCount, window.data()));
    ASSERT_EQ(windowCount, 2);
    EXPECT_EQ(window, toVec(K_RESAMPLE_FWD_WINDOW));

    // resample mode
    hipdnnResampleMode_t resampleMode = HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
    int64_t resampleModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_MODE,
                                       HIPDNN_TYPE_RESAMPLE_MODE,
                                       1,
                                       &resampleModeCount,
                                       &resampleMode));
    ASSERT_EQ(resampleModeCount, 1);
    EXPECT_EQ(resampleMode, HIPDNN_RESAMPLE_MAXPOOL);

    // padding mode
    hipdnnPaddingMode_t paddingMode = HIPDNN_PADDING_NEG_INF_PAD;
    int64_t paddingModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PADDING_MODE,
                                       HIPDNN_TYPE_PADDING_MODE,
                                       1,
                                       &paddingModeCount,
                                       &paddingMode));
    ASSERT_EQ(paddingModeCount, 1);
    EXPECT_EQ(paddingMode, HIPDNN_PADDING_ZERO_PAD);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr; // NOLINT(misc-const-correctness)
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeTensorIndexQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeResampleModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributePaddingModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributePrePaddingQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributePrePaddingQueryThenRetrieve)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Query: get the element count
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);

    // Retrieve: use the queried count to allocate and fetch
    std::vector<int64_t> prePadding(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       elementCount,
                                       &retrievedCount,
                                       prePadding.data()));
    ASSERT_EQ(retrievedCount, 2);
    EXPECT_EQ(prePadding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeResampleModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributePaddingModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    ASSERT_NE(desc->getIndexDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    ASSERT_EQ(desc->getIndexDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("ResampleFwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_RESAMPLE_FWD_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_RESAMPLE_FWD_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("index_uid=" + std::to_string(K_RESAMPLE_FWD_TENSOR_INDEX_UID)),
              std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
}

TEST_F(TestResampleFwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ResampleFwdAttributes);

    auto* poolAttrs = node->attributes.AsResampleFwdAttributes();
    ASSERT_NE(poolAttrs, nullptr);
    ASSERT_EQ(poolAttrs->x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    ASSERT_EQ(poolAttrs->y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    ASSERT_EQ(poolAttrs->index_tensor_uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    EXPECT_EQ(poolAttrs->pre_padding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
    EXPECT_EQ(poolAttrs->stride, toVec(K_RESAMPLE_FWD_STRIDE));
    EXPECT_EQ(poolAttrs->window, toVec(K_RESAMPLE_FWD_WINDOW));
    EXPECT_EQ(poolAttrs->resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(poolAttrs->padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleFwdOperationDescriptor, GetTensorDescriptorsOrderIsXYIndex)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [X, Y, INDEX] matches UIDs [40, 41, 42]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getYDesc());
    EXPECT_EQ(tensors[2], desc->getIndexDesc());
}

TEST_F(TestResampleFwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
}

TEST_F(TestResampleFwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_resamplefwd_op";

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
    EXPECT_STREQ(buffer.data(), "test_resamplefwd_op");
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RESAMPLE_FWD);
}

TEST_F(TestResampleFwdOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_resamplefwd";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_resamplefwd");
}

// =============================================================================
// Compute Data Type Tests
// =============================================================================

TEST_F(TestResampleFwdOperationDescriptor, FinalizeFailsWithoutCompType)
{
    setTensors();
    setResampleParams();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, SetCompType)
{
    auto desc = getDescriptor();
    auto compType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestResampleFwdOperationDescriptor, SetCompTypeWrongAttributeType)
{
    auto desc = getDescriptor();
    auto compType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_INT64, 1, &compType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleFwdOperationDescriptor, GetCompType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnDataType_t compType = HIPDNN_DATA_HALF;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &compType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(compType, HIPDNN_DATA_FLOAT);
}

TEST_F(TestResampleFwdOperationDescriptor, GetCompTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleFwdOperationDescriptor, BuildNodePreservesCompType)
{
    setRequiredAttributes();
    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->compute_data_type, DataType::FLOAT);
}
