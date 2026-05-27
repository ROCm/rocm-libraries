// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/ResampleBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ResampleBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestResampleBwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ResampleBwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ResampleBwdOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_indexDesc);
    }

    void setResampleParams() const
    {
        auto desc = getDescriptor();
        auto prePadding = toVec(K_PRE_PADDING);
        auto postPadding = toVec(K_POST_PADDING);
        auto stride = toVec(K_STRIDE);
        auto window = toVec(K_WINDOW);

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
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);
        auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dyDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _indexDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ResampleBwdOperationDescriptor>();
        _dyDesc = createFinalizedTensor(
            K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
        _dxDesc = createFinalizedTensor(
            K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
        _indexDesc = createFinalizedTensor(
            K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _dyDesc.reset();
        _dxDesc.reset();
        _indexDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeSucceedsWithoutOptionalTensors)
{
    auto desc = getDescriptor();
    // Set only DY and DX (no Index)
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
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

TEST_F(TestResampleBwdOperationDescriptor, GetTensorDescriptorsWithoutOptional)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
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
    EXPECT_EQ(tensors[0]->getData().uid, K_TENSOR_DY_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_TENSOR_DX_UID);
}

TEST_F(TestResampleBwdOperationDescriptor, BuildNodeWithoutOptionalTensors)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
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

    auto* attrs = node->attributes.AsResampleBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_TENSOR_DX_UID);
    EXPECT_FALSE(attrs->index_tensor_uid.has_value());
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutDyTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_indexDesc);
    setResampleParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutDxTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_indexDesc);
    setResampleParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutPrePadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto postPadding = toVec(K_POST_PADDING);
    auto stride = toVec(K_STRIDE);
    auto window = toVec(K_WINDOW);

    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutPostPadding)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_PRE_PADDING);
    auto stride = toVec(K_STRIDE);
    auto window = toVec(K_WINDOW);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutStride)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_PRE_PADDING);
    auto postPadding = toVec(K_POST_PADDING);
    auto window = toVec(K_WINDOW);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, window.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutWindow)
{
    auto desc = getDescriptor();
    setTensors();
    auto prePadding = toVec(K_PRE_PADDING);
    auto postPadding = toVec(K_POST_PADDING);
    auto stride = toVec(K_STRIDE);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutComputeType)
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

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutResampleMode)
{
    setTensors();
    setResampleParams();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, FinalizeFailsWithoutPaddingMode)
{
    setTensors();
    setResampleParams();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, SetTensorDescriptorDy)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().dy_tensor_uid, K_TENSOR_DY_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorDescriptorDx)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc));

    ASSERT_EQ(desc->getData().dx_tensor_uid, K_TENSOR_DX_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorDescriptorIndex)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_indexDesc));

    ASSERT_EQ(desc->getData().index_tensor_uid, K_TENSOR_INDEX_UID);
    ASSERT_NE(desc->getIndexDesc(), nullptr);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_INT64, 1, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Resample Parameters
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, SetResamplePrePadding)
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

TEST_F(TestResampleBwdOperationDescriptor, SetResamplePostPadding)
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

TEST_F(TestResampleBwdOperationDescriptor, SetResampleStride)
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

TEST_F(TestResampleBwdOperationDescriptor, SetResampleWindow)
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

TEST_F(TestResampleBwdOperationDescriptor, SetResampleMode)
{
    auto desc = getDescriptor();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode));

    ASSERT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
}

TEST_F(TestResampleBwdOperationDescriptor, SetResampleModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 2, &resampleMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, SetPaddingMode)
{
    auto desc = getDescriptor();
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode));

    ASSERT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleBwdOperationDescriptor, SetPaddingModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 2, &paddingMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestResampleBwdOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestResampleBwdOperationDescriptor, SetGenerateIndex)
{
    auto desc = getDescriptor();
    bool generateIndex = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT, HIPDNN_TYPE_BOOLEAN, 1, &generateIndex));

    ASSERT_TRUE(desc->getData().generate_index.has_value());
    EXPECT_EQ(desc->getData().generate_index.value(), true);
}

TEST_F(TestResampleBwdOperationDescriptor, SetResampleParamsWrongType)
{
    auto desc = getDescriptor();
    auto padding = toVec(K_PRE_PADDING);

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_CHAR, 2, padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestResampleBwdOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedDy = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedDy)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedDy, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardDy(retrievedDy);
}

// =============================================================================
// GetAttribute Tests - Resample Parameters
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeResampleParams)
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
    EXPECT_EQ(prePadding, toVec(K_PRE_PADDING));

    // post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &postPaddingCount,
                                       postPadding.data()));
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_POST_PADDING));

    // stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data()));
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_STRIDE));

    // window
    std::vector<int64_t> window(2);
    int64_t windowCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, &windowCount, window.data()));
    ASSERT_EQ(windowCount, 2);
    EXPECT_EQ(window, toVec(K_WINDOW));

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

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeGenerateIndex)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    bool generateIndex = true;
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT, HIPDNN_TYPE_BOOLEAN, 1, &generateIndex);
    desc->finalize();

    bool retrieved = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &retrieved));

    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(retrieved, true);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeGenerateIndexUnsetQueryReturnsZero)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = -1;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT, HIPDNN_TYPE_BOOLEAN, 0, &elementCount, nullptr));

    EXPECT_EQ(elementCount, 0);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeTensorDyQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeTensorDxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeTensorIndexQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeResampleModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributePaddingModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributePrePaddingQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributePrePaddingQueryThenRetrieve)
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
    EXPECT_EQ(prePadding, toVec(K_PRE_PADDING));
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeResampleModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributePaddingModeQueryFailsNullElementCount)
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

TEST_F(TestResampleBwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    ASSERT_NE(desc->getIndexDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDyDesc()->getData().uid, K_TENSOR_DY_UID);
    ASSERT_EQ(desc->getDxDesc()->getData().uid, K_TENSOR_DX_UID);
    ASSERT_EQ(desc->getIndexDesc()->getData().uid, K_TENSOR_INDEX_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("ResampleBwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("dy_uid=" + std::to_string(K_TENSOR_DY_UID)), std::string::npos);
    ASSERT_NE(str.find("dx_uid=" + std::to_string(K_TENSOR_DX_UID)), std::string::npos);
    ASSERT_NE(str.find("index_uid=" + std::to_string(K_TENSOR_INDEX_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TENSOR_DY_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_TENSOR_DX_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_TENSOR_INDEX_UID);
}

TEST_F(TestResampleBwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ResampleBwdAttributes);

    auto* poolAttrs = node->attributes.AsResampleBwdAttributes();
    ASSERT_NE(poolAttrs, nullptr);
    ASSERT_EQ(poolAttrs->dy_tensor_uid, K_TENSOR_DY_UID);
    ASSERT_EQ(poolAttrs->dx_tensor_uid, K_TENSOR_DX_UID);
    ASSERT_EQ(poolAttrs->index_tensor_uid, K_TENSOR_INDEX_UID);
    EXPECT_EQ(poolAttrs->pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(poolAttrs->stride, toVec(K_STRIDE));
    EXPECT_EQ(poolAttrs->window, toVec(K_WINDOW));
    EXPECT_EQ(poolAttrs->resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(poolAttrs->padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleBwdOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestResampleBwdOperationDescriptor, GetTensorDescriptorsOrderIsDyDxIndex)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [DY, DX, INDEX] matches UIDs [50, 51, 52]
    EXPECT_EQ(tensors[0], desc->getDyDesc());
    EXPECT_EQ(tensors[1], desc->getDxDesc());
    EXPECT_EQ(tensors[2], desc->getIndexDesc());
}

TEST_F(TestResampleBwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TENSOR_DY_UID);
}

TEST_F(TestResampleBwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _dyDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestResampleBwdOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_resamplebwd_op";

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
    EXPECT_STREQ(buffer.data(), "test_resamplebwd_op");
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RESAMPLE_BWD);
}

TEST_F(TestResampleBwdOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestResampleBwdOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_resamplebwd";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_resamplebwd");
}
