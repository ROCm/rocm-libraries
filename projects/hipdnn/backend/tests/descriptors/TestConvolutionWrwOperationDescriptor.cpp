// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionWrwOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/constants/ConvWrwConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestConvolutionWrwOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ConvolutionWrwOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ConvolutionWrwOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DW, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dwDesc);
    }

    void setConvolutionWrwParams() const
    {
        auto desc = getDescriptor();
        auto prePadding = toVec(K_CONV_PADDING);
        auto postPadding = toVec(K_CONV_PADDING);
        auto stride = toVec(K_CONV_STRIDE);
        auto dilation = toVec(K_CONV_DILATION);

        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setConvolutionWrwParams();
        auto computeType = DataType::FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dyDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dwDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ConvolutionWrwOperationDescriptor>();
        _xDesc = createFinalizedTensor(
            K_WRW_TENSOR_X_UID, toVec(K_WRW_TENSOR_X_DIMS), toVec(K_WRW_TENSOR_X_STRIDES));
        _dyDesc = createFinalizedTensor(
            K_WRW_TENSOR_DY_UID, toVec(K_WRW_TENSOR_DY_DIMS), toVec(K_WRW_TENSOR_DY_STRIDES));
        _dwDesc = createFinalizedTensor(
            K_WRW_TENSOR_DW_UID, toVec(K_WRW_TENSOR_DW_DIMS), toVec(K_WRW_TENSOR_DW_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _dyDesc.reset();
        _dwDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_WRW_DESCRIPTOR);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DW, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dwDesc);
    setConvolutionWrwParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutDYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DW, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dwDesc);
    setConvolutionWrwParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutDWTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    setConvolutionWrwParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutPrePadding)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutPostPadding)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutStride)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> dilation = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutDilation)
{
    auto desc = getDescriptor();
    setTensors();
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {1, 1};

    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setConvolutionWrwParams();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_WRW_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorDescriptorDY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc));

    ASSERT_EQ(desc->getData().dy_tensor_uid, K_WRW_TENSOR_DY_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorDescriptorDW)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_DW, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dwDesc));

    ASSERT_EQ(desc->getData().dw_tensor_uid, K_WRW_TENSOR_DW_UID);
    ASSERT_NE(desc->getDwDesc(), nullptr);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, SetPrePadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> prePadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.pre_padding.size(), 2);
    ASSERT_EQ(data.pre_padding[0], 1);
    ASSERT_EQ(data.pre_padding[1], 1);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetPostPadding)
{
    auto desc = getDescriptor();
    std::vector<int64_t> postPadding = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.post_padding.size(), 2);
    ASSERT_EQ(data.post_padding[0], 1);
    ASSERT_EQ(data.post_padding[1], 1);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetStride)
{
    auto desc = getDescriptor();
    std::vector<int64_t> stride = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.stride.size(), 2);
    ASSERT_EQ(data.stride[0], 1);
    ASSERT_EQ(data.stride[1], 1);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetDilation)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dilation = {1, 1};

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.dilation.size(), 2);
    ASSERT_EQ(data.dilation[0], 1);
    ASSERT_EQ(data.dilation[1], 1);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetConvMode)
{
    auto desc = getDescriptor();
    auto convMode = static_cast<int64_t>(ConvMode::CROSS_CORRELATION);

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 1, &convMode));

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetConvModeWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t convMode = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 2, &convMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetConvolutionwrwParamsWrongType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> padding = {1, 1};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_CHAR, 2, padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionWrwOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       &retrievedX));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeConvolutionwrwParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &prePaddingCount,
                                       prePadding.data()));
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, (std::vector<int64_t>{1, 1}));

    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &postPaddingCount,
                                       postPadding.data()));
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, (std::vector<int64_t>{1, 1}));

    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data()));
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, (std::vector<int64_t>{1, 1}));

    std::vector<int64_t> dilation(2);
    int64_t dilationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilation.data()));
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilation, (std::vector<int64_t>{1, 1}));

    // conv_mode
    int64_t convModeVal = -1;
    int64_t convModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 1, &convModeCount, &convModeVal));
    ASSERT_EQ(convModeCount, 1);
    EXPECT_EQ(convModeVal, static_cast<int64_t>(ConvMode::UNSET));
}

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = DataType::HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    DataType retrieved = DataType::UNSET;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, DataType::HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW_X,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestConvolutionWrwOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getDwDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_WRW_TENSOR_X_UID);
    ASSERT_EQ(desc->getDyDesc()->getData().uid, K_WRW_TENSOR_DY_UID);
    ASSERT_EQ(desc->getDwDesc()->getData().uid, K_WRW_TENSOR_DW_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("ConvolutionWrwOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_WRW_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("dy_uid=" + std::to_string(K_WRW_TENSOR_DY_UID)), std::string::npos);
    ASSERT_NE(str.find("dw_uid=" + std::to_string(K_WRW_TENSOR_DW_UID)), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestConvolutionWrwOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_WRW_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_WRW_TENSOR_DY_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_WRW_TENSOR_DW_UID);
}

TEST_F(TestConvolutionWrwOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = DataType::FLOAT;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ConvolutionWrwAttributes);

    auto* attrs = node->attributes.AsConvolutionWrwAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_WRW_TENSOR_X_UID);
    ASSERT_EQ(attrs->dy_tensor_uid, K_WRW_TENSOR_DY_UID);
    ASSERT_EQ(attrs->dw_tensor_uid, K_WRW_TENSOR_DW_UID);
    ASSERT_EQ(attrs->pre_padding.size(), 2);
    ASSERT_EQ(attrs->post_padding.size(), 2);
    ASSERT_EQ(attrs->stride.size(), 2);
    ASSERT_EQ(attrs->dilation.size(), 2);
}

TEST_F(TestConvolutionWrwOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = DataType::HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestConvolutionWrwOperationDescriptor, GetTensorDescriptorsOrderIsXDYDW)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [X, DY, DW] matches UIDs [20, 21, 22]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getDyDesc());
    EXPECT_EQ(tensors[2], desc->getDwDesc());
}

TEST_F(TestConvolutionWrwOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, 20);
}

TEST_F(TestConvolutionWrwOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsInterface<IGraphOperation>();
    EXPECT_EQ(graphOp, nullptr);
}
