// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvolutionTestUtils.hpp"
#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionBwdOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvDgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestConvolutionBwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ConvolutionBwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ConvolutionBwdOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dyDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_wDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dxDesc);
    }

    void setConvolutionBwdParams() const
    {
        auto desc = getDescriptor();
        std::vector<int64_t> prePadding = {1, 1};
        std::vector<int64_t> postPadding = {1, 1};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};

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
        setConvolutionBwdParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto convMode = HIPDNN_CROSS_CORRELATION;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dyDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _wDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ConvolutionBwdOperationDescriptor>();
        _dyDesc = createFinalizedTensor(
            K_DGRAD_TENSOR_DY_UID, toVec(K_DGRAD_TENSOR_DY_DIMS), toVec(K_DGRAD_TENSOR_DY_STRIDES));
        _wDesc = createFinalizedTensor(
            K_DGRAD_TENSOR_W_UID, toVec(K_DGRAD_TENSOR_W_DIMS), toVec(K_DGRAD_TENSOR_W_STRIDES));
        _dxDesc = createFinalizedTensor(
            K_DGRAD_TENSOR_DX_UID, toVec(K_DGRAD_TENSOR_DX_DIMS), toVec(K_DGRAD_TENSOR_DX_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _dyDesc.reset();
        _wDesc.reset();
        _dxDesc.reset();
        _unfinalizedTensor.reset();
    }
};

namespace
{

// Tensor dims and convolution parameters of one dgrad descriptor. An empty padding vector means
// zeros and an empty dilation vector means ones, one entry per stride entry. The default
// initializers let a test name only the leading members without -Wmissing-field-initializers.
struct DgradShape
{
    // NOLINTBEGIN(readability-redundant-member-init) - the initializers are load-bearing, see above
    std::vector<int64_t> dy;
    std::vector<int64_t> w;
    std::vector<int64_t> dx;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation = {};
    std::vector<int64_t> prePadding = {};
    std::vector<int64_t> postPadding = {};
    // NOLINTEND(readability-redundant-member-init)
};

// Finalizes a dgrad descriptor built from packed tensors with the dims and parameters of @p shape.
void finalizeDgrad(DgradShape shape)
{
    using hipdnn_data_sdk::utilities::generateStrides;

    if(shape.dilation.empty())
    {
        shape.dilation.assign(shape.stride.size(), 1);
    }
    if(shape.prePadding.empty())
    {
        shape.prePadding.assign(shape.stride.size(), 0);
    }
    if(shape.postPadding.empty())
    {
        shape.postPadding.assign(shape.stride.size(), 0);
    }

    auto dyTensor
        = createFinalizedTensor(K_DGRAD_TENSOR_DY_UID, shape.dy, generateStrides(shape.dy));
    auto wTensor = createFinalizedTensor(K_DGRAD_TENSOR_W_UID, shape.w, generateStrides(shape.w));
    auto dxTensor
        = createFinalizedTensor(K_DGRAD_TENSOR_DX_UID, shape.dx, generateStrides(shape.dx));
    HipdnnBackendDescriptor* dyPtr = dyTensor.get();
    HipdnnBackendDescriptor* wPtr = wTensor.get();
    HipdnnBackendDescriptor* dxPtr = dxTensor.get();

    auto wrapper = createDescriptor<ConvolutionBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ConvolutionBwdOperationDescriptor>();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyPtr));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&wPtr));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxPtr));

    setConvolutionParameters(
        *desc, shape.prePadding, shape.postPadding, shape.stride, shape.dilation);

    desc->finalize();
}

// A consistent shape: (4 + 1 + 1 - 3) / 1 + 1 = 4 in both spatial dimensions.
DgradShape paddedBaselineShape()
{
    return {{1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 4, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
}

} // namespace

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutDyTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    setConvolutionBwdParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutWTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    setConvolutionBwdParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutDxTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc);
    setConvolutionBwdParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutPrePadding)
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

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutPostPadding)
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

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutStride)
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

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutDilation)
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

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setConvolutionBwdParams();
    auto convMode = HIPDNN_CROSS_CORRELATION;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithoutConvMode)
{
    setTensors();
    setConvolutionBwdParams();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Finalize Tests - Spatial Shape Consistency
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithHugeStrideInconsistentWithDims)
{
    // dy width must be (4 - 1) / stride + 1 = 1 for any stride > 3, not 4.
    ASSERT_THROW_HIPDNN_STATUS(
        finalizeDgrad({{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 4, 4, 4}, {1, 100000}}),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(
        finalizeDgrad({{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 4, 4, 4}, {1, 1048577}}),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeAcceptsUnitStrideOnSameShape)
{
    ASSERT_NO_THROW(finalizeDgrad({{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 4, 4, 4}, {1, 1}}));
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeAcceptsPaddedBaselineShape)
{
    ASSERT_NO_THROW(finalizeDgrad(paddedBaselineShape()));
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeAcceptsStrideThatFloorsTheOutput)
{
    // H: (6 - 1) / 2 + 1 = 3 (floored), W: (5 - 1) / 2 + 1 = 3 (exact).
    ASSERT_NO_THROW(finalizeDgrad({{1, 1, 3, 3}, {1, 1, 1, 1}, {1, 1, 6, 5}, {2, 2}}));
}

// Each test below changes one parameter so that only the check it names can reject the shape:
// with that check absent or loosened, the rest of the arithmetic accepts it.

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithZeroStride)
{
    // Without the stride bound, 0 / 0 is evaluated.
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 3, 3}, {0, 0}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithNegativeStride)
{
    // (3 - 3) / -1 + 1 = 1 matches dy, so only the stride bound rejects it.
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 3, 3}, {-1, -1}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWithZeroDilation)
{
    // Without the dilation bound, the multiply overflow guard divides by the dilation 0.
    ASSERT_THROW_HIPDNN_STATUS(
        finalizeDgrad({{1, 1, 4, 4}, {1, 1, 3, 3}, {1, 1, 4, 4}, {1, 1}, {0, 0}}),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenInputIsOneSmallerThanDilatedKernel)
{
    // numerator = 2 - 3 = -1, and -1 / 2 truncates to 0, giving the dy extent 1.
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 2, 2}, {2, 2}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenDilatedKernelMultiplyOverflows)
{
    // (5 - 1) * 2^62 = 2^64 wraps to 0, which gives a dilated kernel of 1 and a matching dy.
    const int64_t dilation = int64_t{1} << 62;
    ASSERT_THROW_HIPDNN_STATUS(
        finalizeDgrad({{1, 1, 4, 4}, {1, 1, 5, 5}, {1, 1, 4, 4}, {1, 1}, {dilation, dilation}}),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenPaddedInputAddOverflows)
{
    // 4 + INT64_MAX + INT64_MAX wraps to 2, and (2 - 1) / 1 + 1 = 2 matches dy.
    const int64_t maxPad = std::numeric_limits<int64_t>::max();
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 2, 2},
                                              {1, 1, 1, 1},
                                              {1, 1, 4, 4},
                                              {1, 1},
                                              {1, 1},
                                              {maxPad, maxPad},
                                              {maxPad, maxPad}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenWeightRankDiffers)
{
    // The spatial extents the check reads (indices 2 and 3) are consistent.
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 4, 4}, {1, 1, 1, 1, 7}, {1, 1, 4, 4}, {1, 1}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenInputRankDiffers)
{
    // The spatial extents the check reads (indices 2 and 3) are consistent.
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad({{1, 1, 4, 4}, {1, 1, 1, 1}, {1, 1, 4, 4, 7}, {1, 1}}),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenPrePaddingCountDiffers)
{
    auto shape = paddedBaselineShape();
    shape.prePadding = {1, 1, 0};
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad(shape), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenPostPaddingCountDiffers)
{
    auto shape = paddedBaselineShape();
    shape.postPadding = {1, 1, 0};
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad(shape), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenStrideCountDiffers)
{
    auto shape = paddedBaselineShape();
    shape.stride = {1, 1, 1};
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad(shape), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeFailsWhenDilationCountDiffers)
{
    auto shape = paddedBaselineShape();
    shape.dilation = {1, 1, 1};
    ASSERT_THROW_HIPDNN_STATUS(finalizeDgrad(shape), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizeRejectsRankTwoTensorsForTooFewDimensions)
{
    // A rank-2 shape is also rejected by the parameter-count checks, so the message tells
    // which check fired.
    try
    {
        finalizeDgrad({{4, 4}, {4, 4}, {4, 4}, {1}});
        FAIL() << "Finalizing rank-2 tensors did not throw";
    }
    catch(const HipdnnException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_STATUS_BAD_PARAM);
        EXPECT_NE(e.getMessage().find("at least 3 dimensions"), std::string::npos)
            << e.getMessage();
    }
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorDescriptorDy)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dyDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().dy_tensor_uid, K_DGRAD_TENSOR_DY_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorDescriptorW)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_wDesc));

    ASSERT_EQ(desc->getData().w_tensor_uid, K_DGRAD_TENSOR_W_UID);
    ASSERT_NE(desc->getWDesc(), nullptr);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorDescriptorDx)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dxDesc));

    ASSERT_EQ(desc->getData().dx_tensor_uid, K_DGRAD_TENSOR_DX_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, HIPDNN_TYPE_INT64, 1, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_dyDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, SetPrePadding)
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

TEST_F(TestConvolutionBwdOperationDescriptor, SetPostPadding)
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

TEST_F(TestConvolutionBwdOperationDescriptor, SetStride)
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

TEST_F(TestConvolutionBwdOperationDescriptor, SetDilation)
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

TEST_F(TestConvolutionBwdOperationDescriptor, SetConvMode)
{
    auto desc = getDescriptor();
    auto convMode = HIPDNN_CROSS_CORRELATION;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode));

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetConvModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto convMode = HIPDNN_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 2, &convMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetConvolutionBwdParamsWrongType)
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

TEST_F(TestConvolutionBwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_dyDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionBwdOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawDy = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawDy)));
    const std::unique_ptr<HipdnnBackendDescriptor> retrievedDy(rawDy);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedDy, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeConvolutionBwdParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &prePaddingCount,
                                       prePadding.data()));

    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, (std::vector<int64_t>{1, 1}));

    // post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       2,
                                       &postPaddingCount,
                                       postPadding.data()));
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, (std::vector<int64_t>{1, 1}));

    // stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data()));
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, (std::vector<int64_t>{1, 1}));

    // dilation
    std::vector<int64_t> dilation(2);
    int64_t dilationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilation.data()));
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilation, (std::vector<int64_t>{1, 1}));

    // conv mode
    hipdnnConvolutionMode_t convMode = HIPDNN_CONVOLUTION;
    int64_t convModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       HIPDNN_TYPE_CONVOLUTION_MODE,
                                       1,
                                       &convModeCount,
                                       &convMode));
    ASSERT_EQ(convModeCount, 1);
    EXPECT_EQ(convMode, HIPDNN_CROSS_CORRELATION);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeTensorDyQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeTensorWQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeTensorDxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeConvModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       HIPDNN_TYPE_CONVOLUTION_MODE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributePrePaddingQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributePrePaddingQueryThenRetrieve)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Query: get the element count
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 2);

    // Retrieve: use the queried count to allocate and fetch
    std::vector<int64_t> prePadding(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       HIPDNN_TYPE_INT64,
                                       elementCount,
                                       &retrievedCount,
                                       prePadding.data()));
    ASSERT_EQ(retrievedCount, 2);
    EXPECT_EQ(prePadding, (std::vector<int64_t>{1, 1}));
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetAttributeConvModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getWDesc(), nullptr);
    ASSERT_NE(desc->getDxDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDyDesc()->getData().uid, K_DGRAD_TENSOR_DY_UID);
    ASSERT_EQ(desc->getWDesc()->getData().uid, K_DGRAD_TENSOR_W_UID);
    ASSERT_EQ(desc->getDxDesc()->getData().uid, K_DGRAD_TENSOR_DX_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("ConvolutionBwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("dy_uid=" + std::to_string(K_DGRAD_TENSOR_DY_UID)), std::string::npos);
    ASSERT_NE(str.find("w_uid=" + std::to_string(K_DGRAD_TENSOR_W_UID)), std::string::npos);
    ASSERT_NE(str.find("dx_uid=" + std::to_string(K_DGRAD_TENSOR_DX_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestConvolutionBwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_DGRAD_TENSOR_DY_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_DGRAD_TENSOR_W_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_DGRAD_TENSOR_DX_UID);
}

TEST_F(TestConvolutionBwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::ConvolutionBwdAttributes);

    auto* attrs = node->attributes.AsConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->dy_tensor_uid, K_DGRAD_TENSOR_DY_UID);
    ASSERT_EQ(attrs->w_tensor_uid, K_DGRAD_TENSOR_W_UID);
    ASSERT_EQ(attrs->dx_tensor_uid, K_DGRAD_TENSOR_DX_UID);
    ASSERT_EQ(attrs->pre_padding.size(), 2);
    ASSERT_EQ(attrs->post_padding.size(), 2);
    ASSERT_EQ(attrs->stride.size(), 2);
    ASSERT_EQ(attrs->dilation.size(), 2);
}

TEST_F(TestConvolutionBwdOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestConvolutionBwdOperationDescriptor, GetTensorDescriptorsOrderIsDyWDx)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [DY, W, DX] matches UIDs [10, 11, 12]
    EXPECT_EQ(tensors[0], desc->getDyDesc());
    EXPECT_EQ(tensors[1], desc->getWDesc());
    EXPECT_EQ(tensors[2], desc->getDxDesc());
}

TEST_F(TestConvolutionBwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_DGRAD_TENSOR_DY_UID);
}

TEST_F(TestConvolutionBwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _dyDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
