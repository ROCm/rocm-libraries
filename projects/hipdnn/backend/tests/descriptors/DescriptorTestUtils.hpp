// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>

namespace hipdnn_backend::test_utilities
{

template <typename T>
HipdnnBackendDescriptor* createDescriptorPtr()
{
    return HipdnnBackendDescriptor::packDescriptor(std::make_shared<T>());
}

template <typename T>
std::unique_ptr<HipdnnBackendDescriptor> createDescriptor()
{
    return std::unique_ptr<HipdnnBackendDescriptor>(createDescriptorPtr<T>());
}

// Creates a finalized ConvolutionFwdOperationDescriptor using shared test constants
// from ConvFpropConstants.hpp (padding, stride, dilation) with CROSS_CORRELATION mode.
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedConvOp(HipdnnBackendDescriptor* xDesc,
                          HipdnnBackendDescriptor* wDesc,
                          HipdnnBackendDescriptor* yDesc,
                          hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&wDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));

    auto padding = hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_CONV_PADDING);
    auto stride = hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_CONV_STRIDE);
    auto dilation = hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_CONV_DILATION);

    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);

    desc->finalize();
    return wrapper;
}

} // namespace hipdnn_backend::test_utilities
