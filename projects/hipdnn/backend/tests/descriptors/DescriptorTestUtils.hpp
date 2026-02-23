// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

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

inline std::unique_ptr<HipdnnBackendDescriptor> createFinalizedTensor(
    int64_t uid,
    std::vector<int64_t> dims = hipdnn_tests::toVec(hipdnn_tests::constants::K_TENSOR_X_DIMS),
    std::vector<int64_t> strides = hipdnn_tests::toVec(hipdnn_tests::constants::K_TENSOR_X_STRIDES),
    hipdnn_data_sdk::data_objects::DataType dataType
    = hipdnn_data_sdk::data_objects::DataType::FLOAT)
{
    auto wrapper = createDescriptor<TensorDescriptor>();
    auto desc = wrapper->asDescriptor<TensorDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(dims.size()),
                       dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(strides.size()),
                       strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    desc->finalize();

    return wrapper;
}

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedConvOp(HipdnnBackendDescriptor* xDesc,
                          HipdnnBackendDescriptor* wDesc,
                          HipdnnBackendDescriptor* yDesc,
                          hipdnn_data_sdk::data_objects::DataType computeType
                          = hipdnn_data_sdk::data_objects::DataType::FLOAT)
{
    auto wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc);

    auto padding = hipdnn_tests::toVec(hipdnn_tests::constants::K_CONV_PADDING);
    auto stride = hipdnn_tests::toVec(hipdnn_tests::constants::K_CONV_STRIDE);
    auto dilation = hipdnn_tests::toVec(hipdnn_tests::constants::K_CONV_DILATION);

    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, padding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto convMode
        = static_cast<int64_t>(hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION);
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_INT64, 1, &convMode);

    desc->finalize();
    return wrapper;
}

} // namespace hipdnn_backend::test_utilities
