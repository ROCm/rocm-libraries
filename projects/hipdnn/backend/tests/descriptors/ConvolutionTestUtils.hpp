// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_backend.h"

#include <cstdint>
#include <vector>

namespace hipdnn_backend::test_utilities
{

/// Sets the convolution parameters shared by the forward, backward-data and backward-filter
/// operation descriptors: padding, strides and dilations, each with its own element count, a
/// FLOAT compute type and cross-correlation mode. @p desc is any of the three descriptors.
template <typename DescriptorT>
void setConvolutionParameters(DescriptorT& desc,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& dilation)
{
    desc.setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                      HIPDNN_TYPE_INT64,
                      static_cast<int64_t>(prePadding.size()),
                      prePadding.data());
    desc.setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                      HIPDNN_TYPE_INT64,
                      static_cast<int64_t>(postPadding.size()),
                      postPadding.data());
    desc.setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                      HIPDNN_TYPE_INT64,
                      static_cast<int64_t>(stride.size()),
                      stride.data());
    desc.setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                      HIPDNN_TYPE_INT64,
                      static_cast<int64_t>(dilation.size()),
                      dilation.data());
    const auto computeType = HIPDNN_DATA_FLOAT;
    desc.setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    const auto convMode = HIPDNN_CROSS_CORRELATION;
    desc.setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
}

} // namespace hipdnn_backend::test_utilities
