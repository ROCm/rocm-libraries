// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include <miopen/datatype.hpp>
#include <miopen/softmax/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>
#include <string_view>

namespace miopen {

namespace softmax {

size_t GetStride(const TensorDescriptor& desc, miopenSoftmaxMode_t mode)
{
    size_t stride = 1;
    auto layout   = desc.GetLayoutEnum();
    auto lengths  = desc.GetLengths();
    if(layout.has_value() && layout.value() == miopenTensorNCHW &&
       mode == MIOPEN_SOFTMAX_MODE_CHANNEL)
    {
        stride = lengths[2] * lengths[3]; // stride = H * W
    }
    return stride;
}

size_t GetOuterSize(const TensorDescriptor& desc, miopenSoftmaxMode_t mode)
{
    auto lengths      = desc.GetLengths();
    size_t outer_size = lengths[0]; // outer_size = N
    auto layout       = desc.GetLayoutEnum();
    if(layout.has_value() && layout.value() == miopenTensorNHWC &&
       mode == MIOPEN_SOFTMAX_MODE_CHANNEL)
    {
        outer_size *= lengths[2] * lengths[3]; // outer_size = N * H * W
    }
    return outer_size;
}

size_t GetInnerSize(const TensorDescriptor& desc, miopenSoftmaxMode_t mode)
{
    auto lengths      = desc.GetLengths();
    size_t inner_size = lengths[1]; // inner_size = C
    if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
    {
        inner_size *= lengths[2] * lengths[3]; // inner_size = C * H * W
    }
    return inner_size;
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss(isForward ? "sfmfwd-" : "sfmbwd-");

    // all the tensors must be the same size and types
    // so we can use only one set of values
    const auto& desc = isForward ? xdxDesc : yDesc;
    ss << "forward" << isForward;
    ss << "outer_size" << outer_size << "inner_size" << inner_size << "stride" << stride;
    ss << GetDataType(desc.GetType());
    ss << "a" << alpha;
    ss << "b" << beta;
    ss << "algo" << static_cast<int>(algorithm);
    ss << "mode" << static_cast<int>(mode);
    ss << "x_offset" << x_offset;
    ss << "y_offset" << y_offset;
    ss << "dx_offset" << dx_offset;
    ss << "dy_offset" << dy_offset;

    return NetworkConfig{ss.str()};
}

} // namespace softmax

} // namespace miopen
