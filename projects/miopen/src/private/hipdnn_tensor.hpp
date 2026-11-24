// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// What every forwarded operation reads from a MIOpen tensor descriptor, and how
// it becomes a hipDNN tensor and part of a PlanKey. Shared by the operation
// files (hipdnn_conv.cpp, hipdnn_batchnorm.cpp).
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

#include <hipdnn_frontend.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace miopen {
namespace wrapper {
namespace hipdnn {

struct TensorInfo
{
    miopenDataType_t dataType{};
    // The type set with miopenSetTensorCastType, or dataType when none is set.
    miopenDataType_t castType{};
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    // Above 1 only for NCHWc4, NCHWc8, CHWNc4 and CHWNc8.
    int64_t vectorLength = 1;
};

// Reads lengths and strides at full width (above INT_MAX too), which
// miopenGetTensorDescriptor's int arrays cannot hold.
miopenStatus_t ReadTensor(miopenTensorDescriptor_t desc, TensorInfo& out);

// Appends the length of `values`, then the values, so two different lists never
// flatten to the same sequence.
void AppendList(std::vector<int64_t>& out, const std::vector<int64_t>& values);

// Appends the data type, dims and strides of `tensor`.
void AppendTensor(std::vector<int64_t>& out, const TensorInfo& tensor);

std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
MakeTensor(const TensorInfo& info, hipdnn_frontend::DataType dataType, int64_t uid);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
