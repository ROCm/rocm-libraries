// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// What every forwarded operation needs to read a MIOpen tensor descriptor back
// through the public _impl getters, decline the layouts hipDNN cannot describe,
// and turn the result into a PlanKey entry and a hipDNN tensor attribute.
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
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    // Above 1 only for NCHWc4, NCHWc8, CHWNc4 and CHWNc8.
    int64_t vectorLength = 1;
    // Set when a dim or stride is above INT_MAX, which the int getter cuts down.
    // dims, strides and vectorLength are then unusable.
    bool truncated = false;
};

// miopenStatusSuccess, or miopenStatusInvalidValue when a getter fails.
miopenStatus_t ReadTensor(miopenTensorDescriptor_t desc, TensorInfo& out);

// Reads the mode alone. Every mode the forwarding accepts ignores the
// descriptor's alpha, beta and gamma.
miopenStatus_t ReadActivationMode(miopenActivationDescriptor_t desc, miopenActivationMode_t& mode);

// Exact comparison is the point: the forwarded ops can express no scaling at
// all, so anything but a literal 1.0 or 0.0 has to be declined rather than
// rounded into. A null `value` never matches.
bool ScalarEquals(const void* value, miopenDataType_t type, double expected);

// Returns why hipDNN cannot describe one of these tensors, or null when it can.
// The data type is left to the caller, whose message names the operation.
const char* CheckTensorLayouts(const std::vector<TensorInfo>& tensors);

// Appends `values` preceded by its length, as PlanKey requires.
void AppendList(std::vector<int64_t>& out, const std::vector<int64_t>& values);

// Appends the data type, dims and strides of `tensor`.
void AppendTensor(std::vector<int64_t>& out, const TensorInfo& tensor);

std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
MakeTensor(const TensorInfo& info, hipdnn_frontend::DataType dataType, int64_t uid);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
