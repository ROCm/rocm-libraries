// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <initializer_list>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace miopen_plugin::pointwise_applicability
{

// Validates the MIOpen constraints shared by the pointwise op family (unary activation via
// miopenActivationForward and binary pointwise via miopenOpTensor): both only support fp32/fp16,
// with all IO tensors sharing one dtype. This is NOT a general-purpose validator for the whole
// provider -- batchnorm and convolution also support BFLOAT16 and must not call this.
//
// Throws HipdnnPluginException on any violation. Covers exactly: non-virtual, dtype in
// {FLOAT, HALF}, all listed tensors share one dtype, dims()/strides() non-null, and
// dims()->size() == strides()->size() (vector lengths, not element counts).
// `opName` carries the bare op name (e.g. "Relu", "Binary pointwise") and is NOT prefixed
// with "plan builder:" here -- callers add whatever prefix their own messages use.
void validatePointwiseIoTensors(
    std::initializer_list<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> tensors,
    const std::string& opName);

} // namespace miopen_plugin::pointwise_applicability
