// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Conversions between MIOpen's enums and hipDNN's, for the hipDNN forwarding.
// Each returns false (or a catch-all status) for a value with no hipDNN
// equivalent, so the caller can decline the problem.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>

namespace miopen {
namespace wrapper {
namespace hipdnn {

bool ToHipdnnDataType(miopenDataType_t type, hipdnn_frontend::DataType& out);

// The type to accumulate in for tensors of `type`.
bool ComputeTypeFor(miopenDataType_t type, hipdnn_frontend::DataType& out);

bool ToPointwiseActivation(miopenActivationMode_t mode, hipdnn_frontend::PointwiseMode& out);

miopenStatus_t TranslateHipdnnError(const hipdnn_frontend::Error& error);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
