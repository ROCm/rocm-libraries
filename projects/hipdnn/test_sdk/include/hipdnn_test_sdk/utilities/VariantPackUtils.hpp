// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Selects the pointer representation required by an execute-time variant pack.
/// Runtime pass-by-value tensors always use host memory, even for GPU execution.
inline void* selectVariantPackPointer(hipdnn_data_sdk::utilities::ITensor& tensor,
                                      bool useDevice,
                                      bool isRuntimePassByValue)
{
    return useDevice && !isRuntimePassByValue ? tensor.rawDeviceData() : tensor.rawHostData();
}

} // namespace hipdnn_test_sdk::utilities
