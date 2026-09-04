// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_ROCBLAS_BRIDGE_RUNTIME_H_
#define ROCM_INTERFACES_ROCBLAS_BRIDGE_RUNTIME_H_

#include "rocblas_bridge_generated.h"

namespace rocm::interfaces {

const rocm_rocblas_bridge_v1* rocblas_bridge_table() noexcept;
const rocm_rocblas_bridge_v1* rocblas_bridge_table(rocblas_handle handle) noexcept;
rocblas_handle rocblas_bridge_native_handle(rocblas_handle handle) noexcept;

}  // namespace rocm::interfaces

#endif
