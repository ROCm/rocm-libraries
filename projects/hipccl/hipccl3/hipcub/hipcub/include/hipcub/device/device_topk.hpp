// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_DEVICE_DEVICE_TOPK_HPP_
#define HIPCUB_DEVICE_DEVICE_TOPK_HPP_

#include <hip/hip_common.h>

#ifdef __HIP_PLATFORM_AMD__
    #include "../backend/rocprim/device/device_topk.hpp" // IWYU pragma: export
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include "../backend/cub/device/device_topk.hpp" // IWYU pragma: export
#endif

#endif // HIPCUB_DEVICE_DEVICE_TOPK_HPP_
