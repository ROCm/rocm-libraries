// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef _CCCL_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
#define _CCCL_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_

#include <hip/hip_common.h>

#ifdef __HIP_PLATFORM_AMD__
    #include "../backend/rocprim/device/device_segmented_scan.hpp" // IWYU pragma: export
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include "../backend/cub/device/device_segmented_scan.hpp" // IWYU pragma: export
#endif

#endif // _CCCL_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
