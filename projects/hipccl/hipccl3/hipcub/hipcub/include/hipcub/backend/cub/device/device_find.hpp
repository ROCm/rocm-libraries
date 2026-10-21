// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_CUB_DEVICE_DEVICE_FIND_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_FIND_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <cub/device/device_find.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceFind
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename NumItemsT>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t FindIf(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             ScanOpT         scan_op,
                             NumItemsT       num_items,
                             hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceFind::FindIf(d_temp_storage,
                                                                temp_storage_bytes,
                                                                d_in,
                                                                d_out,
                                                                scan_op,
                                                                num_items,
                                                                stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_FIND_HPP_
