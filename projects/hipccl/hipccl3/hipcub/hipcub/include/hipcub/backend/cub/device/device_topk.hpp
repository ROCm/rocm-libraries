// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_CUB_DEVICE_DEVICE_TOPK_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_TOPK_HPP_

#include "../../../config.hpp"

#include <cub/device/device_topk.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceTopK
{
    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename ValueInputIteratorT,
             typename ValueOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MaxPairs(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               KeyInputIteratorT    d_keys_in,
                               KeyOutputIteratorT   d_keys_out,
                               ValueInputIteratorT  d_values_in,
                               ValueOutputIteratorT d_values_out,
                               NumItemsT            num_items,
                               NumOutItemsT         k,
                               EnvT                 env = {})
    {
        return hipCUDAErrorTohipError(::cub::DeviceTopK::MaxPairs(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys_in,
                                                                  d_keys_out,
                                                                  d_values_in,
                                                                  d_values_out,
                                                                  num_items,
                                                                  k,
                                                                  env));
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename ValueInputIteratorT,
             typename ValueOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MinPairs(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               KeyInputIteratorT    d_keys_in,
                               KeyOutputIteratorT   d_keys_out,
                               ValueInputIteratorT  d_values_in,
                               ValueOutputIteratorT d_values_out,
                               NumItemsT            num_items,
                               NumOutItemsT         k,
                               EnvT                 env = {})
    {
        return hipCUDAErrorTohipError(::cub::DeviceTopK::MinPairs(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys_in,
                                                                  d_keys_out,
                                                                  d_values_in,
                                                                  d_values_out,
                                                                  num_items,
                                                                  k,
                                                                  env));
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MaxKeys(void*              d_temp_storage,
                              size_t&            temp_storage_bytes,
                              KeyInputIteratorT  d_keys_in,
                              KeyOutputIteratorT d_keys_out,
                              NumItemsT          num_items,
                              NumOutItemsT       k,
                              EnvT               env = {})
    {
        return hipCUDAErrorTohipError(::cub::DeviceTopK::MaxKeys(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_keys_in,
                                                                 d_keys_out,
                                                                 num_items,
                                                                 k,
                                                                 env));
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MinKeys(void*              d_temp_storage,
                              size_t&            temp_storage_bytes,
                              KeyInputIteratorT  d_keys_in,
                              KeyOutputIteratorT d_keys_out,
                              NumItemsT          num_items,
                              NumOutItemsT       k,
                              EnvT               env = {})
    {
        return hipCUDAErrorTohipError(::cub::DeviceTopK::MinKeys(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_keys_in,
                                                                 d_keys_out,
                                                                 num_items,
                                                                 k,
                                                                 env));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_TOPK_HPP_
