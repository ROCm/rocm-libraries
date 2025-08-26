/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CUB_AGENT_SINGLE_PASS_SCAN_OPERATORS_HPP_
#define HIPCUB_CUB_AGENT_SINGLE_PASS_SCAN_OPERATORS_HPP_

#include "../../../config.hpp"

#include <hip/hip_runtime.h>

#include <cub/agent/single_pass_scan_operators.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

using ScanTileStatus = cub::ScanTileStatus;

template<typename T, typename ScanOpT>
using BlockScanRunningPrefixOp = cub::BlockScanRunningPrefixOp<T, ScanOpT>;

template<typename T,
         typename ScanOpT,
         typename ScanTileStateT,
         int LEGACY_PTX_ARCH        = 0,
         typename DelayConstructorT = cub::detail::default_delay_constructor_t<T>>
using TilePrefixCallbackOp
    = cub::TilePrefixCallbackOp<T, ScanOpT, ScanTileStateT, LEGACY_PTX_ARCH, DelayConstructorT>;

template<typename ValueT,
         typename KeyT,
         bool SINGLE_WORD
         = (cub::detail::is_primitive<ValueT>::value) && (sizeof(ValueT) + sizeof(KeyT) < 16)>
using ReduceByKeyScanTileState = cub::ReduceByKeyScanTileState<ValueT, KeyT, SINGLE_WORD>;

template<typename T, bool SINGLE_WORD = cub::detail::is_primitive<T>::value>
struct ScanTileState : cub::ScanTileState<T, SINGLE_WORD>
{
    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE
        hipError_t Init(int num_tiles, void* d_temp_storage, size_t temp_storage_bytes)
    {
        // This function *technically* is supported on both host and device, but
        // hipCUDAErrorTohipError only works on host. While the CUB implementation
        // always return cudaSuccess, we'd still want to use the error-type
        // conversion for better compatibility with future versions.
#ifdef __HIP_DEVICE_COMPILE__
        static_cast<cub::ScanTileState<T, SINGLE_WORD>*>(this)->Init(num_tiles,
                                                                     d_temp_storage,
                                                                     temp_storage_bytes);
        return hipSuccess;
#else
        return hipCUDAErrorTohipError(static_cast<cub::ScanTileState<T, SINGLE_WORD>*>(this)
                                          ->Init(num_tiles, d_temp_storage, temp_storage_bytes));
#endif
    }

    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE static hipError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes)
    {
        // This function *technically* is supported on both host and device, but
        // hipCUDAErrorTohipError only works on host. While the CUB implementation
        // always return cudaSuccess, we'd still want to use the error-type
        // conversion for better compatibility with future versions.
#ifdef __HIP_DEVICE_COMPILE__
        cub::ScanTileState<T, SINGLE_WORD>::AllocationSize(num_tiles, temp_storage_bytes);
        return hipSuccess;
#else
        return hipCUDAErrorTohipError(
            cub::ScanTileState<T, SINGLE_WORD>::AllocationSize(num_tiles, temp_storage_bytes));
#endif
    }
};

END_HIPCUB_NAMESPACE

#endif
