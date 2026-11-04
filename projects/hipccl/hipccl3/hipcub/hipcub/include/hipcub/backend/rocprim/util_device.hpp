/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_UTIL_DEVICE_HPP_
#define HIPCUB_ROCPRIM_UTIL_DEVICE_HPP_

#include "../../config.hpp"

#include "util_temporary_storage.hpp"

BEGIN_HIPCUB_NAMESPACE

/// \brief Computes the maximum potential dynamic shared memory size per block for kernel \p kernel_ptr taking into
///        account the amount of kernel's static and CUDA Driver's reserved shared memory.
///
/// \param[out] max_dyn_smem_bytes
///   Maximum dynamic shared memory that can be allocated. Set to -1 in case of error.
///
/// \param[in] kernel_ptr
///   Kernel pointer for which to compute the maximum potential dynamic shared memory.
template<class KernelPtr>
HIPCUB_RUNTIME_FUNCTION
inline hipError_t MaxPotentialDynamicSmemBytes(int&      max_dyn_smem_bytes,
                                               KernelPtr kernel_ptr) noexcept
{
    max_dyn_smem_bytes = -1;

    hipFuncAttributes kernel_attrs{};
    if(const auto error = HipcubDebug(
           hipFuncGetAttributes(&kernel_attrs, reinterpret_cast<const void*>(kernel_ptr))))
    {
        return error;
    }

    int curr_device{};
    if(const auto error = HipcubDebug(hipGetDevice(&curr_device)))
    {
        return error;
    }

    int reserved_smem_per_block{};
    if(const auto error
       = HipcubDebug(hipDeviceGetAttribute(&reserved_smem_per_block,
                                           hipDeviceAttributeReservedSharedMemPerBlock,
                                           curr_device)))
    {
        return error;
    }

    int max_smem_per_block{};
    if(const auto error
       = HipcubDebug(hipDeviceGetAttribute(&max_smem_per_block,
                                           hipDeviceAttributeMaxSharedMemoryPerBlock,
                                           curr_device)))
    {
        return error;
    }

    max_dyn_smem_bytes = max_smem_per_block - reserved_smem_per_block
                         - static_cast<int>(kernel_attrs.sharedSizeBytes);

    return hipSuccess;
}

template<>
HIPCUB_RUNTIME_FUNCTION
inline hipError_t MaxPotentialDynamicSmemBytes(int&           max_dyn_smem_bytes,
                                               std::nullptr_t kernel_ptr) noexcept
{
    max_dyn_smem_bytes = -1;

    hipFuncAttributes kernel_attrs{};
    const auto        error = HipcubDebug(hipFuncGetAttributes(&kernel_attrs, kernel_ptr));
    return error;
}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_DEVICE_HPP_
