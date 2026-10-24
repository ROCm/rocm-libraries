// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCUB_ROCPRIM_DETAIL_ENV_BASED_HPP_
#define HIPCUB_ROCPRIM_DETAIL_ENV_BASED_HPP_

#include "temporary_storage.hpp"

#include _HIPCUB_LIBCXX_INCLUDE(memory_resource)
#include _HIPCUB_LIBCXX_INCLUDE(stream)
#include _HIPCUB_STD_INCLUDE(execution)

BEGIN_HIPCUB_NAMESPACE
namespace detail
{

// TODO: this should be moved to libhipcxx's `hip::__device_memory_resource` when that's added
struct device_memory_resource
{
    void* allocate(size_t bytes, size_t /* alignment */)
    {
        void* ptr{nullptr};
        HIPCUB_TRY_CUDA_API(::hipMalloc, "allocate failed to allocate with hipMalloc", &ptr, bytes);
        return ptr;
    }

    void deallocate(void* ptr, size_t /* bytes */)
    {
        HIPCUB_ASSERT_CUDA_API(::hipFree, "deallocate failed with hipFree", ptr);
    }

    void* allocate(_HIPCUB_LIBCXX::stream_ref stream, size_t bytes)
    {
        void* ptr{nullptr};
        HIPCUB_TRY_CUDA_API(::hipMallocAsync,
                            "allocate failed to allocate with hipMallocAsync",
                            &ptr,
                            bytes,
                            stream.get());
        return ptr;
    }

    void deallocate(_HIPCUB_LIBCXX::stream_ref stream, void* ptr, size_t /* bytes */)
    {
        HIPCUB_ASSERT_CUDA_API(::hipFreeAsync,
                               "deallocate failed with hipFreeAsync",
                               ptr,
                               stream.get());
    }
};

template<typename EnvT, typename FuntionT>
HIPCUB_RUNTIME_FUNCTION
static hipError_t env_invoke(EnvT env, FuntionT&& func)
{
    // Query properties from the environment
    auto stream = _HIPCUB_STD_EXEC::__query_or(env,
                                               _HIPCUB_LIBCXX::get_stream,
                                               _HIPCUB_LIBCXX::stream_ref{hipStream_t{}});
    auto mr     = _HIPCUB_STD_EXEC::__query_or(env,
                                           _HIPCUB_LIBCXX::mr::__get_memory_resource,
                                           detail::device_memory_resource{});

    // Query and allocate temp storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    HIPCUB_RETURN_ON_ERROR(func(d_temp_storage, temp_storage_bytes, stream.get()));

    HIPCUB_RETURN_ON_ERROR(
        detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr));

    // Run
    const hipError_t error = func(d_temp_storage, temp_storage_bytes, stream.get());

    // Deallocate temp storage and return. Function error takes precedence over deallocate error
    const hipError_t deallocate_error
        = detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr);

    if(error != hipSuccess)
    {
        return error;
    }

    return deallocate_error;
}
} // namespace detail
END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DETAIL_ENV_BASED_HPP_
