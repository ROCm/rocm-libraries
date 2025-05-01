/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

// RAII helper class for device memory allocation outside of rocBLAS
//
// rocblas_device_malloc mem(handle, size1, size2, size3, ...);
//
// roblcas_set_optimal_device_memory_size(handle, size1, size2, size3, ...);
//
// if(!mem) return rocblas_status_memory_error;
//
// void* ptr = static_cast<void*>(mem); // Only works if there is one size
//
// rocblas_internal_trsm_template_mem(..., mem, ...)
//
// This header should be included in other projects to use the rocblas_handle
// C++ device memory allocation API. It is unlikely to change very often.

#include "rocblas/rocblas.h"
#include <new>
#include <type_traits>

// Emulate C++17 std::conjunction
template <class...>
struct rocblas_conjunction : std::true_type
{
};
template <class T, class... Ts>
struct rocblas_conjunction<T, Ts...>
    : std::integral_constant<bool, T{} && rocblas_conjunction<Ts...>{}>
{
};

// clang-format off
class [[nodiscard]] rocblas_device_malloc
{
    rocblas_handle              handle;
    rocblas_device_malloc_base* dm_ptr;

public:
    // Allocate memory in a RAII class
    template <
        typename... Ss,
        std::enable_if_t<rocblas_conjunction<std::is_convertible<Ss, size_t>...>{},
                         int> = 0>
    explicit rocblas_device_malloc(rocblas_handle handle, Ss... sizes)
        : handle(handle)
        , dm_ptr(nullptr)
    {
        if(sizeof...(sizes) > 0)
        {
            rocblas_status status = rocblas_device_malloc_alloc(handle, &dm_ptr, sizeof...(sizes), size_t(sizes)...);
            if (status != rocblas_status_success && status != rocblas_status_memory_error)
                throw std::bad_alloc();
        }
    }

    // Move constructor
    rocblas_device_malloc(rocblas_device_malloc&& other) noexcept
        : handle(other.handle)
        , dm_ptr(other.dm_ptr)
    {
        other.dm_ptr = nullptr;
    }

    // Move assignment
    rocblas_device_malloc& operator=(rocblas_device_malloc&& other)
    {
        if(dm_ptr && dm_ptr != other.dm_ptr)
            rocblas_device_malloc_free(dm_ptr);
        handle = other.handle;
        dm_ptr = other.dm_ptr;
        other.dm_ptr = nullptr;
        return *this;
    }

    // Conversion to a pointer type, to get the address of the device memory
    // It is lvalue-qualified to avoid the common mistake of writing:
    // void* ptr = (void*) rocblas_device_malloc(handle, size);
    // ... which is incorrect, since the RAII temporary expression will be
    // destroyed, and the pointer will be left dangling.
    template <typename T>
    explicit operator T*() &
    {
        void* res;
        if(!dm_ptr || rocblas_device_malloc_ptr(dm_ptr, &res) != rocblas_status_success)
            res = nullptr;
        return static_cast<T*>(res);
    }

    // Access a particular element
    // It is lvalue-qualified so that it cannot bind to temporaries
    void* operator[](size_t index) &
    {
        void* res;
        if(!dm_ptr || rocblas_device_malloc_get(dm_ptr, index, &res) != rocblas_status_success)
            res = nullptr;
        return res;
    }

    // Conversion to bool indicates whether allocation succeeded
    // It is lvalue-qualified so that it cannot bind to temporaries
    explicit operator bool() &
    {
        return rocblas_device_malloc_success(dm_ptr);
    }

    // Conversion to rocblas_device_malloc_base reference, to pass to rocBLAS
    // It is lvalue-qualified so that it cannot bind to temporaries
    operator rocblas_device_malloc_base&() &
    {
        return *dm_ptr;
    }

    // Destructor calls the C API to mark the data as freed
    ~rocblas_device_malloc()
    {
        if(dm_ptr)
            rocblas_device_malloc_free(dm_ptr);
    }

    // Copying and assigning to rocblas_device_malloc are deleted
    rocblas_device_malloc(const rocblas_device_malloc&) = delete;
    rocblas_device_malloc& operator=(const rocblas_device_malloc&) = delete;
};
// clang-format on

// Set optimal device memory size in handle
template <
    typename... Ss,
    std::enable_if_t<sizeof...(Ss) && rocblas_conjunction<std::is_convertible<Ss, size_t>...>{},
                     int> = 0>
inline rocblas_status rocblas_set_optimal_device_memory_size(rocblas_handle handle, Ss... sizes)
{
    return rocblas_set_optimal_device_memory_size_impl(handle, sizeof...(sizes), size_t(sizes)...);
}
