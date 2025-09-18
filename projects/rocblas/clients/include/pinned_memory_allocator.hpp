/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include "host_alloc.hpp"

//!
//! @brief  Allocator which requests pinned host memory via hipHostMalloc.
//!         This class can be removed once hipHostRegister has been proven equivalent
//!
template <class T>
struct pinned_memory_allocator
{
    using value_type = T;

    pinned_memory_allocator() = default;

    template <class U>
    pinned_memory_allocator(const pinned_memory_allocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        if(!host_mem_safe(n))
        {
            // exception same as host_alloc helper
            throw std::bad_alloc{};
        }

        T*         ptr;
        hipError_t status = hipHostMalloc(&ptr, sizeof(T) * n, hipHostMallocDefault);
        if(status != hipSuccess)
        {
            ptr = nullptr;
            rocblas_cerr << "rocBLAS pinned_memory_allocator failed to allocate memory: "
                         << hipGetErrorString(status) << std::endl;
            // exception same as host_alloc helper
            throw std::bad_alloc{};
        }

        alloc_ptr_use(ptr, n); // count the same as host memory
        return ptr;
    }

    void deallocate(T* ptr, std::size_t n)
    {
        hipError_t status = hipHostFree(ptr);
        if(status != hipSuccess)
        {
            rocblas_cerr << "rocBLAS pinned_memory_allocator failed to free memory: "
                         << hipGetErrorString(status) << std::endl;
        }

        free_ptr_use(ptr); // release count
    }
};

template <class T, class U>
constexpr bool operator==(const pinned_memory_allocator<T>&, const pinned_memory_allocator<U>&)
{
    return true;
}

template <class T, class U>
constexpr bool operator!=(const pinned_memory_allocator<T>&, const pinned_memory_allocator<U>&)
{
    return false;
}
