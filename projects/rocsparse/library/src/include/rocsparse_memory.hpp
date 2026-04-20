/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocsparse_memory_debug_t.hpp"
#include <hip/hip_runtime_api.h>

hipError_t rocsparse_hipFree(void* p);
hipError_t rocsparse_hipFreeAsync(void* p, hipStream_t stream);

hipError_t rocsparse_hipDeviceSynchronize();

hipError_t rocsparse_hipStreamSynchronize(hipStream_t stream);

hipError_t rocsparse_hipMemset(void* target, int value, size_t size_in_bytes);

hipError_t
    rocsparse_hipMemsetAsync(void* target, int value, size_t size_in_bytes, hipStream_t stream);

hipError_t
    rocsparse_hipMemcpy(void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind);

hipError_t rocsparse_hipMemcpy2DAsync(void*         target,
                                      size_t        tpitch,
                                      const void*   source,
                                      size_t        spitch,
                                      size_t        width,
                                      size_t        height,
                                      hipMemcpyKind kind,
                                      hipStream_t   stream);

hipError_t rocsparse_hipMemcpyAsync(
    void* target, const void* source, size_t size_in_bytes, hipMemcpyKind kind, hipStream_t stream);

hipError_t rocsparse_hipMalloc_impl(void** p, size_t size_in_bytes);
template <typename T>
inline hipError_t rocsparse_hipMalloc(T** p, size_t size_in_bytes)
{
    return rocsparse_hipMalloc_impl(reinterpret_cast<void**>(p), size_in_bytes);
}
hipError_t rocsparse_hipMallocAsync_impl(void** p, size_t size_in_bytes, hipStream_t);

template <typename T>
inline hipError_t rocsparse_hipMallocAsync(T** p, size_t size_in_bytes, hipStream_t stream)
{
    return rocsparse_hipMallocAsync_impl(reinterpret_cast<void**>(p), size_in_bytes, stream);
}
