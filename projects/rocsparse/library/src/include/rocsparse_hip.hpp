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

#include "rocsparse-debugging.h"

#ifndef ROCSPARSE_DEBUGGING

#ifndef ROCSPARSE_WITH_MEMSTAT
#define rocsparse_hipFree(P_) hipFree((P_))
#define rocsparse_hipFreeAsync(P_, S_) hipFreeAsync((P_), (S_))
#define rocsparse_hipMalloc(P, SIZE) hipMalloc(reinterpret_cast<void**>((P)), (SIZE))
#define rocsparse_hipMallocAsync(P, SIZE, STREAM) \
    hipMallocAsync(reinterpret_cast<void**>((P)), (SIZE), (STREAM))
#endif
#define rocsparse_hipMemcpyAsync(TARGET, SOURCE, SIZE, KIND, STREAM) \
    hipMemcpyAsync((TARGET), (SOURCE), (SIZE), (KIND), (STREAM))
#define rocsparse_hipMemcpy(TARGET, SOURCE, SIZE, KIND) \
    hipMemcpy((TARGET), (SOURCE), (SIZE), (KIND))
#define rocsparse_hipMemsetAsync(TARGET, VALUE, SIZE, STREAM) \
    hipMemsetAsync((TARGET), (VALUE), (SIZE), (STREAM))
#define rocsparse_hipMemset(TARGET, VALUE, SIZE, KIND) hipMemset((TARGET), (VALUE), (SIZE))
#define rocsparse_hipDeviceSynchronize() hipDeviceSynchronize()
#define rocsparse_hipStreamSynchronize(STREAM) hipStreamSynchronize((STREAM))
#define rocsparse_hipMemcpy2DAsync(TARGET, TPITCH, SOURCE, SPITCH, WIDTH, HEIGHT, KIND, STREAM) \
    hipMemcpy2DAsync((TARGET), (TPITCH), (SOURCE), (SPITCH), (WIDTH), (HEIGHT), (KIND), (STREAM))
#define rocsparse_hipLaunchKernelGGL(K, G, T, M, S, ...) \
    hipLaunchKernelGGL((K), (G), (T), (M), (S), __VA_ARGS__)

#else

namespace rocsparse
{
    namespace hip
    {
        hipError_t hipMemcpy(void*         target,
                             const void*   source,
                             size_t        size_in_bytes,
                             hipMemcpyKind kind,
                             const char*   function_name,
                             const char*   file_name,
                             int32_t       file_line);

        hipError_t hipMemcpyAsync(void*         target,
                                  const void*   source,
                                  size_t        size_in_bytes,
                                  hipMemcpyKind kind,
                                  hipStream_t   stream,
                                  const char*   function_name,
                                  const char*   file_name,
                                  int32_t       file_line);

        hipError_t hipMemcpy2DAsync(void*         target,
                                    size_t        tpitch,
                                    const void*   source,
                                    size_t        spitch,
                                    size_t        width,
                                    size_t        height,
                                    hipMemcpyKind kind,
                                    hipStream_t   stream,
                                    const char*   function_name,
                                    const char*   file_name,
                                    int32_t       file_line);

        hipError_t hipMemset(void*       target,
                             int32_t     value,
                             size_t      size_in_bytes,
                             const char* function_name,
                             const char* file_name,
                             int32_t     file_line);

        hipError_t hipMemsetAsync(void*       target,
                                  int         value,
                                  size_t      size_in_bytes,
                                  hipStream_t stream,
                                  const char* function_name,
                                  const char* file_name,
                                  int32_t     file_line);

        hipError_t hipDeviceSynchronize(const char* function_name,
                                        const char* file_name,
                                        int32_t     file_line);

        hipError_t hipStreamSynchronize(hipStream_t stream,
                                        const char* function_name,
                                        const char* file_name,
                                        int32_t     file_line);

        hipError_t hipMallocAsync(void**      p_that,
                                  size_t      size_in_bytes,
                                  hipStream_t stream,
                                  const char* function_name,
                                  const char* file_name,
                                  int32_t     file_line);

        hipError_t hipFreeAsync(void*       that,
                                hipStream_t stream,
                                const char* function_name,
                                const char* file_name,
                                int32_t     file_line);
        hipError_t hipFree(void*       that,
                           const char* function_name,
                           const char* file_name,
                           int32_t     file_line);

        hipError_t hipMalloc(void**      p_that,
                             size_t      size_in_bytes,
                             const char* function_name,
                             const char* file_name,
                             int32_t     file_line);

        void tag_hipLaunchKernelGGL(hipStream_t stream,
                                    const char* function_name,
                                    const char* file_name,
                                    int32_t     file_line);

    }

}

#define rocsparse_hipFree(P) rocsparse::hip::hipFree((P), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipFreeAsync(P, S) \
    rocsparse::hip::hipFreeAsync((P), (S), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMemcpyAsync(TARGET, SOURCE, SIZE, KIND, STREAM) \
    rocsparse::hip::hipMemcpyAsync(                                  \
        (TARGET), (SOURCE), (SIZE), (KIND), (STREAM), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMemcpy(TARGET, SOURCE, SIZE, KIND) \
    rocsparse::hip::hipMemcpy((TARGET), (SOURCE), (SIZE), (KIND), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMemsetAsync(TARGET, VALUE, SIZE, STREAM) \
    rocsparse::hip::hipMemsetAsync(                           \
        (TARGET), (VALUE), (SIZE), (STREAM), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMemset(TARGET, VALUE, SIZE, KIND) \
    rocsparse::hip::hipMemset((TARGET), (VALUE), (SIZE), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipDeviceSynchronize() \
    rocsparse::hip::hipDeviceSynchronize(__FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipStreamSynchronize(STREAM) \
    rocsparse::hip::hipStreamSynchronize((STREAM), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMemcpy2DAsync(TARGET, TPITCH, SOURCE, SPITCH, WIDTH, HEIGHT, KIND, STREAM) \
    rocsparse::hip::hipMemcpy2DAsync((TARGET),                                                  \
                                     (TPITCH),                                                  \
                                     (SOURCE),                                                  \
                                     (SPITCH),                                                  \
                                     (WIDTH),                                                   \
                                     (HEIGHT),                                                  \
                                     (KIND),                                                    \
                                     (STREAM),                                                  \
                                     __FUNCTION__,                                              \
                                     __FILE__,                                                  \
                                     __LINE__)
#define rocsparse_hipMalloc(P, SIZE) \
    rocsparse::hip::hipMalloc(       \
        reinterpret_cast<void**>((P)), (SIZE), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipMallocAsync(P, SIZE, STREAM) \
    rocsparse::hip::hipMallocAsync(               \
        reinterpret_cast<void**>((P)), (SIZE), (STREAM), __FUNCTION__, __FILE__, __LINE__)
#define rocsparse_hipLaunchKernelGGL(K, G, T, M, S, ...)                                   \
    do                                                                                     \
    {                                                                                      \
        auto local_S = (S);                                                                \
        rocsparse::hip::tag_hipLaunchKernelGGL(local_S, __FUNCTION__, __FILE__, __LINE__); \
        hipLaunchKernelGGL((K), (G), (T), (M), local_S, __VA_ARGS__);                      \
    } while(false)

#endif
