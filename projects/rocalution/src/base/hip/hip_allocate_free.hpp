/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_ALLOCATE_FREE_HPP_
#define ROCALUTION_HIP_ALLOCATE_FREE_HPP_

#include <hip/hip_runtime_api.h>
#include <iostream>

namespace rocalution
{
    template <typename DataType>
    void allocate_hip(int64_t n, DataType** ptr);

    template <typename DataType>
    void free_hip(DataType** ptr);

    template <typename DataType>
    void set_to_zero_hip(
        int blocksize, int64_t n, DataType* ptr, bool async = false, hipStream_t stream = 0);

    template <typename DataType>
    void set_to_one_hip(
        int blocksize, int64_t n, DataType* ptr, bool async = false, hipStream_t stream = 0);

    template <typename DataType>
    void set_to_value_hip(int         blocksize,
                          int64_t     n,
                          DataType*   ptr,
                          DataType    value,
                          bool        async  = false,
                          hipStream_t stream = 0);

    template <typename DataType>
    void copy_d2h(
        int64_t n, const DataType* src, DataType* dst, bool async = false, hipStream_t stream = 0);

    template <typename DataType>
    void copy_h2d(
        int64_t n, const DataType* src, DataType* dst, bool async = false, hipStream_t stream = 0);

    template <typename DataType>
    void copy_d2d(
        int64_t n, const DataType* src, DataType* dst, bool async = false, hipStream_t stream = 0);
} // namespace rocalution

#endif // ROCALUTION_HIP_ALLOCATE_FREE_HPP_
