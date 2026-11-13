// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_BACKEND_CUB_THREAD_SCAN_HPP_
#define HIPCUB_BACKEND_CUB_THREAD_SCAN_HPP_

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <cub/thread/thread_scan.cuh> // CUB thread scan

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<int LENGTH,
         typename T,
         typename ScanOp>
 HIPCUB_DEVICE
HIPCUB_FORCEINLINE
    T ThreadScanInclusive(T      inclusive, ///< [in] Initial value for inclusive aggregate
                          T*     input, ///< [in] Input array
                          T*     output, ///< [out] Output array (may be aliased to \p input)
                          ScanOp scan_op, ///< [in] Binary scan operator
                          ::hipcub::detail::int_constant_t<LENGTH> /*length*/)
{
    return cub::detail::ThreadScanInclusive(inclusive,
                                            input,
                                            output,
                                            scan_op,
                                            ::hipcub::detail::int_constant_t<LENGTH>());
}

template<int LENGTH,
         typename T,
         typename ScanOp>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE
    T ThreadScanInclusive(T*     input, ///< [in] Input array
                          T*     output, ///< [out] Output array (may be aliased to \p input)
                          ScanOp scan_op) ///< [in] Binary scan operator
{
    return cub::detail::ThreadScanInclusive<LENGTH>(input, output, scan_op);
}

template<int LENGTH,
         typename T,
         typename ScanOp>
 HIPCUB_DEVICE
HIPCUB_FORCEINLINE
    T ThreadScanInclusive(T (&input)[LENGTH], ///< [in] Input array
                          T (&output)[LENGTH], ///< [out] Output array (may be aliased to \p input)
                          ScanOp scan_op) ///< [in] Binary scan operator
{
    return cub::detail::ThreadScanInclusive<LENGTH>(input, output, scan_op);
}

template<int LENGTH,
         typename T,
         typename ScanOp>
 HIPCUB_DEVICE
HIPCUB_FORCEINLINE T ThreadScanInclusive(
    T*     input, ///< [in] Input array
    T*     output, ///< [out] Output array (may be aliased to \p input)
    ScanOp scan_op, ///< [in] Binary scan operator
    T      prefix, ///< [in] Prefix to seed scan with
    bool   apply_prefix
    = true) ///< [in] Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
    return cub::detail::ThreadScanInclusive<LENGTH>(input, output, scan_op, prefix, apply_prefix);
}

template<int LENGTH,
         typename T,
         typename ScanOp>
 HIPCUB_DEVICE
HIPCUB_FORCEINLINE T ThreadScanInclusive(
    T (&input)[LENGTH], ///< [in] Input array
    T (&output)[LENGTH], ///< [out] Output array (may be aliased to \p input)
    ScanOp scan_op, ///< [in] Binary scan operator
    T      prefix, ///< [in] Prefix to seed scan with
    bool   apply_prefix
    = true) ///< [in] Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
    return cub::detail::ThreadScanInclusive<LENGTH>(input, output, scan_op, prefix, apply_prefix);
}

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_BACKEND_CUB_THREAD_SCAN_HPP_
