/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class OutputIterator, class UnaryOp>
constexpr OutputIterator
transform(Iterator first1, Iterator last1, OutputIterator out, UnaryOp unary_op)
{
    for(; first1 != last1; ++out, ++first1)
        *out = unary_op(*first1);

    return out;
}

template <class Iterator1, class Iterator2, class OutputIterator, class BinaryOp>
constexpr OutputIterator transform(
    Iterator1 first1, Iterator1 last1, Iterator2 first2, OutputIterator out, BinaryOp binary_op)
{
    for(; first1 != last1; ++out, ++first1, ++first2)
        *out = binary_op(*first1, *first2);

    return out;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP
