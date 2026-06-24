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
#ifndef ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/copy.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator1, class Iterator2, class OutputIterator, class Compare>
constexpr OutputIterator merge(Iterator1 first1,
                               Iterator1 last1,
                               Iterator2 first2,
                               Iterator2 last2,
                               OutputIterator d_first,
                               Compare comp)
{
    for(; first1 != last1; ++d_first)
    {
        if(first2 == last2)
            return copy(first1, last1, d_first);

        if(comp(*first2, *first1))
        {
            *d_first = *first2;
            ++first2;
        }
        else
        {
            *d_first = *first1;
            ++first1;
        }
    }
    return copy(first2, last2, d_first);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP
