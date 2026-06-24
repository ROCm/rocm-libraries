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
#ifndef ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP

#include <rocm/config.hpp>
#include <rocm/assert.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/rotate.hpp>
#include <rocm/algorithm/upper_bound.hpp>
#include <rocm/algorithm/is_sorted.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr void stable_sort(Iterator first, Iterator last, Compare comp)
{
    if(first == last)
        return;
    for(auto i = first; i != last; ++i)
        rotate(upper_bound(first, i, *i, comp), i, i + 1);
    ROCM_ASSERT(is_sorted(first, last, comp));
}

template <class Iterator>
constexpr void stable_sort(Iterator first, Iterator last)
{
    stable_sort(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP
