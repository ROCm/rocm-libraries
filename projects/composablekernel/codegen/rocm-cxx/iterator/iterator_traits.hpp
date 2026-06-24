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
#ifndef ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP
#define ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>
#include <rocm/type_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

struct input_iterator_tag
{
};

struct output_iterator_tag
{
};

struct forward_iterator_tag : input_iterator_tag
{
};

struct bidirectional_iterator_tag : forward_iterator_tag
{
};

struct random_access_iterator_tag : bidirectional_iterator_tag
{
};

template <class Iterator>
struct iterator_traits
{
    using difference_type   = typename Iterator::difference_type;
    using value_type        = typename Iterator::value_type;
    using pointer           = typename Iterator::pointer;
    using reference         = typename Iterator::reference;
    using iterator_category = typename Iterator::iterator_category;
};

template <class T>
struct iterator_traits<T*>
{
    using difference_type   = ptrdiff_t;
    using value_type        = remove_cv_t<T>;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = random_access_iterator_tag;
};

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP
