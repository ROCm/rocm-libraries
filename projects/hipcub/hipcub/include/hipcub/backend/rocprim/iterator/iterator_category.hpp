// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_ITERATOR_CATEGORY_HPP
#define HIPCUB_ROCPRIM_ITERATOR_CATEGORY_HPP

#include "../../../config.hpp"

#if(THRUST_VERSION >= 100700)
    // This iterator is compatible with Thrust API 1.7 and newer
    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

// Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
template<typename ValueType, typename Reference, bool AnySystemTag = true>
struct IteratorCategory
{
    using system_tag
        = std::conditional<AnySystemTag, thrust::any_system_tag, thrust::device_system_tag>::type;
    using type =
        typename thrust::detail::iterator_facade_category<system_tag,
                                                          thrust::random_access_traversal_tag,
                                                          ValueType,
                                                          Reference>::type;
};

} // namespace detail

END_HIPCUB_NAMESPACE

#else

    #include <iterator>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<typename ValueType, typename Reference, bool AnySystemTag = true>
struct IteratorCategory
{
    using type = typename std::random_access_iterator_tag;
};

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // THRUST_VERSION

#endif // HIPCUB_ROCPRIM_ITERATOR_CATEGORY_HPP
