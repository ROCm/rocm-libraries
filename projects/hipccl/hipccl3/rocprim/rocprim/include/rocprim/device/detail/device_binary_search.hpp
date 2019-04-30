// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_BINARY_SEARCH_HPP_

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Size>
ROCPRIM_DEVICE inline
Size get_binary_search_middle(Size left, Size right)
{
    // Instead of `/ 2` we use `* 33 / 64`, i.e. the middle is slightly moved.
    // This greatly reduces address aliasing and hence cache misses for (nearly-)power-of-two
    // sizes of haystack (when addresses are mapped to the same cache line).
    // For random needles and (nearly-)power-of-two sizes, this change increases performance
    // 4-20 times making it equal to performance of arbitrary sizes of haystack.
    // See https://www.pvk.ca/Blog/2012/07/30/binary-search-is-a-pathological-case-for-caches/
    const Size d = right - left;
    return left + d / 2 + d / 64;
}

template<class RandomAccessIterator, class Size, class T, class BinaryPredicate>
ROCPRIM_DEVICE inline
Size lower_bound_n(RandomAccessIterator first,
                   Size size,
                   const T& value,
                   BinaryPredicate compare_op)
{
    Size left = 0;
    Size right = size;
    while(left < right)
    {
        const Size mid = get_binary_search_middle(left, right);
        if(compare_op(first[mid], value))
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

template<class RandomAccessIterator, class Size, class T, class BinaryPredicate>
ROCPRIM_DEVICE inline
Size upper_bound_n(RandomAccessIterator first,
                   Size size,
                   const T& value,
                   BinaryPredicate compare_op)
{
    Size left = 0;
    Size right = size;
    while(left < right)
    {
        const Size mid = get_binary_search_middle(left, right);
        if(compare_op(value, first[mid]))
        {
            right = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

struct lower_bound_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE inline
    Size operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        return lower_bound_n(haystack, size, value, compare_op);
    }
};

struct upper_bound_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE inline
    Size operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        return upper_bound_n(haystack, size, value, compare_op);
    }
};

struct binary_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE inline
    bool operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        const Size n = lower_bound_n(haystack, size, value, compare_op);
        return n != size && !compare_op(value, haystack[n]);
    }
};

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_BINARY_SEARCH_HPP_
