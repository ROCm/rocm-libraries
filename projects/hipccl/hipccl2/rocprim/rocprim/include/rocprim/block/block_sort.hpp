// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_sort_shared.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_sort primitive.
enum class block_sort_algorithm
{
    using_shared,
    default_algorithm = using_shared,
};

namespace detail
{

// Selector for block_sort algorithm which gives block sort implementation
// type based on passed block_sort_algorithm enum
template<block_sort_algorithm Algorithm>
struct select_block_sort_impl;

template<>
struct select_block_sort_impl<block_sort_algorithm::using_shared>
{
    template<class Key, unsigned int BlockSize, class Value>
    using type = block_sort_shared<Key, BlockSize, Value>;
};

} // end namespace detail

template<
    class Key,
    unsigned int BlockSize,
    class Value = empty_type,
    block_sort_algorithm Algorithm = block_sort_algorithm::default_algorithm
>
class block_sort
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSize, Value>
#endif
{
    using base_type = typename detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSize, Value>;
public:
    using storage_type = typename base_type::storage_type;

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, compare_function);
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(
            thread_key, storage, compare_function
        );
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, compare_function);
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(
            thread_key, thread_value, storage, compare_function
        );
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SORT_HPP_
