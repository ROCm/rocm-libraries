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

#ifndef ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
#define ROCPRIM_BLOCK_BLOCK_SCAN_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_scan_warp_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_scan primitive.
enum class block_scan_algorithm
{
    using_warp_scan
};

namespace detail
{

// Selector for block_scan algorithm which gives block scan implementation
// type based on passed block_scan_algorithm enum
template<block_scan_algorithm Algorithm>
struct select_block_scan_impl;

template<>
struct select_block_scan_impl<block_scan_algorithm::using_warp_scan>
{
    template<class T, unsigned int BlockSize>
    using type = block_scan_warp_scan<T, BlockSize>;
};

} // end namespace detail

/// \brief Parallel scan primitive for block.
template<
    class T,
    unsigned int BlockSize,
    block_scan_algorithm Algorithm = block_scan_algorithm::using_warp_scan
>
class block_scan : detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>
{
    using base_type = typename detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>;
public:
    static constexpr block_scan_algorithm algorithm = Algorithm;
    using storage_type = typename base_type::storage_type;

    template<class BinaryFunction = ::rocprim::plus<T>>
    T inclusive_scan(T thread_value, BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        return base_type::inclusive_scan(thread_value, scan_op);
    }

    // Optionally we can add "count" parameter, so scan would be performed
    // only for the first count elements in the block. That gives runtime
    // option for a user to limit scan size.
    template<class BinaryFunction = ::rocprim::plus<T>>
    T inclusive_scan(T thread_value,
                     storage_type& temporary_storage,
                     BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        return base_type::inclusive_scan(
            thread_value, temporary_storage, scan_op
        );
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
