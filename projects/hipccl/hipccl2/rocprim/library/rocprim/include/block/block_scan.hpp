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
#include "detail/block_scan_reduce_then_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_scan primitive.
enum class block_scan_algorithm
{
    using_warp_scan,
    default_algorithm = using_warp_scan,
    reduce_then_scan
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

template<>
struct select_block_scan_impl<block_scan_algorithm::reduce_then_scan>
{
    template<class T, unsigned int BlockSize>
    using type = block_scan_reduce_then_scan<T, BlockSize>;
};

} // end namespace detail

/// \brief Parallel scan primitive for block.
template<
    class T,
    unsigned int BlockSize,
    block_scan_algorithm Algorithm = block_scan_algorithm::default_algorithm
>
class block_scan : private detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>
{
    using base_type = typename detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>;
public:
    using storage_type = typename base_type::storage_type;

    template<class BinaryFunction = ::rocprim::plus<T>>
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, storage, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void inclusive_scan(T input,
                        T& output,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, reduction, scan_op);
    }

    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T input,
                        T& output,
                        PrefixCallback& prefix_callback,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, prefix_callback, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, reduction, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        PrefixCallback& prefix_callback,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::inclusive_scan(input, output, prefix_callback, storage, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    template<class BinaryFunction = ::rocprim::plus<T>>
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, reduction, scan_op);
    }

    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        PrefixCallback& prefix_callback,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, prefix_callback, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, reduction, scan_op);
    }

    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        PrefixCallback& prefix_callback,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction()) [[hc]]
    {
        base_type::exclusive_scan(input, output, init, prefix_callback, storage, scan_op);
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
