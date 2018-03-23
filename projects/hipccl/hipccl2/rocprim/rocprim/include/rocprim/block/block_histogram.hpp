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

#ifndef ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
#define ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_histogram_atomic.hpp"
#include "detail/block_histogram_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup blockmodule
/// @{

enum class block_histogram_algorithm
{
    using_atomic,
    using_sort,
    default_algorithm = using_atomic,
};

namespace detail
{

// Selector for block_histogram algorithm which gives block histogram implementation
// type based on passed block_histogram_algorithm enum
template<block_histogram_algorithm Algorithm>
struct select_block_histogram_impl;

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_atomic>
{
    template<class T, unsigned BlockSize, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_atomic<T, BlockSize, ItemsPerThread, Bins>;
};

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_sort>
{
    template<class T, unsigned BlockSize, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_sort<T, BlockSize, ItemsPerThread, Bins>;
};

} // end namespace detail

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Bins,
    block_histogram_algorithm Algorithm = block_histogram_algorithm::default_algorithm
>
class block_histogram
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSize, ItemsPerThread, Bins>
#endif
{
    using base_type = typename detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSize, ItemsPerThread, Bins>;
public:
    using storage_type = typename base_type::storage_type;

    template<class Counter>
    ROCPRIM_DEVICE inline
    void init_histogram(Counter hist[Bins])
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id();

        #pragma unroll
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                hist[offset_tid] = Counter();
            }
        }
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        base_type::composite(input, hist);
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist);
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        base_type::composite(input, hist, storage);
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist, storage);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
