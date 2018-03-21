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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../block_radix_sort.hpp"
#include "../block_discontinuity.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Bins
>
class block_histogram_sort
{
private:
    using radix_sort = block_radix_sort<T, BlockSize, ItemsPerThread>;
    using discontinuity = block_discontinuity<T, BlockSize>;

public:
    union storage_type
    {
        typename radix_sort::storage_type sort;
        struct
        {
            typename discontinuity::storage_type flag;
            unsigned int start[Bins];
            unsigned int end[Bins];
        };
    };

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter (&hist)[Bins])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->composite(input, hist, storage);
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter (&hist)[Bins],
                   storage_type& storage)
    {
        constexpr auto tile_size = BlockSize * ItemsPerThread;
        constexpr bool check = (Bins % BlockSize == 0);
        const auto flat_tid = ::rocprim::flat_block_thread_id();
        unsigned int head_flags[ItemsPerThread];
        unsigned int offset = 0;
        discontinuity_op flags_op(storage);

        radix_sort().sort(input, storage.sort);
        ::rocprim::syncthreads();

        #pragma unroll
        for(offset = 0; offset + BlockSize <= Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            storage.start[offset_tid] = tile_size;
            storage.end[offset_tid] = tile_size;
        }

        if((offset + flat_tid < Bins) && check)
        {
            const unsigned int offset_tid = offset + flat_tid;
            storage.start[offset_tid] = tile_size;
            storage.end[offset_tid] = tile_size;
        }
        ::rocprim::syncthreads();

        discontinuity().flag_heads(head_flags, input, flags_op, storage.flag);

        if(flat_tid == 0)
        {
            storage.start[input[0]] = 0;
        }
        ::rocprim::syncthreads();

        offset = 0;

        #pragma unroll
        for(offset = 0; offset + BlockSize <= Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            Counter count = static_cast<Counter>(storage.end[offset_tid] - storage.start[offset_tid]);
            hist[offset_tid] += count;
        }

        if((offset + flat_tid < Bins) && check)
        {
            const unsigned int offset_tid = offset + flat_tid;
            Counter count = static_cast<Counter>(storage.end[offset_tid] - storage.start[offset_tid]);
            hist[offset_tid] += count;
        }
        ::rocprim::syncthreads();
    }

private:
    struct discontinuity_op
    {
        storage_type &storage;

        ROCPRIM_HOST_DEVICE inline
        discontinuity_op(storage_type &storage) : storage(storage)
        {
        }

        ROCPRIM_HOST_DEVICE inline
        bool test(const T& a, const T& b, unsigned int b_index) const
        {
            storage.start[b] = b_index;
            storage.end[a] = b_index;
            return true;
        }

        ROCPRIM_HOST_DEVICE inline
        constexpr bool operator()(const T& a, const T& b, unsigned int b_index) const
        {
            return (a != b) ? test(a, b, b_index) : false;
        }
    };
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_
