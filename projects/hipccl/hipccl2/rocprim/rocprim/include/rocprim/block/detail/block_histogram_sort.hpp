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
    static_assert(
        std::is_convertible<T, unsigned int>::value,
        "T must be convertible to unsigned int"
    );

private:
    using radix_sort = block_radix_sort<T, BlockSize, ItemsPerThread>;
    using discontinuity = block_discontinuity<T, BlockSize>;

public:
    union storage_type_
    {
        typename radix_sort::storage_type sort;
        struct
        {
            typename discontinuity::storage_type flag;
            unsigned int start[Bins];
            unsigned int end[Bins];
        };
    };

    using storage_type = detail::raw_storage<storage_type_>;

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->composite(input, hist, storage);
    }

    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        static_assert(
            std::is_convertible<unsigned int, Counter>::value,
            "unsigned int must be convertible to Counter"
        );
        constexpr auto tile_size = BlockSize * ItemsPerThread;
        const auto flat_tid = ::rocprim::flat_block_thread_id();
        unsigned int head_flags[ItemsPerThread];
        discontinuity_op flags_op(storage);
        storage_type_& storage_ = storage.get();

        radix_sort().sort(input, storage_.sort);
        ::rocprim::syncthreads(); // Fix race condition that appeared on Vega10 hardware, storage LDS is reused below.

        #pragma unroll
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                storage_.start[offset_tid] = tile_size;
                storage_.end[offset_tid] = tile_size;
            }
        }
        ::rocprim::syncthreads();

        discontinuity().flag_heads(head_flags, input, flags_op, storage_.flag);
        
        // ::rocprim::syncthreads() isn't required here as input is sorted by this point
        // and it's impossible that flags_op will be called where b = input[0] and a != b
        if(flat_tid == 0)
        {
            storage_.start[static_cast<unsigned int>(input[0])] = 0;
        }
        ::rocprim::syncthreads();

        #pragma unroll
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                Counter count = static_cast<Counter>(storage_.end[offset_tid] - storage_.start[offset_tid]);
                hist[offset_tid] += count;
            }
        }
    }

private:
    struct discontinuity_op
    {
        storage_type &storage;

        ROCPRIM_DEVICE inline
        discontinuity_op(storage_type &storage) : storage(storage)
        {
        }

        ROCPRIM_DEVICE inline
        bool operator()(const T& a, const T& b, unsigned int b_index) const
        {
            storage_type_& storage_ = storage.get();
            if(a != b)
            {
                storage_.start[static_cast<unsigned int>(b)] = b_index;
                storage_.end[static_cast<unsigned int>(a)] = b_index;
                return true;
            }
            else
            {
                return false;
            }
        }
    };
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_
