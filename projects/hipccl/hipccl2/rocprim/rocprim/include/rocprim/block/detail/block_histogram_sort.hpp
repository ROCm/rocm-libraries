// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    unsigned int ItemsPerThread,
    unsigned int Bins
>
class block_histogram_sort
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static_assert(
        std::is_convertible<T, unsigned int>::value,
        "T must be convertible to unsigned int"
    );

private:
    using radix_sort = block_radix_sort<T, BlockSizeX, ItemsPerThread, empty_type, BlockSizeY, BlockSizeZ>;
    using discontinuity = block_discontinuity<T, BlockSizeX, BlockSizeY, BlockSizeZ>;

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
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->composite(input, hist, storage);
    }

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        // TODO: Check, MSVC rejects the code with the static assertion, yet compiles fine for all tested types. Predicate likely too strict
        //static_assert(
        //    std::is_convertible<unsigned int, Counter>::value,
        //    "unsigned int must be convertible to Counter"
        //);
        constexpr auto tile_size = BlockSize * ItemsPerThread;
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        unsigned int head_flags[ItemsPerThread];
        discontinuity_op flags_op(storage);
        storage_type_& storage_ = storage.get();

        radix_sort().sort(input, storage_.sort);
        ::rocprim::syncthreads(); // Fix race condition that appeared on Vega10 hardware, storage LDS is reused below.

        ROCPRIM_UNROLL
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
        ::rocprim::syncthreads();

        // The start of the first bin is not overwritten since the input is sorted
        // and the starts are based on the second item.
        // The very first item is never used as `b` in the operator
        // This means that this should not need synchromization, but in practice it does.
        if(flat_tid == 0)
        {
            storage_.start[static_cast<unsigned int>(input[0])] = 0;
        }
        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
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

        ROCPRIM_DEVICE ROCPRIM_INLINE
        discontinuity_op(storage_type &storage) : storage(storage)
        {
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        bool operator()(const T& a, const T& b, unsigned int b_index) const
        {
            storage_type_& storage_ = storage.get();
            if(static_cast<unsigned int>(a) != static_cast<unsigned int>(b))
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
