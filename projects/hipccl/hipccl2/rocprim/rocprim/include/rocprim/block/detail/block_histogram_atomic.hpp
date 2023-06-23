// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_ATOMIC_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_ATOMIC_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

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
class block_histogram_atomic
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static_assert(
        std::is_convertible<T, unsigned int>::value,
        "T must be convertible to unsigned int"
    );

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        static_assert(
            std::is_same<Counter, unsigned int>::value || std::is_same<Counter, int>::value ||
            std::is_same<Counter, float>::value || std::is_same<Counter, unsigned long long>::value,
            "Counter must be type that is supported by atomics (float, int, unsigned int, unsigned long long)"
        );
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int bin = static_cast<unsigned int>(input[i]);

            // Get a mask with the threads that have the same value for `bin`.
            ::rocprim::lane_mask_type peer_mask = ballot(1);
            ROCPRIM_UNROLL
            for(unsigned int b = 1; b < Bins; b <<= 1)
            {
                const unsigned int bit_set      = bin & b;
                const auto         bit_set_mask = ballot(bit_set);
                peer_mask &= (bit_set ? bit_set_mask : ~bit_set_mask);
            }

            // The total number of threads in the warp which also have this digit.
            const unsigned int bin_count = bit_count(peer_mask);

            // The number of threads in the warp that have the same digit AND whose lane id is lower
            // than the current thread's.
            const unsigned int peer_digit_prefix = masked_bit_count(peer_mask);

            // Set counter value.
            if(peer_digit_prefix == 0)
            {
                detail::atomic_add(&hist[bin], Counter(bin_count));
            }
        }
        ::rocprim::syncthreads();
    }

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        (void) storage;
        this->composite(input, hist);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_ATOMIC_HPP_
