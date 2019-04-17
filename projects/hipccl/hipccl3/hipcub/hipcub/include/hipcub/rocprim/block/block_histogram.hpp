/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_

#include <type_traits>

#include "../../config.hpp"

#include "../thread/thread_operators.hpp"

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_histogram_algorithm>::type
    to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm v)
    {
        using utype = std::underlying_type<::rocprim::block_histogram_algorithm>::type;
        return static_cast<utype>(v);
    }
}

enum BlockHistogramAlgorithm
{
    BLOCK_HISTO_ATOMIC
        = detail::to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm::using_atomic),
    BLOCK_HISTO_SORT
        = detail::to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm::using_sort)
};

template<
    typename T,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    int BINS,
    BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_SORT,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH /* ignored */
>
class BlockHistogram
    : private ::rocprim::block_histogram<
        T,
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
        ITEMS_PER_THREAD,
        BINS,
        static_cast<::rocprim::block_histogram_algorithm>(ALGORITHM)
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_histogram<
            T,
            BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
            ITEMS_PER_THREAD,
            BINS,
            static_cast<::rocprim::block_histogram_algorithm>(ALGORITHM)
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockHistogram() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockHistogram(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void InitHistogram(CounterT histogram[BINS])
    {
        base_type::init_histogram(histogram);
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void Composite(T (&items)[ITEMS_PER_THREAD],
                   CounterT histogram[BINS])
    {
        base_type::composite(items, histogram, temp_storage_);
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void Histogram(T (&items)[ITEMS_PER_THREAD],
                   CounterT histogram[BINS])
    {
        base_type::init_histogram(histogram);
        CTA_SYNC();
        base_type::composite(items, histogram, temp_storage_);
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
