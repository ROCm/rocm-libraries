/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_

#include <type_traits>

#include <rocprim/block/block_reduce.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_reduce_algorithm>::type
    to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm v)
    {
        using utype = std::underlying_type<::rocprim::block_reduce_algorithm>::type;
        return static_cast<utype>(v);
    }
}

enum BlockReduceAlgorithm
{
    BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::raking_reduce),
    BLOCK_REDUCE_RAKING
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::raking_reduce),
    BLOCK_REDUCE_WARP_REDUCTIONS
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::using_warp_reduce)
};

template<
    typename T,
    int BLOCK_DIM_X,
    BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH /* ignored */
>
class BlockReduce
    : private ::rocprim::block_reduce<
        T,
        BLOCK_DIM_X,
        static_cast<::rocprim::block_reduce_algorithm>(ALGORITHM),
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_reduce<
            T,
            BLOCK_DIM_X,
            static_cast<::rocprim::block_reduce_algorithm>(ALGORITHM),
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockReduce() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockReduce(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    T Sum(T input)
    {
        base_type::reduce(input, input, temp_storage_);
        return input;
    }

    HIPCUB_DEVICE inline
    T Sum(T input, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_);
        return input;
    }

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    T Sum(T(&input)[ITEMS_PER_THREAD])
    {
        T output;
        base_type::reduce(input, output, temp_storage_);
        return output;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T input, ReduceOp reduce_op)
    {
        base_type::reduce(input, input, temp_storage_, reduce_op);
        return input;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T input, ReduceOp reduce_op, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_, reduce_op);
        return input;
    }

    template<int ITEMS_PER_THREAD, typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T(&input)[ITEMS_PER_THREAD], ReduceOp reduce_op)
    {
        T output;
        base_type::reduce(input, output, temp_storage_, reduce_op);
        return output;
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

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
