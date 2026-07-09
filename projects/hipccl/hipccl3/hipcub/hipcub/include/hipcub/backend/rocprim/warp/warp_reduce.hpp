/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2026, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_WARP_WARP_REDUCE_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_REDUCE_HPP_

#include "../../../config.hpp"

#include "../thread/thread_reduce.hpp"
#include "../util_ptx.hpp"

#include <rocprim/warp/warp_reduce.hpp> // IWYU pragma: export

#include _HIPCUB_LIBCXX_INCLUDE(functional)
#include _HIPCUB_STD_INCLUDE(functional)

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
constexpr auto sum_reduce_op = _HIPCUB_STD::plus<>{};

constexpr auto max_reduce_op =
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
    _HIPCUB_LIBCXX::maximum<>{}
#else
    [](auto a, auto b) { return a > b ? a : b; }
#endif
;

constexpr auto min_reduce_op =
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
    _HIPCUB_LIBCXX::minimum<>{}
#else
    [](auto a, auto b) { return a > b ? b : a; }
#endif
;
} // namespace detail

template<typename T, int LogicalWarpThreads = HIPCUB_DEVICE_WARP_THREADS>
class WarpReduce : private ::rocprim::warp_reduce<T, LogicalWarpThreads>
{
    static_assert(LogicalWarpThreads >= 1 && LogicalWarpThreads <= HIPCUB_DEVICE_WARP_THREADS,
                  "LogicalWarpThreads must be in the range [1, HIPCUB_DEVICE_WARP_THREADS]");
    using base_type = typename ::rocprim::warp_reduce<T, LogicalWarpThreads>;

    typename base_type::storage_type &temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    WarpReduce(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    T Sum(T input)
    {
        base_type::reduce(input, input, temp_storage_);
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Sum(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::sum_reduce_op);
        base_type::reduce(thread_reduction, thread_reduction, temp_storage_);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Sum(T input, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_);
        return input;
    }

    HIPCUB_DEVICE
    inline T Max(T input)
    {
        base_type::reduce(input, input, temp_storage_, detail::max_reduce_op);
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Max(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::max_reduce_op);
        base_type::reduce(thread_reduction, thread_reduction, temp_storage_, detail::max_reduce_op);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Max(T input, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_, detail::max_reduce_op);
        return input;
    }

    HIPCUB_DEVICE
    inline T Min(T input)
    {
        base_type::reduce(input, input, temp_storage_, detail::min_reduce_op);
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Min(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::min_reduce_op);
        base_type::reduce(thread_reduction, thread_reduction, temp_storage_, detail::min_reduce_op);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Min(T input, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_, detail::min_reduce_op);
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE inline
    T HeadSegmentedSum(T input, FlagT head_flag)
    {
        base_type::head_segmented_reduce(input, input, head_flag, temp_storage_);
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE inline
    T TailSegmentedSum(T input, FlagT tail_flag)
    {
        base_type::tail_segmented_reduce(input, input, tail_flag, temp_storage_);
        return input;
    }

    template<typename InputT,
             typename ReduceOp,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Reduce(const InputT& input, ReduceOp reduce_op)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, reduce_op);
        base_type::reduce(thread_reduction, thread_reduction, temp_storage_, reduce_op);
        return thread_reduction;
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

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE inline
    T HeadSegmentedReduce(T input, FlagT head_flag, ReduceOp reduce_op)
    {
        base_type::head_segmented_reduce(
            input, input, head_flag, temp_storage_, reduce_op
        );
        return input;
    }

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE inline
    T TailSegmentedReduce(T input, FlagT tail_flag, ReduceOp reduce_op)
    {
        base_type::tail_segmented_reduce(
            input, input, tail_flag, temp_storage_, reduce_op
        );
        return input;
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
template<typename T>
class WarpReduce<T, 1>
{
private:
    using _TempStorage = hipcub::NullType;

public:
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    HIPCUB_DEVICE
    inline WarpReduce(TempStorage& /*temp_storage*/)
    {}

    HIPCUB_DEVICE
    inline T Sum(T input)
    {
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Sum(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::sum_reduce_op);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Sum(T input, int /*valid_items*/)
    {
        return input;
    }

    HIPCUB_DEVICE
    inline T Max(T input)
    {
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Max(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::max_reduce_op);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Max(T input, int /*valid_items*/)
    {
        return input;
    }

    HIPCUB_DEVICE
    inline T Min(T input)
    {
        return input;
    }

    template<typename InputT,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Min(const InputT& input)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, detail::min_reduce_op);
        return thread_reduction;
    }

    HIPCUB_DEVICE
    inline T Min(T input, int /*valid_items*/)
    {
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE
    inline T HeadSegmentedSum(T input, FlagT /*head_flag*/)
    {
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE
    inline T TailSegmentedSum(T input, FlagT /*tail_flag*/)
    {
        return input;
    }

    template<typename InputT,
             typename ReduceOp,
             HIPCUB_REQUIRES(HIPCUB_TRAIT(detail::is_fixed_size_random_access_range, InputT))>
    HIPCUB_DEVICE
    inline T Reduce(const InputT& input, ReduceOp reduce_op)
    {
        auto thread_reduction = HIPCUB_NAMESPACE::ThreadReduce(input, reduce_op);
        return thread_reduction;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE
    inline T Reduce(T input, ReduceOp /*reduce_op*/)
    {
        return input;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE
    inline T Reduce(T input, ReduceOp /*reduce_op*/, int /*valid_items*/)
    {
        return input;
    }

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE
    inline T HeadSegmentedReduce(T input, FlagT /*head_flag*/, ReduceOp /*reduce_op*/)
    {
        return input;
    }

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE
    inline T TailSegmentedReduce(T input, FlagT /*tail_flag*/, ReduceOp /*reduce_op*/)
    {
        return input;
    }
};

#endif

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_REDUCE_HPP_
