/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_

#include "../../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

namespace internal
{

template<typename AccumType, typename InputType, typename ReductionOp, typename PrefixType>
[[nodiscard]]
__device__ __forceinline__
AccumType
    ThreadReduceSequential(const InputType& input, ReductionOp reduction_op, PrefixType prefix)
{
    AccumType     retval = static_cast<AccumType>(prefix);
    constexpr int length = ::hipcub::detail::static_size_v<InputType>();
#pragma unroll
    for(int i = 0; i < length; ++i)
    {
        retval = reduction_op(retval, input[i]);
    }
    return retval;
}

template<typename AccumType, typename InputType, typename ReductionOp>
[[nodiscard]]
__device__ __forceinline__
AccumType ThreadReduceSequential(const InputType& input, ReductionOp reduction_op)
{
    AccumType     retval = input[0];
    constexpr int length = ::hipcub::detail::static_size_v<InputType>();
#pragma unroll
    for(int i = 1; i < length; ++i)
    {
        retval = reduction_op(retval, input[i]);
    }
    return retval;
}

// This function is not used, it is just here for compatibility
template<typename AccumType, typename InputType, typename ReductionOp>
[[nodiscard]] [[deprecated("This function is an internal API, using it directly is not "
                           "recommended.")]]
__device__ __forceinline__
AccumType ThreadReduceBinaryTree(const InputType& input, ReductionOp reduction_op)
{
    constexpr auto length = ::hipcub::detail::static_size_v<InputType>();
#pragma unroll
    for(int i = 1; i < length; i *= 2)
    {
#pragma unroll
        for(int j = 0; j + i < length; j += i * 2)
        {
            input[j] = reduction_op(input[j], input[j + i]);
        }
    }
    return input[0];
}

template<typename AccumType, typename InputType, typename ReductionOp>
[[nodiscard]] [[deprecated("This function is an internal API, using it directly is not "
                           "recommended.")]]
__device__ __forceinline__
AccumType ThreadReduceTernaryTree(const InputType& input, ReductionOp reduction_op)
{
    constexpr auto length = ::hipcub::detail::static_size_v<InputType>();
#pragma unroll
    for(int i = 1; i < length; i *= 3)
    {
#pragma unroll
        for(int j = 0; j + i < length; j += i * 3)
        {
            auto value = reduction_op(input[j], input[j + i]);
            input[j]   = (j + i * 2 < length) ? reduction_op(value, input[j + i * 2]) : value;
        }
    }
    return input[0];
}

// TODO: we should also implement ThreadReduceSimd after simd intrinsics are available.

} // namespace internal

template<typename InputType,
         typename ReductionOp,
         typename ValueType = typename ::std::remove_cv<typename ::std::remove_reference<
             decltype(::std::declval<InputType>()[0])>::type>::type,
         typename AccumType = ::rocprim::accumulator_t<ReductionOp, ValueType>>
[[nodiscard]]
__device__ __forceinline__
AccumType ThreadReduce(const InputType& input, ReductionOp reduction_op)
{
    static_assert(::hipcub::detail::is_fixed_size_random_access_range<InputType>::value,
                  "InputType must support the subscript operator[] and have a compile-time size");
    static_assert(std::is_invocable<ReductionOp, ValueType, ValueType>::value,
                  "ReductionOp must be invocable with operator()(ValueType, ValueType)");
    return ::hipcub::internal::ThreadReduceSequential<AccumType>(input, reduction_op);
}

template<typename InputType,
         typename ReductionOp,
         typename PrefixType,
         typename ValueType = typename ::std::remove_cv<typename ::std::remove_reference<
             decltype(::std::declval<InputType>()[0])>::type>::type,
         typename AccumType = ::rocprim::accumulator_t<ReductionOp, ValueType, PrefixType>>
[[nodiscard]]
__device__ __forceinline__
AccumType ThreadReduce(const InputType& input, ReductionOp reduction_op, PrefixType prefix)
{
    static_assert(::hipcub::detail::is_fixed_size_random_access_range<InputType>::value,
                  "InputType must support the subscript operator[] and have a compile-time size");
    static_assert(std::is_invocable<ReductionOp, ValueType, ValueType>::value,
                  "ReductionOp must be invocable with operator()(ValueType, ValueType)");
    constexpr auto length = ::hipcub::detail::static_size_v<InputType>();
    return ::hipcub::internal::ThreadReduceSequential<AccumType>(input, reduction_op, prefix);
}

template<int Length,
         typename T,
         typename ReductionOp,
         typename AccumType = ::rocprim::accumulator_t<ReductionOp, T>>
[[nodiscard]]
__device__ __forceinline__
AccumType ThreadReduce(const T* input, ReductionOp reduction_op)
{
    static_assert(Length > 0, "Length must be greater than 0");
    static_assert(std::is_invocable<ReductionOp, T, T>::value,
                  "ReductionOp must have the binary call operator: operator(V1, V2)");
    auto array = reinterpret_cast<const T(*)[Length]>(input);
    return ::hipcub::ThreadReduce(*array, reduction_op);
}

template<int Length,
         typename T,
         typename ReductionOp,
         typename PrefixType,
         typename AccumType = ::rocprim::accumulator_t<ReductionOp, T, PrefixType>,
         typename std::enable_if<(Length > 0), int>::type = 0>
[[nodiscard]]
__device__ __forceinline__
AccumType ThreadReduce(const T* input, ReductionOp reduction_op, PrefixType prefix)
{
    static_assert(std::is_invocable<ReductionOp, T, T>::value,
                  "ReductionOp must have the binary call operator: operator(V1, V2)");
    auto array = reinterpret_cast<const T(*)[Length]>(input);
    return ::hipcub::ThreadReduce(*array, reduction_op, prefix);
}

template<int Length,
         typename T,
         typename ReductionOp,
         typename PrefixType,
         typename std::enable_if<(Length == 0), int>::type = 0>
[[nodiscard]]
__device__ __forceinline__
T ThreadReduce(const T*, ReductionOp, PrefixType prefix)
{
    return prefix;
}

END_HIPCUB_NAMESPACE

#endif
