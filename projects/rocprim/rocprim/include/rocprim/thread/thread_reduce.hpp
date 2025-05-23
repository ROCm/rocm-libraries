/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_THREAD_THREAD_REDUCE_HPP_
#define ROCPRIM_THREAD_THREAD_REDUCE_HPP_

#include "../config.hpp"

#include <variant>

BEGIN_ROCPRIM_NAMESPACE

/// \defgroup thread_reduce Thread Reduce Functions
/// \ingroup threadmodule

/// \addtogroup thread_reduce
/// @{

/// \brief Carry out a reduction on an array of elements in one thread
/// \tparam Length Length of the array to be reduced
/// \tparam T the input/output type
/// \tparam ReductionOp Binary Operation that used to carry out the reduction
/// \param input [in] Pointer to the first element of the array to be reduced
/// \param reduction_op [in] Instance of the reduction operator functor
/// \param prefix [in] Optional value to be used as prefix
/// \return Value obtained from reduction of input array
template<int Length, typename T, typename ReductionOp, typename Prefix = std::monostate>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto thread_reduce(T* input, ReductionOp reduction_op, Prefix prefix = {})
{
    T retval = input[0];

    if constexpr(std::is_same_v<Prefix, std::monostate>)
    {
        retval = input[0];
    }
    else
    {
        retval = prefix;
    }

    ROCPRIM_UNROLL
    for(int i = 1; i < Length; ++i)
    {
        retval = reduction_op(retval, input[i]);
    }

    return retval;
}

/// \brief Carry out a reduction on an array of elements in one thread
/// \tparam Length Length of the array to be reduced
/// \tparam T the input/output type
/// \tparam ReductionOp Binary Operation that used to carry out the reduction
/// \param input [in] Pointer to the first element of the array to be reduced
/// \param reduction_op [in] Instance of the reduction operator functor
/// \param prefix [in] Optional value to be used as prefix
/// \return Value obtained from reduction of input array
template<int Length, typename T, typename ReductionOp, typename Prefix = std::monostate>
ROCPRIM_DEVICE ROCPRIM_INLINE
T thread_reduce(T (&input)[Length], ReductionOp reduction_op, Prefix prefix = {})
{
    return thread_reduce<Length>(static_cast<T*>(input), reduction_op, prefix);
}

/// @}
// end of group thread_reduce

END_ROCPRIM_NAMESPACE

#endif
