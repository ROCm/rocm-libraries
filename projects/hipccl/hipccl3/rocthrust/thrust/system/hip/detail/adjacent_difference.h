/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/system/hip/config.h>

#include <thrust/detail/temporary_array.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/transform.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>

// rocprim include
#include <rocprim/rocprim.hpp>

#include <cstdint>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
THRUST_HOST_DEVICE OutputIterator
adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                    InputIterator                                               first,
                    InputIterator                                               last,
                    OutputIterator                                              result,
                    BinaryFunction                                              binary_op);

namespace hip_rocprim
{
namespace __adjacent_difference
{
    template <typename InputIt, typename OutputIt, typename BinaryOp>
    hipError_t THRUST_HIP_RUNTIME_FUNCTION __adjecent_difference(
        void* const       temporary_storage,
        size_t&           storage_size,
        const InputIt     input,
        const OutputIt    output,
        const size_t      size,
        const BinaryOp    op,
        const hipStream_t stream,
        const bool        debug_synchronous,
        thrust::detail::integral_constant<bool, true> /* iterators are comparable */)
    {
        if(input != output)
        {
            return rocprim::adjacent_difference(temporary_storage,
                                                      storage_size,
                                                      input,
                                                      output,
                                                      size,
                                                      op,
                                                      stream,
                                                      debug_synchronous);
        }
        else
        {
            return rocprim::adjacent_difference_inplace(temporary_storage,
                                                        storage_size,
                                                        input,
                                                        output,
                                                        size,
                                                        op,
                                                        stream,
                                                        debug_synchronous);
        }
    }

    template <typename InputIt, typename OutputIt, typename BinaryOp>
    hipError_t THRUST_HIP_RUNTIME_FUNCTION __adjecent_difference(
        void* const       temporary_storage,
        size_t&           storage_size,
        const InputIt     input,
        const OutputIt    output,
        const size_t      size,
        const BinaryOp    op,
        const hipStream_t stream,
        const bool        debug_synchronous,
        thrust::detail::integral_constant<bool, false> /* iterators are not comparable */)
    {
        return rocprim::adjacent_difference_inplace(
            temporary_storage, storage_size, input, output, size, op, stream, debug_synchronous);
    }

    template <typename Derived, typename InputIt, typename OutputIt, typename BinaryOp>
    static OutputIt THRUST_HIP_RUNTIME_FUNCTION
    adjacent_difference(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt                   result,
                        BinaryOp                   binary_op)
    {
        using size_type = typename iterator_traits<InputIt>::difference_type;

        size_type   num_items    = thrust::distance(first, last);
        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items <= 0)
        {
            return result;
        }

        // Check if iterators can be compared
        using unwrap_input_iterator  = thrust::try_unwrap_contiguous_iterator_t<InputIt>;
        using unwrap_output_iterator = thrust::try_unwrap_contiguous_iterator_t<OutputIt>;

        using input_value_type  = thrust::iterator_value_t<unwrap_input_iterator>;
        using output_value_type = thrust::iterator_value_t<unwrap_output_iterator>;

        constexpr bool can_compare_iterators
            = std::is_pointer<unwrap_input_iterator>::value
              && std::is_pointer<unwrap_output_iterator>::value
              && std::is_same<input_value_type, output_value_type>::value;

        // Unwrap iterators to make them comparable
        auto first_unwrap  = thrust::try_unwrap_contiguous_iterator(first);
        auto result_unwrap = thrust::try_unwrap_contiguous_iterator(result);

        thrust::detail::integral_constant<bool, can_compare_iterators> comparable;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(__adjecent_difference(nullptr,
                                                            storage_size,
                                                            first_unwrap,
                                                            result_unwrap,
                                                            static_cast<size_t>(num_items),
                                                            binary_op,
                                                            stream,
                                                            debug_sync,
                                                            comparable),
                                    "adjacent_difference failed on 1st step");

        // Allocate temporary storage.
        thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy,
                                                                                storage_size);
        void* ptr = static_cast<void*>(tmp.data().get());

        hip_rocprim::throw_on_error(__adjecent_difference(ptr,
                                                            storage_size,
                                                            first_unwrap,
                                                            result_unwrap,
                                                            static_cast<size_t>(num_items),
                                                            binary_op,
                                                            stream,
                                                            debug_sync,
                                                            comparable),
                                    "adjacent_difference failed on 2nd step");

        hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(policy));

        return result + num_items;
    }

} // namespace __adjacent_difference

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class OutputIt, class BinaryOp>
OutputIt THRUST_HIP_FUNCTION
adjacent_difference(execution_policy<Derived>& policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result,
                    BinaryOp                   binary_op)
{
    // struct workaround is required for HIP-clang
    struct workaround
    {
        THRUST_HOST
        static void par(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt&                  result,
                        BinaryOp                   binary_op)
        {
            result = __adjacent_difference::adjacent_difference(
                policy, first, last, result, binary_op);
        }
        THRUST_DEVICE
        static void seq(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt&                  result,
                        BinaryOp                   binary_op)
        {
            result = thrust::adjacent_difference(
               cvt_to_seq(derived_cast(policy)),
               first,
               last,
               result,
               binary_op
            );
        }
    };
    #if __THRUST_HAS_HIPRT__
    workaround::par(policy, first, last, result, binary_op);
    #else
    workaround::seq(policy, first, last, result, binary_op);
    #endif

    return result;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
adjacent_difference(execution_policy<Derived>& policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result)
{
    using input_type = typename iterator_traits<InputIt>::value_type;
    return hip_rocprim::adjacent_difference(policy, first, last, result, minus<input_type>());
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

//
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
