 /******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/hip/config.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator __host__ __device__
unique(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
       ForwardIterator                                             first,
       ForwardIterator                                             last,
       BinaryPredicate                                             binary_pred);

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator __host__ __device__
unique_copy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
            InputIterator                                               first,
            InputIterator                                               last,
            OutputIterator                                              result,
            BinaryPredicate                                             binary_pred);

namespace hip_rocprim
{
namespace __unique
{
    template <class Policy, class ItemsInputIt, class ItemsOutputIt, class BinaryPred>
    ItemsOutputIt THRUST_HIP_RUNTIME_FUNCTION
    unique(Policy&       policy,
           ItemsInputIt  items_first,
           ItemsInputIt  items_last,
           ItemsOutputIt items_result,
           BinaryPred    binary_pred)
    {
        typedef size_t size_type;

        size_type num_items = static_cast<size_type>(thrust::distance(items_first, items_last));
        void*     d_temp_storage       = NULL;
        size_t    temp_storage_bytes   = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        size_type*  d_num_selected_out = NULL;
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return items_result;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::unique(d_temp_storage,
                                                    temp_storage_bytes,
                                                    items_first,
                                                    items_result,
                                                    d_num_selected_out,
                                                    num_items,
                                                    binary_pred,
                                                    stream,
                                                    debug_sync),
                                    "unique failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage
            = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes + sizeof(size_type));
        hip_rocprim::throw_on_error(hipGetLastError(), "unique failed to get memory buffer");

        d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(d_temp_storage) + temp_storage_bytes);

        hip_rocprim::throw_on_error(rocprim::unique(d_temp_storage,
                                                    temp_storage_bytes,
                                                    items_first,
                                                    items_result,
                                                    d_num_selected_out,
                                                    num_items,
                                                    binary_pred,
                                                    stream,
                                                    debug_sync),
                                    "unique failed on 2nd step");

        size_type num_selected = get_value(policy, d_num_selected_out);

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(), "unique failed to return memory buffer");

        return items_result + num_selected;
    }
} // namespace __unique

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class OutputIt,
                                        class BinaryPred>
OutputIt THRUST_HIP_FUNCTION
unique_copy(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            OutputIt                   result,
            BinaryPred                 binary_pred)
{
    OutputIt ret = result;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__unique::unique<execution_policy<Derived>, InputIt, OutputIt, BinaryPred>)
    );
#if __THRUST_HAS_HIPRT__
    ret = __unique::unique(policy, first, last, result, binary_pred);
#else
    ret = thrust::unique_copy(
        cvt_to_seq(derived_cast(policy)), first, last, result, binary_pred
    );
#endif
    return ret;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
unique_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
    typedef typename iterator_traits<InputIt>::value_type input_type;
    return hip_rocprim::unique_copy(policy, first, last, result, equal_to<input_type>());
}

__thrust_exec_check_disable__ template <class Derived, class InputIt, class BinaryPred>
InputIt THRUST_HIP_FUNCTION
unique(execution_policy<Derived>& policy,
       InputIt                    first,
       InputIt                    last,
       BinaryPred                 binary_pred)
{
    InputIt ret = first;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (unique_copy<Derived, InputIt, InputIt, BinaryPred>)
    );
#if __THRUST_HAS_HIPRT__
    ret = hip_rocprim::unique_copy(policy, first, last, first, binary_pred);
#else
    ret = thrust::unique(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);
#endif
    return ret;
}

template <class Derived, class InputIt>
InputIt THRUST_HIP_FUNCTION unique(execution_policy<Derived>& policy,
                                   InputIt                    first,
                                   InputIt                    last)
{
    typedef typename iterator_traits<InputIt>::value_type input_type;
    return hip_rocprim::unique(policy, first, last, equal_to<input_type>());
}

} // namespace hip_rocprim
END_NS_THRUST

#include <thrust/memory.h>
#include <thrust/unique.h>
#endif
