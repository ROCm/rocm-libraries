/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST

// XXX declare generic copy_if interface
// to avoid circulular dependency from thrust/copy.h
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator __host__ __device__
copy_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
        InputIterator                                               first,
        InputIterator                                               last,
        OutputIterator                                              result,
        Predicate                                                   pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator __host__ __device__
copy_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
        InputIterator1                                              first,
        InputIterator1                                              last,
        InputIterator2                                              stencil,
        OutputIterator                                              result,
        Predicate                                                   pred);

namespace hip_rocprim
{
namespace __copy_if
{
    template <class Policy, class InputIt, class OutputIt, class Predicate>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    copy_if(Policy& policy, InputIt first, InputIt last, OutputIt output, Predicate predicate)
    {
        typedef typename iterator_traits<InputIt>::difference_type size_type;

        size_type   num_items          = thrust::distance(first, last);
        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        size_type*  d_num_selected_out = NULL;
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return output;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::select(d_temp_storage,
                                                    temp_storage_bytes,
                                                    first,
                                                    output,
                                                    d_num_selected_out,
                                                    num_items,
                                                    predicate,
                                                    stream,
                                                    debug_sync),
                                    "copy_if failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage
            = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes + sizeof(size_type));
        hip_rocprim::throw_on_error(hipGetLastError(), "copy_if failed to get memory buffer");

        d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(d_temp_storage) + temp_storage_bytes);

        hip_rocprim::throw_on_error(rocprim::select(d_temp_storage,
                                                    temp_storage_bytes,
                                                    first,
                                                    output,
                                                    d_num_selected_out,
                                                    num_items,
                                                    predicate,
                                                    stream,
                                                    debug_sync),
                                    "copy_if failed on 2nd step");

        size_type num_selected = get_value(policy, d_num_selected_out);

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "copy_if failed to return memory buffer");

        return output + num_selected;
    }

    template <class Policy, class InputIt, class StencilIt, class OutputIt, class Predicate>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    copy_if(Policy&   policy,
            InputIt   first,
            InputIt   last,
            StencilIt stencil,
            OutputIt  output,
            Predicate predicate)
    {
        typedef typename iterator_traits<InputIt>::difference_type size_type;

        size_type   num_items          = thrust::distance(first, last);
        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        size_type*  d_num_selected_out = NULL;
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return output;

        auto flags = thrust::make_transform_iterator(stencil, predicate);

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::select(d_temp_storage,
                                                    temp_storage_bytes,
                                                    first,
                                                    flags,
                                                    output,
                                                    d_num_selected_out,
                                                    num_items,
                                                    stream,
                                                    debug_sync),
                                    "copy_if failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage
            = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes + sizeof(size_type));
        hip_rocprim::throw_on_error(hipGetLastError(), "copy_if failed to get memory buffer");

        d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(d_temp_storage) + temp_storage_bytes);

        hip_rocprim::throw_on_error(rocprim::select(d_temp_storage,
                                                    temp_storage_bytes,
                                                    first,
                                                    flags,
                                                    output,
                                                    d_num_selected_out,
                                                    num_items,
                                                    stream,
                                                    debug_sync),
                                    "copy_if failed on 2nd step");

        size_type num_selected = get_value(policy, d_num_selected_out);

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "copy_if failed to return memory buffer");

        return output + num_selected;
    }

} // namespace __copy_if

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIterator, class OutputIterator, class Predicate>
OutputIterator THRUST_HIP_FUNCTION
copy_if(execution_policy<Derived>& policy,
        InputIterator              first,
        InputIterator              last,
        OutputIterator             result,
        Predicate                  pred)
{
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__copy_if::copy_if<execution_policy<Derived>,
                            InputIterator,
                            OutputIterator,
                            Predicate>)
    );
#if __THRUST_HAS_HIPRT__
    return __copy_if::copy_if(policy, first, last, result, pred);
#else
    return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, result, pred);
#endif
} // func copy_if

template <class Derived,
          class InputIterator,
          class StencilIterator,
          class OutputIterator,
          class Predicate>
OutputIterator THRUST_HIP_FUNCTION
copy_if(execution_policy<Derived>& policy,
        InputIterator              first,
        InputIterator              last,
        StencilIterator            stencil,
        OutputIterator             result,
        Predicate                  pred)
{
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__copy_if::copy_if<execution_policy<Derived>,
                           InputIterator,
                           StencilIterator,
                           OutputIterator,
                           Predicate>)
    );
#if __THRUST_HAS_HIPRT__
    return __copy_if::copy_if(policy, first, last, stencil, result, pred);
#else
    return thrust::copy_if(
        cvt_to_seq(derived_cast(policy)), first, last, stencil, result, pred);
#endif
} // func copy_if

} // namespace hip_rocprim
END_NS_THRUST

#include <thrust/copy.h>
#endif
