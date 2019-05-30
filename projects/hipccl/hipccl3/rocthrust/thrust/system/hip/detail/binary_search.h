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
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/execution_policy.h>

#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST
namespace hip_rocprim
{
namespace __binary_search
{
    template <class Policy, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    lower_bound(Policy&    policy,
                HaystackIt haystack_begin,
                HaystackIt haystack_end,
                NeedlesIt  needles_begin,
                NeedlesIt  needles_end,
                OutputIt   result,
                CompareOp  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        void*       d_temp_storage     = nullptr;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::lower_bound(d_temp_storage,
                                                         temp_storage_bytes,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         compare_op,
                                                         stream,
                                                         debug_sync),
                                    "lower_bound: failed on 1st call");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "lower_bound: failed to get memory buffer");

        hip_rocprim::throw_on_error(rocprim::lower_bound(d_temp_storage,
                                                         temp_storage_bytes,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         compare_op,
                                                         stream,
                                                         debug_sync),
                                    "lower_bound: failed on 2nt call");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "lower_bound: failed to return memory buffer");

        return result + needles_size;
    }

    template <class Policy, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    upper_bound(Policy&    policy,
                HaystackIt haystack_begin,
                HaystackIt haystack_end,
                NeedlesIt  needles_begin,
                NeedlesIt  needles_end,
                OutputIt   result,
                CompareOp  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        void*       d_temp_storage     = nullptr;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::upper_bound(d_temp_storage,
                                                         temp_storage_bytes,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         compare_op,
                                                         stream,
                                                         debug_sync),
                                    "upper_bound: failed on 1st call");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "upper_bound: failed to get memory buffer");

        hip_rocprim::throw_on_error(rocprim::upper_bound(d_temp_storage,
                                                         temp_storage_bytes,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         compare_op,
                                                         stream,
                                                         debug_sync),
                                    "upper_bound: failed on 2nt call");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "upper_bound: failed to return memory buffer");

        return result + needles_size;
    }

    template <class Policy, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    binary_search(Policy&    policy,
                  HaystackIt haystack_begin,
                  HaystackIt haystack_end,
                  NeedlesIt  needles_begin,
                  NeedlesIt  needles_end,
                  OutputIt   result,
                  CompareOp  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        void*       d_temp_storage     = nullptr;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::binary_search(d_temp_storage,
                                                           temp_storage_bytes,
                                                           haystack_begin,
                                                           needles_begin,
                                                           result,
                                                           haystack_size,
                                                           needles_size,
                                                           compare_op,
                                                           stream,
                                                           debug_sync),
                                    "binary_search: failed on 1st call");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "binary_search: failed to get memory buffer");

        hip_rocprim::throw_on_error(rocprim::binary_search(d_temp_storage,
                                                           temp_storage_bytes,
                                                           haystack_begin,
                                                           needles_begin,
                                                           result,
                                                           haystack_size,
                                                           needles_size,
                                                           compare_op,
                                                           stream,
                                                           debug_sync),
                                    "binary_search: failed on 2nt call");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "binary_search: failed to return memory buffer");

        return result + needles_size;
    }

} // namespace __binary_search

//-------------------------
// Thrust API entry points
//-------------------------

// Vector functions

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result,
            CompareOp                  compare_op)
{
    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::lower_bound<execution_policy<Derived>,
                                          HaystackIt,
                                          NeedlesIt,
                                          OutputIt,
                                          CompareOp>)
        );
        return first;
#else
        return __binary_search::lower_bound(
            policy, first, last, values_first, values_last, result, compare_op
        );
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        return thrust::lower_bound(cvt_to_seq(derived_cast(policy)),
                                   first,
                                   last,
                                   values_first,
                                   values_last,
                                   result,
                                   compare_op);
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result)
{
    return hip_rocprim::lower_bound(
        policy, first, last, values_first, values_last, result, rocprim::less<>()
    );
}

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
upper_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result,
            CompareOp                  compare_op)
{
    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::upper_bound<execution_policy<Derived>,
                                          HaystackIt,
                                          NeedlesIt,
                                          OutputIt,
                                          CompareOp>)
        );
        return first;
#else
        return __binary_search::upper_bound(
            policy, first, last, values_first, values_last, result, compare_op
        );
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        return thrust::upper_bound(cvt_to_seq(derived_cast(policy)),
                                   first,
                                   last,
                                   values_first,
                                   values_last,
                                   result,
                                   compare_op);
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
upper_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result)
{
    return hip_rocprim::upper_bound(
        policy, first, last, values_first, values_last, result, rocprim::less<>()
    );
}

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
binary_search(execution_policy<Derived>& policy,
              HaystackIt                 first,
              HaystackIt                 last,
              NeedlesIt                  values_first,
              NeedlesIt                  values_last,
              OutputIt                   result,
              CompareOp                  compare_op)
{
    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::binary_search<execution_policy<Derived>,
                                            HaystackIt,
                                            NeedlesIt,
                                            OutputIt,
                                            CompareOp>)
        );
        return first;
#else
        return __binary_search::binary_search(
            policy, first, last, values_first, values_last, result, compare_op
        );
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        return thrust::binary_search(cvt_to_seq(derived_cast(policy)),
                                     first,
                                     last,
                                     values_first,
                                     values_last,
                                     result,
                                     compare_op);
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

template <class Derived, class HaystackIt, class NeedlesIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
binary_search(execution_policy<Derived>& policy,
              HaystackIt                 first,
              HaystackIt                 last,
              NeedlesIt                  values_first,
              NeedlesIt                  values_last,
              OutputIt                   result)
{
    return hip_rocprim::binary_search(
        policy, first, last, values_first, values_last, result, rocprim::less<>()
    );
}

// Scalar functions

// We use these custom implementations instead of thrust/system/detail/generic/binary_search.inl
// because HIP support of device-side memory allocation is under development
// (it is used in generic/binary_search.inl by thrust::detail::temporary_array).

template<typename Derived, typename HaystackIt, typename T, typename CompareOp>
THRUST_HIP_FUNCTION
HaystackIt lower_bound(execution_policy<Derived>& policy,
                       HaystackIt                 first,
                       HaystackIt                 last,
                       const T&                   value,
                       CompareOp                  compare_op)
{
    using difference_type = typename thrust::iterator_traits<HaystackIt>::difference_type;
    using values_type = typename thrust::detail::temporary_array<T, Derived>;
    using results_type = typename thrust::detail::temporary_array<difference_type, Derived>;

    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::lower_bound<execution_policy<Derived>,
                                          HaystackIt,
                                          typename values_type::iterator,
                                          typename results_type::iterator,
                                          CompareOp>)
        );
        return first;
#else
        values_type values(policy, 1);
        results_type result(policy, 1);

        values[0] = value;

        __binary_search::lower_bound(
            policy, first, last, values.begin(), values.end(), result.begin(), compare_op
        );

        return first + result[0];
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        difference_type result;
        thrust::lower_bound(cvt_to_seq(derived_cast(policy)),
                            first,
                            last,
                            &value,
                            &value + 1,
                            &result,
                            compare_op);
        return first + result;
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

template<typename Derived, typename HaystackIt, typename T, typename CompareOp>
THRUST_HIP_FUNCTION
HaystackIt upper_bound(execution_policy<Derived>& policy,
                       HaystackIt                 first,
                       HaystackIt                 last,
                       const T&                   value,
                       CompareOp                  compare_op)
{
    using difference_type = typename thrust::iterator_traits<HaystackIt>::difference_type;
    using values_type = typename thrust::detail::temporary_array<T, Derived>;
    using results_type = typename thrust::detail::temporary_array<difference_type, Derived>;

    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::upper_bound<execution_policy<Derived>,
                                          HaystackIt,
                                          typename values_type::iterator,
                                          typename results_type::iterator,
                                          CompareOp>)
        );
        return first;
#else
        values_type values(policy, 1);
        results_type result(policy, 1);

        values[0] = value;

        __binary_search::upper_bound(
            policy, first, last, values.begin(), values.end(), result.begin(), compare_op
        );

        return first + result[0];
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        difference_type result;
        thrust::upper_bound(cvt_to_seq(derived_cast(policy)),
                            first,
                            last,
                            &value,
                            &value + 1,
                            &result,
                            compare_op);
        return first + result;
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

template<typename Derived, typename HaystackIt, typename T, typename CompareOp>
THRUST_HIP_FUNCTION
bool binary_search(execution_policy<Derived>& policy,
                   HaystackIt                 first,
                   HaystackIt                 last,
                   const T&                   value,
                   CompareOp                  compare_op)
{
    using values_type = typename thrust::detail::temporary_array<T, Derived>;
    using results_type = typename thrust::detail::temporary_array<int, Derived>;

    auto workaround_par = [=]() mutable
    {
#if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__binary_search::binary_search<execution_policy<Derived>,
                                            HaystackIt,
                                            typename values_type::iterator,
                                            typename results_type::iterator,
                                            CompareOp>)
        );
        return first;
#else
        values_type values(policy, 1);
        results_type result(policy, 1);

        values[0] = value;

        __binary_search::binary_search(
            policy, first, last, values.begin(), values.end(), result.begin(), compare_op
        );

        return result[0] != 0;
#endif
    };
    (void)workaround_par;

    auto workaround_seq = [=]() mutable
    {
        bool result;
        thrust::binary_search(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              &value,
                              &value + 1,
                              &result,
                              compare_op);
        return result;
    };
    (void)workaround_seq;

#if __THRUST_HAS_HIPRT__
    return workaround_par();
#else
    return workaround_seq();
#endif
}

} // namespace hip_rocprim
END_NS_THRUST

#endif
