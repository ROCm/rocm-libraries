/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/execution_policy.h>

#include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace __binary_search
{
    template <typename Derived, typename HaystackIt, typename NeedlesIt, typename OutputIt, typename CompareOp>
    THRUST_HIP_RUNTIME_FUNCTION OutputIt
    lower_bound(execution_policy<Derived>& policy,
                HaystackIt                 haystack_begin,
                HaystackIt                 haystack_end,
                NeedlesIt                  needles_begin,
                NeedlesIt                  needles_end,
                OutputIt                   result,
                CompareOp                  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        thrust::detail::wrapped_function<CompareOp, bool> wrapped_op(compare_op);

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::lower_bound(NULL,
                                                         storage_size,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         wrapped_op,
                                                         stream,
                                                         debug_sync),
                                    "lower_bound: failed on 1st call");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        hip_rocprim::throw_on_error(rocprim::lower_bound(ptr,
                                                         storage_size,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         wrapped_op,
                                                         stream,
                                                         debug_sync),
                                    "lower_bound: failed on 2nt call");

        hip_rocprim::throw_on_error(
            hip_rocprim::synchronize_optional(policy),
            "lower_bound: failed to synchronize"
        );

        return result + needles_size;
    }

    template <typename Derived, typename HaystackIt, typename NeedlesIt, typename OutputIt, typename CompareOp>
    THRUST_HIP_RUNTIME_FUNCTION OutputIt
    upper_bound(execution_policy<Derived>& policy,
                HaystackIt                 haystack_begin,
                HaystackIt                 haystack_end,
                NeedlesIt                  needles_begin,
                NeedlesIt                  needles_end,
                OutputIt                   result,
                CompareOp                  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        thrust::detail::wrapped_function<CompareOp, bool> wrapped_op(compare_op);

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::upper_bound(NULL,
                                                         storage_size,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         wrapped_op,
                                                         stream,
                                                         debug_sync),
                                    "upper_bound: failed on 1st call");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        hip_rocprim::throw_on_error(rocprim::upper_bound(ptr,
                                                         storage_size,
                                                         haystack_begin,
                                                         needles_begin,
                                                         result,
                                                         haystack_size,
                                                         needles_size,
                                                         wrapped_op,
                                                         stream,
                                                         debug_sync),
                                    "upper_bound: failed on 2nt call");

        hip_rocprim::throw_on_error(
            hip_rocprim::synchronize_optional(policy),
            "upper_bound: failed to synchronize"
        );

        return result + needles_size;
    }

    template <typename Derived, typename HaystackIt, typename NeedlesIt, typename OutputIt, typename CompareOp>
    THRUST_HIP_RUNTIME_FUNCTION OutputIt
    binary_search(execution_policy<Derived>& policy,
                  HaystackIt                 haystack_begin,
                  HaystackIt                 haystack_end,
                  NeedlesIt                  needles_begin,
                  NeedlesIt                  needles_end,
                  OutputIt                   result,
                  CompareOp                  compare_op)
    {
        using size_type = typename iterator_traits<NeedlesIt>::difference_type;

        const size_type needles_size  = thrust::distance(needles_begin, needles_end);
        const size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

        if(needles_size == 0)
            return result;

        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        thrust::detail::wrapped_function<CompareOp, bool> wrapped_op(compare_op);

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::binary_search(NULL,
                                                           storage_size,
                                                           haystack_begin,
                                                           needles_begin,
                                                           result,
                                                           haystack_size,
                                                           needles_size,
                                                           wrapped_op,
                                                           stream,
                                                           debug_sync),
                                    "binary_search: failed on 1st call");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        hip_rocprim::throw_on_error(rocprim::binary_search(ptr,
                                                           storage_size,
                                                           haystack_begin,
                                                           needles_begin,
                                                           result,
                                                           haystack_size,
                                                           needles_size,
                                                           wrapped_op,
                                                           stream,
                                                           debug_sync),
                                    "binary_search: failed on 2nt call");

        hip_rocprim::throw_on_error(
            hip_rocprim::synchronize_optional(policy),
            "binary_search: failed to synchronize"
        );

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
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__ static OutputIt par(execution_policy<Derived>& policy,
                                     HaystackIt                 first,
                                     HaystackIt                 last,
                                     NeedlesIt                  values_first,
                                     NeedlesIt                  values_last,
                                     OutputIt                   result,
                                     CompareOp                  compare_op)
        {
            return __binary_search::lower_bound(
                policy, first, last, values_first, values_last, result, compare_op);
        }

        __device__ static OutputIt seq(execution_policy<Derived>& policy,
                                       HaystackIt                 first,
                                       HaystackIt                 last,
                                       NeedlesIt                  values_first,
                                       NeedlesIt                  values_last,
                                       OutputIt                   result,
                                       CompareOp                  compare_op)
        {
            return thrust::lower_bound(cvt_to_seq(derived_cast(policy)),
                                       first,
                                       last,
                                       values_first,
                                       values_last,
                                       result,
                                       compare_op);
        }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, last, values_first, values_last, result, compare_op);
  #else
    return workaround::seq(policy, first, last, values_first, values_last, result, compare_op);
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
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__ static OutputIt par(execution_policy<Derived>& policy,
                                     HaystackIt                 first,
                                     HaystackIt                 last,
                                     NeedlesIt                  values_first,
                                     NeedlesIt                  values_last,
                                     OutputIt                   result,
                                     CompareOp                  compare_op)
        {
            return __binary_search::upper_bound(
                policy, first, last, values_first, values_last, result, compare_op);
        }

        __device__ static OutputIt seq(execution_policy<Derived>& policy,
                                       HaystackIt                 first,
                                       HaystackIt                 last,
                                       NeedlesIt                  values_first,
                                       NeedlesIt                  values_last,
                                       OutputIt                   result,
                                       CompareOp                  compare_op)
        {
            return thrust::upper_bound(cvt_to_seq(derived_cast(policy)),
                                       first,
                                       last,
                                       values_first,
                                       values_last,
                                       result,
                                       compare_op);
        }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, last, values_first, values_last, result, compare_op);
  #else
    return workaround::seq(policy, first, last, values_first, values_last, result, compare_op);
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
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__ static OutputIt par(execution_policy<Derived>& policy,
                                     HaystackIt                 first,
                                     HaystackIt                 last,
                                     NeedlesIt                  values_first,
                                     NeedlesIt                  values_last,
                                     OutputIt                   result,
                                     CompareOp                  compare_op)
        {
            return __binary_search::binary_search(
                policy, first, last, values_first, values_last, result, compare_op);
        }

        __device__ static OutputIt seq(execution_policy<Derived>& policy,
                                       HaystackIt                 first,
                                       HaystackIt                 last,
                                       NeedlesIt                  values_first,
                                       NeedlesIt                  values_last,
                                       OutputIt                   result,
                                       CompareOp                  compare_op)
        {
            return thrust::binary_search(cvt_to_seq(derived_cast(policy)),
                                         first,
                                         last,
                                         values_first,
                                         values_last,
                                         result,
                                         compare_op);
        }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, last, values_first, values_last, result, compare_op);
  #else
    return workaround::seq(policy, first, last, values_first, values_last, result, compare_op);
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

    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static HaystackIt par(execution_policy<Derived>& policy,
                              HaystackIt                 first,
                              HaystackIt                 last,
                              const T&                   value,
                              CompareOp                  compare_op)
        {
            values_type  values(policy, 1);
            results_type result(policy, 1);

            {
                typedef typename thrust::iterator_system<const T*>::type value_in_system_t;
                value_in_system_t                                        value_in_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(
                        thrust::detail::derived_cast(thrust::detail::strip_const(value_in_system)),
                        thrust::detail::derived_cast(thrust::detail::strip_const(policy))),
                    &value,
                    1,
                    values.begin());
            }

            __binary_search::lower_bound(
                policy, first, last, values.begin(), values.end(), result.begin(), compare_op);

            difference_type h_result;
            {
                typedef
                    typename thrust::iterator_system<difference_type*>::type result_out_system_t;
                result_out_system_t                                          result_out_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(thrust::detail::derived_cast(thrust::detail::strip_const(policy)),
                                  thrust::detail::derived_cast(
                                      thrust::detail::strip_const(result_out_system))),
                    result.begin(),
                    1,
                    &h_result);
            }

            return first + h_result;
        }

        __device__
        static HaystackIt seq(execution_policy<Derived>& policy,
                            HaystackIt                 first,
                            HaystackIt                 last,
                            const T&                   value,
                            CompareOp                  compare_op)
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
        }
    };

    #if __THRUST_HAS_HIPRT__
      return workaround::par(policy, first, last, value, compare_op);
    #else
      return workaround::seq(policy, first, last, value, compare_op);
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

    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static HaystackIt par(execution_policy<Derived>& policy,
                              HaystackIt                 first,
                              HaystackIt                 last,
                              const T&                   value,
                              CompareOp                  compare_op)
        {
          values_type values(policy, 1);
          results_type result(policy, 1);

          {
                typedef typename thrust::iterator_system<const T*>::type value_in_system_t;
                value_in_system_t                                        value_in_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(
                        thrust::detail::derived_cast(thrust::detail::strip_const(value_in_system)),
                        thrust::detail::derived_cast(thrust::detail::strip_const(policy))),
                    &value,
                    1,
                    values.begin());
          }

          __binary_search::upper_bound(
              policy, first, last, values.begin(), values.end(), result.begin(), compare_op
          );

          difference_type h_result;
          {
                typedef
                    typename thrust::iterator_system<difference_type*>::type result_out_system_t;
                result_out_system_t                                          result_out_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(thrust::detail::derived_cast(thrust::detail::strip_const(policy)),
                                  thrust::detail::derived_cast(
                                      thrust::detail::strip_const(result_out_system))),
                    result.begin(),
                    1,
                    &h_result);
          }

          return first + h_result;
        }

        __device__
        static HaystackIt seq(execution_policy<Derived>& policy,
                            HaystackIt                 first,
                            HaystackIt                 last,
                            const T&                   value,
                            CompareOp                  compare_op)
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
        }
    };

    #if __THRUST_HAS_HIPRT__
      return workaround::par(policy, first, last, value, compare_op);
    #else
      return workaround::seq(policy, first, last, value, compare_op);
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

    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static bool par(execution_policy<Derived>& policy,
                              HaystackIt                 first,
                              HaystackIt                 last,
                              const T&                   value,
                              CompareOp                  compare_op)
        {
          values_type values(policy, 1);
          results_type result(policy, 1);

          {
                typedef typename thrust::iterator_system<const T*>::type value_in_system_t;
                value_in_system_t                                        value_in_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(
                        thrust::detail::derived_cast(thrust::detail::strip_const(value_in_system)),
                        thrust::detail::derived_cast(thrust::detail::strip_const(policy))),
                    &value,
                    1,
                    values.begin());
          }

          __binary_search::binary_search(
              policy, first, last, values.begin(), values.end(), result.begin(), compare_op
          );

          int h_result;
          {
                typedef typename thrust::iterator_system<int*>::type result_out_system_t;
                result_out_system_t                                  result_out_system;
                using thrust::system::detail::generic::select_system;
                thrust::copy_n(
                    select_system(thrust::detail::derived_cast(thrust::detail::strip_const(policy)),
                                  thrust::detail::derived_cast(
                                      thrust::detail::strip_const(result_out_system))),
                    result.begin(),
                    1,
                    &h_result);
          }

          return h_result != 0;
        }

        __device__
        static bool seq(execution_policy<Derived>& policy,
                            HaystackIt                 first,
                            HaystackIt                 last,
                            const T&                   value,
                            CompareOp                  compare_op)
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
        }
    };

    #if __THRUST_HAS_HIPRT__
      return workaround::par(policy, first, last, value, compare_op);
    #else
      return workaround::seq(policy, first, last, value, compare_op);
    #endif
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#endif
