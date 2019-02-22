// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/hip/execution_policy.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>

#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST
namespace hip_rocprim {

namespace __binary_search {

template <class Policy,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
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

  if (needles_size == 0)
    return result;

  void*       d_temp_storage     = nullptr;
  size_t      temp_storage_bytes = 0;
  hipStream_t stream             = hip_rocprim::stream(policy);
  bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;
  status = rocprim::lower_bound(d_temp_storage,
                                temp_storage_bytes,
                                haystack_begin,
                                needles_begin,
                                result,
                                haystack_size,
                                needles_size,
                                compare_op,
                                stream,
                                debug_sync);
  hip_rocprim::throw_on_error(status, "lower_bound: failed on 1st call");

  d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
  hip_rocprim::throw_on_error(hipGetLastError(), "lower_bound: failed to get memory buffer");

  status = rocprim::lower_bound(d_temp_storage,
                                temp_storage_bytes,
                                haystack_begin,
                                needles_begin,
                                result,
                                haystack_size,
                                needles_size,
                                compare_op,
                                stream,
                                debug_sync);
  hip_rocprim::throw_on_error(status, "lower_bound: failed on 2nt call");

  status = hip_rocprim::synchronize(policy);
  hip_rocprim::throw_on_error(status, "lower_bound: failed to synchronize");

  hip_rocprim::return_memory_buffer(policy, d_temp_storage);
  hip_rocprim::throw_on_error(hipGetLastError(), "lower_bound: failed to return memory buffer");

  return result + needles_size;
}

template <class Policy,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
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

  if (needles_size == 0)
    return result;

  void*       d_temp_storage     = nullptr;
  size_t      temp_storage_bytes = 0;
  hipStream_t stream             = hip_rocprim::stream(policy);
  bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;
  status = rocprim::upper_bound(d_temp_storage,
                                temp_storage_bytes,
                                haystack_begin,
                                needles_begin,
                                result,
                                haystack_size,
                                needles_size,
                                compare_op,
                                stream,
                                debug_sync);
  hip_rocprim::throw_on_error(status, "upper_bound: failed on 1st call");

  d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
  hip_rocprim::throw_on_error(hipGetLastError(), "upper_bound: failed to get memory buffer");

  status = rocprim::upper_bound(d_temp_storage,
                                temp_storage_bytes,
                                haystack_begin,
                                needles_begin,
                                result,
                                haystack_size,
                                needles_size,
                                compare_op,
                                stream,
                                debug_sync);
  hip_rocprim::throw_on_error(status, "upper_bound: failed on 2nt call");

  status = hip_rocprim::synchronize(policy);
  hip_rocprim::throw_on_error(status, "upper_bound: failed to synchronize");

  hip_rocprim::return_memory_buffer(policy, d_temp_storage);
  hip_rocprim::throw_on_error(hipGetLastError(), "upper_bound: failed to return memory buffer");

  return result + needles_size;
}

template <class Policy,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
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

  if (needles_size == 0)
    return result;

  void*       d_temp_storage     = nullptr;
  size_t      temp_storage_bytes = 0;
  hipStream_t stream             = hip_rocprim::stream(policy);
  bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;
  status = rocprim::binary_search(d_temp_storage,
                                  temp_storage_bytes,
                                  haystack_begin,
                                  needles_begin,
                                  result,
                                  haystack_size,
                                  needles_size,
                                  compare_op,
                                  stream,
                                  debug_sync);
  hip_rocprim::throw_on_error(status, "binary_search: failed on 1st call");

  d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
  hip_rocprim::throw_on_error(hipGetLastError(), "binary_search: failed to get memory buffer");

  status = rocprim::binary_search(d_temp_storage,
                                  temp_storage_bytes,
                                  haystack_begin,
                                  needles_begin,
                                  result,
                                  haystack_size,
                                  needles_size,
                                  compare_op,
                                  stream,
                                  debug_sync);
  hip_rocprim::throw_on_error(status, "binary_search: failed on 2nt call");

  status = hip_rocprim::synchronize(policy);
  hip_rocprim::throw_on_error(status, "binary_search: failed to synchronize");

  hip_rocprim::return_memory_buffer(policy, d_temp_storage);
  hip_rocprim::throw_on_error(hipGetLastError(), "binary_search: failed to return memory buffer");

  return result + needles_size;
}

} // namespace __binary_search

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
OutputIt THRUST_HIP_FUNCTION
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result,
            CompareOp                  compare_op)
{
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    __binary_search::lower_bound<
      execution_policy<Derived>, HaystackIt, NeedlesIt, OutputIt, CompareOp
    >
  ));
#if __THRUST_HAS_HIPRT__
  return __binary_search::lower_bound(policy,
                                      first,
                                      last,
                                      values_first,
                                      values_last,
                                      result,
                                      compare_op);
#else
  return thrust::lower_bound(cvt_to_seq(derived_cast(policy)),
                             first,
                             last,
                             values_first,
                             values_last,
                             result,
                             compare_op);
#endif
}

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt>
OutputIt THRUST_HIP_FUNCTION
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result)
{
  return hip_rocprim::lower_bound(policy,
                                  first,
                                  last,
                                  values_first,
                                  values_last,
                                  result,
                                  rocprim::less<>());
}

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
OutputIt THRUST_HIP_FUNCTION
upper_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result,
            CompareOp                  compare_op)
{
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    __binary_search::upper_bound<
      execution_policy<Derived>, HaystackIt, NeedlesIt, OutputIt, CompareOp
    >
  ));
#if __THRUST_HAS_HIPRT__
  return __binary_search::upper_bound(policy,
                                      first,
                                      last,
                                      values_first,
                                      values_last,
                                      result,
                                      compare_op);
#else
  return thrust::upper_bound(cvt_to_seq(derived_cast(policy)),
                             first,
                             last,
                             values_first,
                             values_last,
                             result,
                             compare_op);
#endif
}

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt>
OutputIt THRUST_HIP_FUNCTION
upper_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result)
{
  return hip_rocprim::upper_bound(policy,
                                  first,
                                  last,
                                  values_first,
                                  values_last,
                                  result,
                                  rocprim::less<>());
}

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
OutputIt THRUST_HIP_FUNCTION
binary_search(execution_policy<Derived>& policy,
              HaystackIt                 first,
              HaystackIt                 last,
              NeedlesIt                  values_first,
              NeedlesIt                  values_last,
              OutputIt                   result,
              CompareOp                  compare_op)
{
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    __binary_search::binary_search<
      execution_policy<Derived>, HaystackIt, NeedlesIt, OutputIt, CompareOp
    >
  ));
#if __THRUST_HAS_HIPRT__
  return __binary_search::binary_search(policy,
                                        first,
                                        last,
                                        values_first,
                                        values_last,
                                        result,
                                        compare_op);
#else
  return thrust::binary_search(cvt_to_seq(derived_cast(policy)),
                               first,
                               last,
                               values_first,
                               values_last,
                               result,
                               compare_op);
#endif
}

template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt>
OutputIt THRUST_HIP_FUNCTION
binary_search(execution_policy<Derived>& policy,
              HaystackIt                 first,
              HaystackIt                 last,
              NeedlesIt                  values_first,
              NeedlesIt                  values_last,
              OutputIt                   result)
{
  return hip_rocprim::binary_search(policy,
                                    first,
                                    last,
                                    values_first,
                                    values_last,
                                    result,
                                    rocprim::less<>());
}

} // namespace hip_rocprim
END_NS_THRUST

#endif
