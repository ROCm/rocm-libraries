/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

// this system has no special version of this algorithm

/*


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/cuda/config.h>

#include <thrust/system/hip/detail/util.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>

BEGIN_NS_THRUST

template <typename DerivedPolicy,
          typename ForwardIterator1,
          typename ForwardIterator2>
__host__ __device__ thrust::pair<ForwardIterator1, ForwardIterator2>
unique_by_key(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    ForwardIterator1                                            keys_first,
    ForwardIterator1                                            keys_last,
    ForwardIterator2                                            values_first);
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
unique_by_key_copy(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    InputIterator1                                              keys_first,
    InputIterator1                                              keys_last,
    InputIterator2                                              values_first,
    OutputIterator1                                             keys_result,
    OutputIterator2                                             values_result);

namespace hip_rocprim {

// XXX  it should be possible to unify unique & unique_by_key into a single
//      agent with various specializations, similar to what is done
//      with partition
namespace __unique_by_key {


  template <class Policy,
            class KeyInputIt,
            class ValInputIt,
            class KeyOutputIt,
            class ValOutputIt,
            class BinaryPred>
  pair<KeyOutputIt, ValOutputIt> THRUST_RUNTIME_FUNCTION
  unique_by_key(Policy &    policy,
                KeyInputIt  keys_first,
                KeyInputIt  keys_last,
                ValInputIt  values_first,
                KeyOutputIt keys_result,
                ValOutputIt values_result,
                BinaryPred  binary_pred)
  {

    //  typedef typename iterator_traits<KeyInputIt>::difference_type size_type;
    typedef int size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(keys_first, keys_last));
    char *       d_temp_storage     = NULL;
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = hip_rocprim::stream(policy);
    size_type *  d_num_selected_out = NULL;
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = __unique_by_key::doit_step(d_temp_storage,
                                        temp_storage_bytes,
                                        keys_first,
                                        values_first,
                                        keys_result,
                                        values_result,
                                        binary_pred,
                                        d_num_selected_out,
                                        num_items,
                                        stream,
                                        debug_sync);
    hip_rocprim::throw_on_error(status, "unique_by_key: failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;
    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);

    void *ptr = hip_rocprim::get_memory_buffer(policy, storage_size);
    hip_rocprim::throw_on_error(cudaGetLastError(),
                             "unique_by_key: failed to get memory buffer");

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);

    d_num_selected_out = (size_type *)allocations[0];
    d_temp_storage     = (char *)allocations[1];

    status = __unique_by_key::doit_step(d_temp_storage,
                                        temp_storage_bytes,
                                        keys_first,
                                        values_first,
                                        keys_result,
                                        values_result,
                                        binary_pred,
                                        d_num_selected_out,
                                        num_items,
                                        stream,
                                        debug_sync);
    hip_rocprim::throw_on_error(status, "unique_by_key: failed on 2nd step");


    status = hip_rocprim::synchronize(policy);
    hip_rocprim::throw_on_error(status, "unique_by_key: failed to synchronize");

    size_type num_selected = get_value(policy, d_num_selected_out);

    hip_rocprim::return_memory_buffer(policy, ptr);
    hip_rocprim::throw_on_error(cudaGetLastError(),
                             "unique_by_key: failed to return memory buffer");

    return thrust::make_pair(keys_result + num_selected, values_result + num_selected);
  }
} // namespace __unique_by_key

//-------------------------
// Thrust API entry points
//-------------------------


__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt,
          class BinaryPred>
pair<KeyOutputIt, ValOutputIt> __host__ __device__
unique_by_key_copy(execution_policy<Derived> &policy,
                   KeyInputIt                 keys_first,
                   KeyInputIt                 keys_last,
                   ValInputIt                 values_first,
                   KeyOutputIt                keys_result,
                   ValOutputIt                values_result,
                   BinaryPred                 binary_pred)
{
  pair<KeyOutputIt, ValOutputIt> ret = thrust::make_pair(keys_result, values_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __unique_by_key::unique_by_key(policy,
                                         keys_first,
                                         keys_last,
                                         values_first,
                                         keys_result,
                                         values_result,
                                         binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::unique_by_key_copy(cvt_to_seq(derived_cast(policy)),
                                     keys_first,
                                     keys_last,
                                     values_first,
                                     keys_result,
                                     values_result,
                                     binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt>
pair<KeyOutputIt, ValOutputIt> __host__ __device__
unique_by_key_copy(execution_policy<Derived> &policy,
                   KeyInputIt                 keys_first,
                   KeyInputIt                 keys_last,
                   ValInputIt                 values_first,
                   KeyOutputIt                keys_result,
                   ValOutputIt                values_result)
{
  typedef typename iterator_traits<KeyInputIt>::value_type key_type;
  return hip_rocprim::unique_by_key_copy(policy,
                                         keys_first,
                                         keys_last,
                                         values_first,
                                         keys_result,
                                         values_result,
                                         equal_to<key_type>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class BinaryPred>
pair<KeyInputIt, ValInputIt> __host__ __device__
unique_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              BinaryPred                 binary_pred)
{
  pair<KeyInputIt, ValInputIt> ret = thrust::make_pair(keys_first, values_first);
  if (__THRUST_HAS_CUDART__)
  {
    ret = hip_rocprim::unique_by_key_copy(policy,
                                          keys_first,
                                          keys_last,
                                          values_first,
                                          keys_first,
                                          values_first,
                                          binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::unique_by_key(cvt_to_seq(derived_cast(policy)),
                                keys_first,
                                keys_last,
                                values_first,
                                binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt>
pair<KeyInputIt, ValInputIt> __host__ __device__
unique_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first)
{
  typedef typename iterator_traits<KeyInputIt>::value_type key_type;
  return hip_rocprim::unique_by_key(policy,
                                    keys_first,
                                    keys_last,
                                    values_first,
                                    equal_to<key_type>());
}



}    // namespace hip_rocprim
END_NS_THRUST

#include <thrust/memory.h>
#include <thrust/unique.h>

#endif*/
