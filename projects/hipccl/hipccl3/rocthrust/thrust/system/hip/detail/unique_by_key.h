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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/hip/config.h>

#include <thrust/system/hip/detail/util.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>
#include <thrust/detail/range/head_flags.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST

template <typename DerivedPolicy,
          typename ForwardIterator1,
          typename ForwardIterator2>
THRUST_HIP_FUNCTION thrust::pair<ForwardIterator1, ForwardIterator2>
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
THRUST_HIP_FUNCTION thrust::pair<OutputIterator1, OutputIterator2>
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

  template <class KeyType,
            class ValueType,
            class Predicate>
  struct predicate_wrapper
  {
      Predicate predicate;
      typedef rocprim::tuple<KeyType, ValueType> pair_type;

      THRUST_HIP_FUNCTION
      predicate_wrapper(Predicate p) : predicate(p) {}

      bool THRUST_HIP_DEVICE_FUNCTION
      operator()(pair_type const &lhs, pair_type const &rhs) const
      {
          return predicate(rocprim::get<0>(lhs), rocprim::get<0>(rhs));
      }
  };    // struct predicate_wrapper


  template <class Policy,
            class KeyInputIt,
            class ValInputIt,
            class KeyOutputIt,
            class ValOutputIt,
            class BinaryPred>
  pair<KeyOutputIt, ValOutputIt> THRUST_HIP_RUNTIME_FUNCTION
  unique_by_key(Policy &    policy,
                KeyInputIt  keys_first,
                KeyInputIt  keys_last,
                ValInputIt  values_first,
                KeyOutputIt keys_result,
                ValOutputIt values_result,
                BinaryPred  binary_pred)
  {
    typedef size_t size_type;

    typedef typename iterator_traits<KeyInputIt>::value_type KeyType;
    typedef typename iterator_traits<ValInputIt>::value_type ValueType;

    predicate_wrapper<KeyType, ValueType, BinaryPred> wrapped_binary_pred(binary_pred);

    size_type    num_items          = static_cast<size_type>(thrust::distance(keys_first, keys_last));
    void *       d_temp_storage     = NULL;
    size_t       temp_storage_bytes = 0;
    hipStream_t  stream             = hip_rocprim::stream(policy);
    size_type *  d_num_selected_out = NULL;
    bool         debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

    if (num_items == 0)
      return thrust::make_pair(keys_result, values_result);



    hipError_t status;
    status = rocprim::unique(d_temp_storage,
                             temp_storage_bytes,
                             rocprim::make_zip_iterator(rocprim::make_tuple(keys_first, values_first)),
                             rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, values_result)),
                             d_num_selected_out,
                             num_items,
                             wrapped_binary_pred,
                             stream,
                             debug_sync);
    hip_rocprim::throw_on_error(status, "unique_by_key failed on 1st step");

    temp_storage_bytes = rocprim::detail::align_size(temp_storage_bytes);
    d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes + sizeof(size_type));
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "unique_by_key failed to get memory buffer");

    d_num_selected_out = reinterpret_cast<size_type *>(
      reinterpret_cast<char *>(d_temp_storage) + temp_storage_bytes);

    status = rocprim::unique(d_temp_storage,
                             temp_storage_bytes,
                             rocprim::make_zip_iterator(rocprim::make_tuple(keys_first, values_first)),
                             rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, values_result)),
                             d_num_selected_out,
                             num_items,
                             wrapped_binary_pred,
                             stream,
                             debug_sync);
    hip_rocprim::throw_on_error(status, "unique_by_key failed on 2nd step");

    size_type num_selected = get_value(policy, d_num_selected_out);

    hip_rocprim::return_memory_buffer(policy, d_temp_storage);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "unique_by_key failed to return memory buffer");

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
pair<KeyOutputIt, ValOutputIt> THRUST_HIP_FUNCTION
unique_by_key_copy(execution_policy<Derived> &policy,
                   KeyInputIt                 keys_first,
                   KeyInputIt                 keys_last,
                   ValInputIt                 values_first,
                   KeyOutputIt                keys_result,
                   ValOutputIt                values_result,
                   BinaryPred                 binary_pred)
{
  pair<KeyOutputIt, ValOutputIt> ret = thrust::make_pair(keys_result, values_result);
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
      __unique_by_key::unique_by_key<Derived, KeyInputIt, ValInputIt, KeyOutputIt, ValOutputIt, BinaryPred>
  ));
#if __THRUST_HAS_HIPRT__
    ret = __unique_by_key::unique_by_key(policy,
                                         keys_first,
                                         keys_last,
                                         values_first,
                                         keys_result,
                                         values_result,
                                         binary_pred);
#else
    ret = thrust::unique_by_key_copy(cvt_to_seq(derived_cast(policy)),
                                     keys_first,
                                     keys_last,
                                     values_first,
                                     keys_result,
                                     values_result,
                                     binary_pred);
#endif
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt>
pair<KeyOutputIt, ValOutputIt> THRUST_HIP_FUNCTION
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
pair<KeyInputIt, ValInputIt> THRUST_HIP_FUNCTION
unique_by_key(execution_policy<Derived> &policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              BinaryPred                 binary_pred)
{
  pair<KeyInputIt, ValInputIt> ret = thrust::make_pair(keys_first, values_first);
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
      hip_rocprim::unique_by_key_copy<Derived, KeyInputIt, ValInputIt, KeyInputIt, ValInputIt>
  ));
#if __THRUST_HAS_HIPRT__
    ret = hip_rocprim::unique_by_key_copy(policy,
                                          keys_first,
                                          keys_last,
                                          values_first,
                                          keys_first,
                                          values_first,
                                          binary_pred);
#else
    ret = thrust::unique_by_key(cvt_to_seq(derived_cast(policy)),
                                keys_first,
                                keys_last,
                                values_first,
                                binary_pred);
#endif
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt>
pair<KeyInputIt, ValInputIt> THRUST_HIP_FUNCTION
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

#endif
