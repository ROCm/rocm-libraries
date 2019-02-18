/*******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/system/hip/detail/reduce.h>
//
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

BEGIN_NS_THRUST

namespace hip_rocprim {

/// min element

// custom reduce function
/*auto min_op =
    [] __host__ __device__ (int a, int b) -> int
    {
       return a < b ? a : b;
    };*/
    
__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
ItemsIt __host__ __device__
min_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
  ItemsIt ret = first;
#if __THRUST_HAS_HIPRT__
    ItemsIt* ret_ptr = NULL;
    size_t tmp_size = 0;
    const ItemsIt& start_value = first;
    size_t count_values = last - first;
/*    hip_rocprim::throw_on_error(
      rocprim::reduce(nullptr,
                      tmp_size,
                      first,
                      ret_ptr,
                      start_value,
                      count_values,
                      binary_pred),
      "min reduction");*/
    
#else // __THRUST_HAS_HIPRT__
    ret = thrust::min_element(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              binary_pred);
#endif // __THRUST_HAS_HIPRT__

  return ret;
}

template <class Derived,
          class ItemsIt>
ItemsIt __host__ __device__
min_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return hip_rocprim::min_element(policy, first, last, less<value_type>());
}

/// max element

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
ItemsIt __host__ __device__
max_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
  ItemsIt ret = first;
#if __THRUST_HAS_HIPRT__
/*  ret = __extrema::element<__extrema::arg_max_f>(policy,
                                                 first,
                                                 last,
                                                 binary_pred);*/
#else // __THRUST_HAS_HIPRT__
  ret = thrust::max_element(cvt_to_seq(derived_cast(policy)),
                            first,
                            last,
                            binary_pred);
#endif // __THRUST_HAS_HIPRT__
  return ret;
}

template <class Derived,
          class ItemsIt>
ItemsIt __host__ __device__
max_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return hip_rocprim::max_element(policy, first, last, less<value_type>());
}

/// minmax element

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
pair<ItemsIt, ItemsIt> __host__ __device__
minmax_element(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last,
               BinaryPred                 binary_pred)
{
  pair<ItemsIt, ItemsIt> ret = thrust::make_pair(first, first);

#if __THRUST_HAS_HIPRT__
/*  if (first == last)
    return thrust::make_pair(last, last);

  typedef typename iterator_traits<ItemsIt>::value_type      InputType;
  typedef typename iterator_traits<ItemsIt>::difference_type IndexType;

  IndexType num_items = static_cast<IndexType>(thrust::distance(first, last));

  typedef tuple<ItemsIt, counting_iterator_t<IndexType> > iterator_tuple;
  typedef zip_iterator<iterator_tuple> zip_iterator;

  iterator_tuple iter_tuple = make_tuple(first, counting_iterator_t<IndexType>(0));

  typedef __extrema::arg_minmax_f<InputType, IndexType, BinaryPred> arg_minmax_t;
  typedef typename arg_minmax_t::two_pairs_type  two_pairs_type;
  typedef typename arg_minmax_t::duplicate_tuple duplicate_t;
  typedef transform_input_iterator_t<two_pairs_type,
                                     zip_iterator,
                                     duplicate_t>
          transform_t;

  zip_iterator   begin  = make_zip_iterator(iter_tuple);
  two_pairs_type result = __extrema::extrema(policy,
                                             transform_t(begin, duplicate_t()),
                                             num_items,
                                             arg_minmax_t(binary_pred),
                                             (two_pairs_type *)(NULL));
  ret = thrust::make_pair(first + get<1>(get<0>(result)),
                          first + get<1>(get<1>(result))); */
#else // __THRUST_HAS_HIPRT__
  ret = thrust::minmax_element(cvt_to_seq(derived_cast(policy)),
                               first,
                               last,
                               binary_pred);
#endif // __THRUST_HAS_HIPRT__
  return ret;
}

template <class Derived,
          class ItemsIt>
pair<ItemsIt, ItemsIt> __host__ __device__
minmax_element(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return hip_rocprim::minmax_element(policy, first, last, less<value_type>());
}


} // namespace hip_rocprim
END_NS_THRUST
#endif
