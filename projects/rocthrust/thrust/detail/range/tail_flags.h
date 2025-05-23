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

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence_access.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename RandomAccessIterator,
          typename BinaryPredicate = thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator>::type>,
          typename ValueType       = bool,
          typename IndexType       = typename thrust::iterator_difference<RandomAccessIterator>::type>
class tail_flags
{
  // XXX WAR cudafe bug
  // private:

public:
  struct tail_flag_functor
  {
    BinaryPredicate binary_pred; // this must be the first member for performance reasons
    RandomAccessIterator iter;
    IndexType n;

    using result_type = ValueType;

    THRUST_HOST_DEVICE tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last)
        : binary_pred()
        , iter(first)
        , n(last - first)
    {}

    THRUST_HOST_DEVICE
    tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
        : binary_pred(binary_pred)
        , iter(first)
        , n(last - first)
    {}

    THRUST_HOST_DEVICE THRUST_FORCEINLINE result_type operator()(const IndexType& i)
    {
      return (i == (n - 1) || !binary_pred(iter[i], iter[i + 1]));
    }
  };

  using counting_iterator = thrust::counting_iterator<IndexType>;

public:
  using iterator = thrust::transform_iterator<tail_flag_functor, counting_iterator>;

  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE tail_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(
        thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), tail_flag_functor(first, last)))
      , m_end(m_begin + (last - first))
  {}

  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(thrust::make_transform_iterator(
        thrust::counting_iterator<IndexType>(0), tail_flag_functor(first, last, binary_pred)))
      , m_end(m_begin + (last - first))
  {}

  THRUST_HOST_DEVICE iterator begin() const
  {
    return m_begin;
  }

  THRUST_HOST_DEVICE iterator end() const
  {
    return m_end;
  }

  template <typename OtherIndex>
  THRUST_HOST_DEVICE typename iterator::reference operator[](OtherIndex i)
  {
    return *(begin() + i);
  }

  THRUST_SYNTHESIZE_SEQUENCE_ACCESS(tail_flags, iterator);

private:
  iterator m_begin, m_end;
};

template <typename RandomAccessIterator, typename BinaryPredicate>
THRUST_HOST_DEVICE tail_flags<RandomAccessIterator, BinaryPredicate>
make_tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return tail_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}

template <typename RandomAccessIterator>
THRUST_HOST_DEVICE tail_flags<RandomAccessIterator>
make_tail_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return tail_flags<RandomAccessIterator>(first, last);
}

} // namespace detail
THRUST_NAMESPACE_END
