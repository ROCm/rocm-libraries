// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include _THRUST_STD_INCLUDE(__algorithm/min.h)
#include _THRUST_STD_INCLUDE(tuple)

// Contributed by Erich Elsen

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator, typename T>
THRUST_HOST_DEVICE InputIterator
find(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, const T& value)
{
  using thrust::placeholders::_1;

  return thrust::find_if(exec, first, last, _1 == value);
} // end find()

template <typename TupleType>
struct find_if_functor
{
  THRUST_HOST_DEVICE TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (_THRUST_STD::get<0>(lhs) && _THRUST_STD::get<0>(rhs))
    {
      return TupleType(true, (_THRUST_STD::min)(_THRUST_STD::get<1>(lhs), _THRUST_STD::get<1>(rhs)));
    }
    else if (_THRUST_STD::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
THRUST_HOST_DEVICE InputIterator
find_if(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  using difference_type = thrust::detail::it_difference_t<InputIterator>;
  using result_type     = typename _THRUST_STD::tuple<bool, difference_type>;

  // empty sequence
  if (first == last)
  {
    return last;
  }

  const difference_type n = _THRUST_STD::distance(first, last);

  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found

  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const difference_type interval_threshold = 1 << 20;
  const difference_type interval_size      = (_THRUST_STD::min)(interval_threshold, n);

  // force transform_iterator output to bool
  using XfrmIterator  = thrust::transform_iterator<Predicate, InputIterator, bool>;
  using IteratorTuple = _THRUST_STD::tuple<XfrmIterator, thrust::counting_iterator<difference_type>>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  IteratorTuple iter_tuple =
    _THRUST_STD::make_tuple(XfrmIterator(first, pred), thrust::counting_iterator<difference_type>(0));

  ZipIterator begin = thrust::make_zip_iterator(iter_tuple);
  ZipIterator end   = begin + n;

  for (ZipIterator interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    ZipIterator interval_end = interval_begin + interval_size;
    if (end < interval_end)
    {
      interval_end = end;
    } // end if

    result_type result = thrust::reduce(
      exec, interval_begin, interval_end, result_type(false, interval_end - begin), find_if_functor<result_type>());

    // see if we found something
    if (_THRUST_STD::get<0>(result))
    {
      return first + _THRUST_STD::get<1>(result);
    }
  }

  // nothing was found if we reach here...
  return first + n;
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
THRUST_HOST_DEVICE InputIterator
find_if_not(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, _THRUST_STD::not_fn(pred));
} // end find()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
