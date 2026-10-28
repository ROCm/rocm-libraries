/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2026, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_HAS_HIP_COMPILER()
#  include <thrust/system/hip/config.h>

#  include <thrust/distance.h>
#  include <thrust/iterator/counting_iterator.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/type_traits/unwrap_contiguous_iterator.h>
#  if !_THRUST_HAS_DEVICE_SYSTEM_STD
#    include <thrust/detail/algorithm_wrapper.h>

#    include <iterator>
#  endif

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

// XXX forward declare to circumvent circular dependency
template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class T>
InputIt THRUST_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value);

}; // namespace hip_rocprim
THRUST_NAMESPACE_END

#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/hip/detail/reduce.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

namespace __find_if
{

template <typename TupleType>
struct functor
{
  THRUST_HIP_DEVICE_FUNCTION TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (thrust::get<0>(lhs) && thrust::get<0>(rhs))
    {
      return TupleType(true, (_THRUST_STD::min)(thrust::get<1>(lhs), thrust::get<1>(rhs)));
    }
    else if (thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <class ValueType, class InputIt, class UnaryOp>
struct transform_input_iterator_t
{
  using self_t            = transform_input_iterator_t;
  using difference_type   = thrust::detail::it_difference_t<InputIt>;
  using value_type        = ValueType;
  using pointer           = void;
  using reference         = value_type;
  using iterator_category = _THRUST_STD::random_access_iterator_tag;

  InputIt input;
  mutable UnaryOp op;

  THRUST_HOST_DEVICE THRUST_FORCEINLINE transform_input_iterator_t(InputIt input, UnaryOp op)
      : input(input)
      , op(op)
  {}

  transform_input_iterator_t(const self_t&) = default;

  // UnaryOp might not be copy assignable, such as when it is a lambda.  Define
  // an explicit copy assignment operator that doesn't try to assign it.
  THRUST_HOST_DEVICE self_t& operator=(const self_t& o)
  {
    input = o.input;
    return *this;
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t operator++(int)
  {
    self_t retval = *this;
    ++input;
    return retval;
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t& operator++()
  {
    ++input;
    return *this;
  }

  // Pass the dereferenced value straight to op (like ::cuda::transform_iterator) instead of materializing the
  // value_type first, so value types need not be constructible on the device. See:
  // https://github.com/NVIDIA/cccl/issues/3591
  THRUST_HOST_DEVICE THRUST_FORCEINLINE reference operator*() const
  {
    return op(*input);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE reference operator*()
  {
    return op(*input);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t operator+(difference_type n) const
  {
    return self_t(input + n, op);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t& operator+=(difference_type n)
  {
    input += n;
    return *this;
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t operator-(difference_type n) const
  {
    return self_t(input - n, op);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE self_t& operator-=(difference_type n)
  {
    input -= n;
    return *this;
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE difference_type operator-(self_t other) const
  {
    return input - other.input;
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE reference operator[](difference_type n) const
  {
    return op(input[n]);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE bool operator==(const self_t& rhs) const
  {
    return (input == rhs.input);
  }

  THRUST_HOST_DEVICE THRUST_FORCEINLINE bool operator!=(const self_t& rhs) const
  {
    return (input != rhs.input);
  }
};
} // namespace __find_if

template <class Derived, class InputIt, class Size, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_n(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  using result_type = typename thrust::tuple<bool, Size>;

  // empty sequence
  if (num_items == 0)
  {
    return first;
  }

  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found
  //
  // XXX compose find_if from a look-back prefix scan algorithm
  //     and abort kernel when the first element is found

  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const Size interval_threshold = 1 << 20;
  const Size interval_size      = (_THRUST_STD::min)(interval_threshold, num_items);

  // Upstream uses ::cuda::make_transform_iterator here. thrust::transform_iterator converts to the value_type before
  // calling the functor (https://github.com/NVIDIA/cccl/issues/3591), so keep a dedicated iterator that does not.
  auto unwrapped_first = thrust::try_unwrap_contiguous_iterator(first);
  using XfrmIterator   = __find_if::transform_input_iterator_t<bool, decltype(unwrapped_first), Predicate>;

  const auto begin = thrust::make_zip_iterator(XfrmIterator(unwrapped_first, predicate), counting_iterator<Size>(0));
  const auto end = begin + num_items;

  for (auto interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    auto interval_end = interval_begin + interval_size;
    if (end < interval_end)
    {
      interval_end = end;
    } // end if

    result_type result = reduce(
      policy, interval_begin, interval_end, result_type(false, interval_end - begin), __find_if::functor<result_type>());

    // see if we found something
    if (thrust::get<0>(result))
    {
      return first + thrust::get<1>(result);
    }
  }

  // nothing was found if we reach here...
  return first + num_items;
}

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return hip_rocprim::find_if_n(policy, first, _THRUST_STD::distance(first, last), predicate);
}

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return hip_rocprim::find_if(policy, first, last, _THRUST_STD::not_fn(predicate));
}

template <class Derived, class InputIt, class T>
InputIt THRUST_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value)
{
  using thrust::placeholders::_1;

  return hip_rocprim::find_if(policy, first, last, _1 == value);
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
