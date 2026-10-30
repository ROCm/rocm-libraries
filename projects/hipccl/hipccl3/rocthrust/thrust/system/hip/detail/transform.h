/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/hip/detail/dispatch.h>
#  include <thrust/system/hip/detail/parallel_for.h>
#  include <thrust/system/hip/detail/util.h>
#  include <thrust/zip_function.h>

#  include _THRUST_LIBCXX_INCLUDE(functional)
#  include _THRUST_STD_INCLUDE(__iterator/distance.h)
#  include _THRUST_STD_INCLUDE(__iterator/incrementable_traits.h)
#  include _THRUST_STD_INCLUDE(cstdint)

THRUST_NAMESPACE_BEGIN

namespace hip_rocprim
{
namespace __transform
{
struct no_stencil_tag
{};

struct always_true_predicate
{
  template <class T>
  constexpr bool THRUST_HIP_DEVICE_FUNCTION operator()(T const&) const
  {
    return true;
  }
};

template <class InputIt, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct unary_transform_f
{
  InputIt input;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  THRUST_HIP_FUNCTION
  unary_transform_f(InputIt input_, OutputIt output_, StencilIt stencil_, TransformOp op_, Predicate pred_)
      : input(input_)
      , output(output_)
      , stencil(stencil_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input[idx]));
    }
  }
};

template <class InputIt, class OutputIt, class TransformOp, class Predicate>
struct unary_transform_f<InputIt, OutputIt, no_stencil_tag, TransformOp, Predicate>
{
  InputIt input;
  OutputIt output;
  TransformOp op;
  Predicate pred;

  THRUST_HIP_FUNCTION
  unary_transform_f(InputIt input_, OutputIt output_, no_stencil_tag, TransformOp op_, Predicate pred_)
      : input(input_)
      , output(output_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(input[idx])))
    {
      output[idx] = op(raw_reference_cast(input[idx]));
    }
  }
};

template <class InputIt1, class InputIt2, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct binary_transform_f
{
  InputIt1 input1;
  InputIt2 input2;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  THRUST_HIP_FUNCTION
  binary_transform_f(
    InputIt1 input1_, InputIt2 input2_, OutputIt output_, StencilIt stencil_, TransformOp op_, Predicate pred_)
      : input1(input1_)
      , input2(input2_)
      , output(output_)
      , stencil(stencil_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input1[idx]), raw_reference_cast(input2[idx]));
    }
  }
};

template <class InputIt1, class InputIt2, class OutputIt, class TransformOp, class Predicate>
struct binary_transform_f<InputIt1, InputIt2, OutputIt, no_stencil_tag, TransformOp, Predicate>
{
  InputIt1 input1;
  InputIt2 input2;
  OutputIt output;
  TransformOp op;
  Predicate pred;

  THRUST_HIP_FUNCTION
  binary_transform_f(
    InputIt1 input1_, InputIt2 input2_, OutputIt output_, no_stencil_tag, TransformOp op_, Predicate pred_)
      : input1(input1_)
      , input2(input2_)
      , output(output_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(input1[idx])))
    {
      output[idx] = op(raw_reference_cast(input1[idx]), raw_reference_cast(input2[idx]));
    }
  }
};

// EAN 2024-10-04: when force-inlined, gcc's optimizer will generate bad code
// for this function:
template <class Policy, class InputIt, class Size, class OutputIt, class StencilIt, class TransformOp, class Predicate>
OutputIt THRUST_HOST_DEVICE inline unary(
  Policy& policy,
  InputIt items,
  OutputIt result,
  Size num_items,
  StencilIt stencil,
  TransformOp transform_op,
  Predicate predicate)
{
  if (num_items == 0)
  {
    return result;
  }

  using unary_transform_t = unary_transform_f<InputIt, OutputIt, StencilIt, TransformOp, Predicate>;

  // We use 'parallel_for' instead of 'rocprim::transform', since 'thrust::transform' allows
  // the function to modify the input iterator! 'rocprim::transform' does not write any
  // effects on the input iterator back to memory.
  hip_rocprim::parallel_for(policy, unary_transform_t{items, result, stencil, transform_op, predicate}, num_items);
  return result + num_items;
}

// EAN 2024-10-04: when force-inlined, gcc's optimizer will generate bad code
// for this function:
template <class Policy,
          class InputIt1,
          class InputIt2,
          class Size,
          class OutputIt,
          class StencilIt,
          class TransformOp,
          class Predicate>
OutputIt THRUST_HOST_DEVICE inline binary(
  Policy& policy,
  InputIt1 items1,
  InputIt2 items2,
  OutputIt result,
  Size num_items,
  StencilIt stencil,
  TransformOp transform_op,
  Predicate predicate)
{
  if (num_items == 0)
  {
    return result;
  }

  using binary_transform_t = binary_transform_f<InputIt1, InputIt2, OutputIt, StencilIt, TransformOp, Predicate>;

  // We use 'parallel_for' instead of 'rocprim::transform', since 'thrust::transform' allows
  // the function to modify the input iterator! 'rocprim::transform' does not write any
  // effects on the input iterator back to memory.
  hip_rocprim::parallel_for(
    policy, binary_transform_t{items1, items2, result, stencil, transform_op, predicate}, num_items);
  return result + num_items;
}

} // namespace __transform

//  one input data stream

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_HIP_FUNCTION OutputIt
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op)
{
  return __transform::unary(
    policy,
    first,
    result,
    _THRUST_STD::distance(first, last),
    __transform::no_stencil_tag{},
    transform_op,
    __transform::always_true_predicate{});
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_HIP_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> num_items,
  OutputIt result,
  TransformOp transform_op)
{
  return __transform::unary(
    policy, first, result, num_items, __transform::no_stencil_tag{}, transform_op, __transform::always_true_predicate{});
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp, typename Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return __transform::unary(
    policy, first, result, _THRUST_STD::distance(first, last), __transform::no_stencil_tag{}, transform_op, predicate);
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp, typename Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> count,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return __transform::unary(policy, first, result, count, __transform::no_stencil_tag{}, transform_op, predicate);
}

//  one input data stream + stencil

template <class Derived, class InputIt, class OutputIt, class StencilInputIt, class TransformOp, class Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return __transform::unary(policy, first, result, _THRUST_STD::distance(first, last), stencil, transform_op, predicate);
}

template <typename Derived,
          typename InputIt,
          typename StencilInputIt,
          typename OutputIt,
          typename TransformOp,
          typename Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> num_items,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return __transform::unary(policy, first, result, num_items, stencil, transform_op, predicate);
}

// two input data streams

template <typename Derived, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryTransformOp>
THRUST_HIP_FUNCTION OutputIt transform(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  return __transform::binary(
    policy,
    first1,
    first2,
    result,
    _THRUST_STD::distance(first1, last1),
    __transform::no_stencil_tag{},
    transform_op,
    __transform::always_true_predicate{});
}

template <typename Derived, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryTransformOp>
THRUST_HIP_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  _THRUST_STD::iter_difference_t<InputIt1> num_items,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  return __transform::binary(
    policy,
    first1,
    first2,
    result,
    num_items,
    __transform::no_stencil_tag{},
    transform_op,
    __transform::always_true_predicate{});
}

// two input data streams + stencil

template <typename Derived,
          typename InputIt1,
          typename InputIt2,
          typename StencilInputIt,
          typename OutputIt,
          typename BinaryTransformOp,
          typename Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  StencilInputIt stencil,
  OutputIt result,
  BinaryTransformOp transform_op,
  Predicate predicate)
{
  return __transform::binary(
    policy, first1, first2, result, _THRUST_STD::distance(first1, last1), stencil, transform_op, predicate);
}

template <typename Derived,
          typename InputIt1,
          typename InputIt2,
          typename StencilInputIt,
          typename OutputIt,
          typename BinaryTransformOp,
          typename Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  _THRUST_STD::iter_difference_t<InputIt1> num_items,
  InputIt2 first2,
  StencilInputIt stencil,
  OutputIt result,
  BinaryTransformOp transform_op,
  Predicate predicate)
{
  return __transform::binary(policy, first1, first2, result, num_items, stencil, transform_op, predicate);
}

} // namespace hip_rocprim

THRUST_NAMESPACE_END
#endif
