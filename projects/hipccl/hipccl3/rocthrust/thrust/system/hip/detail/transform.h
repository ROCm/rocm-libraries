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

#  include <thrust/distance.h>
#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/hip/detail/dispatch.h>
#  include <thrust/system/hip/detail/parallel_for.h>
#  include <thrust/system/hip/detail/util.h>
#  include <thrust/zip_function.h>

#  include _THRUST_LIBCXX_INCLUDE(functional)
#  include _THRUST_STD_INCLUDE(__algorithm_)
#  include _THRUST_STD_INCLUDE(cstdint)

// rocprim include
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

namespace hip_rocprim
{
namespace __transform
{
template <class InputIt, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct unary_transform_f
{
  InputIt input;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input[idx]));
    }
  }
};

// EAN 2024-10-04: when force-inlined, gcc's optimizer will generate bad code
// for this function:
template <class Policy, class InputIt, class Size, class OutputIt, class StencilIt, class TransformOp, class Predicate>
OutputIt _CCCL_HOST_DEVICE unary_if_with_stencil(
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

template <class InputIt1, class InputIt2, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct binary_transform_f
{
  InputIt1 input1;
  InputIt2 input2;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  template <class Size>
  void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input1[idx]), raw_reference_cast(input2[idx]));
    }
  }
};

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
OutputIt THRUST_HOST_DEVICE inline binary_if_with_stencil(
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

THRUST_EXEC_CHECK_DISABLE
template <class Derived,
          class Offset,
          class... InputIts,
          class OutputIt,
          class TransformOp,
          class Predicate = rocprim::detail::always_true_predicate>
OutputIt THRUST_HOST_DEVICE hip_transform_many(
  execution_policy<Derived>& policy,
  rocprim::tuple<InputIts...> firsts,
  OutputIt result,
  Offset num_items,
  TransformOp transform_op,
  Predicate pred = {})
{
  if (num_items == 0)
  {
    return result;
  }

  hipStream_t stream = hip_rocprim::stream(policy);
  bool debug_sync    = THRUST_HIP_DEBUG_SYNC_FLAG;

  hip_rocprim::throw_on_error(
    rocprim::transform(firsts, result, static_cast<size_t>(num_items), pred, transform_op, stream, debug_sync),
    "transform: failed on 1st call");

  hipError_t status = hip_rocprim::synchronize_optional(policy);
  throw_on_error(status, "transform: failed to synchronize");

  return result + num_items;
}

// Necessary because when using libhipcxx thrust::tuple will not be rocprim::tuple but hip::std::tuple,
// and that'll mess up rocPRIM's overload resolution
template <typename... Ts, std::size_t... Is>
THRUST_HOST_DEVICE auto
convert_to_rocprim_tuple(thrust::tuple<Ts...> t, std::index_sequence<Is...>) -> rocprim::tuple<Ts...>
{
  return rocprim::tuple<Ts...>{thrust::get<Is>(t)...};
}

// unwrap zip_iterator and zip_function into their underlying iterators so transform can optimize them
template <class Derived, class Offset, class... InputIts, class OutputIt, class TransformOp>
OutputIt THRUST_HOST_DEVICE hip_transform_many(
  execution_policy<Derived>& policy,
  rocprim::tuple<zip_iterator<thrust::tuple<InputIts...>>> firsts,
  OutputIt result,
  Offset num_items,
  thrust::zip_function<TransformOp> transform_op)
{
  return hip_transform_many(
    policy,
    convert_to_rocprim_tuple(get<0>(firsts).get_iterator_tuple(), std::index_sequence_for<InputIts...>{}),
    result,
    num_items,
    transform_op.underlying_function());
}

template <typename F>
struct raw_reference_cast_args
{
  mutable F f; // mutable to support non-const F::operator()

  template <typename... Ts>
  THRUST_HIP_FUNCTION decltype(auto) operator()(Ts&&... args) const
  {
    return f(raw_reference_cast(std::forward<Ts>(args))...);
  }
};
} // namespace __transform

//-------------------------
// Thrust API entry points
//-------------------------

//  one input data stream

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_HIP_FUNCTION OutputIt
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt& result, TransformOp transform_op)
    {
      return __transform::hip_transform_many(
        policy, rocprim::make_tuple(first), result, _THRUST_STD::distance(first, last), transform_op);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt& result, TransformOp transform_op)
    {
      return _THRUST_STD::transform(
        first, last, result, __transform::raw_reference_cast_args<TransformOp>{transform_op});
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first, last, result, transform_op);
#  else
  return workaround::seq(policy, first, last, result, transform_op);
#  endif
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_HIP_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> num_items,
  OutputIt result,
  TransformOp transform_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy,
        InputIt first,
        _THRUST_STD::iter_difference_t<InputIt> num_items,
        OutputIt& result,
        TransformOp transform_op)
    {
      return __transform::hip_transform_many(policy, rocprim::make_tuple(first), result, num_items, transform_op);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy,
        InputIt first,
        _THRUST_STD::iter_difference_t<InputIt> num_items,
        OutputIt& result,
        TransformOp transform_op)
    {
      return _THRUST_STD::transform(
        first, first + num_items, result, __transform::raw_reference_cast_args<TransformOp>{transform_op});
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first, num_items, result, transform_op);
#  else
  return workaround::seq(policy, first, num_items, result, transform_op);
#  endif
}

template <class Derived, class InputIt, class OutputIt, class TransformOp, class Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        OutputIt& result,
        TransformOp transform_op,
        Predicate predicate)
    {
      return __transform::hip_transform_many(
        policy, rocprim::make_tuple(first), result, _THRUST_STD::distance(first, last), transform_op, predicate);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        OutputIt& result,
        TransformOp transform_op,
        Predicate predicate)
    {
      while (first != last)
      {
        if (predicate(raw_reference_cast(*first)))
        {
          *result = transform_op(raw_reference_cast(*first));
        }
        ++first;
        ++result;
      }
      return result;
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first, last, result, transform_op, predicate);
#  else
  return workaround::seq(policy, first, last, result, transform_op, predicate);
#  endif
}

template <class Derived, class InputIt, class OutputIt, class TransformOp, class Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> num_items,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy,
        InputIt first,
        _THRUST_STD::iter_difference_t<InputIt> num_items,
        OutputIt& result,
        TransformOp transform_op,
        Predicate predicate)
    {
      return __transform::hip_transform_many(
        policy, rocprim::make_tuple(first), result, num_items, transform_op, predicate);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy,
        InputIt first,
        _THRUST_STD::iter_difference_t<InputIt> num_items,
        OutputIt& result,
        TransformOp transform_op,
        Predicate predicate)
    {
      for (decltype(num_items) i = 0; i < num_items; i++)
      {
        {
          if (predicate(raw_reference_cast(*first)))
          {
            *result = transform_op(raw_reference_cast(*first));
          }
          ++first;
          ++result;
        }
        return result;
      }
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first, num_items, result, transform_op, predicate);
#  else
  return workaround::seq(policy, first, num_items, result, transform_op, predicate);
#  endif
}

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
  return __transform::unary_if_with_stencil(
    policy, first, result, _THRUST_STD::distance(first, last), stencil, transform_op, predicate);
}

template <class Derived, class InputIt, class OutputIt, class StencilInputIt, class TransformOp, class Predicate>
THRUST_HIP_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  _THRUST_STD::iter_difference_t<InputIt> num_items,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return __transform::unary_if_with_stencil(policy, first, result, num_items, stencil, transform_op, predicate);
}

// two input data streams

template <class Derived, class InputIt1, class InputIt2, class OutputIt, class BinaryTransformOp>
THRUST_HIP_FUNCTION OutputIt transform(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy,
        InputIt1 first1,
        InputIt1 last1,
        InputIt2 first2,
        OutputIt result,
        BinaryTransformOp transform_op)
    {
      return __transform::hip_transform_many(
        policy, rocprim::make_tuple(first1, first2), result, _THRUST_STD::distance(first1, last1), transform_op);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy,
        InputIt1 first1,
        InputIt1 last1,
        InputIt2 first2,
        OutputIt result,
        BinaryTransformOp transform_op)
    {
      return _THRUST_STD::transform(
        first1, last1, first2, result, __transform::raw_reference_cast_args<BinaryTransformOp>{transform_op});
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first1, last1, first2, result, transform_op);
#  else
  return workaround::seq(policy, first1, last1, first2, result, transform_op);
#  endif
}

template <class Derived, class InputIt1, class InputIt2, class OutputIt, class BinaryTransformOp>
THRUST_HIP_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  _THRUST_STD::iter_difference_t<InputIt1> num_items,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy,
        InputIt1 first1,
        _THRUST_STD::iter_difference_t<InputIt1> num_items,
        InputIt2 first2,
        OutputIt result,
        BinaryTransformOp transform_op)
    {
      return __transform::hip_transform_many(
        policy, rocprim::make_tuple(first1, first2), result, num_items, transform_op);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy,
        InputIt1 first1,
        _THRUST_STD::iter_difference_t<InputIt1> num_items,
        InputIt2 first2,
        OutputIt result,
        BinaryTransformOp transform_op)
    {
      return _THRUST_STD::transform(
        first1,
        first1 + num_items,
        first2,
        result,
        __transform::raw_reference_cast_args<BinaryTransformOp>{transform_op});
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(policy, first1, num_items, first2, result, transform_op);
#  else
  return workaround::seq(policy, first1, num_items, first2, result, transform_op);
#  endif
}

template <class Derived,
          class InputIt1,
          class InputIt2,
          class StencilInputIt,
          class OutputIt,
          class BinaryTransformOp,
          class Predicate>
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
  return __transform::binary_if_with_stencil(
    policy, first1, first2, result, _THRUST_STD::distance(first1, last1), stencil, transform_op, predicate);
}

template <class Derived,
          class InputIt1,
          class InputIt2,
          class StencilInputIt,
          class OutputIt,
          class BinaryTransformOp,
          class Predicate>
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
  return __transform::binary_if_with_stencil(
    policy, first1, first2, result, num_items, stencil, transform_op, predicate);
}
} // namespace hip_rocprim

THRUST_NAMESPACE_END
#endif
