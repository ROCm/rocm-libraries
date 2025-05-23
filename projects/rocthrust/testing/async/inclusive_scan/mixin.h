/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if THRUST_CPP_DIALECT >= 2017

#  include <thrust/async/scan.h>
#  include <thrust/scan.h>

#  include <async/mixin.h>

namespace testing
{
namespace async
{
namespace inclusive_scan
{

namespace mixin
{

namespace postfix_args_init
{

template <typename ValueType, typename AltBinaryOp = thrust::maximum<>>
struct all_overloads
{
  using postfix_args_type = std::tuple< // List any extra arg overloads:
    std::tuple<ValueType, AltBinaryOp> // - initial_value with binary_op
    >;

  static postfix_args_type generate_postfix_args()
  {
    return postfix_args_type{std::make_tuple(ValueType{42}, AltBinaryOp{})};
  }
};

} // namespace postfix_args_init

//------------------------------------------------------------------------------
namespace postfix_args
{

template <typename alternate_binary_op = thrust::maximum<>>
struct all_overloads
{
  using postfix_args_type = std::tuple< // List any extra arg overloads:
    std::tuple<>, // - no extra args
    std::tuple<alternate_binary_op> // - binary_op
    >;

  static postfix_args_type generate_postfix_args()
  {
    return postfix_args_type{std::tuple<>{}, std::make_tuple(alternate_binary_op{})};
  }
};

} // namespace postfix_args

//------------------------------------------------------------------------------
namespace invoke_reference
{

template <typename input_value_type, typename output_value_type = input_value_type>
struct host_synchronous
{
  template <typename InputType, typename OutputType, typename PostfixArgTuple, std::size_t... PostfixArgIndices>
  static void invoke_reference(
    InputType const& input,
    OutputType& output,
    PostfixArgTuple&& postfix_tuple,
    std::index_sequence<PostfixArgIndices...>)
  {
    // Create host versions of the input/output:
    thrust::host_vector<input_value_type> host_input(input.cbegin(), input.cend());
    thrust::host_vector<output_value_type> host_output(host_input.size());

    // Run host synchronous algorithm to generate reference.
    thrust::inclusive_scan(
      host_input.cbegin(),
      host_input.cend(),
      host_output.begin(),
      std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);

    // Copy back to device.
    output = host_output;
  }
};

} // namespace invoke_reference

//------------------------------------------------------------------------------
namespace invoke_async
{

struct simple
{
  template <typename PrefixArgTuple,
            std::size_t... PrefixArgIndices,
            typename InputType,
            typename OutputType,
            typename PostfixArgTuple,
            std::size_t... PostfixArgIndices>
  static auto invoke_async(
    PrefixArgTuple&& prefix_tuple,
    std::index_sequence<PrefixArgIndices...>,
    InputType const& input,
    OutputType& output,
    PostfixArgTuple&& postfix_tuple,
    std::index_sequence<PostfixArgIndices...>)
  {
    auto e = thrust::async::inclusive_scan(
      std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
      input.cbegin(),
      input.cend(),
      output.begin(),
      std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
    return e;
  }
};

} // namespace invoke_async

} // namespace mixin
} // namespace inclusive_scan
} // namespace async
} // namespace testing

#endif // C++14
