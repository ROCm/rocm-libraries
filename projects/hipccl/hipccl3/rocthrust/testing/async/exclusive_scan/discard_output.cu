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

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2017

#  include <async/exclusive_scan/mixin.h>
#  include <async/test_policy_overloads.h>

// Compilation test with discard iterators. No runtime validation is actually
// performed, other than testing whether the algorithm completes without
// exception.

template <typename input_value_type,
          typename initial_value_type  = input_value_type,
          typename alternate_binary_op = thrust::maximum<>>
struct discard_invoker
    : testing::async::mixin::input::device_vector<input_value_type>
    , testing::async::mixin::output::discard_iterator
    , testing::async::exclusive_scan::mixin::postfix_args::all_overloads<initial_value_type, alternate_binary_op>
    , testing::async::mixin::invoke_reference::noop
    , testing::async::exclusive_scan::mixin::invoke_async::simple
    , testing::async::mixin::compare_outputs::noop
{
  static std::string description()
  {
    return "discard output";
  }
};

template <typename T>
struct test_discard
{
  void operator()(std::size_t num_values) const
  {
    testing::async::test_policy_overloads<discard_invoker<T>>::run(num_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(test_discard, NumericTypes);

#endif // C++14
