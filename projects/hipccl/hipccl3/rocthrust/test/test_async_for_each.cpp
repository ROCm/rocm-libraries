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

#  include <thrust/async/for_each.h>
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>

#  include "test_param_fixtures.hpp"
#  include "test_utils.hpp"

THRUST_SUPPRESS_DEPRECATED_PUSH

#  define DEFINE_ASYNC_FOR_EACH_CALLABLE(name, ...)                                            \
    struct THRUST_PP_CAT2(name, _fn)                                                           \
    {                                                                                          \
      template <typename ForwardIt, typename Sentinel, typename UnaryFunction>                 \
      THRUST_HOST auto operator()(ForwardIt&& first, Sentinel&& last, UnaryFunction&& f) const \
        THRUST_RETURNS(::thrust::async::for_each(                                              \
          __VA_ARGS__ THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__)) THRUST_FWD(first),      \
          THRUST_FWD(last),                                                                    \
          THRUST_FWD(f)))                                                                      \
    };                                                                                         \
    /**/

DEFINE_ASYNC_FOR_EACH_CALLABLE(invoke_async_for_each);
DEFINE_ASYNC_FOR_EACH_CALLABLE(invoke_async_for_each_device, thrust::device);

#  undef DEFINE_ASYNC_FOR_EACH_CALLABLE

///////////////////////////////////////////////////////////////////////////////

struct inplace_divide_by_2
{
  template <typename T>
  THRUST_HOST_DEVICE void operator()(T& x) const
  {
    x /= 2;
  }
};

///////////////////////////////////////////////////////////////////////////////

TESTS_DEFINE(AsyncForEachTests, NumericalTestsParams);

template <typename T, typename AsyncForEachCallable, typename UnaryFunction>
THRUST_HOST void test_async_for_each()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0_data(random_integers<T>(size));
    thrust::device_vector<T> d0_data(h0_data);

    thrust::for_each(h0_data.begin(), h0_data.end(), UnaryFunction{});

    auto f0 = AsyncForEachCallable{}(d0_data.begin(), d0_data.end(), UnaryFunction{});

    f0.wait();

    ASSERT_EQ(h0_data, d0_data);
  }
}

TYPED_TEST(AsyncForEachTests, test_async_for_each)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_for_each<T, invoke_async_for_each_fn, inplace_divide_by_2>();
}

TYPED_TEST(AsyncForEachTests, test_async_for_each_policy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_for_each<T, invoke_async_for_each_device_fn, inplace_divide_by_2>();
}

#endif
