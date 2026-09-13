/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#  include <thrust/future.h>

#  include "test_param_fixtures.hpp"
#  include "test_real_assertions.hpp"
#  include "test_utils.hpp"

// note: there is no matching THRUST_SUPPRESS_DEPRECATED_POP, so the warning suppression leaks into more content of the
// generated cudafe1.stub.c file.
THRUST_SUPPRESS_DEPRECATED_PUSH

struct mock
{};

using future_value_types = ::testing::Types<
  Params<char>,
  Params<signed char>,
  Params<unsigned char>,
  Params<short>,
  Params<unsigned short>,
  Params<int>,
  Params<unsigned int>,
  Params<long>,
  Params<unsigned long>,
  Params<long long>,
  Params<unsigned long long>,
  Params<float>,
  Params<double>,
  Params<float2>,
  Params<mock>>;

TESTS_DEFINE(FutureTests, future_value_types);

///////////////////////////////////////////////////////////////////////////////

TYPED_TEST(FutureTests, test_future_default_constructed)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  THRUST_STATIC_ASSERT((std::is_same<thrust::future<decltype(thrust::device), T>,
                                     thrust::unique_eager_future<decltype(thrust::device), T>>::value));

  THRUST_STATIC_ASSERT((std::is_same<thrust::future<decltype(thrust::device), T>, thrust::device_future<T>>::value));

  THRUST_STATIC_ASSERT((std::is_same<thrust::device_future<T>, thrust::device_unique_eager_future<T>>::value));

  thrust::device_future<T> f0;

  ASSERT_EQ(false, f0.valid_stream());
  ASSERT_EQ(false, f0.valid_content());

  ASSERT_THROWS_EQ(f0.wait(), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));

  ASSERT_THROWS_EQ(f0.stream(), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));

  ASSERT_THROWS_EQ(f0.get(), thrust::event_error, thrust::event_error(thrust::event_errc::no_content));

  ASSERT_THROWS_EQ(
    THRUST_UNUSED_VAR(f0.extract()), thrust::event_error, thrust::event_error(thrust::event_errc::no_content));
}

///////////////////////////////////////////////////////////////////////////////

TYPED_TEST(FutureTests, test_future_new_stream)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  auto f0 = thrust::device_future<T>(thrust::new_stream);

  ASSERT_EQ(true, f0.valid_stream());
  ASSERT_EQ(false, f0.valid_content());

  ASSERT_NE_QUIET(nullptr, f0.stream().native_handle());

  TEST_EVENT_WAIT(f0);

  ASSERT_EQ(true, f0.ready());

  ASSERT_THROWS_EQ(f0.get(), thrust::event_error, thrust::event_error(thrust::event_errc::no_content));

  ASSERT_THROWS_EQ(
    THRUST_UNUSED_VAR(f0.extract()), thrust::event_error, thrust::event_error(thrust::event_errc::no_content));
}

///////////////////////////////////////////////////////////////////////////////

TYPED_TEST(FutureTests, test_future_convert_to_event)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  auto f0 = thrust::device_future<T>(thrust::new_stream);

  auto const f0_stream = f0.stream().native_handle();

  ASSERT_EQ(true, f0.valid_stream());
  ASSERT_EQ(false, f0.valid_content());

  ASSERT_NE_QUIET(nullptr, f0_stream);

  auto f1 = thrust::device_event(std::move(f0));

  ASSERT_EQ(false, f0.valid_stream());
  ASSERT_EQ(true, f1.valid_stream());

  ASSERT_EQ(f0_stream, f1.stream().native_handle());
}

///////////////////////////////////////////////////////////////////////////////

TYPED_TEST(FutureTests, test_future_when_all)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  // Create futures with new streams.
  auto f0 = thrust::device_future<T>(thrust::new_stream);
  auto f1 = thrust::device_future<T>(thrust::new_stream);
  auto f2 = thrust::device_future<T>(thrust::new_stream);
  auto f3 = thrust::device_future<T>(thrust::new_stream);
  auto f4 = thrust::device_future<T>(thrust::new_stream);
  auto f5 = thrust::device_future<T>(thrust::new_stream);
  auto f6 = thrust::device_future<T>(thrust::new_stream);
  auto f7 = thrust::device_future<T>(thrust::new_stream);

  auto const f0_stream = f0.stream().native_handle();

  ASSERT_EQ(true, f0.valid_stream());
  ASSERT_EQ(true, f1.valid_stream());
  ASSERT_EQ(true, f2.valid_stream());
  ASSERT_EQ(true, f3.valid_stream());
  ASSERT_EQ(true, f4.valid_stream());
  ASSERT_EQ(true, f5.valid_stream());
  ASSERT_EQ(true, f6.valid_stream());
  ASSERT_EQ(true, f7.valid_stream());

  ASSERT_EQ(false, f0.valid_content());
  ASSERT_EQ(false, f1.valid_content());
  ASSERT_EQ(false, f2.valid_content());
  ASSERT_EQ(false, f3.valid_content());
  ASSERT_EQ(false, f4.valid_content());
  ASSERT_EQ(false, f5.valid_content());
  ASSERT_EQ(false, f6.valid_content());
  ASSERT_EQ(false, f7.valid_content());

  ASSERT_NE_QUIET(nullptr, f0_stream);
  ASSERT_NE_QUIET(nullptr, f1.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f2.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f3.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f4.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f5.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f6.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, f7.stream().native_handle());

  auto e0 = thrust::when_all(f0, f1, f2, f3, f4, f5, f6, f7);

  ASSERT_EQ(false, f0.valid_stream());
  ASSERT_EQ(false, f1.valid_stream());
  ASSERT_EQ(false, f2.valid_stream());
  ASSERT_EQ(false, f3.valid_stream());
  ASSERT_EQ(false, f4.valid_stream());
  ASSERT_EQ(false, f5.valid_stream());
  ASSERT_EQ(false, f6.valid_stream());
  ASSERT_EQ(false, f7.valid_stream());

  ASSERT_EQ(false, f0.valid_content());
  ASSERT_EQ(false, f1.valid_content());
  ASSERT_EQ(false, f2.valid_content());
  ASSERT_EQ(false, f3.valid_content());
  ASSERT_EQ(false, f4.valid_content());
  ASSERT_EQ(false, f5.valid_content());
  ASSERT_EQ(false, f6.valid_content());
  ASSERT_EQ(false, f7.valid_content());

  ASSERT_EQ(true, e0.valid_stream());

  ASSERT_EQ(f0_stream, e0.stream().native_handle());

  TEST_EVENT_WAIT(e0);

  ASSERT_EQ(false, f0.ready());
  ASSERT_EQ(false, f1.ready());
  ASSERT_EQ(false, f2.ready());
  ASSERT_EQ(false, f3.ready());
  ASSERT_EQ(false, f4.ready());
  ASSERT_EQ(false, f5.ready());
  ASSERT_EQ(false, f6.ready());
  ASSERT_EQ(false, f7.ready());

  ASSERT_EQ(true, e0.ready());
}

///////////////////////////////////////////////////////////////////////////////

#endif
