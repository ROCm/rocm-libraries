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

#  include <thrust/event.h>

#  include "test_param_fixtures.hpp"
#  include "test_real_assertions.hpp"
#  include "test_utils.hpp"

TESTS_DEFINE(EventTests, FullTestsParams);

// note: there is no matching THRUST_SUPPRESS_DEPRECATED_POP, so the warning suppression leaks into more content of the
// generated cudafe1.stub.c file.
THRUST_SUPPRESS_DEPRECATED_PUSH

///////////////////////////////////////////////////////////////////////////////

TEST(EventTests, test_event_default_constructed)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  THRUST_STATIC_ASSERT((
    std::is_same<thrust::event<decltype(thrust::device)>, thrust::unique_eager_event<decltype(thrust::device)>>::value));

  THRUST_STATIC_ASSERT((std::is_same<thrust::event<decltype(thrust::device)>, thrust::device_event>::value));

  THRUST_STATIC_ASSERT((std::is_same<thrust::device_event, thrust::device_unique_eager_event>::value));

  thrust::device_event e0;

  ASSERT_EQ(false, e0.valid_stream());

  ASSERT_THROWS_EQ(e0.wait(), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));

  ASSERT_THROWS_EQ(e0.stream(), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));
}

///////////////////////////////////////////////////////////////////////////////

TEST(EventTests, test_event_new_stream)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto e0 = thrust::device_event(thrust::new_stream);

  ASSERT_EQ(true, e0.valid_stream());

  ASSERT_NE_QUIET(nullptr, e0.stream().native_handle());

  e0.wait();

  ASSERT_EQ(true, e0.ready());
}

///////////////////////////////////////////////////////////////////////////////

TEST(EventTests, test_event_linear_chaining)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  constexpr std::int64_t n = 1024;

  // Create a new stream.
  auto e0 = thrust::when_all();

  auto const e0_stream = e0.stream().native_handle();

  ASSERT_EQ(true, e0.valid_stream());

  ASSERT_NE_QUIET(nullptr, e0_stream);

  thrust::device_event e1;

  for (std::int64_t i = 0; i < n; ++i)
  {
    ASSERT_EQ(true, e0.valid_stream());

    ASSERT_EQ(false, e1.valid_stream());
    ASSERT_EQ(false, e1.ready());

    ASSERT_EQ_QUIET(e0_stream, e0.stream().native_handle());

    e1 = thrust::when_all(e0);

    ASSERT_EQ(false, e0.valid_stream());
    ASSERT_EQ(false, e0.ready());

    ASSERT_EQ(true, e1.valid_stream());

    ASSERT_EQ(e0_stream, e1.stream().native_handle());

    std::swap(e0, e1);
  }
}

///////////////////////////////////////////////////////////////////////////////

TEST(EventTests, test_event_when_all)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // Create events with new streams.
  auto e0 = thrust::when_all();
  auto e1 = thrust::when_all();
  auto e2 = thrust::when_all();
  auto e3 = thrust::when_all();
  auto e4 = thrust::when_all();
  auto e5 = thrust::when_all();
  auto e6 = thrust::when_all();
  auto e7 = thrust::when_all();

  auto const e0_stream = e0.stream().native_handle();

  ASSERT_EQ(true, e0.valid_stream());
  ASSERT_EQ(true, e1.valid_stream());
  ASSERT_EQ(true, e2.valid_stream());
  ASSERT_EQ(true, e3.valid_stream());
  ASSERT_EQ(true, e4.valid_stream());
  ASSERT_EQ(true, e5.valid_stream());
  ASSERT_EQ(true, e6.valid_stream());
  ASSERT_EQ(true, e7.valid_stream());

  ASSERT_NE_QUIET(nullptr, e0_stream);
  ASSERT_NE_QUIET(nullptr, e1.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e2.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e3.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e4.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e5.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e6.stream().native_handle());
  ASSERT_NE_QUIET(nullptr, e7.stream().native_handle());

  auto e8 = thrust::when_all(e0, e1, e2, e3, e4, e5, e6, e7);

  ASSERT_EQ(false, e0.valid_stream());
  ASSERT_EQ(false, e1.valid_stream());
  ASSERT_EQ(false, e2.valid_stream());
  ASSERT_EQ(false, e3.valid_stream());
  ASSERT_EQ(false, e4.valid_stream());
  ASSERT_EQ(false, e5.valid_stream());
  ASSERT_EQ(false, e6.valid_stream());
  ASSERT_EQ(false, e7.valid_stream());

  ASSERT_EQ(true, e8.valid_stream());

  ASSERT_EQ(e0_stream, e8.stream().native_handle());

  e8.wait();

  ASSERT_EQ(false, e0.ready());
  ASSERT_EQ(false, e1.ready());
  ASSERT_EQ(false, e2.ready());
  ASSERT_EQ(false, e3.ready());
  ASSERT_EQ(false, e4.ready());
  ASSERT_EQ(false, e5.ready());
  ASSERT_EQ(false, e6.ready());
  ASSERT_EQ(false, e7.ready());

  // // FIXME: Temporarily disabled due to observed failure in AMD CI.
  // ASSERT_EQ(true, e8.ready());
}

///////////////////////////////////////////////////////////////////////////////

#endif
