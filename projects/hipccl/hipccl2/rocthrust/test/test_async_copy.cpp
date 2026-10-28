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

// need to suppress deprecation warnings inside a lot of thrust headers
THRUST_SUPPRESS_DEPRECATED_PUSH

#if THRUST_CPP_DIALECT >= 2014

#  include <thrust/async/copy.h>
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>
#  include <thrust/limits.h>

#  include "test_param_fixtures.hpp"
#  include "test_real_assertions.hpp"
#  include "test_utils.hpp"

using BuiltinNumericTypes = ::testing::Types<
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
  Params<double>>;

TESTS_DEFINE(AsyncCopyTests, BuiltinNumericTypes);

#  define DEFINE_ASYNC_COPY_CALLABLE(name, ...)                                                \
    struct THRUST_PP_CAT2(name, _fn)                                                           \
    {                                                                                          \
      template <typename ForwardIt, typename Sentinel, typename OutputIt>                      \
      THRUST_HOST auto operator()(ForwardIt&& first, Sentinel&& last, OutputIt&& output) const \
        THRUST_RETURNS(::thrust::async::copy(                                                  \
          __VA_ARGS__ THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__)) THRUST_FWD(first),      \
          THRUST_FWD(last),                                                                    \
          THRUST_FWD(output)))                                                                 \
    };                                                                                         \
    /**/

DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy);

DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_host, thrust::host);
DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_device, thrust::device);

DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_host_to_device, thrust::host, thrust::device);
DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_device_to_host, thrust::device, thrust::host);
DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_host_to_host, thrust::host, thrust::host);
DEFINE_ASYNC_COPY_CALLABLE(invoke_async_copy_device_to_device, thrust::device, thrust::device);

#  undef DEFINE_ASYNC_COPY_CALLABLE

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename AsyncCopyCallable>
THRUST_HOST void test_async_copy_host_to_device()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h0(
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed));
      thrust::device_vector<T> d0(size);

      auto f0 = AsyncCopyCallable{}(h0.begin(), h0.end(), d0.begin());

      f0.wait();

      ASSERT_EQ(h0, d0);
    }
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_trivially_relocatable_elements_host_to_device)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_host_to_device<T, invoke_async_copy_fn>();
}

TYPED_TEST(AsyncCopyTests, test_async_copy_trivially_relocatable_elements_host_to_device_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_host_to_device<T, invoke_async_copy_host_to_device_fn>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename AsyncCopyCallable>
THRUST_HOST void test_async_copy_device_to_host()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h0(
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed));
      thrust::host_vector<T> h1(size);
      thrust::device_vector<T> d0(size);

      thrust::copy(h0.begin(), h0.end(), d0.begin());

      ASSERT_EQ(h0, d0);

      auto f0 = AsyncCopyCallable{}(d0.begin(), d0.end(), h1.begin());

      f0.wait();

      ASSERT_EQ(h0, d0);
      ASSERT_EQ(d0, h1);
    }
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_trivially_relocatable_elements_device_to_host)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_device_to_host<T, invoke_async_copy_fn>();
}

TYPED_TEST(AsyncCopyTests, test_async_copy_trivially_relocatable_elements_device_to_host_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_device_to_host<T, invoke_async_copy_device_to_host_fn>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename AsyncCopyCallable>
THRUST_HOST void test_async_copy_device_to_device()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));
    thrust::device_vector<T> d0(size);
    thrust::device_vector<T> d1(size);

    thrust::copy(h0.begin(), h0.end(), d0.begin());

    ASSERT_EQ(h0, d0);

    auto f0 = AsyncCopyCallable{}(d0.begin(), d0.end(), d1.begin());

    f0.wait();

    ASSERT_EQ(h0, d0);
    ASSERT_EQ(d0, d1);
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_device_to_device)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_device_to_device<T, invoke_async_copy_fn>();
}

TYPED_TEST(AsyncCopyTests, test_async_copy_device_to_device_policy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_device_to_device<T, invoke_async_copy_device_fn>();
}

TYPED_TEST(AsyncCopyTests, test_async_copy_device_to_device_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_device_to_device<T, invoke_async_copy_device_to_device_fn>();
}

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename T, typename AsyncCopyCallable>
THRUST_HOST void test_async_copy_counting_iterator_input_to_device_vector()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::counting_iterator<T> first(0);
    thrust::counting_iterator<T> last(truncate_to_max_representable<T>(size));

    thrust::device_vector<T> d0(size);
    thrust::device_vector<T> d1(size);

    thrust::copy(first, last, d0.begin());

    auto f0 = AsyncCopyCallable{}(first, last, d1.begin());

    f0.wait();

    ASSERT_EQ(d0, d1);
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_device_vector<T, invoke_async_copy_fn>();
}

TYPED_TEST(AsyncCopyTests,
           test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_device_vector<T, invoke_async_copy_device_fn>();
}

TYPED_TEST(AsyncCopyTests,
           test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_device_vector<T, invoke_async_copy_device_to_device_fn>();
}

TYPED_TEST(AsyncCopyTests, test_async_copy_counting_iterator_input_host_to_device_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_device_vector<T, invoke_async_copy_host_to_device_fn>();
}

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename T, typename AsyncCopyCallable>
THRUST_HOST void test_async_copy_counting_iterator_input_to_host_vector()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::counting_iterator<T> first(0);
    thrust::counting_iterator<T> last(truncate_to_max_representable<T>(size));

    thrust::host_vector<T> d0(size);
    thrust::host_vector<T> d1(size);

    thrust::copy(first, last, d0.begin());

    auto f0 = AsyncCopyCallable{}(first, last, d1.begin());

    f0.wait();

    ASSERT_EQ(d0, d1);

#  if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_INTEL)
    // ICC fails this for some unknown reason - see #1468.
    GTEST_NONFATAL_FAILURE_("Known failure");
#  endif
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_host_vector<T, invoke_async_copy_fn>();
}

TYPED_TEST(AsyncCopyTests,
           test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host_policies)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_counting_iterator_input_to_host_vector<T, invoke_async_copy_device_to_host_fn>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
THRUST_HOST void test_async_copy_roundtrip()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));
    thrust::device_vector<T> d0(size);

    auto e0 = thrust::async::copy(thrust::host, thrust::device, h0.begin(), h0.end(), d0.begin());

    auto e1 = thrust::async::copy(thrust::device.after(e0), thrust::host, d0.begin(), d0.end(), h0.begin());

    TEST_EVENT_WAIT(e1);

    ASSERT_EQ(h0, d0);
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_trivially_relocatable_elements_roundtrip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_roundtrip<T>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
THRUST_HOST void test_async_copy_after()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));
    thrust::host_vector<T> h1(size);
    thrust::device_vector<T> d0(size);
    thrust::device_vector<T> d1(size);
    thrust::device_vector<T> d2(size);

    auto e0 = thrust::async::copy(h0.begin(), h0.end(), d0.begin());

    ASSERT_EQ(true, e0.valid_stream());

    auto const e0_stream = e0.stream().native_handle();

    auto e1 = thrust::async::copy(thrust::device.after(e0), d0.begin(), d0.end(), d1.begin());

    // Verify that double consumption of a future produces an exception.
    ASSERT_THROWS_EQ(auto x = thrust::async::copy(thrust::device.after(e0), d0.begin(), d0.end(), d1.begin());
                     THRUST_UNUSED_VAR(x), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));

    ASSERT_EQ_QUIET(e0_stream, e1.stream().native_handle());

    auto after_policy2 = thrust::device.after(e1);

    auto e2 = thrust::async::copy(thrust::host, after_policy2, h0.begin(), h0.end(), d2.begin());

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQ(auto x = thrust::async::copy(thrust::host, after_policy2, h0.begin(), h0.end(), d2.begin());
                     THRUST_UNUSED_VAR(x), thrust::event_error, thrust::event_error(thrust::event_errc::no_state));

    ASSERT_EQ_QUIET(e0_stream, e2.stream().native_handle());

    auto e3 = thrust::async::copy(thrust::device.after(e2), thrust::host, d1.begin(), d1.end(), h1.begin());

    ASSERT_EQ_QUIET(e0_stream, e3.stream().native_handle());

    TEST_EVENT_WAIT(e3);

    ASSERT_EQ(h0, h1);
    ASSERT_EQ(h0, d0);
    ASSERT_EQ(h0, d1);
    ASSERT_EQ(h0, d2);
  }
}

TYPED_TEST(AsyncCopyTests, test_async_copy_after_test)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_copy_after<T>();
}

///////////////////////////////////////////////////////////////////////////////

// TODO: device_to_device NonContiguousIterator output (discard_iterator).

// TODO: host_to_device non trivially relocatable.

// TODO: device_to_host non trivially relocatable.

// TODO: host_to_device NonContiguousIterator input (counting_iterator).

// TODO: host_to_device NonContiguousIterator output (discard_iterator).

// TODO: device_to_host NonContiguousIterator input (counting_iterator).

// TODO: device_to_host NonContiguousIterator output (discard_iterator).

// TODO: Mixed types, needs loosening of `is_trivially_relocatable_to` logic.

// TODO: H->D copy, then dependent D->H copy (round trip).
// Can't do this today because we can't do cross-system with explicit policies.

#endif

// we need to leak the suppression on clang/MSVC to suppresses warnings from the cudafe1.stub.c file
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_CLANG && THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
THRUST_SUPPRESS_DEPRECATED_POP
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_CLANG && THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
