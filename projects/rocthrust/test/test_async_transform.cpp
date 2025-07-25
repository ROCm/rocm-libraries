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

#  include <thrust/async/copy.h>
#  include <thrust/async/transform.h>
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>

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

TESTS_DEFINE(AsyncTransformTests, NumericalTestsParams);
TESTS_DEFINE(AsyncTransformBuiltinTests, BuiltinNumericTypes);

THRUST_SUPPRESS_DEPRECATED_PUSH

template <typename T>
struct divide_by_2
{
  THRUST_HOST_DEVICE T operator()(T x) const
  {
    return x / 2;
  }
};

#  define DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(NAME, MEMBERS, CTOR, DTOR, VALIDATE, ...)             \
    template <typename T>                                                                                     \
    struct NAME                                                                                               \
    {                                                                                                         \
      MEMBERS                                                                                                 \
                                                                                                              \
      NAME()                                                                                                  \
      {                                                                                                       \
        CTOR                                                                                                  \
      }                                                                                                       \
                                                                                                              \
      ~NAME()                                                                                                 \
      {                                                                                                       \
        DTOR                                                                                                  \
      }                                                                                                       \
                                                                                                              \
      template <typename Event>                                                                               \
      void validate_event(Event& e)                                                                           \
      {                                                                                                       \
        THRUST_UNUSED_VAR(e);                                                                                 \
        VALIDATE                                                                                              \
      }                                                                                                       \
                                                                                                              \
      template <typename ForwardIt, typename Sentinel, typename OutputIt, typename UnaryOperation>            \
      THRUST_HOST auto operator()(ForwardIt&& first, Sentinel&& last, OutputIt&& output, UnaryOperation&& op) \
        THRUST_DECLTYPE_RETURNS(::thrust::async::transform(__VA_ARGS__))                                      \
    };                                                                                                        \
    /**/

#  define DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                                            \
    DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(                                                   \
      NAME, THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), __VA_ARGS__) \
    /**/

#  define DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                                                      \
    template <typename T>                                                                                     \
    struct NAME                                                                                               \
    {                                                                                                         \
      template <typename ForwardIt, typename Sentinel, typename OutputIt, typename UnaryOperation>            \
      THRUST_HOST auto operator()(ForwardIt&& first, Sentinel&& last, OutputIt&& output, UnaryOperation&& op) \
        THRUST_RETURNS(::thrust::transform(__VA_ARGS__))                                                      \
    };                                                                                                        \
    /**/

DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker, THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output), THRUST_FWD(op));
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device,
  thrust::device,
  THRUST_FWD(first),
  THRUST_FWD(last),
  THRUST_FWD(output),
  THRUST_FWD(op));
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator,
  thrust::device(thrust::device_allocator<void>{}),
  THRUST_FWD(first),
  THRUST_FWD(last),
  THRUST_FWD(output),
  THRUST_FWD(op));
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_on
  // Members.
  ,
  SPECIALIZE_DEVICE_RESOURCE_NAME(Stream_t) stream_;
  // Constructor.
  ,
  thrust::THRUST_DEVICE_BACKEND_DETAIL::throw_on_error(SPECIALIZE_DEVICE_RESOURCE_NAME(StreamCreateWithFlags)(
    &stream_, SPECIALIZE_DEVICE_RESOURCE_NAME(StreamNonBlocking)));
  // Destructor.
  ,
  thrust::THRUST_DEVICE_BACKEND_DETAIL::throw_on_error(SPECIALIZE_DEVICE_RESOURCE_NAME(StreamDestroy)(stream_));
  // `validate_event` member.
  ,
  ASSERT_EQ_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::transform`.
  ,
  thrust::device.on(stream_),
  THRUST_FWD(first),
  THRUST_FWD(last),
  THRUST_FWD(output),
  THRUST_FWD(op));
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator_on
  // Members.
  ,
  SPECIALIZE_DEVICE_RESOURCE_NAME(Stream_t) stream_;
  // Constructor.
  ,
  thrust::THRUST_DEVICE_BACKEND_DETAIL::throw_on_error(SPECIALIZE_DEVICE_RESOURCE_NAME(StreamCreateWithFlags)(
    &stream_, SPECIALIZE_DEVICE_RESOURCE_NAME(StreamNonBlocking)));
  // Destructor.
  ,
  thrust::THRUST_DEVICE_BACKEND_DETAIL::throw_on_error(SPECIALIZE_DEVICE_RESOURCE_NAME(StreamDestroy)(stream_));
  // `validate_event` member.
  ,
  ASSERT_EQ_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::transform`.
  ,
  thrust::device(thrust::device_allocator<void>{}).on(stream_),
  THRUST_FWD(first),
  THRUST_FWD(last),
  THRUST_FWD(output),
  THRUST_FWD(op));

DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_sync_invoker, THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output), THRUST_FWD(op));

///////////////////////////////////////////////////////////////////////////////

template <typename T,
          template <typename>
          class AsyncTransformUnaryInvoker,
          template <typename>
          class SyncTransformUnaryInvoker,
          template <typename>
          class UnaryOperation>
THRUST_HOST void test_async_transform_unary()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));

    thrust::device_vector<T> d0a(h0);
    thrust::device_vector<T> d0b(h0);
    thrust::device_vector<T> d0c(h0);
    thrust::device_vector<T> d0d(h0);

    thrust::host_vector<T> h1(size);

    thrust::device_vector<T> d1a(size);
    thrust::device_vector<T> d1b(size);
    thrust::device_vector<T> d1c(size);
    thrust::device_vector<T> d1d(size);

    AsyncTransformUnaryInvoker<T> invoke_async;
    SyncTransformUnaryInvoker<T> invoke_sync;

    UnaryOperation<T> op;

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);
    ASSERT_EQ(h0, d0c);
    ASSERT_EQ(h0, d0d);

    auto f0a = invoke_async(d0a.begin(), d0a.end(), d1a.begin(), op);
    auto f0b = invoke_async(d0b.begin(), d0b.end(), d1b.begin(), op);
    auto f0c = invoke_async(d0c.begin(), d0c.end(), d1c.begin(), op);
    auto f0d = invoke_async(d0d.begin(), d0d.end(), d1d.begin(), op);

    invoke_async.validate_event(f0a);
    invoke_async.validate_event(f0b);
    invoke_async.validate_event(f0c);
    invoke_async.validate_event(f0d);

    // This potentially runs concurrently with the copies.
    invoke_sync(h0.begin(), h0.end(), h1.begin(), op);

    TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);
    ASSERT_EQ(h0, d0c);
    ASSERT_EQ(h0, d0d);

    ASSERT_EQ(h1, d1a);
    ASSERT_EQ(h1, d1b);
    ASSERT_EQ(h1, d1c);
    ASSERT_EQ(h1, d1d);
  }
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<T, transform_unary_async_invoker, transform_unary_sync_invoker, divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_policy_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<T, transform_unary_async_invoker_device, transform_unary_sync_invoker, divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_policy_allocator_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<T,
                             transform_unary_async_invoker_device_allocator,
                             transform_unary_sync_invoker,
                             divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_policy_on_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<T, transform_unary_async_invoker_device_on, transform_unary_sync_invoker, divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_policy_allocator_on_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<T,
                             transform_unary_async_invoker_device_allocator_on,
                             transform_unary_sync_invoker,
                             divide_by_2>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T,
          template <typename>
          class AsyncTransformUnaryInvoker,
          template <typename>
          class SyncTransformUnaryInvoker,
          template <typename>
          class UnaryOperation>
THRUST_HOST void test_async_transform_unary_inplace()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));

    thrust::device_vector<T> d0a(h0);
    thrust::device_vector<T> d0b(h0);
    thrust::device_vector<T> d0c(h0);
    thrust::device_vector<T> d0d(h0);

    AsyncTransformUnaryInvoker<T> invoke_async;
    SyncTransformUnaryInvoker<T> invoke_sync;

    UnaryOperation<T> op;

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);
    ASSERT_EQ(h0, d0c);
    ASSERT_EQ(h0, d0d);

    auto f0a = invoke_async(d0a.begin(), d0a.end(), d0a.begin(), op);
    auto f0b = invoke_async(d0b.begin(), d0b.end(), d0b.begin(), op);
    auto f0c = invoke_async(d0c.begin(), d0c.end(), d0c.begin(), op);
    auto f0d = invoke_async(d0d.begin(), d0d.end(), d0d.begin(), op);

    invoke_async.validate_event(f0a);
    invoke_async.validate_event(f0b);
    invoke_async.validate_event(f0c);
    invoke_async.validate_event(f0d);

    // This potentially runs concurrently with the copies.
    invoke_sync(h0.begin(), h0.end(), h0.begin(), op);

    TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);
    ASSERT_EQ(h0, d0c);
    ASSERT_EQ(h0, d0d);
  }
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_inplace_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<T, transform_unary_async_invoker, transform_unary_sync_invoker, divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_inplace_policy_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<T, transform_unary_async_invoker_device, transform_unary_sync_invoker, divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_inplace_policy_allocator_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<T,
                                     transform_unary_async_invoker_device_allocator,
                                     transform_unary_sync_invoker,
                                     divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_inplace_policy_on_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<T,
                                     transform_unary_async_invoker_device_on,
                                     transform_unary_sync_invoker,
                                     divide_by_2>();
}

TYPED_TEST(AsyncTransformTests, test_async_transform_unary_inplace_policy_allocator_on_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<T,
                                     transform_unary_async_invoker_device_allocator_on,
                                     transform_unary_sync_invoker,
                                     divide_by_2>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T,
          template <typename>
          class AsyncTransformUnaryInvoker,
          template <typename>
          class SyncTransformUnaryInvoker,
          template <typename>
          class UnaryOperation>
THRUST_HOST void test_async_transform_unary_counting_iterator()
{
  constexpr std::size_t n = 15 * sizeof(T);

  ASSERT_LE(T(n), truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T> first(0);
  thrust::counting_iterator<T> last(n);

  thrust::host_vector<T> h0(n);

  thrust::device_vector<T> d0a(n);
  thrust::device_vector<T> d0b(n);
  thrust::device_vector<T> d0c(n);
  thrust::device_vector<T> d0d(n);

  AsyncTransformUnaryInvoker<T> invoke_async;
  SyncTransformUnaryInvoker<T> invoke_sync;

  UnaryOperation<T> op;

  auto f0a = invoke_async(first, last, d0a.begin(), op);
  auto f0b = invoke_async(first, last, d0b.begin(), op);
  auto f0c = invoke_async(first, last, d0c.begin(), op);
  auto f0d = invoke_async(first, last, d0d.begin(), op);

  invoke_async.validate_event(f0a);
  invoke_async.validate_event(f0b);
  invoke_async.validate_event(f0c);
  invoke_async.validate_event(f0d);

  // This potentially runs concurrently with the copies.
  invoke_sync(first, last, h0.begin(), op);

  TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

  ASSERT_EQ(h0, d0a);
  ASSERT_EQ(h0, d0b);
  ASSERT_EQ(h0, d0c);
  ASSERT_EQ(h0, d0d);
}

TYPED_TEST(AsyncTransformBuiltinTests, test_async_transform_unary_counting_iterator_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_counting_iterator<T,
                                               transform_unary_async_invoker,
                                               transform_unary_sync_invoker,
                                               divide_by_2>();
}

TYPED_TEST(AsyncTransformBuiltinTests, test_async_transform_unary_counting_iterator_policy_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_counting_iterator<T,
                                               transform_unary_async_invoker_device,
                                               transform_unary_sync_invoker,
                                               divide_by_2>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename> class UnaryOperation>
THRUST_HOST void test_async_transform_using()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h0(random_integers<T>(size));

    thrust::device_vector<T> d0a(h0);
    thrust::device_vector<T> d0b(h0);

    thrust::host_vector<T> h1(size);

    thrust::device_vector<T> d1a(size);
    thrust::device_vector<T> d1b(size);

    UnaryOperation<T> op;

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);

    thrust::device_event f0a;
    thrust::device_event f0b;

    // When you import the customization points into the global namespace,
    // they should be selected instead of the synchronous algorithms.
    {
      THRUST_SUPPRESS_DEPRECATED_PUSH
      using namespace thrust::async;
      f0a = transform(d0a.begin(), d0a.end(), d1a.begin(), op);
      THRUST_SUPPRESS_DEPRECATED_POP
    }
    {
      THRUST_SUPPRESS_DEPRECATED_PUSH
      using thrust::async::transform;
      f0b = transform(d0b.begin(), d0b.end(), d1b.begin(), op);
      THRUST_SUPPRESS_DEPRECATED_POP
    }

    // ADL should find the synchronous algorithms.
    // This potentially runs concurrently with the copies.
    transform(h0.begin(), h0.end(), h1.begin(), op);

    TEST_EVENT_WAIT(thrust::when_all(f0a, f0b));

    ASSERT_EQ(h0, d0a);
    ASSERT_EQ(h0, d0b);

    ASSERT_EQ(h1, d1a);
    ASSERT_EQ(h1, d1b);
  }
}

TYPED_TEST(AsyncTransformTests, test_async_transform_using_divide_by_2)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_using<T, divide_by_2>();
}

///////////////////////////////////////////////////////////////////////////////

#endif
