#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)

#include <thrust/async/transform.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

template <typename T>
struct divide_by_2
{
  __host__ __device__
  T operator()(T x) const
  {
    return x / 2;
  }
};

#define DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(                        \
    NAME, MEMBERS, CTOR, DTOR, VALIDATE, ...                                  \
  )                                                                           \
  template <typename T>                                                       \
  struct NAME                                                                 \
  {                                                                           \
    MEMBERS                                                                   \
                                                                              \
    NAME() { CTOR }                                                           \
                                                                              \
    ~NAME() { DTOR }                                                          \
                                                                              \
    template <typename Event>                                                 \
    void validate_event(Event& e)                                             \
    {                                                                         \
      THRUST_UNUSED_VAR(e);                                                   \
      VALIDATE                                                                \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel, typename OutputIt                \
    , typename UnaryOperation                                                 \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    , UnaryOperation&& op                                                     \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::transform(                                             \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

#define DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                       \
  DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(                              \
    NAME                                                                      \
  , THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY()\
  , __VA_ARGS__                                                               \
  )                                                                           \
  /**/

#define DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                        \
  template <typename T>                                                       \
  struct NAME                                                                 \
  {                                                                           \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel, typename OutputIt                \
    , typename UnaryOperation                                                 \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    , UnaryOperation&& op                                                     \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::transform(                                                    \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_on
  // Members.
, hipStream_t stream_;
  // Constructor.
, thrust::hip_rocprim::throw_on_error(
    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)
  );
  // Destructor.
, thrust::hip_rocprim::throw_on_error(
    hipStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQ_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::transform`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator_on
  // Members.
, hipStream_t stream_;
  // Constructor.
, thrust::hip_rocprim::throw_on_error(
    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)
  );
  // Destructor.
, thrust::hip_rocprim::throw_on_error(
    hipStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQ_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::transform`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);

DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_sync_invoker
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);

///////////////////////////////////////////////////////////////////////////////

TESTS_DEFINE(AsyncTransformTests, NumericalTestsParams);

template <
  typename T
, template <typename> class AsyncTransformUnaryInvoker
, template <typename> class SyncTransformUnaryInvoker
, template <typename> class UnaryOperation
>
void test_async_transform_unary()
{
  for(auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
      unsigned int seed_value
        = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
      SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

      thrust::host_vector<T> h0 = get_random_data<T>(size, T(-1000), T(1000), seed_value);

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

      //TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

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
};

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnary)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<
      T,
    transform_unary_async_invoker
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<
      T,
    transform_unary_async_invoker_device
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryDeviceAllocator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<
      T,
    transform_unary_async_invoker_device_allocator
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryDeviceOn)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<
      T,
    transform_unary_async_invoker_device_on
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryDeviceAllocatorOn)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary<
      T,
    transform_unary_async_invoker_device_allocator_on
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

///////////////////////////////////////////////////////////////////////////////

template <
  typename T
, template <typename> class AsyncTransformUnaryInvoker
, template <typename> class SyncTransformUnaryInvoker
, template <typename> class UnaryOperation
>
void test_async_transform_unary_inplace()
{
  for(auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
      unsigned int seed_value
        = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
      SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

      thrust::host_vector<T> h0 = get_random_data<T>(size, T(-1000), T(1000), seed_value);

      thrust::device_vector<T> d0a(h0);
      thrust::device_vector<T> d0b(h0);
      thrust::device_vector<T> d0c(h0);
      thrust::device_vector<T> d0d(h0);

      AsyncTransformUnaryInvoker<T> invoke_async;
      SyncTransformUnaryInvoker<T>  invoke_sync;

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

      //TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

      ASSERT_EQ(h0, d0a);
      ASSERT_EQ(h0, d0b);
      ASSERT_EQ(h0, d0c);
      ASSERT_EQ(h0, d0d);
    }
  }
};

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryInplace)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<
    T,
    transform_unary_async_invoker
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryInplaceDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<
    T,
    transform_unary_async_invoker_device
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryInplaceDeviceAllocator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<
    T,
    transform_unary_async_invoker_device_allocator
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryInplaceDeviceOn)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<
    T,
    transform_unary_async_invoker_device_on
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

TYPED_TEST(AsyncTransformTests, TestAsyncTransformUnaryInplaceDeviceAllocatorOn)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  test_async_transform_unary_inplace<
    T,
    transform_unary_async_invoker_device_allocator_on
  , transform_unary_sync_invoker
  , divide_by_2
  >();
}

///////////////////////////////////////////////////////////////////////////////

#endif
