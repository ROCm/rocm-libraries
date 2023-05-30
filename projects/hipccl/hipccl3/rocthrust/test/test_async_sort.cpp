#include <thrust/detail/config.h>

#include <thrust/async/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncSortTests, NumericalTestsParams);

enum wait_policy
{
  wait_for_futures
, do_not_wait_for_futures
};

template <typename T>
struct custom_greater
{
  __host__ __device__
  bool operator()(T rhs, T lhs) const
  {
    return lhs > rhs;
  }
};

#define DEFINE_SORT_INVOKER(name, ...)                                        \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_INVOKER(
  sort_invoker
);
DEFINE_SORT_INVOKER(
  sort_invoker_device, thrust::device
);

#define DEFINE_SORT_OP_INVOKER(name, op, ...)                                 \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_OP_INVOKER(
  sort_invoker_less,        thrust::less
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_less_device, thrust::less, thrust::device
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater,        thrust::greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater_device, thrust::greater, thrust::device
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater,        custom_greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater_device, custom_greater, thrust::device
);

#undef DEFINE_SORT_INVOKER
#undef DEFINE_SORT_OP_INVOKER

TYPED_TEST(AsyncSortTests, AsyncSortInstance)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d0_data(h0_data);

            ASSERT_EQ(h0_data, d0_data);

            thrust::sort(
                h0_data.begin(), h0_data.end()
            );

            auto f0 = thrust::async::sort(
               d0_data.begin(), d0_data.end()
            );

            f0.wait();

            ASSERT_EQ(h0_data, d0_data);
        }
    }
};

TYPED_TEST(AsyncSortTests, AsyncSortWithPolicyInstance)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d0_data(h0_data);

            ASSERT_EQ(h0_data, d0_data);

            thrust::sort(
                h0_data.begin(), h0_data.end()
            );

            auto f0 = thrust::async::sort(
              thrust::device, d0_data.begin(), d0_data.end()
            );

            f0.wait();

            ASSERT_EQ(h0_data, d0_data);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template < typename T, template <typename> class SortInvoker, wait_policy WaitPolicy>
void TestAsyncSort()
{
  for(auto size : get_sizes())
  {
      SCOPED_TRACE(testing::Message() << "with size = " << size);
      for(auto seed : get_seeds())
      {
          SCOPED_TRACE(testing::Message() << "with seed= " << seed);

          thrust::host_vector<T>   h0_data = get_random_data<T>(
              size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
          thrust::device_vector<T> d0_data(h0_data);

          SortInvoker<T>::sync(
            h0_data.begin(), h0_data.end()
          );

          auto f0 = SortInvoker<T>::async(
            d0_data.begin(), d0_data.end()
          );

          if (wait_for_futures == WaitPolicy)
          {
            f0.wait();

            ASSERT_EQ(h0_data, d0_data);
          }
      }
  }
}

TYPED_TEST(AsyncSortTests, AsyncSort)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortPolicy)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_device, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortLess)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_less, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortLessNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_less, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortPolicyLess)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_less_device, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortGreater)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_greater, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortGreaterNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_greater, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortPolicyGreaterNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_greater_device, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortCustomGreater)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_custom_greater, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortCustomGreaterNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_custom_greater, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortPolicyCustomGreater)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_custom_greater_device, wait_for_futures>();
};

TYPED_TEST(AsyncSortTests, AsyncSortPolicyCustomGreaterNoWait)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    TestAsyncSort<T, sort_invoker_custom_greater_device, do_not_wait_for_futures>();
};

///////////////////////////////////////////////////////////////////////////////

// TODO: Async copy then sort.

// TODO: Test future return type.
