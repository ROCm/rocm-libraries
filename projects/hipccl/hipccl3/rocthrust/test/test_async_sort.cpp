#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/async/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncSortTests, NumericalTestsParams);

template <typename T>
struct custom_greater
{
  __host__ __device__
  bool operator()(T rhs, T lhs) const
  {
    return lhs > rhs;
  }
};

TYPED_TEST(AsyncSortTests, AsyncSortInstance)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);
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
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);
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

template <typename T, typename Operator>
void AsyncSortWithOperator()
{
  for(auto size : get_sizes())
  {
      SCOPED_TRACE(testing::Message() << "with size = " << size);
      for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
      {
          unsigned int seed_value
              = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
          SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

          thrust::host_vector<T>   h0_data = get_random_data<T>(
              size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);
          thrust::device_vector<T> d0_data(h0_data);

          ASSERT_EQ(h0_data, d0_data);

          Operator op{};

          thrust::sort(
            h0_data.begin(), h0_data.end(), op
          );

          auto f0 = thrust::async::sort(
            d0_data.begin(), d0_data.end(), op
          );

          f0.wait();

          ASSERT_EQ(h0_data, d0_data);
      }
  }
}

TYPED_TEST(AsyncSortTests, AsyncSortWithOpCustomGreater)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithOperator<T, custom_greater<T>>();
}

TYPED_TEST(AsyncSortTests, AsyncSortWithOpThrustGreater)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithOperator<T, thrust::greater<T>>();
}

TYPED_TEST(AsyncSortTests, AsyncSortWithOpThrustLess)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithOperator<T, thrust::less<T>>();
}

template <typename T, typename Operator>
void AsyncSortWithPolicyOperator(void)
{
  for(auto size : get_sizes())
  {
      SCOPED_TRACE(testing::Message() << "with size = " << size);
      for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
      {
          unsigned int seed_value
              = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
          SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

          thrust::host_vector<T>   h0_data = get_random_data<T>(
              size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);
          thrust::device_vector<T> d0_data(h0_data);

          ASSERT_EQ(h0_data, d0_data);

          Operator op{};

          thrust::sort(
            h0_data.begin(), h0_data.end(), op
          );

          auto f0 = thrust::async::sort(
            thrust::device, d0_data.begin(), d0_data.end(), op
          );

          f0.wait();

          ASSERT_EQ(h0_data, d0_data);
      }
  }
}

TYPED_TEST(AsyncSortTests, AsyncSortWithPolicyOpCustomGreater)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithPolicyOperator<T, custom_greater<T>>();
}

TYPED_TEST(AsyncSortTests, AsyncSortWithPolicyOpThrustGreater)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithPolicyOperator<T, thrust::greater<T>>();
}

TYPED_TEST(AsyncSortTests, AsyncSortWithPolicyOpThrustLess)
{
    using T = typename TestFixture::input_type;
    AsyncSortWithPolicyOperator<T, thrust::less<T>>();
}


// TODO: Async copy then sort.

// TODO: Test future return type.

#endif // THRUST_CPP_DIALECT >= 2011
