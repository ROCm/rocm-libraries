#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/async/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncReduceTests, NumericalTestsParams);

template <typename T>
struct custom_plus
{
  __host__ __device__
  T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

TYPED_TEST(AsyncReduceTests, TestAsyncReduce)
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

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end()
            );

            auto f0 = thrust::async::reduce(
              d0_data.begin(), d0_data.end()
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
}

TYPED_TEST(AsyncReduceTests, TestAsyncReduceWithPolicy)
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

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end()
            );

            auto f0 = thrust::async::reduce(
              thrust::device, d0_data.begin(), d0_data.end()
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
}

TYPED_TEST(AsyncReduceTests, TestAsyncReduceWithInit)
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

            T const init = get_random_data<T>(
                1, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value + 1
            )[0];

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end(), init
            );

            auto f0 = thrust::async::reduce(
              d0_data.begin(), d0_data.end(), init
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
}

TYPED_TEST(AsyncReduceTests, TestAsyncReduceWithPolicyInit)
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

            T const init = get_random_data<T>(
                1, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value + 1
            )[0];

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end(), init
            );

            auto f0 = thrust::async::reduce(
              thrust::device, d0_data.begin(), d0_data.end(), init
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
}

TYPED_TEST(AsyncReduceTests, TestAsyncReduceWithInitOp)
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

            T const init = get_random_data<T>(
                1, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value + 1
            )[0];
            custom_plus<T> op{};

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end(), init, op
            );

            auto f0 = thrust::async::reduce(
              d0_data.begin(), d0_data.end(), init, op
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
};

TYPED_TEST(AsyncReduceTests, TestAsyncReduceWithPolicyInitOp)
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

            T const init = get_random_data<T>(
                1, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value + 1
            )[0];
            custom_plus<T> op{};

            auto r0 = thrust::reduce(
              h0_data.begin(), h0_data.end(), init, op
            );

            auto f0 = thrust::async::reduce(
              thrust::device, d0_data.begin(), d0_data.end(), init, op
            );

            auto r1 = std::move(f0).get();

            ASSERT_EQ(r0, r1);
        }
    }
}

// TODO: Async copy then reduce.

// TODO: Device-side reduction usage.

// TODO: Make random_integers more generic.

#endif // THRUST_CPP_DIALECT >= 2011
