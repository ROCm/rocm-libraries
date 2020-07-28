#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncCopyTests, NumericalTestsParams);

TYPED_TEST(AsyncCopyTests, TestAsyncCopyHostToDeviceTriviallyRelocatable)
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
            thrust::device_vector<T> d0_data(size);

            auto f0 = thrust::async::copy(
                h0_data.begin(), h0_data.end(), d0_data.begin()
            );

            std::move(f0).get();

            ASSERT_EQ(h0_data, d0_data);
        }
    }
};

TYPED_TEST(AsyncCopyTests, TestAsyncCopyDeviceToHostTriviallyRelocatable)
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
            thrust::device_vector<T> h1_data(size);
            thrust::device_vector<T> d0_data(size);

            thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

            ASSERT_EQ(h0_data, d0_data);

            auto f0 = thrust::async::copy(
                d0_data.begin(), d0_data.end(), h1_data.begin()
            );

            std::move(f0).get();

            ASSERT_EQ(h0_data, d0_data);
            ASSERT_EQ(d0_data, h1_data);
        }
    }
};

TYPED_TEST(AsyncCopyTests, TestAsyncCopyDevicetoDevice)
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
            thrust::device_vector<T> d0_data(size);
            thrust::device_vector<T> d1_data(size);

            thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

            ASSERT_EQ(h0_data, d0_data);

            auto f0 = thrust::async::copy(d0_data.begin(), d0_data.end(), d1_data.begin());

            std::move(f0).get();

            ASSERT_EQ(h0_data, d0_data);
            ASSERT_EQ(d0_data, d1_data);
        }
    }
}

TYPED_TEST(AsyncCopyTests, TestAsyncCopyDevicetoDeviceWithPolicy)
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
            thrust::device_vector<T> d0_data(size);
            thrust::device_vector<T> d1_data(size);

            thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

            ASSERT_EQ(h0_data, d0_data);

            auto f0 = thrust::async::copy(
              thrust::device, d0_data.begin(), d0_data.end(), d1_data.begin()
            );

            std::move(f0).get();

            ASSERT_EQ(h0_data, d0_data);
            ASSERT_EQ(d0_data, d1_data);
        }
    }
};

// TODO: device_to_device implicit.

// TODO: device_to_device NonContiguousIterator input (counting_iterator).

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

#endif // THRUST_CPP_DIALECT >= 2011
