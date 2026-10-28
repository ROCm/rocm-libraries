// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_TOPK_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_TOPK_HPP_

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_topk.hpp>
#include <hipcub/util_type.hpp>

// required test headers
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

// std library
#include <algorithm>
#include <iostream>
#include <vector>

#include _HIPCUB_LIBCXX_INCLUDE(__execution/output_ordering.h)
#include _HIPCUB_LIBCXX_INCLUDE(__execution/determinism.h)
#include _HIPCUB_LIBCXX_INCLUDE(__execution/require.h)
#include _HIPCUB_STD_INCLUDE(__execution/env.h)
#include _HIPCUB_STD_INCLUDE(__iterator/reverse_iterator.h)
#include _HIPCUB_STD_INCLUDE(__functional/identity.h)

template<class KeyType,
         class ValueType,
         class SizeType   = size_t,
         bool Descending  = false,
         bool UseIterator = false>
struct DeviceTopkParams
{
    using key_type                     = KeyType;
    using value_type                   = ValueType;
    using size_type                    = SizeType;
    static constexpr bool descending   = Descending;
    static constexpr bool use_iterator = UseIterator;
};

template<class Params>
class HipcubDeviceTopkTests : public ::testing::Test
{
public:
    using key_type                     = typename Params::key_type;
    using value_type                   = typename Params::value_type;
    using size_type                    = typename Params::size_type;
    static constexpr bool descending   = Params::descending;
    static constexpr bool use_iterator = Params::use_iterator;
};

template<bool Descending,
         typename KeyInputIteratorT,
         typename KeyOutputIteratorT,
         typename NumItemsT,
         typename NumOutItemsT,
         typename EnvT = _HIPCUB_STD_EXEC::env<>>
auto invoke_topk(void*              d_temporary_storage,
                 size_t&            temporary_storage_bytes,
                 KeyInputIteratorT  d_keys_in,
                 KeyOutputIteratorT d_keys_out,
                 NumItemsT          size,
                 NumOutItemsT       k,
                 EnvT               env = {}) -> std::enable_if_t<Descending /*Max*/, hipError_t>
{
    return hipcub::DeviceTopK::MaxKeys(d_temporary_storage,
                                       temporary_storage_bytes,
                                       d_keys_in,
                                       d_keys_out,
                                       size,
                                       k,
                                       env);
}

template<bool Descending,
         typename KeyInputIteratorT,
         typename KeyOutputIteratorT,
         typename NumItemsT,
         typename NumOutItemsT,
         typename EnvT = _HIPCUB_STD_EXEC::env<>>
auto invoke_topk(void*              d_temporary_storage,
                 size_t&            temporary_storage_bytes,
                 KeyInputIteratorT  d_keys_in,
                 KeyOutputIteratorT d_keys_out,
                 NumItemsT          size,
                 NumOutItemsT       k,
                 EnvT               env = {}) -> std::enable_if_t<!Descending /*Min*/, hipError_t>
{
    return hipcub::DeviceTopK::MinKeys(d_temporary_storage,
                                       temporary_storage_bytes,
                                       d_keys_in,
                                       d_keys_out,
                                       size,
                                       k,
                                       env);
}

template<bool Descending,
         typename KeyInputIteratorT,
         typename KeyOutputIteratorT,
         typename ValueInputIteratorT,
         typename ValueOutputIteratorT,
         typename NumItemsT,
         typename NumOutItemsT,
         typename EnvT = _HIPCUB_STD_EXEC::env<>>
auto invoke_topk(void*                d_temporary_storage,
                 size_t&              temporary_storage_bytes,
                 KeyInputIteratorT    d_keys_in,
                 KeyOutputIteratorT   d_keys_out,
                 ValueInputIteratorT  d_values_in,
                 ValueOutputIteratorT d_values_out,
                 NumItemsT            size,
                 NumOutItemsT         k,
                 EnvT                 env = {}) -> std::enable_if_t<Descending /*Max*/, hipError_t>
{
    return hipcub::DeviceTopK::MaxPairs(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_in,
                                        d_keys_out,
                                        d_values_in,
                                        d_values_out,
                                        size,
                                        k,
                                        env);
}

template<bool Descending,
         typename KeyInputIteratorT,
         typename KeyOutputIteratorT,
         typename ValueInputIteratorT,
         typename ValueOutputIteratorT,
         typename NumItemsT,
         typename NumOutItemsT,
         typename EnvT = _HIPCUB_STD_EXEC::env<>>
auto invoke_topk(void*                d_temporary_storage,
                 size_t&              temporary_storage_bytes,
                 KeyInputIteratorT    d_keys_in,
                 KeyOutputIteratorT   d_keys_out,
                 ValueInputIteratorT  d_values_in,
                 ValueOutputIteratorT d_values_out,
                 NumItemsT            size,
                 NumOutItemsT         k,
                 EnvT                 env = {}) -> std::enable_if_t<!Descending /*Min*/, hipError_t>
{
    return hipcub::DeviceTopK::MinPairs(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_in,
                                        d_keys_out,
                                        d_values_in,
                                        d_values_out,
                                        size,
                                        k,
                                        env);
}

template<class T, class SizeT>
using size_grouped_params = ::testing::Types<
    // Non-deterministic, unsorted
    // Min
    DeviceTopkParams<T, size_t, SizeT, false /*Descending*/>,
    // Max
    DeviceTopkParams<T, size_t, SizeT, true /*Descending*/>,

    // Non-deterministic, unsorted, with iterators
    // Min
    DeviceTopkParams<T, size_t, SizeT, false /*Descending*/, true /*UseIterator*/>,
    // Max
    DeviceTopkParams<T, size_t, SizeT, true /*Descending*/, true /*UseIterator*/>>;

template<class T>
using grouped_params = typename test_utils::merge_sequence<size_grouped_params<T, uint32_t>,
                                                           size_grouped_params<T, uint64_t>>::type;

using HipcubDeviceTopkTestsParams =
    typename test_utils::merge_sequence<grouped_params<uint8_t>
                                        // ,
                                        // grouped_params<uint64_t>,
                                        // grouped_params<float>
                                        // ,
                                        // grouped_params<rocprim::uint128_t>
                                        // ,
                                        // grouped_params<rocprim::half>
                                        >::type;

TYPED_TEST_SUITE(HipcubDeviceTopkTests, HipcubDeviceTopkTestsParams);

template<bool Wrap, typename T, typename OpT>
inline auto wrap_in_transform_iterator(T* ptr, OpT op) ->
    typename std::enable_if<Wrap, test_utils::transform_iterator<T*, OpT>>::type
{
    return test_utils::transform_iterator<T*, OpT>(ptr, op);
}

template<bool Wrap, typename T, typename OpT>
inline auto wrap_in_transform_iterator(T* ptr, OpT /*op*/) ->
    typename std::enable_if<!Wrap, T*>::type
{
    return ptr;
}

template<bool Wrap, typename T>
inline auto wrap_in_counting_iterator(T* /*ptr*/, T init) ->
    typename std::enable_if<Wrap, test_utils::counting_iterator<T>>::type
{
    return test_utils::counting_iterator<T>(init);
}

template<bool Wrap, typename T>
inline auto wrap_in_counting_iterator(T* ptr, T /*init*/) ->
    typename std::enable_if<!Wrap, T*>::type
{
    return ptr;
}

TYPED_TEST(HipcubDeviceTopkTests, TopkKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type              = typename TestFixture::key_type;
    using size_type             = typename TestFixture::size_type;
    constexpr bool descending   = TestFixture::descending;
    constexpr bool use_iterator = TestFixture::use_iterator;

    hipStream_t stream     = 0; // default
    auto        stream_env = _HIPCUB_STD_EXEC::prop{_HIPCUB_LIBCXX::get_stream_t{},
                                             _HIPCUB_LIBCXX::stream_ref{stream}};
    auto        requirements
        = _HIPCUB_LIBCXX::execution::require(_HIPCUB_LIBCXX::execution::determinism::not_guaranteed,
                                             _HIPCUB_LIBCXX::execution::output_ordering::unsorted);
    auto env = _HIPCUB_STD_EXEC::env{stream_env, requirements};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_type size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            size_type k = 0;
            if(size > 0)
            {
                k = test_utils::get_random_value<size_type>(0, size - 1, seed_value);
            }
            SCOPED_TRACE(testing::Message() << "with k = " << k);

            const auto memory_size     = size * sizeof(key_type);
            const int  out_memory_size = k * sizeof(key_type);
            const auto op              = _HIPCUB_STD::identity{};

            // Generate data
            std::vector<key_type> keys(size);
            std::iota(keys.begin(), keys.end(), static_cast<key_type>(0));

            // Calculate expected results on host
            std::vector<key_type> expected = keys;
            std::stable_sort(expected.begin(),
                             expected.end(),
                             test_utils::key_comparator<key_type,
                                                        descending,
                                                        0 /*StartBit*/,
                                                        sizeof(key_type) * 8 /*EndBit*/>());

            // Allocate device memory
            key_type* d_keys;
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys, memory_size));
            HIP_CHECK(hipMemcpy(d_keys, keys.data(), memory_size, hipMemcpyHostToDevice));

            key_type* d_keys_out;
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys_out, out_memory_size));

            auto d_keys_it = wrap_in_transform_iterator<use_iterator>(d_keys, op);

            // Get size of temporary storage
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_topk<descending>(nullptr,
                                              temporary_storage_bytes,
                                              d_keys_it,
                                              d_keys_out,
                                              size,
                                              k,
                                              env));

            // Allocate temporary storage
            ASSERT_GT(temporary_storage_bytes, 0);

            void* d_temporary_storage;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            // Run
            HIP_CHECK(invoke_topk<descending>(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys_it,
                                              d_keys_out,
                                              size,
                                              k,
                                              env));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Get results from device. Order them for easier verification
            std::vector<key_type> output(size);
            HIP_CHECK(hipMemcpy(output.data(), d_keys_out, out_memory_size, hipMemcpyDeviceToHost));
            std::stable_sort(output.begin(),
                             output.begin() + k,
                             test_utils::key_comparator<key_type,
                                                        descending,
                                                        0 /*StartBit*/,
                                                        sizeof(key_type) * 8 /*EndBit*/>());

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_keys_out));
            HIP_CHECK(hipFree(d_temporary_storage));

            // Verify results
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, k /*max_length*/));
        }
    }
}

TYPED_TEST(HipcubDeviceTopkTests, TopkPairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type              = typename TestFixture::key_type;
    using value_type            = typename TestFixture::value_type;
    using key_value             = std::pair<key_type, value_type>;
    using size_type             = typename TestFixture::size_type;
    constexpr bool descending   = TestFixture::descending;
    constexpr bool use_iterator = TestFixture::use_iterator;

    hipStream_t stream     = 0; // default
    auto        stream_env = _HIPCUB_STD_EXEC::prop{_HIPCUB_LIBCXX::get_stream_t{},
                                             _HIPCUB_LIBCXX::stream_ref{stream}};
    auto        requirements
        = _HIPCUB_LIBCXX::execution::require(_HIPCUB_LIBCXX::execution::determinism::not_guaranteed,
                                             _HIPCUB_LIBCXX::execution::output_ordering::unsorted);
    auto env = _HIPCUB_STD_EXEC::env{stream_env, requirements};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_type size : test_utils::get_sizes(seed_value))
        {
            size
                = std::min<size_t>(static_cast<size_t>(size), std::numeric_limits<key_type>::max());
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            size_type k = 0;
            if(size > 0)
            {
                k = test_utils::get_random_value<size_type>(0, size - 1, seed_value);
            }
            SCOPED_TRACE(testing::Message() << "with k = " << k);

            const auto keys_memory_size       = size * sizeof(key_type);
            const int  out_keys_memory_size   = k * sizeof(key_type);
            const auto values_memory_size     = size * sizeof(value_type);
            const int  out_values_memory_size = k * sizeof(value_type);
            const auto op                     = _HIPCUB_STD::identity{};

            // Generate data
            std::vector<key_type> keys(size);
            std::iota(keys.begin(), keys.end(), static_cast<key_type>(0));

            std::vector<value_type> values(size);
            std::iota(values.begin(), values.end(), static_cast<value_type>(0));

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_type i = 0; i < size; i++)
            {
                expected[i] = key_value(keys[i], values[i]);
            }
            std::stable_sort(expected.begin(),
                             expected.end(),
                             test_utils::key_value_comparator<key_type,
                                                              value_type,
                                                              descending,
                                                              0 /*StartBit*/,
                                                              sizeof(key_type) * 8 /*EndBit*/>());

            // Allocate device memory
            key_type* d_keys;
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys, keys_memory_size));
            HIP_CHECK(hipMemcpy(d_keys, keys.data(), keys_memory_size, hipMemcpyHostToDevice));
            key_type* d_keys_out;
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys_out, out_keys_memory_size));

            value_type* d_values = nullptr;
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_values, values_memory_size));
            HIP_CHECK(
                hipMemcpy(d_values, values.data(), values_memory_size, hipMemcpyHostToDevice));
            value_type* d_values_out = nullptr;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_values_out, out_values_memory_size));

            auto d_keys_it   = wrap_in_transform_iterator<use_iterator>(d_keys, op);
            auto d_values_it = wrap_in_counting_iterator<use_iterator>(d_values, value_type{});

            // Get size of temporary storage
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_topk<descending>(nullptr,
                                              temporary_storage_bytes,
                                              d_keys_it,
                                              d_keys_out,
                                              d_values_it,
                                              d_values_out,
                                              size,
                                              k,
                                              env));

            // Allocate temporary storage
            ASSERT_GT(temporary_storage_bytes, 0);

            void* d_temporary_storage;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            // Run
            HIP_CHECK(invoke_topk<descending>(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys_it,
                                              d_keys_out,
                                              d_values_it,
                                              d_values_out,
                                              size,
                                              k,
                                              env));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Get results from device. Order them for easier verification
            HIP_CHECK(
                hipMemcpy(keys.data(), d_keys_out, out_keys_memory_size, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(values.data(),
                                d_values_out,
                                out_values_memory_size,
                                hipMemcpyDeviceToHost));
            std::vector<key_value> output(size);
            for(size_type i = 0; i < k; i++)
            {
                output[i] = key_value(keys[i], values[i]);
            }
            std::stable_sort(output.begin(),
                             output.begin() + k,
                             test_utils::key_value_comparator<key_type,
                                                              value_type,
                                                              descending,
                                                              0 /*StartBit*/,
                                                              sizeof(key_type) * 8 /*EndBit*/>());

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_keys_out));
            HIP_CHECK(hipFree(d_values));
            HIP_CHECK(hipFree(d_values_out));
            HIP_CHECK(hipFree(d_temporary_storage));

            // Verify results
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, k /*max_length*/));
        }
    }
}

template<bool Descending>
void topk_keys_large_sizes_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using size_type = uint64_t;
    using key_type  = size_type;

    hipStream_t stream     = 0; // default
    auto        stream_env = _HIPCUB_STD_EXEC::prop{_HIPCUB_LIBCXX::get_stream_t{},
                                             _HIPCUB_LIBCXX::stream_ref{stream}};
    auto        requirements
        = _HIPCUB_LIBCXX::execution::require(_HIPCUB_LIBCXX::execution::determinism::not_guaranteed,
                                             _HIPCUB_LIBCXX::execution::output_ordering::unsorted);
    auto env = _HIPCUB_STD_EXEC::env{stream_env, requirements};

    unsigned int seed_value = seeds[1];
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    std::vector<size_type> sizes = {(size_type{1} << 30) - 1, (size_type{1} << 32)};

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_type k = 1;
        SCOPED_TRACE(testing::Message() << "with k = " << k);

        const int out_memory_size = k * sizeof(key_type);

        // Allocate device memory
        auto      d_keys = test_utils::counting_iterator<key_type>{0};
        key_type* d_keys_out;
        HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys_out, out_memory_size));

        // Get size of temporary storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_topk<Descending>(nullptr,
                                          temporary_storage_bytes,
                                          d_keys,
                                          d_keys_out,
                                          size,
                                          k,
                                          env));

        // Allocate temporary storage
        ASSERT_GT(temporary_storage_bytes, 0);

        void* d_temporary_storage;
        HIP_CHECK_MEMORY(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

        // Run
        HIP_CHECK(invoke_topk<Descending>(d_temporary_storage,
                                          temporary_storage_bytes,
                                          d_keys,
                                          d_keys_out,
                                          size,
                                          k,
                                          env));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Get results from device
        key_type output;
        HIP_CHECK(hipMemcpy(&output, d_keys_out, out_memory_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_keys_out));
        HIP_CHECK(hipFree(d_temporary_storage));

        // Verify results
        if constexpr(Descending)
        {
            test_utils::assert_eq(output, size_type{size - 1});
        }
        else
        {
            test_utils::assert_eq(output, size_type{0});
        }
    }
}

template<bool Descending>
void topk_pairs_large_sizes_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using size_type  = uint64_t;
    using key_type   = size_type;
    using value_type = size_type;

    hipStream_t stream     = 0; // default
    auto        stream_env = _HIPCUB_STD_EXEC::prop{_HIPCUB_LIBCXX::get_stream_t{},
                                             _HIPCUB_LIBCXX::stream_ref{stream}};
    auto        requirements
        = _HIPCUB_LIBCXX::execution::require(_HIPCUB_LIBCXX::execution::determinism::not_guaranteed,
                                             _HIPCUB_LIBCXX::execution::output_ordering::unsorted);
    auto env = _HIPCUB_STD_EXEC::env{stream_env, requirements};

    unsigned int seed_value = seeds[1];
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    std::vector<size_type> sizes = {(size_type{1} << 30) - 1, (size_type{1} << 32)};

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_type k = 1;
        SCOPED_TRACE(testing::Message() << "with k = " << k);

        const int out_memory_size = k * sizeof(key_type);

        // Allocate device memory
        auto      d_keys = test_utils::counting_iterator<key_type>{0};
        key_type* d_keys_out;
        HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys_out, out_memory_size));

        auto        d_values = _HIPCUB_STD::make_reverse_iterator(d_keys + size);
        value_type* d_values_out;
        HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_values_out, out_memory_size));

        // Get size of temporary storage
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_topk<Descending>(nullptr,
                                          temporary_storage_bytes,
                                          d_keys,
                                          d_keys_out,
                                          d_values,
                                          d_values_out,
                                          size,
                                          k,
                                          env));

        // Allocate temporary storage
        ASSERT_GT(temporary_storage_bytes, 0);

        void* d_temporary_storage;
        HIP_CHECK_MEMORY(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

        // Run
        HIP_CHECK(invoke_topk<Descending>(d_temporary_storage,
                                          temporary_storage_bytes,
                                          d_keys,
                                          d_keys_out,
                                          d_values,
                                          d_values_out,
                                          size,
                                          k,
                                          env));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Get results from device
        key_type   output_key;
        value_type output_value;
        HIP_CHECK(hipMemcpy(&output_key, d_keys_out, out_memory_size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&output_value, d_values_out, out_memory_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_keys_out));
        HIP_CHECK(hipFree(d_values_out));
        HIP_CHECK(hipFree(d_temporary_storage));

        // Verify results
        if constexpr(Descending)
        {
            test_utils::assert_eq(output_key, size_type{size - 1});
            // test_utils::assert_eq(output_value, size_type{0});
        }
        else
        {
            test_utils::assert_eq(output_key, size_type{0});
            test_utils::assert_eq(output_value, size_type{size - 1});
        }
    }
}

TEST(HipcubDeviceTopkTests, TopkKeysLargeSizesMin)
{
    topk_keys_large_sizes_test<false>();
}
TEST(HipcubDeviceTopkTests, TopkKeysLargeSizesMax)
{
    topk_keys_large_sizes_test<true>();
}

TEST(HipcubDeviceTopkTests, TopkPairsLargeSizesMin)
{
    topk_pairs_large_sizes_test<false>();
}
TEST(HipcubDeviceTopkTests, TopkPairsLargeSizesMax)
{
    topk_pairs_large_sizes_test<true>();
}

#endif // HIPCUB_TEST_HIPCUB_DEVICE_TOPK_HPP_
