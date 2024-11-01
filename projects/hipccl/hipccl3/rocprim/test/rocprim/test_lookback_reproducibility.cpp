// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>
#include <rocprim/iterator/discard_iterator.hpp>

#include "test_utils_types.hpp"

template<typename T>
std::vector<size_t> get_sizes(T seed_value)
{
    std::vector<size_t> sizes = {(1 << 16) - 1220,
                                 (1 << 17) + 23567,
                                 (1 << 18) - 11111,
                                 (1 << 19) + 997452,
                                 (1 << 20) + 123};

    const std::vector<size_t> random_sizes1
        = test_utils::get_random_data<size_t>(2, 2, 1 << 20, seed_value);
    sizes.insert(sizes.end(), random_sizes1.begin(), random_sizes1.end());

    const std::vector<size_t> random_sizes2
        = test_utils::get_random_data<size_t>(3, 2, 1 << 17, seed_value);
    sizes.insert(sizes.end(), random_sizes2.begin(), random_sizes2.end());

    std::sort(sizes.begin(), sizes.end());

    return sizes;
}

template<typename InputType, typename ScanOp = rocprim::plus<InputType>>
struct TestParams
{
    using input_type   = InputType;
    using scan_op_type = ScanOp;
};

template<typename Params>
struct RocprimLookbackReproducibilityTests : public testing::Test
{
    using input_type             = typename Params::input_type;
    using scan_op_type           = typename Params::scan_op_type;
    const bool debug_synchronous = false;
};

using Suite = testing::Types<TestParams<int>, // Sanity check
                             TestParams<rocprim::bfloat16>,
                             TestParams<rocprim::half>,
                             TestParams<float>,
                             TestParams<double>,
                             TestParams<test_utils::custom_test_type<double>>>;

TYPED_TEST_SUITE(RocprimLookbackReproducibilityTests, Suite);

template<typename S, typename F>
void test_reproducibility(S scan_op, F run_test)
{
    bool* d_enable_sleep;
    HIP_CHECK(hipMalloc(&d_enable_sleep, sizeof(*d_enable_sleep)));

    // Delay the operator by a semi-random amount to increase the likelyhood
    // of changing the number of lookback steps between the runs.
    auto eepy_scan_op = [scan_op, d_enable_sleep](auto a, auto b)
    {
        if(*d_enable_sleep)
        {
            for(unsigned int i = 0; i < blockIdx.x * 3001 % 64; ++i)
            {
                __builtin_amdgcn_s_sleep(63);
            }
        }
        return scan_op(a, b);
    };

    bool enable_sleep = false;
    HIP_CHECK(
        hipMemcpy(d_enable_sleep, &enable_sleep, sizeof(enable_sleep), hipMemcpyHostToDevice));
    auto first = run_test(eepy_scan_op);

    enable_sleep = true;
    HIP_CHECK(
        hipMemcpy(d_enable_sleep, &enable_sleep, sizeof(enable_sleep), hipMemcpyHostToDevice));
    auto second = run_test(eepy_scan_op);
    // We want the result to be bitwise equal, even if the inputs/outputs are floats.
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(first, second));

    HIP_CHECK(hipFree(d_enable_sleep));
}

template<typename T>
std::vector<T>
    generate_segments(size_t seed, size_t n, size_t min_segment_length, size_t max_segment_length)
{
    std::default_random_engine            gen(seed);
    std::uniform_int_distribution<size_t> key_count_dis(min_segment_length, max_segment_length);

    std::vector<T> values(n);
    size_t         i = 0;
    while(i < n)
    {
        const size_t key_count = key_count_dis(gen);
        for(size_t j = 0; j < std::min(key_count, n - i); ++j)
        {
            values[i + j] = i;
        }
        i += key_count;
    }

    return values;
}

TYPED_TEST(RocprimLookbackReproducibilityTests, Scan)
{
    using T            = typename TestFixture::input_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    using Config       = rocprim::default_config;

    const bool debug_synchronous = TestFixture::debug_synchronous;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const hipStream_t stream = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, -1000, 1000, seed_value);

            T* d_input;
            T* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(T)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

            scan_op_type scan_op;

            test_reproducibility(
                scan_op,
                [&](auto test_scan_op)
                {
                    size_t temp_storage_size_bytes;
                    void*  d_temp_storage = nullptr;
                    HIP_CHECK(rocprim::deterministic_inclusive_scan<Config>(d_temp_storage,
                                                                            temp_storage_size_bytes,
                                                                            d_input,
                                                                            d_output,
                                                                            input.size(),
                                                                            scan_op,
                                                                            stream,
                                                                            debug_synchronous));

                    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage,
                                                                 temp_storage_size_bytes));

                    HIP_CHECK(rocprim::deterministic_inclusive_scan<Config>(d_temp_storage,
                                                                            temp_storage_size_bytes,
                                                                            d_input,
                                                                            d_output,
                                                                            input.size(),
                                                                            test_scan_op,
                                                                            stream,
                                                                            debug_synchronous));
                    HIP_CHECK(hipGetLastError());

                    std::vector<T> output(input.size());
                    HIP_CHECK(hipMemcpy(output.data(),
                                        d_output,
                                        output.size() * sizeof(T),
                                        hipMemcpyDeviceToHost));
                    HIP_CHECK(hipFree(d_temp_storage));
                    return output;
                });

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
        }
    }
}

TYPED_TEST(RocprimLookbackReproducibilityTests, ScanByKey)
{
    using K               = unsigned int; // key type
    using V               = typename TestFixture::input_type;
    using scan_op_type    = typename TestFixture::scan_op_type;
    using compare_op_type = rocprim::equal_to<K>;
    using Config          = rocprim::default_config;

    const size_t min_segment_length = 1000;
    const size_t max_segment_length = 10000;

    const bool debug_synchronous = TestFixture::debug_synchronous;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const hipStream_t stream = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<V> input = test_utils::get_random_data<V>(size, -1000, 1000, seed_value);
            std::vector<K> keys
                = generate_segments<K>(seed_value, size, min_segment_length, max_segment_length);

            K* d_keys;
            V* d_input;
            V* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(V)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(V)));
            HIP_CHECK(
                hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(K), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(V), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            scan_op_type    scan_op;
            compare_op_type compare_op;

            test_reproducibility(
                scan_op,
                [&](auto test_scan_op)
                {
                    size_t temp_storage_size_bytes;
                    void*  d_temp_storage = nullptr;
                    HIP_CHECK(rocprim::deterministic_inclusive_scan_by_key<Config>(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_keys,
                        d_input,
                        d_output,
                        input.size(),
                        scan_op,
                        compare_op,
                        stream,
                        debug_synchronous));

                    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage,
                                                                 temp_storage_size_bytes));

                    HIP_CHECK(rocprim::deterministic_inclusive_scan_by_key<Config>(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_keys,
                        d_input,
                        d_output,
                        input.size(),
                        test_scan_op,
                        compare_op,
                        stream,
                        debug_synchronous));
                    HIP_CHECK(hipGetLastError());

                    std::vector<V> output(input.size());
                    HIP_CHECK(hipMemcpy(output.data(),
                                        d_output,
                                        output.size() * sizeof(V),
                                        hipMemcpyDeviceToHost));
                    HIP_CHECK(hipFree(d_temp_storage));
                    return output;
                });

            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_input));
        }
    }
}

TYPED_TEST(RocprimLookbackReproducibilityTests, ReduceByKey)
{
    using K               = unsigned int; // key type
    using V               = typename TestFixture::input_type;
    using scan_op_type    = typename TestFixture::scan_op_type;
    using compare_op_type = rocprim::equal_to<K>;
    using Config          = rocprim::default_config;

    // Large segments seem required to trigger issues with reduce by key and rocprim::half.
    const size_t min_segment_length = 10000;
    const size_t max_segment_length = 100000;

    const bool debug_synchronous = false;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const hipStream_t stream = 0;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<V> input = test_utils::get_random_data<V>(size, -1000, 1000, seed_value);
            std::vector<K> keys
                = generate_segments<K>(seed_value, size, min_segment_length, max_segment_length);

            K*      d_keys;
            V*      d_input;
            V*      d_output;
            size_t* d_unique_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(V)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(V)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_unique_count_output,
                                                         sizeof(*d_unique_count_output)));
            HIP_CHECK(
                hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(K), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(V), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            scan_op_type    scan_op;
            compare_op_type compare_op;

            // We don't really care about these values. This is tested by the
            // reduce_by_key tests.
            auto d_discard_unique_output = rocprim::make_discard_iterator();

            test_reproducibility(
                scan_op,
                [&](auto test_scan_op)
                {
                    size_t temp_storage_size_bytes;
                    void*  d_temp_storage = nullptr;
                    HIP_CHECK(rocprim::deterministic_reduce_by_key<Config>(d_temp_storage,
                                                                           temp_storage_size_bytes,
                                                                           d_keys,
                                                                           d_input,
                                                                           input.size(),
                                                                           d_discard_unique_output,
                                                                           d_output,
                                                                           d_unique_count_output,
                                                                           test_scan_op,
                                                                           compare_op,
                                                                           stream,
                                                                           debug_synchronous));

                    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage,
                                                                 temp_storage_size_bytes));

                    HIP_CHECK(rocprim::deterministic_reduce_by_key<Config>(d_temp_storage,
                                                                           temp_storage_size_bytes,
                                                                           d_keys,
                                                                           d_input,
                                                                           input.size(),
                                                                           d_discard_unique_output,
                                                                           d_output,
                                                                           d_unique_count_output,
                                                                           test_scan_op,
                                                                           compare_op,
                                                                           stream,
                                                                           debug_synchronous));
                    HIP_CHECK(hipGetLastError());

                    size_t unique_count_output;
                    HIP_CHECK(hipMemcpy(&unique_count_output,
                                        d_unique_count_output,
                                        sizeof(unique_count_output),
                                        hipMemcpyDeviceToHost));

                    std::vector<V> output(unique_count_output);
                    HIP_CHECK(hipMemcpy(output.data(),
                                        d_output,
                                        output.size() * sizeof(V),
                                        hipMemcpyDeviceToHost));
                    HIP_CHECK(hipFree(d_temp_storage));
                    return output;
                });

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_unique_count_output));
        }
    }
}
