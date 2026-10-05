// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_functional.hpp"
#include "test_utils_thread_operators.hpp"
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_segmented_reduce.hpp>
#include <hipcub/thread/thread_operators.hpp>
#include <hipcub/util_type.hpp>

#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include _HIPCUB_STD_INCLUDE(functional)

template<class InputType, class OutputType = InputType>
struct ThreadOperatorsParams
{
    using input_type  = InputType;
    using output_type = OutputType;
};

template<class Params>
class HipcubThreadOperatorsTests : public ::testing::Test
{
public:
    using input_type  = typename Params::input_type;
    using output_type = typename Params::output_type;
};

using ThreadOperatorsParameters = ::testing::Types<
    ThreadOperatorsParams<int, short>,
    ThreadOperatorsParams<int, long>,
    ThreadOperatorsParams<int, float>,
    ThreadOperatorsParams<int, double>,
    ThreadOperatorsParams<unsigned long>,
    ThreadOperatorsParams<short, long>,
    ThreadOperatorsParams<short, float>,
    ThreadOperatorsParams<short, double>,
    ThreadOperatorsParams<long, float>,
    ThreadOperatorsParams<long, double>,
    ThreadOperatorsParams<float, double>,
    ThreadOperatorsParams<test_utils::half, test_utils::half>,
    ThreadOperatorsParams<test_utils::bfloat16, test_utils::bfloat16>,
    ThreadOperatorsParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>,
    ThreadOperatorsParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>
#ifdef __HIP_PLATFORM_AMD__
    ,
    ThreadOperatorsParams<test_utils::half, float>, // Doesn't work on NVIDIA / CUB
    ThreadOperatorsParams<test_utils::bfloat16, float> // Doesn't work on NVIDIA / CUB
#endif
    >;

TYPED_TEST_SUITE(HipcubThreadOperatorsTests, ThreadOperatorsParameters);

// Commutative operators tests.

/// \brief Shared code for ArgMin/ArgMax operators.
template<typename ArgOpT, typename InputT>
void arg_op_test(bool is_max)
{
    using input_pair_type = hipcub::KeyValuePair<int, InputT>;

    ArgOpT op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        // Generate random initial and input values.
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
        std::vector<InputT> generated_values
            = test_utils::get_random_data<InputT>(2, 1.0f, 100.0f, seed_value);

        InputT input_val = generated_values[0];
        InputT init_val  = generated_values[1];
        InputT output_val
            = is_max ? test_utils::max(init_val, input_val) : test_utils::min(init_val, input_val);

        input_pair_type init_pair(0, init_val);
        input_pair_type input_pair(0, input_val);
        input_pair_type output_pair = op(init_pair, input_pair);

        // Check result.
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_pair.value, output_val));
    }
}

TYPED_TEST(HipcubThreadOperatorsTests, ArgMax)
{
    using input_type = typename TestFixture::input_type;
    using ArgMax     = typename ArgMaxSelector<input_type>::type;

    arg_op_test<ArgMax, input_type>(true);
}

TYPED_TEST(HipcubThreadOperatorsTests, ArgMin)
{
    using input_type = typename TestFixture::input_type;
    using ArgMin     = typename ArgMinSelector<input_type>::type;

    arg_op_test<ArgMin, input_type>(false);
}

// Non-commutative operators.

template<class Params>
class HipcubNCThreadOperatorsTests : public ::testing::Test
{
public:
    using input_type  = typename Params::input_type;
    using output_type = typename Params::output_type;
};

using NCThreadOperatorsParameters = ::testing::Types<ThreadOperatorsParams<int, short>,
                                                     ThreadOperatorsParams<int, long>,
                                                     ThreadOperatorsParams<int, float>,
                                                     ThreadOperatorsParams<int, double>,
                                                     ThreadOperatorsParams<short, long>,
                                                     ThreadOperatorsParams<short, float>,
                                                     ThreadOperatorsParams<short, double>,
                                                     ThreadOperatorsParams<long, float>,
                                                     ThreadOperatorsParams<long, double>,
                                                     ThreadOperatorsParams<float, double>>;

std::vector<size_t> get_sizes()
{
    // We generate size 208 as a maximum so the sum $\sum_{i = n/2 + 1}^n i$ does not overflow for sort type.
    // This overflow does not happen for an unsigned int size n iff (3 * n^2 + 2 * n)/4 <= 32767 iff n <= 208.
    std::vector<size_t>       sizes        = {1, 8, 10, 53, 208};
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 208, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(HipcubNCThreadOperatorsTests, NCThreadOperatorsParameters);

/// \brief Shared code for scan operators.
template<typename InputT, typename OutputT, typename ScanOpT>
void scan_op_test(std::vector<InputT>  h_input,
                  std::vector<OutputT> h_expected,
                  ScanOpT              op,
                  size_t               input_size)
{
    // Set HIP device.
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0;

    // Allocate input and output on device and copy input from host.
    InputT*  d_input{};
    OutputT* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input_size * sizeof(InputT)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input_size * sizeof(OutputT)));
    HIP_CHECK(
        hipMemcpy(d_input, h_input.data(), input_size * sizeof(InputT), hipMemcpyHostToDevice));

    // Get size of temporary storage on device.
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                op,
                                                input_size,
                                                stream));

    // Size of temporary storage must be > 0.
    ASSERT_GT(temp_storage_size_bytes, 0U);

    // Allocate temporary storage.
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run kernel.
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                op,
                                                input_size,
                                                stream));
    HIP_CHECK(hipGetLastError());

    // Copy output to host.
    std::vector<OutputT> h_output(input_size);
    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, input_size * sizeof(OutputT), hipMemcpyDeviceToHost));

    // Check output.
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_near(h_output,
                                h_expected,
                                test_utils::precision<InputT>::value * input_size));

    // Check output type.
    for(size_t i = 0; i < input_size; ++i)
    {
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_type(h_output[i], h_expected[i]))
            << "where index = " << i;
    }

    // Free resources.
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

TYPED_TEST(HipcubNCThreadOperatorsTests, SwizzleScanOp)
{
    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;

    // Generate input data.
    const std::vector<size_t> sizes = get_sizes();
    for(auto input_size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << input_size);

        std::vector<input_type> h_input(input_size);
        std::iota(h_input.begin(), h_input.end(), static_cast<input_type>(1));

        // Scan function: SwizzleScanOp.
        test_utils::plus                        sum_op{};
        hipcub::SwizzleScanOp<test_utils::plus> scan_op(sum_op);

        // Calculate expected results on host.
        std::vector<output_type> h_expected(input_size);
        test_utils::host_inclusive_scan(h_input.begin(),
                                        h_input.end(),
                                        h_expected.begin(),
                                        scan_op);

        scan_op_test<input_type, output_type>(h_input, h_expected, scan_op, input_size);
    }
}

TYPED_TEST(HipcubNCThreadOperatorsTests, ReduceBySegmentOp)
{
    using key_type    = int;
    using input_type  = typename TestFixture::input_type;
    using output_type = input_type;
    using pair_type   = hipcub::KeyValuePair<key_type, input_type>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto segment_size : sizes)
    {
        constexpr size_t segment_count = 2;
        const size_t     input_size    = segment_count * segment_size;

        SCOPED_TRACE(testing::Message() << "with size = " << input_size);

        // Generate data. We generate the input {1, 2, 3, ... , n} and we want to compute the
        // output {1 + 2 + ... + n/2, (n/2 + 1) + (n/2 + 2) + ... + n}.
        std::vector<input_type> input_values(input_size);
        std::iota(input_values.begin(), input_values.end(), static_cast<input_type>(1));

        std::vector<key_type> input_keys(input_size);
        std::iota(input_keys.begin(), input_keys.begin() + segment_size, static_cast<key_type>(0));
        std::iota(input_keys.begin() + segment_size, input_keys.end(), static_cast<key_type>(0));

        std::vector<pair_type> input{};
        for(size_t i = 0; i < input_size; ++i)
        {
            input.push_back(pair_type(input_keys[i], input_values[i]));
        }

        // Reduce and scan operators.
        test_utils::plus                            sum_op{};
        hipcub::ReduceBySegmentOp<test_utils::plus> op(sum_op);

        // Calculate expected results on host.
        std::vector<pair_type> expected{};
        pair_type              init(0, 0);
        for(size_t offset = 0; offset < input_size; offset += segment_size)
        {
            const size_t end       = _HIPCUB_STD::min(input_size, offset + segment_size);
            pair_type    aggregate = init;
            for(size_t i = offset; i < end; ++i)
            {
                pair_type   input_pair   = input[i];
                key_type    expected_key = sum_op(aggregate.key, input_pair.key);
                output_type expected_value
                    = input_pair.key ? input_pair.value : sum_op(aggregate.value, input_pair.value);
                aggregate = pair_type(expected_key, expected_value);
            }
            expected.push_back(aggregate);
        }

        // Get output on host.
        std::vector<pair_type> output{};
        for(size_t offset = 0; offset < input_size; offset += segment_size)
        {
            const size_t end       = _HIPCUB_STD::min(input_size, offset + segment_size);
            pair_type    aggregate = init;
            for(size_t i = offset; i < end; ++i)
            {
                aggregate = op(aggregate, input[i]);
            }
            output.push_back(aggregate);
        }

        // Check if output pairs are as expected.
        for(size_t i = 0; i < segment_count; i++)
        {
            // Check keys.
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected[i].key, output[i].key))
                << "where index = " << i;

            // Check values.
            auto tolerance = test_utils::precision<input_type>::value * i;
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(expected[i].value, output[i].value, tolerance))
                << "where index = " << i;
        }
    }
}

TYPED_TEST(HipcubNCThreadOperatorsTests, ReduceByKeyOp)
{
    using key_type    = int;
    using input_type  = typename TestFixture::input_type;
    using output_type = input_type;
    using pair_type   = hipcub::KeyValuePair<key_type, input_type>;

    // Set HIP device.
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0;

    const std::vector<size_t> sizes = get_sizes();
    for(auto input_size : sizes)
    {
        const size_t h_unique_keys = input_size / 2 + (input_size % 2);

        // Generate data. We generate the input {1, 2, 3, ... , n} and we want to compute the
        // output {1 + 2, 3 + 4, ... , (n - 1) + n}.
        std::vector<input_type> h_input(input_size);
        std::iota(h_input.begin(), h_input.end(), static_cast<input_type>(1));

        std::vector<key_type> h_keys(input_size);
        for(size_t i = 0; i < input_size; ++i)
        {
            h_keys[i] = (i % 2) ? h_keys[i - 1] : i / 2;
        }

        // Reduce operators.
        test_utils::plus                        sum_op;
        hipcub::ReduceByKeyOp<test_utils::plus> op{};

        // Calculate output on host.
        std::vector<output_type> h_output(h_unique_keys);
        std::vector<key_type>    h_keys_output(h_unique_keys);
        h_keys_output[0] = h_keys[0];
        h_output[0]      = h_input[0];
        pair_type first(h_keys[0], h_input[0]);
        for(size_t i = 1; i < input_size; ++i)
        {
            pair_type second(h_keys[i], h_input[i]);
            pair_type result         = (first.key == second.key) ? op(first, second) : second;
            h_keys_output[h_keys[i]] = result.key;
            h_output[h_keys[i]]      = result.value;
            first                    = pair_type(second);
        }

        // Allocate input, keys and expected results on device and copy input and keys from host.
        input_type* d_input{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input_size * sizeof(input_type)));
        HIP_CHECK(hipMemcpy(d_input,
                            h_input.data(),
                            input_size * sizeof(input_type),
                            hipMemcpyHostToDevice));

        key_type* d_keys{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, input_size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(d_keys, h_keys.data(), input_size * sizeof(key_type), hipMemcpyHostToDevice));

        key_type*    d_keys_expected{};
        output_type* d_expected{};
        size_t*      d_unique_keys_expected{};
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_keys_expected, h_unique_keys * sizeof(key_type)));
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_expected, h_unique_keys * sizeof(output_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_unique_keys_expected, sizeof(size_t)));

        // Get size of temporary storage on device.
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        HIP_CHECK(hipcub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_keys,
                                                    d_keys_expected,
                                                    d_input,
                                                    d_expected,
                                                    d_unique_keys_expected,
                                                    sum_op,
                                                    input_size,
                                                    stream));

        // Size of temporary storage must be > 0.
        ASSERT_GT(temp_storage_size_bytes, 0U);

        // Allocate temporary storage.
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

        // Run kernel.
        HIP_CHECK(hipcub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_keys,
                                                    d_keys_expected,
                                                    d_input,
                                                    d_expected,
                                                    d_unique_keys_expected,
                                                    sum_op,
                                                    input_size,
                                                    stream));
        HIP_CHECK(hipGetLastError());

        // Copy expected results to host.
        std::vector<key_type> h_keys_expected(h_unique_keys);
        HIP_CHECK(hipMemcpy(h_keys_expected.data(),
                            d_keys_expected,
                            h_unique_keys * sizeof(key_type),
                            hipMemcpyDeviceToHost));

        std::vector<output_type> h_expected(h_unique_keys);
        HIP_CHECK(hipMemcpy(h_expected.data(),
                            d_expected,
                            h_unique_keys * sizeof(output_type),
                            hipMemcpyDeviceToHost));

        std::vector<size_t> h_unique_keys_expected(1);
        HIP_CHECK(hipMemcpy(h_unique_keys_expected.data(),
                            d_unique_keys_expected,
                            sizeof(size_t),
                            hipMemcpyDeviceToHost));

        // Check if output values are as expected.
        // Check number of unique keys.
        ASSERT_EQ(h_unique_keys_expected[0], h_unique_keys);

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_keys_expected, h_keys_output));
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(h_expected,
                                    h_output,
                                    test_utils::precision<input_type>::value * input_size));

        // Free resources.
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_keys));
        HIP_CHECK(hipFree(d_keys_expected));
        HIP_CHECK(hipFree(d_expected));
        HIP_CHECK(hipFree(d_unique_keys_expected));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}

// Unary operators tests.

TYPED_TEST(HipcubNCThreadOperatorsTests, CastOp)
{
    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using IteratorType
        = test_utils::transform_iterator<input_type*, hipcub::CastOp<output_type>, output_type>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto input_size : sizes)
    {
        // Generate data.
        std::vector<input_type> input(input_size);
        std::iota(input.begin(), input.end(), static_cast<input_type>(0));

        std::vector<output_type> expected(input_size);
        std::iota(expected.begin(), expected.end(), static_cast<output_type>(0));

        // Scan operator: CastOp.
        hipcub::CastOp<output_type> op{};

        // Transform input applying the casting operator.
        auto output = IteratorType(input.data(), op);

        // Check output.
        for(size_t i = 0; i < input_size; ++i)
        {
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output[i],
                                        expected[i],
                                        test_utils::precision<input_type>::value));
        }

        // Check output type.
        for(size_t i = 0; i < input_size; ++i)
        {
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_type(output[i], expected[i]))
                << "where index = " << i;
        }
    }
}
