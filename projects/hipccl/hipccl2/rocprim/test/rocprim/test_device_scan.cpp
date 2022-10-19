// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

// required rocprim headers
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"

#include <functional>
#include <iterator>
#include <numeric>

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class ScanOp = ::rocprim::plus<InputType>,
    // Tests output iterator with void value_type (OutputIterator concept)
    // scan-by-key primitives don't support output iterator with void value_type
    bool UseIdentityIteratorIfSupported = false,
    size_t SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using scan_op_type = ScanOp;
    static constexpr bool use_identity_iterator = UseIdentityIteratorIfSupported;
    static constexpr size_t size_limit = SizeLimit;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceScanTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using scan_op_type = typename Params::scan_op_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr size_t size_limit = Params::size_limit;
};

typedef ::testing::Types<
    // Small
    DeviceScanParams<char>,
    DeviceScanParams<unsigned short>,
    DeviceScanParams<short, int>,
    DeviceScanParams<int>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, 512>,
    DeviceScanParams<float, float, rocprim::maximum<float>>,
    DeviceScanParams<float, float, rocprim::plus<float>, false, 1024>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, 524288>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, 1048576>,
    DeviceScanParams<int8_t, int8_t, rocprim::maximum<int8_t>>,
    DeviceScanParams<uint8_t, uint8_t, rocprim::maximum<uint8_t>>,
    DeviceScanParams<rocprim::half, rocprim::half, rocprim::maximum<rocprim::half>>,
    DeviceScanParams<rocprim::half, float, rocprim::plus<float>>,
    DeviceScanParams<rocprim::bfloat16, rocprim::bfloat16, rocprim::maximum<rocprim::bfloat16>>,
    DeviceScanParams<rocprim::bfloat16, float, rocprim::plus<float>>,
    // Large
    DeviceScanParams<int, double, rocprim::plus<int>>,
    DeviceScanParams<int, double, rocprim::plus<double>>,
    DeviceScanParams<int, long long, rocprim::plus<long long>>,
    DeviceScanParams<unsigned int, unsigned long long, rocprim::plus<unsigned long long>>,
    DeviceScanParams<long long, long long, rocprim::maximum<long long>>,
    DeviceScanParams<double, double, rocprim::plus<double>, true>,
    DeviceScanParams<signed char, long, rocprim::plus<long>>,
    DeviceScanParams<float, double, rocprim::minimum<double>>,
    DeviceScanParams<test_utils::custom_test_type<int>>,
    DeviceScanParams<test_utils::custom_test_type<double>,
                     test_utils::custom_test_type<double>,
                     rocprim::plus<test_utils::custom_test_type<double>>,
                     true>,
    DeviceScanParams<test_utils::custom_test_type<int>>,
    DeviceScanParams<test_utils::custom_test_array_type<long long, 5>>,
    DeviceScanParams<test_utils::custom_test_array_type<int, 10>>>
    RocprimDeviceScanTestsParams;

template <unsigned int SizeLimit>
struct size_limit_config {
    using type = rocprim::scan_config<256,
                                      16,
                                      ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
                                      rocprim::block_load_method::block_load_transpose,
                                      rocprim::block_store_method::block_store_transpose,
                                      rocprim::block_scan_algorithm::using_warp_scan,
                                      SizeLimit>;
};

template <>
struct size_limit_config<ROCPRIM_GRID_SIZE_LIMIT> {
    using type = rocprim::default_config;
};

template <unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<SizeLimit>::type;

// use float for accumulation of bfloat16 and half inputs if operator is plus
template <typename input_type, typename input_op_type> struct accum_type {
    static constexpr bool is_low_precision =
        std::is_same<input_type, ::rocprim::half>::value ||
        std::is_same<input_type, ::rocprim::bfloat16>::value;
    static constexpr bool is_plus = test_utils::is_plus_operator<input_op_type>::value;
    using type = typename std::conditional_t<is_low_precision && is_plus, float, input_type>;
};

TYPED_TEST_SUITE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanEmptyInput)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using acc_type = typename accum_type<T, scan_op_type>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0; // default

    U * d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(U)));

    test_utils::out_of_bounds_flag out_of_bounds;
    test_utils::bounds_checking_iterator<U> d_checking_output(
        d_output,
        out_of_bounds.device_pointer(),
        0
    );

    // scan function
    scan_op_type scan_op;

    auto input_iterator = rocprim::make_transform_iterator(
        rocprim::make_constant_iterator<T>(T(345)),
        [] (T in) { return static_cast<acc_type>(in); });

    // temp storage
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::inclusive_scan(
            d_temp_storage, temp_storage_size_bytes,
            input_iterator, d_checking_output, 
            0, scan_op, stream, debug_synchronous
        )
    );

    // allocate temporary storage
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::inclusive_scan(
            d_temp_storage, temp_storage_size_bytes,
            input_iterator, d_checking_output, 
            0, scan_op, stream, debug_synchronous
        )
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_FALSE(out_of_bounds.get());

    hipFree(d_output);
    hipFree(d_temp_storage);
}

TYPED_TEST(RocprimDeviceScanTests, InclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size(), U{0});

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan(
                input.begin(), input.end(),
                expected.begin(), scan_op
            );

            auto input_iterator = rocprim::make_transform_iterator(
                d_input, [] (T in) { return static_cast<acc_type>(in); });

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::inclusive_scan<Config>(
                    d_temp_storage, temp_storage_size_bytes, input_iterator,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), scan_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::inclusive_scan<Config>(
                    d_temp_storage, temp_storage_size_bytes, input_iterator,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), scan_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    // for non-associative operations in exclusive scan
    // intermediate results use the type of initial value, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    acc_type        initial_value;
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size());

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            initial_value = test_utils::get_random_value<acc_type>(1, 10, seed_value);
            test_utils::host_exclusive_scan(
                input.begin(), input.end(),
                initial_value, expected.begin(),
                scan_op
            );

            auto input_iterator = rocprim::make_transform_iterator(
                d_input, [] (T in) { return static_cast<acc_type>(in); });

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::exclusive_scan<Config>(
                    d_temp_storage, temp_storage_size_bytes, input_iterator,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    initial_value, input.size(), scan_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::exclusive_scan<Config>(
                    d_temp_storage, temp_storage_size_bytes, input_iterator,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    initial_value, input.size(), scan_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // key type
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const bool use_unique_keys = bool(test_utils::get_random_value<int>(0, 1, seed_value));

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 9, seed_value);
            std::vector<K> keys;
            if(use_unique_keys)
            {
                keys = test_utils::get_random_data<K>(size, 0, 16, seed_value);
                std::sort(keys.begin(), keys.end());
            }
            else
            {
                keys = test_utils::get_random_data<K>(size, 0, 3, seed_value);
            }
            std::vector<U> output(input.size(), U{0});

            T * d_input;
            K * d_keys;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_keys, keys.data(),
                    keys.size() * sizeof(K),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;
            // key compare function
            rocprim::equal_to<K> keys_compare_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan_by_key(
                input.begin(), input.end(), keys.begin(),
                expected.begin(),
                scan_op, keys_compare_op
            );

            auto input_iterator = rocprim::make_transform_iterator(
                d_input, [] (T in) { return static_cast<acc_type>(in); }); 

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::inclusive_scan_by_key<Config>(
                    d_temp_storage, temp_storage_size_bytes, d_keys, input_iterator, 
                    d_output, input.size(), scan_op, keys_compare_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::inclusive_scan_by_key<Config>(
                    d_temp_storage, temp_storage_size_bytes, d_keys, input_iterator,
                    d_output, input.size(), scan_op, keys_compare_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            hipFree(d_keys);
            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // key type
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    acc_type        initial_value;
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const bool use_unique_keys = bool(test_utils::get_random_value<int>(0, 1, seed_value));

            // Generate data
            initial_value        = test_utils::get_random_value<acc_type>(1, 100, seed_value);
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 9, seed_value);
            std::vector<K> keys;
            if(use_unique_keys)
            {
                keys = test_utils::get_random_data<K>(size, 0, 16, seed_value);
                std::sort(keys.begin(), keys.end());
            }
            else
            {
                keys = test_utils::get_random_data<K>(size, 0, 3, seed_value);
            }
            std::vector<U> output(input.size(), U{0});

            T * d_input;
            K * d_keys;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_keys, keys.data(),
                    keys.size() * sizeof(K),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // key compare function
            rocprim::equal_to<K> keys_compare_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan_by_key(
                input.begin(), input.end(), keys.begin(),
                initial_value, expected.begin(),
                scan_op, keys_compare_op
            );

            auto input_iterator = rocprim::make_transform_iterator(
                d_input, [] (T in) { return static_cast<acc_type>(in); }); 

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::exclusive_scan_by_key<Config>(
                    d_temp_storage, temp_storage_size_bytes, d_keys, input_iterator,
                    d_output, initial_value, input.size(), scan_op, keys_compare_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::exclusive_scan_by_key<Config>(
                    d_temp_storage, temp_storage_size_bytes, d_keys, input_iterator,
                    d_output, initial_value, input.size(), scan_op, keys_compare_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            hipFree(d_keys);
            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

template <typename T>
class single_index_iterator {
private:
    class conditional_discard_value {
    public:
        __host__ __device__ explicit conditional_discard_value(T* const value, bool keep)
            : value_{value}
            , keep_{keep}
        {
        }

        __host__ __device__ conditional_discard_value& operator=(T value) {
            if(keep_) {
                *value_ = value;
            }
            return *this;
        }
    private:
        T* const   value_;
        const bool keep_;
    };

    T*     value_;
    size_t expected_index_;
    size_t index_;

public:
    using value_type        = conditional_discard_value;
    using reference         = conditional_discard_value;
    using pointer           = conditional_discard_value*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    __host__ __device__ single_index_iterator(T* value, size_t expected_index, size_t index = 0)
        : value_{value}
        , expected_index_{expected_index}
        , index_{index}
    {
    }

    __host__ __device__ single_index_iterator(const single_index_iterator&) = default;
    __host__ __device__ single_index_iterator& operator=(const single_index_iterator&) = default;

    // clang-format off
    __host__ __device__ bool operator==(const single_index_iterator& rhs) { return index_ == rhs.index_; }
    __host__ __device__ bool operator!=(const single_index_iterator& rhs) { return !(this == rhs);       }

    __host__ __device__ reference operator*() { return value_type{value_, index_ == expected_index_}; }

    __host__ __device__ reference operator[](const difference_type distance) { return *(*this + distance); }

    __host__ __device__ single_index_iterator& operator+=(const difference_type rhs) { index_ += rhs; return *this; }
    __host__ __device__ single_index_iterator& operator-=(const difference_type rhs) { index_ -= rhs; return *this; }

    __host__ __device__ difference_type operator-(const single_index_iterator& rhs) const { return index_ - rhs.index_; }

    __host__ __device__ single_index_iterator operator+(const difference_type rhs) const { return single_index_iterator(*this) += rhs; }
    __host__ __device__ single_index_iterator operator-(const difference_type rhs) const { return single_index_iterator(*this) -= rhs; }

    __host__ __device__ single_index_iterator& operator++() { ++index_; return *this; }
    __host__ __device__ single_index_iterator& operator--() { --index_; return *this; }

    __host__ __device__ single_index_iterator operator++(int) { return ++single_index_iterator{*this}; }
    __host__ __device__ single_index_iterator operator--(int) { return --single_index_iterator{*this}; }
    // clang-format on
};

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = size_t;
    using Iterator = typename rocprim::counting_iterator<T>;
    using OutputIterator = single_index_iterator<T>;
    const bool debug_synchronous = false;

    const hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Create counting_iterator<U> with random starting point
            Iterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value ^ size));

            SCOPED_TRACE(testing::Message() << "with starting point = " << *input_begin);

            T   output;
            T * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
            HIP_CHECK(hipDeviceSynchronize());

            OutputIterator output_it{d_output, size - 1};

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;

            // Get temporary array size
            HIP_CHECK(
                rocprim::inclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    input_begin, output_it, size,
                    ::rocprim::plus<T>(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::inclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    input_begin, output_it, size,
                    ::rocprim::plus<T>(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(T), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Sum of 'size' increasing numbers starting at 'n' is size * (2n + size - 1)
            // The division is not integer division but either (size) or (2n + size - 1) has to be even.
            const T multiplicand_1 = size;
            const T multiplicand_2 = 2 * (*input_begin) + size - 1;
            const T expected_output = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                                : multiplicand_1 * (multiplicand_2 / 2);

            ASSERT_EQ(output, expected_output);

            hipFree(d_temp_storage);
            hipFree(d_output);
        }
    }
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = size_t;
    using Iterator = typename rocprim::counting_iterator<T>;
    using OutputIterator = single_index_iterator<T>;
    const bool debug_synchronous = false;

    const hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Create counting_iterator<U> with random starting point
            Iterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value ^ size));
            T initial_value = test_utils::get_random_value<T>(1, 10, seed_value ^ *input_begin);

            SCOPED_TRACE(testing::Message() << "with starting point = " << *input_begin);
            SCOPED_TRACE(testing::Message() << "with initial value = " << initial_value);

            T  output;
            T* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
            HIP_CHECK(hipDeviceSynchronize());

            OutputIterator output_it{d_output, size - 1};

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;

            // Get temporary array size
            HIP_CHECK(
                rocprim::exclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    input_begin, output_it,
                    initial_value, size,
                    ::rocprim::plus<T>(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::exclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    input_begin, output_it,
                    initial_value, size,
                    ::rocprim::plus<T>(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(T), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Sum of 'size' - 1 increasing numbers starting at 'n' is (size - 1) * (2n + size - 2)
            // The division is not integer division but either (size - 1) or (2n + size - 2) has to be even.
            const T multiplicand_1 = size - 1;
            const T multiplicand_2 = 2 * (*input_begin) + size - 2;

            const T product = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                        : multiplicand_1 * (multiplicand_2 / 2);

            const T expected_output = initial_value + product;

            ASSERT_EQ(output, expected_output);

            hipFree(d_temp_storage);
            hipFree(d_output);
        }
    }
}

/// \brief This iterator keeps track of the current index. Upon dereference, a \p CheckValue object
/// is created and besides the current index, the provided \p rocprim::tuple<Args...> is passed
/// to its constructor.
template<class CheckValue, class... Args>
class check_run_iterator
{
public:
    using args_t            = rocprim::tuple<Args...>;
    using value_type        = CheckValue;
    using reference         = CheckValue;
    using pointer           = CheckValue*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    ROCPRIM_HOST_DEVICE
    check_run_iterator(const args_t args) : current_index_(0), args_(args) {}

    ROCPRIM_HOST_DEVICE bool operator==(const check_run_iterator& rhs)
    {
        return current_index_ == rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE bool operator!=(const check_run_iterator& rhs)
    {
        return !(*this == rhs);
    }
    ROCPRIM_HOST_DEVICE reference operator*()
    {
        return value_type{current_index_, args_};
    }
    ROCPRIM_HOST_DEVICE reference operator[](const difference_type distance)
    {
        return *(*this + distance);
    }
    ROCPRIM_HOST_DEVICE check_run_iterator& operator+=(const difference_type rhs)
    {
        current_index_ += rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator& operator-=(const difference_type rhs)
    {
        current_index_ -= rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE difference_type operator-(const check_run_iterator& rhs) const
    {
        return current_index_ - rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator operator+(const difference_type rhs) const
    {
        return check_run_iterator(*this) += rhs;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator operator-(const difference_type rhs) const
    {
        return check_run_iterator(*this) -= rhs;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator& operator++()
    {
        ++current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator& operator--()
    {
        --current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_run_iterator operator++(int)
    {
        return ++check_run_iterator{*this};
    }
    ROCPRIM_HOST_DEVICE check_run_iterator operator--(int)
    {
        return --check_run_iterator{*this};
    }

private:
    size_t current_index_;
    args_t args_;
};

/// \brief Checks if the \p size_t values written to it are part of the expected sequence.
/// The expected sequence is the sequence [0, 1, 2, 3, ...) inclusively scanned in runs of
/// \p run_length.
/// If a discrepancy is found, \p incorrect_flag is atomically set to 1.
struct check_value_inclusive
{
    size_t                                current_index_{};
    rocprim::tuple<size_t, unsigned int*> args_; // run_length, incorrect flag

    ROCPRIM_HOST_DEVICE
    size_t operator=(const size_t value)
    {
        const size_t run_start    = current_index_ - (current_index_ % rocprim::get<0>(args_));
        const size_t index_in_run = current_index_ - run_start + 1;
        const size_t expected_sum = (run_start + current_index_) * index_in_run / 2;
        if(value != expected_sum)
        {
            rocprim::detail::atomic_exch(rocprim::get<1>(args_), 1);
        }
        return value;
    }
};

/// \brief Checks if the \p size_t values written to it are part of the expected sequence.
/// The expected sequence is the sequence [0, 1, 2, 3, ...) exclusively scanned in runs of
/// \p run_length, with \p initial_value.
/// If a discrepancy is found, \p incorrect_flag is atomically set to 1.
struct check_value_exclusive
{
    size_t current_index_{};
    rocprim::tuple<size_t, size_t, unsigned int*>
        args_; // run_length, initial_value, incorrect flag

    ROCPRIM_HOST_DEVICE
    size_t operator=(const size_t value)
    {
        const size_t run_start    = current_index_ - (current_index_ % rocprim::get<0>(args_));
        const size_t index_in_run = current_index_ - run_start;
        const size_t expected_sum
            = rocprim::get<1>(args_) + (run_start + current_index_ - 1) * index_in_run / 2;
        if(value != expected_sum)
        {
            rocprim::detail::atomic_exch(rocprim::get<2>(args_), 1);
        }
        return value;
    }
};

using check_run_inclusive_iterator
    = check_run_iterator<check_value_inclusive, size_t, unsigned int*>;
using check_run_exclusive_iterator
    = check_run_iterator<check_value_exclusive, size_t, size_t, unsigned int*>;

/// \p brief Provides a skeleton to both the inclusive and exclusive scan large indices tests.
/// The call to the appropriate scan function must be implemented in \p scan_by_key_fun.
template<class ScanByKeyFun>
void large_indices_scan_by_key_test(ScanByKeyFun scan_by_key_fun)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    constexpr bool        debug_synchronous = false;
    constexpr hipStream_t stream            = 0;

    const int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    const auto size = test_utils::get_large_sizes(seed_value).back();
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    unsigned int* d_incorrect_flag;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_incorrect_flag, sizeof(*d_incorrect_flag)));
    HIP_CHECK(hipMemset(d_incorrect_flag, 0, sizeof(*d_incorrect_flag)));

    const size_t run_length = test_utils::get_random_value<size_t>(1, 10000, seed_value);
    SCOPED_TRACE(testing::Message() << "with run_length = " << run_length);

    const auto keys_input = rocprim::make_transform_iterator(rocprim::counting_iterator<size_t>(0),
                                                             [run_length](const auto value)
                                                             { return value / run_length; });
    const auto values_input = rocprim::counting_iterator<size_t>(0);

    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    HIP_CHECK(scan_by_key_fun(d_temp_storage,
                              temp_storage_size_bytes,
                              keys_input,
                              values_input,
                              run_length,
                              d_incorrect_flag,
                              size,
                              stream,
                              debug_synchronous,
                              seed_value));
    ASSERT_GT(temp_storage_size_bytes, 0);
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(scan_by_key_fun(d_temp_storage,
                              temp_storage_size_bytes,
                              keys_input,
                              values_input,
                              run_length,
                              d_incorrect_flag,
                              size,
                              stream,
                              debug_synchronous,
                              seed_value));
    HIP_CHECK(hipGetLastError());

    unsigned int incorrect_flag;
    HIP_CHECK(hipMemcpy(&incorrect_flag,
                        d_incorrect_flag,
                        sizeof(incorrect_flag),
                        hipMemcpyDeviceToHost));

    ASSERT_EQ(0, incorrect_flag);

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_incorrect_flag));
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanByKey)
{
    auto inclusive_scan_by_key = [](void*         d_temp_storage,
                                    size_t&       temp_storage_size_bytes,
                                    auto          keys_input,
                                    auto          values_input,
                                    size_t        run_length,
                                    unsigned int* d_incorrect_flag,
                                    size_t        size,
                                    hipStream_t   stream,
                                    bool          debug_synchronous,
                                    int /*seed_value*/) -> hipError_t
    {
        const check_run_inclusive_iterator output_it(
            rocprim::make_tuple(run_length, d_incorrect_flag));

        return rocprim::inclusive_scan_by_key(d_temp_storage,
                                              temp_storage_size_bytes,
                                              keys_input,
                                              values_input,
                                              output_it,
                                              size,
                                              rocprim::plus<size_t>{},
                                              rocprim::equal_to<size_t>{},
                                              stream,
                                              debug_synchronous);
    };
    large_indices_scan_by_key_test(inclusive_scan_by_key);
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScanByKey)
{
    auto exclusive_scan_by_key = [](void*         d_temp_storage,
                                    size_t&       temp_storage_size_bytes,
                                    auto          keys_input,
                                    auto          values_input,
                                    size_t        run_length,
                                    unsigned int* d_incorrect_flag,
                                    size_t        size,
                                    hipStream_t   stream,
                                    bool          debug_synchronous,
                                    int           seed_value) -> hipError_t
    {
        const size_t initial_value = test_utils::get_random_value<size_t>(0, 10000, seed_value);
        const check_run_exclusive_iterator output_it(
            rocprim::make_tuple(run_length, initial_value, d_incorrect_flag));
        return rocprim::exclusive_scan_by_key(d_temp_storage,
                                              temp_storage_size_bytes,
                                              keys_input,
                                              values_input,
                                              output_it,
                                              initial_value,
                                              size,
                                              rocprim::plus<size_t>{},
                                              rocprim::equal_to<size_t>{},
                                              stream,
                                              debug_synchronous);
    };
    large_indices_scan_by_key_test(exclusive_scan_by_key);
}

using RocprimDeviceScanFutureTestsParams
    = ::testing::Types<DeviceScanParams<char>,
                       DeviceScanParams<int>,
                       DeviceScanParams<float, double, rocprim::minimum<double>>,
                       DeviceScanParams<double, double, rocprim::plus<double>, true>,
                       DeviceScanParams<test_utils::custom_test_type<int>>,
                       DeviceScanParams<test_utils::custom_test_array_type<long long, 5>>>;

template <typename Params>
class RocprimDeviceScanFutureTests : public RocprimDeviceScanTests<Params>
{
};

TYPED_TEST_SUITE(RocprimDeviceScanFutureTests, RocprimDeviceScanFutureTestsParams);

TYPED_TEST(RocprimDeviceScanFutureTests, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;

    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            constexpr int future_size = 2048;
            const float   precision
                = test_utils::precision<T> * future_size
                  + (is_plus_op::value ? test_utils::precision<acc_type> * size : 0);

            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            const hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const std::vector<T> future_input
                = test_utils::get_random_data<T>(future_size, 1, 10, ~seed_value);
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U>       output(input.size());

            T* d_input;
            U* d_output;
            T* d_future_input;
            T* d_initial_value;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_future_input,
                                                         future_input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_initial_value, sizeof(T)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_future_input,
                                future_input.data(),
                                future_input.size() * sizeof(T),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            const acc_type initial_value = std::accumulate(future_input.begin(), future_input.end(), T(0));

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan(
                input.begin(), input.end(), initial_value, expected.begin(), scan_op);

            const auto future_iter
                = test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_initial_value);
            const auto future_initial_value
                = rocprim::future_value<acc_type, std::remove_const_t<decltype(future_iter)>>{
                    future_iter};

            auto input_iterator = rocprim::make_transform_iterator(
                d_input, [] (T in) { return static_cast<acc_type>(in); });

            // temp storage
            size_t temp_storage_size_bytes;
            char*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::exclusive_scan<Config>(
                nullptr, temp_storage_size_bytes, input_iterator, 
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                future_initial_value,
                input.size(),
                scan_op,
                stream,
                debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            size_t temp_storage_reduce = 0;
            HIP_CHECK(rocprim::reduce(
                nullptr, temp_storage_reduce, d_future_input, d_initial_value, 2048));

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(
                &d_temp_storage, temp_storage_size_bytes + temp_storage_reduce));
            HIP_CHECK(hipDeviceSynchronize());

            // Fill initial value on the device
            HIP_CHECK(rocprim::reduce(d_temp_storage + temp_storage_size_bytes,
                                      temp_storage_reduce,
                                      d_future_input,
                                      d_initial_value,
                                      2048));

            // Run
            HIP_CHECK(rocprim::exclusive_scan<Config>(
                d_temp_storage, temp_storage_size_bytes, input_iterator,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                future_initial_value,
                input.size(),
                scan_op,
                stream,
                debug_synchronous));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(
                output.data(), d_output, output.size() * sizeof(U), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, precision));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_future_input);
            hipFree(d_initial_value);
            hipFree(d_temp_storage);
        }
    }
}
