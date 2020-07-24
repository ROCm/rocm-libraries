// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

// required rocprim headers
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>
#include <rocprim/iterator/constant_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class ScanOp = ::rocprim::plus<OutputType>,
    // Tests output iterator with void value_type (OutputIterator concept)
    // scan-by-key primitives don't support output iterator with void value_type
    bool UseIdentityIteratorIfSupported = false
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using scan_op_type = ScanOp;
    static constexpr bool use_identity_iterator = UseIdentityIteratorIfSupported;
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
};

typedef ::testing::Types<
    // Small
    DeviceScanParams<char>,
    DeviceScanParams<unsigned short>,
    DeviceScanParams<short, int>,
    DeviceScanParams<int>,
    DeviceScanParams<float, float, rocprim::maximum<float> >,
    DeviceScanParams<int8_t, int8_t, rocprim::maximum<int8_t>>,
    DeviceScanParams<uint8_t, uint8_t, rocprim::maximum<uint8_t>>,
#ifndef __HIP__
    // hip-clang does provide host comparison operators
    DeviceScanParams<rocprim::half, rocprim::half, test_utils::half_maximum>,
    // hip-clang does not allow to convert half to float
    DeviceScanParams<rocprim::half, float>,
#endif
    // Large
    DeviceScanParams<int, double, rocprim::plus<int> >,
    DeviceScanParams<int, double, rocprim::plus<double> >,
    DeviceScanParams<int, long long, rocprim::plus<long long> >,
    DeviceScanParams<unsigned int, unsigned long long, rocprim::plus<unsigned long long> >,
    DeviceScanParams<long long, long long, rocprim::maximum<long long> >,
    DeviceScanParams<double, double, rocprim::plus<double>, true>,
    DeviceScanParams<signed char, long, rocprim::plus<long> >,
    DeviceScanParams<float, double, rocprim::minimum<double> >,
    DeviceScanParams<test_utils::custom_test_type<int> >,
    DeviceScanParams<
        test_utils::custom_test_type<double>, test_utils::custom_test_type<double>,
        rocprim::plus<test_utils::custom_test_type<double> >, true
    >,
    DeviceScanParams<test_utils::custom_test_type<int> >,
    DeviceScanParams<test_utils::custom_test_array_type<long long, 5> >,
    DeviceScanParams<test_utils::custom_test_array_type<int, 10> >
> RocprimDeviceScanTestsParams;

std::vector<size_t> get_sizes(int seed_value)
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 18),
        (1 << 20) - 12345
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(3, 1, 100000, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanEmptyInput)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0; // default

    U * d_output;
    HIP_CHECK(hipMalloc(&d_output, sizeof(U)));

    test_utils::out_of_bounds_flag out_of_bounds;
    test_utils::bounds_checking_iterator<U> d_checking_output(
        d_output,
        out_of_bounds.device_pointer(),
        0
    );

    // temp storage
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::inclusive_scan(
            d_temp_storage, temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_checking_output,
            0, scan_op_type(), stream, debug_synchronous
        )
    );

    // allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::inclusive_scan(
            d_temp_storage, temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_checking_output,
            0, scan_op_type(), stream, debug_synchronous
        )
    );
    HIP_CHECK(hipPeekAtLastError());
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
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size(), 0);

            T * d_input;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
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

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::inclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), scan_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::inclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), scan_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage));

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
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size());

            T * d_input;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
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
            T initial_value = test_utils::get_random_value<T>(1, 10, seed_value);
            test_utils::host_exclusive_scan(
                input.begin(), input.end(),
                initial_value, expected.begin(),
                scan_op
            );

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::exclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    initial_value, input.size(), scan_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::exclusive_scan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    initial_value, input.size(), scan_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage));

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
    const bool debug_synchronous = TestFixture::debug_synchronous;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
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
            std::vector<U> output(input.size(), 0);

            T * d_input;
            K * d_keys;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
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
            test_utils::host_inclusive_scan(
                rocprim::make_zip_iterator(
                    rocprim::make_tuple(input.begin(), keys.begin())
                ),
                rocprim::make_zip_iterator(
                    rocprim::make_tuple(input.end(), keys.end())
                ),
                rocprim::make_zip_iterator(
                    rocprim::make_tuple(expected.begin(), rocprim::make_discard_iterator())
                ),
                [scan_op, keys_compare_op](const rocprim::tuple<U, K>& t1,
                                        const rocprim::tuple<U, K>& t2)
                    -> rocprim::tuple<U, K>
                {
                    if(keys_compare_op(rocprim::get<1>(t1), rocprim::get<1>(t2)))
                    {
                        return rocprim::make_tuple(
                            scan_op(rocprim::get<0>(t1), rocprim::get<0>(t2)),
                            rocprim::get<1>(t2)
                        );
                    }
                    return t2;
                }
            );

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::inclusive_scan_by_key(
                    d_temp_storage, temp_storage_size_bytes,
                    d_keys, d_input, d_output, input.size(),
                    scan_op, keys_compare_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::inclusive_scan_by_key(
                    d_temp_storage, temp_storage_size_bytes,
                    d_keys, d_input, d_output, input.size(),
                    scan_op, keys_compare_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage));

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
    const bool debug_synchronous = TestFixture::debug_synchronous;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const bool use_unique_keys = bool(test_utils::get_random_value<int>(0, 1, seed_value));

            // Generate data
            T initial_value = test_utils::get_random_value<T>(1, 100, seed_value);
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
            std::vector<U> output(input.size(), 0);

            T * d_input;
            K * d_keys;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
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

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::exclusive_scan_by_key(
                    d_temp_storage, temp_storage_size_bytes,
                    d_keys, d_input, d_output, initial_value, input.size(),
                    scan_op, keys_compare_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::exclusive_scan_by_key(
                    d_temp_storage, temp_storage_size_bytes,
                    d_keys, d_input, d_output, initial_value, input.size(),
                    scan_op, keys_compare_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage));

            hipFree(d_keys);
            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}
