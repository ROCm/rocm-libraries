// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Google Test
#include <gtest/gtest.h>
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    // Tests output iterator with void value_type (OutputIterator concept)
    // scan-by-key primitives don't support output iterator with void value_type
    bool UseIdentityIteratorIfSupported = false
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
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
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DeviceScanParams<int>,
    DeviceScanParams<double, double, true>,
    DeviceScanParams<short, int>,
    DeviceScanParams<long, double>,
    DeviceScanParams<test_utils::custom_test_type<double>, test_utils::custom_test_type<double>, true>
> RocprimDeviceScanTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 18)
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanSum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
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
        ::rocprim::plus<U> plus_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        test_utils::host_inclusive_scan(
            input.begin(), input.end(),
            expected.begin(), plus_op
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
                input.size(), plus_op, stream, debug_synchronous
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
                input.size(), plus_op, stream, debug_synchronous
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScanSum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
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
        ::rocprim::plus<U> plus_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        T initial_value = test_utils::get_random_value<T>(1, 10);
        test_utils::host_exclusive_scan(
            input.begin(), input.end(),
            initial_value, expected.begin(),
            plus_op
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
                initial_value, input.size(), plus_op, stream, debug_synchronous
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
                initial_value, input.size(), plus_op, stream, debug_synchronous
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // key type
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
        std::vector<K> keys = test_utils::get_random_data<K>(size, 1, 16);
        std::sort(keys.begin(), keys.end());
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
        rocprim::plus<U> scan_op;
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_keys);
        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // key type
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        T initial_value = test_utils::get_random_value<T>(1, 1);
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
        std::vector<K> keys = test_utils::get_random_data<K>(size, 1, 16);
        std::sort(keys.begin(), keys.end());
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
        rocprim::plus<U> scan_op;
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_keys);
        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}
