// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// hipcub API
#include <hipcub/device/device_scan.hpp>

#include "single_index_iterator.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_data_generation.hpp"

// Params for tests
template<class InputType,
         class OutputType = InputType,
         class ScanOp     = test_utils::plus,
         class KeyType    = int,
         bool UseGraphs   = false>
struct DeviceScanParams
{
    using input_type   = InputType;
    using output_type  = OutputType;
    using scan_op_type = ScanOp;

    static_assert(std::is_integral<KeyType>::value, "Keys must be integral");
    using key_type                   = KeyType;
    static constexpr bool use_graphs = UseGraphs;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceScanTests : public ::testing::Test
{
public:
    using input_type                 = typename Params::input_type;
    using output_type                = typename Params::output_type;
    using scan_op_type               = typename Params::scan_op_type;
    using key_type                   = typename Params::key_type;
    static constexpr bool use_graphs = Params::use_graphs;
};

using HipcubDeviceScanTestsParams = ::testing::Types<
    DeviceScanParams<int, long>,
    DeviceScanParams<unsigned long long, unsigned long long, test_utils::minimum>,
    DeviceScanParams<unsigned long>,
    DeviceScanParams<short, float, test_utils::maximum>,
    DeviceScanParams<int, double>,
    DeviceScanParams<test_utils::bfloat16, test_utils::bfloat16, test_utils::maximum>,
    DeviceScanParams<test_utils::half, test_utils::half, test_utils::maximum>,
    DeviceScanParams<int, long, test_utils::plus, int, true>>;

// use float for accumulation of bfloat16 and half inputs if operator is plus
template<typename input_type, typename input_op_type>
struct accum_type
{
    static constexpr bool is_low_precision = std::is_same_v<input_type, test_utils::half>
                                             || std::is_same_v<input_type, test_utils::bfloat16>;
    static constexpr bool is_add = test_utils::is_add_operator<input_op_type>::value;
    using type = typename std::conditional_t<is_low_precision && is_add, float, input_type>;
};

TYPED_TEST_SUITE(HipcubDeviceScanTests, HipcubDeviceScanTestsParams);

namespace
{
template<typename T>
std::vector<T>
    generate_segments(const size_t size, const size_t max_segment_length, const int seed_value)
{
    static_assert(std::is_integral<T>::value, "Key type must be integral");

    std::default_random_engine            prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    std::uniform_int_distribution<T>      key_distribution(_HIPCUB_STD::numeric_limits<T>::max());
    std::vector<T>                        keys(size);

    size_t keys_start_index = 0;
    while(keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end
            = _HIPCUB_STD::min(size, keys_start_index + new_segment_length);
        const T      key                = key_distribution(prng);
        std::fill(std::next(keys.begin(), keys_start_index),
                  std::next(keys.begin(), new_segment_end),
                  key);
        keys_start_index += new_segment_length;
    }
    return keys;
}
} // namespace

TYPED_TEST(HipcubDeviceScanTests, AccumulatorTypeTest)
{
    using T = hipcub::detail::accumulator_t<typename TestFixture::scan_op_type,
                                            typename TestFixture::input_type>;
    using U = typename TestFixture::input_type;
    static_assert(std::is_same_v<T, U>, "accumulator type mismatch");
    ASSERT_TRUE(true);
}

TYPED_TEST(HipcubDeviceScanTests, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType     = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    constexpr bool inplace = std::is_same_v<T, U> && std::is_same_v<acc_type, T>;

    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size(), test_utils::convert_to_device<U>(0));

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            if constexpr(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            }
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan(input.begin(), input.end(), expected.begin(), scan_op);

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if constexpr(std::is_same_v<scan_op_type, test_utils::plus>)
                {
                    if constexpr(inplace)
                    {
                        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp_storage,
                                                                   temp_storage_size_bytes,
                                                                   d_input,
                                                                   input.size(),
                                                                   stream));
                    }
                    else
                    {
                        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp_storage,
                                                                   temp_storage_size_bytes,
                                                                   input_iterator,
                                                                   d_output,
                                                                   input.size(),
                                                                   stream));
                    }
                }
                else
                {
                    if constexpr(inplace)
                    {
                        HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    d_input,
                                                                    scan_op,
                                                                    input.size(),
                                                                    stream));
                    }
                    else
                    {
                        HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    input_iterator,
                                                                    d_output,
                                                                    scan_op,
                                                                    input.size(),
                                                                    stream));
                    }
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            if constexpr(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceScanTests, InclusiveScanInit)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType     = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    constexpr bool inplace = std::is_same_v<T, U> && std::is_same_v<acc_type, T>;

    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size(), test_utils::convert_to_device<U>(0));

            const T initial_value
                = test_utils::get_random_value<T>(test_utils::convert_to_device<T>(1),
                                                  test_utils::convert_to_device<T>(100),
                                                  seed_value + seed_value_addition);

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            if constexpr(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            }
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan_init(input.begin(),
                                                 input.end(),
                                                 expected.begin(),
                                                 initial_value,
                                                 scan_op);

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if constexpr(inplace)
                {
                    HIP_CHECK(hipcub::DeviceScan::InclusiveScanInit(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    d_input,
                                                                    d_input,
                                                                    scan_op,
                                                                    initial_value,
                                                                    input.size(),
                                                                    stream));
                }
                else
                {
                    HIP_CHECK(hipcub::DeviceScan::InclusiveScanInit(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    input_iterator,
                                                                    d_output,
                                                                    scan_op,
                                                                    initial_value,
                                                                    input.size(),
                                                                    stream));
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if(TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            if constexpr(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(HipcubDeviceScanTests, InclusiveScanByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using K = typename TestFixture::key_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;
    constexpr size_t max_segment_length = 100;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            const std::vector<K> keys = generate_segments<K>(size, max_segment_length, seed_value);
            const std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size(), test_utils::convert_to_device<U>(0));

            T* d_input;
            U* d_output;
            K* d_keys;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(K), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan_by_key(input.begin(),
                                                   input.end(),
                                                   keys.begin(),
                                                   expected.begin(),
                                                   scan_op,
                                                   test_utils::equal());

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            // temp storage
            size_t temp_storage_size_bytes{};
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if(std::is_same_v<scan_op_type, test_utils::plus>)
            {
                HIP_CHECK(hipcub::DeviceScan::InclusiveSumByKey(d_temp_storage,
                                                                temp_storage_size_bytes,
                                                                d_keys,
                                                                input_iterator,
                                                                d_output,
                                                                static_cast<int>(input.size()),
                                                                test_utils::equal(),
                                                                stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceScan::InclusiveScanByKey(d_temp_storage,
                                                                 temp_storage_size_bytes,
                                                                 d_keys,
                                                                 input_iterator,
                                                                 d_output,
                                                                 scan_op,
                                                                 static_cast<int>(input.size()),
                                                                 test_utils::equal(),
                                                                 stream));
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            if(std::is_same_v<scan_op_type, test_utils::plus>)
            {
                HIP_CHECK(hipcub::DeviceScan::InclusiveSumByKey(d_temp_storage,
                                                                temp_storage_size_bytes,
                                                                d_keys,
                                                                input_iterator,
                                                                d_output,
                                                                static_cast<int>(input.size()),
                                                                test_utils::equal(),
                                                                stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceScan::InclusiveScanByKey(d_temp_storage,
                                                                 temp_storage_size_bytes,
                                                                 d_keys,
                                                                 input_iterator,
                                                                 d_output,
                                                                 scan_op,
                                                                 static_cast<int>(input.size()),
                                                                 test_utils::equal(),
                                                                 stream));
            }

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceScanTests, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType     = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    constexpr bool inplace = std::is_same_v<T, U> && std::is_same_v<acc_type, T>;

    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size());

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            if constexpr(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            }
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            const T        initial_value
                = std::is_same_v<scan_op_type, test_utils::plus>
                      ? test_utils::convert_to_device<T>(0)
                      : test_utils::get_random_value<T>(test_utils::convert_to_device<T>(1),
                                                        test_utils::convert_to_device<T>(100),
                                                        seed_value + seed_value_addition);
            test_utils::host_exclusive_scan(input.begin(),
                                            input.end(),
                                            initial_value,
                                            expected.begin(),
                                            scan_op);

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if constexpr(std::is_same_v<scan_op_type, test_utils::plus>)
                {
                    if constexpr(inplace)
                    {
                        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                   temp_storage_size_bytes,
                                                                   d_input,
                                                                   input.size(),
                                                                   stream));
                    }
                    else
                    {
                        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp_storage,
                                                                   temp_storage_size_bytes,
                                                                   input_iterator,
                                                                   d_output,
                                                                   input.size(),
                                                                   stream));
                    }
                }
                else
                {
                    if constexpr(inplace)
                    {
                        HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    d_input,
                                                                    scan_op,
                                                                    initial_value,
                                                                    input.size(),
                                                                    stream));
                    }
                    else
                    {
                        HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                                    temp_storage_size_bytes,
                                                                    input_iterator,
                                                                    d_output,
                                                                    scan_op,
                                                                    initial_value,
                                                                    input.size(),
                                                                    stream));
                    }
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            if constexpr(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceScanTests, ExclusiveScanByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using K = typename TestFixture::key_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;
    constexpr size_t max_segment_length = 100;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            const std::vector<K> keys = generate_segments<K>(size, max_segment_length, seed_value);
            const std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size(), test_utils::convert_to_device<U>(0));

            std::vector<T> initial_value_vector
                = test_utils::get_random_data<T>(1,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            T initial_value = initial_value_vector.front();
            if(std::is_same_v<scan_op_type, test_utils::plus>)
            {
                initial_value = test_utils::convert_to_device<T>(0);
            }

            T* d_input;
            U* d_output;
            K* d_keys;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(K), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan_by_key(input.begin(),
                                                   input.end(),
                                                   keys.begin(),
                                                   initial_value,
                                                   expected.begin(),
                                                   scan_op,
                                                   test_utils::equal());

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if(std::is_same_v<scan_op_type, test_utils::plus>)
            {
                HIP_CHECK(hipcub::DeviceScan::ExclusiveSumByKey(d_temp_storage,
                                                                temp_storage_size_bytes,
                                                                d_keys,
                                                                input_iterator,
                                                                d_output,
                                                                static_cast<int>(input.size()),
                                                                test_utils::equal(),
                                                                stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceScan::ExclusiveScanByKey(d_temp_storage,
                                                                 temp_storage_size_bytes,
                                                                 d_keys,
                                                                 input_iterator,
                                                                 d_output,
                                                                 scan_op,
                                                                 initial_value,
                                                                 static_cast<int>(input.size()),
                                                                 test_utils::equal(),
                                                                 stream));
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            if(std::is_same_v<scan_op_type, test_utils::plus>)
            {
                HIP_CHECK(hipcub::DeviceScan::ExclusiveSumByKey(d_temp_storage,
                                                                temp_storage_size_bytes,
                                                                d_keys,
                                                                input_iterator,
                                                                d_output,
                                                                static_cast<int>(input.size()),
                                                                test_utils::equal(),
                                                                stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceScan::ExclusiveScanByKey(d_temp_storage,
                                                                 temp_storage_size_bytes,
                                                                 d_keys,
                                                                 input_iterator,
                                                                 d_output,
                                                                 scan_op,
                                                                 initial_value,
                                                                 static_cast<int>(input.size()),
                                                                 test_utils::equal(),
                                                                 stream));
            }

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(HipcubDeviceScanTests, LargeIndicesInclusiveScan)
{
    using T              = unsigned int;
    using InputIterator  = test_utils::counting_iterator<T>;
    using OutputIterator = test_utils::single_index_iterator<T>;

    const size_t size = (1ul << 31) + 1ul;

    hipStream_t stream = 0; // default

    unsigned int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    // Create test_utils::counting_iterator<U> with random starting point
    InputIterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value));

    T* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());
    OutputIterator output_it(d_output, size - 1);

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;

    // Get temporary array size
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                input_begin,
                                                output_it,
                                                test_utils::plus{},
                                                size,
                                                stream));

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Run
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                input_begin,
                                                output_it,
                                                test_utils::plus{},
                                                size,
                                                stream));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    T actual_output;
    HIP_CHECK(hipMemcpy(&actual_output, d_output, sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Validating results
    // Sum of 'size' increasing numbers starting at 'n' is size * (2n + size - 1)
    // The division is not integer division but either (size) or (2n + size - 1) has to be even.
    const T multiplicand_1  = size;
    const T multiplicand_2  = 2 * (*input_begin) + size - 1;
    const T expected_output = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                        : multiplicand_1 * (multiplicand_2 / 2);
    ASSERT_EQ(expected_output, actual_output);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

TEST(HipcubDeviceScanTests, LargeIndicesExclusiveScan)
{
    using T              = unsigned int;
    using InputIterator  = test_utils::counting_iterator<T>;
    using OutputIterator = test_utils::single_index_iterator<T>;

    const size_t size = (1ul << 31) + 1ul;

    hipStream_t stream = 0; // default

    unsigned int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    // Create test_utils::counting_iterator<U> with random starting point
    InputIterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value));
    T             initial_value = test_utils::get_random_value<T>(1, 10, seed_value);

    T* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());
    OutputIterator output_it(d_output, size - 1);

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;

    // Get temporary array size
    HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                input_begin,
                                                output_it,
                                                test_utils::plus{},
                                                initial_value,
                                                size,
                                                stream));

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Run
    HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                input_begin,
                                                output_it,
                                                test_utils::plus{},
                                                initial_value,
                                                size,
                                                stream));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    T actual_output;
    HIP_CHECK(hipMemcpy(&actual_output, d_output, sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Validating results
    // Sum of 'size' - 1 increasing numbers starting at 'n' is (size - 1) * (2n + size - 2)
    // The division is not integer division but either (size - 1) or (2n + size - 2) has to be even.
    const T multiplicand_1 = size - 1;
    const T multiplicand_2 = 2 * (*input_begin) + size - 2;

    const T product = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                : multiplicand_1 * (multiplicand_2 / 2);

    const T expected_output = initial_value + product;

    ASSERT_EQ(expected_output, actual_output);

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

template<typename T>
static __global__
void fill_initial_value(T* ptr, const T initial_value)
{
    *ptr = initial_value;
}

TYPED_TEST(HipcubDeviceScanTests, ExclusiveScanFuture)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // if scan_op_type is plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_add_op    = test_utils::is_add_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;
    using IteratorType = test_utils::transform_iterator<T*, hipcub::CastOp<acc_type>>;
    // for non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision
        = is_add_op::value ? test_utils::precision<acc_type>::value : 0;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            const std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(10),
                                                 seed_value);
            std::vector<U> output(input.size());

            T* d_input;
            U* d_output;
            U* d_initial_value;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_initial_value, sizeof(U)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            const scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            const U        initial_value
                = (U)test_utils::get_random_value<T>(test_utils::convert_to_device<U>(1),
                                                     test_utils::convert_to_device<U>(100),
                                                     seed_value + seed_value_addition);
            test_utils::host_exclusive_scan(input.begin(),
                                            input.end(),
                                            initial_value,
                                            expected.begin(),
                                            scan_op);

            // Scan operator: CastOp.
            hipcub::CastOp<acc_type> op{};

            // Transform input applying the casting operator.
            auto input_iterator = IteratorType(d_input, op);

            const auto future_initial_value = hipcub::FutureValue<U>{d_initial_value};

            // Check the provided aliases to be correct at compile-time
            static_assert(std::is_same_v<typename decltype(future_initial_value)::value_type, U>,
                          "The futures value type is expected to be U");

            static_assert(
                std::is_same_v<typename decltype(future_initial_value)::iterator_type, U*>,
                "The futures iterator type is expected to be U*");

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        input_iterator,
                                                        d_output,
                                                        scan_op,
                                                        future_initial_value,
                                                        input.size(),
                                                        stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Fill initial value
            fill_initial_value<<<1, 1, 0, stream>>>(d_initial_value, initial_value);
            HIP_CHECK(hipGetLastError());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        input_iterator,
                                                        d_output,
                                                        scan_op,
                                                        future_initial_value,
                                                        input.size(),
                                                        stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_initial_value));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

// Params for invalid data tests
template<typename SegmentOffsetT>
struct segment;

template<class SegmentOffsetT = unsigned int,
         class InputType      = segment<SegmentOffsetT>,
         class CountT         = unsigned int>
struct InvalidDeviceScanParams
{
    using segment_offset_type = SegmentOffsetT;
    using input_type          = InputType;
    using count_type          = CountT;

    static_assert(std::is_same_v<input_type, segment<SegmentOffsetT>>
                      || (sizeof(segment<SegmentOffsetT>) == sizeof(input_type)),
                  "If input is a primitive type, it must be bit-casteable to segment");
};

// -----------------------------------------------------------------------
// Test for scan ops using non-consecutive partial results for the scan op
// -----------------------------------------------------------------------

template<class Params>
class HipcubInvalidDeviceScanTests : public ::testing::Test
{
public:
    using segment_offset_type = typename Params::segment_offset_type;
    using input_type          = typename Params::input_type;
    using count_type          = typename Params::count_type;
};

using HipcubInvalidDeviceScanTestsParams
    = ::testing::Types<InvalidDeviceScanParams<>,
                       InvalidDeviceScanParams<unsigned int, unsigned long long>>;

TYPED_TEST_SUITE(HipcubInvalidDeviceScanTests, HipcubInvalidDeviceScanTestsParams);

template<typename Destination, typename Source>
HIPCUB_HOST_DEVICE
Destination bit_cast(const Source& source)
{
    static_assert(sizeof(Source) == sizeof(Destination),
                  "Destination and Source types must have the same size");
    Destination dest;
    memcpy(static_cast<void*>(&dest), &source, sizeof(Destination));
    return dest;
}

template<bool Inclusive,
         bool WithInitValue,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class InitValueType>
void invoke_scan(void*          d_temp_storage,
                 size_t&        temp_storage_size_bytes,
                 InputIterator  input,
                 OutputIterator output,
                 BinaryFunction scan_op,
                 InitValueType  init,
                 size_t         size,
                 hipStream_t    stream)
{
    if constexpr(Inclusive)
    {
        if constexpr(WithInitValue)
        {
            HIP_CHECK(hipcub::DeviceScan::InclusiveScanInit(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            input,
                                                            output,
                                                            scan_op,
                                                            init,
                                                            size,
                                                            stream));
        }
        else
        {

            HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        input,
                                                        output,
                                                        scan_op,
                                                        size,
                                                        stream));
        }
    }
    else
    {
        HIP_CHECK(hipcub::DeviceScan::ExclusiveScan(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    input,
                                                    output,
                                                    scan_op,
                                                    init,
                                                    size,
                                                    stream));
    }
}

template<typename SegmentOffsetT>
struct segment
{
    // Make sure that default constructed segments can not be merged
    SegmentOffsetT begin = _HIPCUB_STD::numeric_limits<SegmentOffsetT>::min();
    SegmentOffsetT end   = _HIPCUB_STD::numeric_limits<SegmentOffsetT>::max();

    HIPCUB_HOST_DEVICE
    segment()
        = default;

    HIPCUB_HOST_DEVICE
    constexpr segment(SegmentOffsetT b, SegmentOffsetT e)
        : begin{b}, end{e}
    {}

    HIPCUB_HOST_DEVICE
    friend bool operator==(segment left, segment right)
    {
        return left.begin == right.begin && left.end == right.end;
    }
};

template<typename SegmentOffsetT, typename WrapperT>
struct tuple_to_wrapper_op
{
    HIPCUB_HOST_DEVICE
    auto operator()(test_utils::tuple<SegmentOffsetT, SegmentOffsetT> interval) const
    {
#if defined(__HIP_PLATFORM_AMD__)
        SegmentOffsetT begin = rocprim::get<0>(interval);
        SegmentOffsetT end   = rocprim::get<1>(interval);
#elif defined(__HIP_PLATFORM_NVIDIA__)
        const auto [begin, end] = interval;
#endif
        return bit_cast<WrapperT>(segment{begin, end});
    }
};

template<typename CountT>
struct counts_t
{
    CountT default_init; // Partial results that were default-initialized
    CountT zero_init; // Partial results that were zero-initialized
    CountT other; // Duplicate elements or valid elements combined in an invalid order
    CountT cascade; // Count down-the-line errors
};

template<typename CountT, typename SegmentOffsetT>
struct segments_op
{
    using seg_t       = segment<SegmentOffsetT>;
    using counts_type = counts_t<CountT>;

    static constexpr auto minus_1  = static_cast<SegmentOffsetT>(-1);
    static constexpr auto cascaded = seg_t{minus_1, minus_1};
    counts_type*          counts;

    HIPCUB_HOST_DEVICE
    segments_op(counts_type* counts_)
        : counts{counts_}
    {}

    HIPCUB_DEVICE
    bool check_inputs(seg_t left, seg_t right) const
    {
        // left == right to detect zero-initialized segments
        if(left.end != right.begin || left == right)
        {
            CountT* count_ptr = &counts->other;
            if(left == cascaded || right == cascaded)
            {
                count_ptr = &counts->cascade;
            }
            else if(left == seg_t{} || right == seg_t{})
            {
                count_ptr = &counts->default_init;
            }
            else if(left == seg_t{0, 0} || right == seg_t{0, 0})
            {
                count_ptr = &counts->zero_init;
            }
            atomicAdd(count_ptr, 1);
            return true;
        }
        return false;
    }

    HIPCUB_DEVICE
    seg_t operator()(seg_t left, seg_t right) const
    {
        if(check_inputs(left, right))
        {
            return cascaded;
        }
        return {left.begin, right.end};
    }

    template<typename PrimitiveT>
    HIPCUB_DEVICE
    PrimitiveT operator()(PrimitiveT p_left, PrimitiveT p_right) const
    {
        const auto left  = bit_cast<seg_t>(p_left);
        const auto right = bit_cast<seg_t>(p_right);
        return bit_cast<PrimitiveT>(this->operator()(left, right));
    }
};

template<typename SegmentOffsetT>
struct segments_host_op
{
    using seg_t = segment<SegmentOffsetT>;
    HIPCUB_HOST
    seg_t operator()(seg_t left, seg_t right) const
    {
        return {left.begin, right.end};
    }

    template<typename PrimitiveT>
    HIPCUB_HOST
    PrimitiveT operator()(PrimitiveT p_left, PrimitiveT p_right) const
    {
        const auto left  = bit_cast<seg_t>(p_left);
        const auto right = bit_cast<seg_t>(p_right);
        return bit_cast<PrimitiveT>(this->operator()(left, right));
    }
};

template<typename TestFixture, bool Inclusive, bool WithInitValue>
void invalid_data_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                 = typename TestFixture::input_type;
    using U                 = T;
    using count_t           = typename TestFixture::count_type;
    using segment_offset_t  = typename TestFixture::segment_offset_type;
    using counts_type       = counts_t<count_t>;
    using scan_op_type      = segments_op<count_t, segment_offset_t>;
    using scan_host_op_type = segments_host_op<segment_offset_t>;

    using segment_t = segment<segment_offset_t>;

    using counting_it_t    = test_utils::counting_iterator<segment_offset_t>;
    using iterator_tuple_t = test_utils::tuple<counting_it_t, counting_it_t>;
    using zip_iterator_t   = test_utils::zip_iterator<iterator_tuple_t>;
    using transform_op_t   = tuple_to_wrapper_op<segment_offset_t, T>;
    using transform_it_t   = test_utils::transform_iterator<zip_iterator_t, transform_op_t, U>;

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            const auto initial_value
                = bit_cast<U>((Inclusive && !WithInitValue) ? segment_t{0, 0} : segment_t{0, 1});
            const auto input_iterator = transform_it_t(
                zip_iterator_t(test_utils::make_tuple(counting_it_t{1}, counting_it_t{2})),
                transform_op_t());
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(U)));

            counts_type* d_counts;
            HIP_CHECK(hipMalloc(&d_counts, sizeof(counts_type)));
            HIP_CHECK(hipMemset(d_counts, 0, sizeof(counts_type)));

            // scan function
            scan_op_type scan_op{d_counts};

            const auto h_input_iterator = transform_it_t(
                zip_iterator_t(test_utils::make_tuple(counting_it_t{1}, counting_it_t{2})),
                transform_op_t());

            // Calculate expected results on host
            scan_host_op_type scan_host_op{};
            std::vector<U>    expected(size);
            if constexpr(Inclusive)
            {
                if constexpr(WithInitValue)
                {
                    test_utils::host_inclusive_scan_init(h_input_iterator,
                                                         h_input_iterator + size,
                                                         expected.begin(),
                                                         initial_value,
                                                         scan_host_op);
                }
                else
                {
                    test_utils::host_inclusive_scan(h_input_iterator,
                                                    h_input_iterator + size,
                                                    expected.begin(),
                                                    scan_host_op);
                }
            }
            else
            {
                test_utils::host_exclusive_scan(h_input_iterator,
                                                h_input_iterator + size,
                                                initial_value,
                                                expected.begin(),
                                                scan_host_op);
            }

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;

            // Get size of d_temp_storage
            invoke_scan<Inclusive, WithInitValue>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  input_iterator,
                                                  d_output,
                                                  scan_op,
                                                  initial_value,
                                                  size,
                                                  stream);
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            invoke_scan<Inclusive, WithInitValue>(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  input_iterator,
                                                  d_output,
                                                  scan_op,
                                                  initial_value,
                                                  size,
                                                  stream);

            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(d_temp_storage));

            // Copy output to host
            std::vector<U> output(size);
            counts_type    counts{};
            HIP_CHECK(hipMemcpy(&counts, d_counts, sizeof(counts_type), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(d_counts));
            HIP_CHECK(hipFree(d_output));

            // Output values should still be as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            // The actual core requirement expected to fail
            auto is_zero = [](counts_type c) -> bool
            {
                return c.default_init == count_t{0} && c.zero_init == count_t{0}
                       && c.other == count_t{0} && c.cascade == count_t{0};
            };
            if(!is_zero(counts))
            {
                printf("This test is allowed to fail (and it does), because device_*_scan calls "
                       "the scan operator on non-consecutive partial results\n");
                break;
            }
            else
            {
                ASSERT_NO_FATAL_FAILURE(ASSERT_TRUE(is_zero(counts)));
            }
        }
    }
}

TYPED_TEST(HipcubInvalidDeviceScanTests, InvalidDataInclusiveScan)
{
    invalid_data_test<TestFixture, true /*Inclusive*/, false /*WithInitValue*/>();
}

TYPED_TEST(HipcubInvalidDeviceScanTests, InvalidDataInclusiveScanInit)
{
    invalid_data_test<TestFixture, true /*Inclusive*/, true /*WithInitValue*/>();
}

TYPED_TEST(HipcubInvalidDeviceScanTests, InvalidDataExclusiveScan)
{
    invalid_data_test<TestFixture, false /*Inclusive*/, false /*WithInitValue*/>();
}

TYPED_TEST(HipcubInvalidDeviceScanTests, InvalidDataExclusiveScanInit)
{
    invalid_data_test<TestFixture, false /*Inclusive*/, true /*WithInitValue*/>();
}
