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

#include "common_test_header.hpp"

// Utils
#include "test_utils.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"

// hipcub API
#include <hipcub/device/device_find.hpp>

#include <algorithm>

// Params for tests
template<class RangeType,
         class ValuesType    = RangeType,
         class OutputType    = RangeType,
         class CompareOpType = _HIPCUB_STD::less<ValuesType>>
struct DeviceBinarySearchParams
{
    using range_type      = RangeType;
    using values_type     = ValuesType;
    using output_type     = OutputType;
    using compare_op_type = CompareOpType;
};

template<class Params>
class HipcubDeviceBinarySearchTests : public ::testing::Test
{
public:
    using range_type      = typename Params::range_type;
    using values_type     = typename Params::values_type;
    using output_type     = typename Params::output_type;
    using compare_op_type = typename Params::compare_op_type;
};

using HipcubDeviceBinarySearchTestsParams = ::testing::Types<DeviceBinarySearchParams<uint8_t>,
                                                             DeviceBinarySearchParams<int16_t>,
                                                             DeviceBinarySearchParams<uint32_t>,
                                                             DeviceBinarySearchParams<double>>;

TYPED_TEST_SUITE(HipcubDeviceBinarySearchTests, HipcubDeviceBinarySearchTestsParams);

template<bool IsLower, class... Args>
auto invoke_binary_search(Args&&... args) -> std::enable_if_t<IsLower, hipError_t>
{
    return hipcub::DeviceFind::LowerBound(std::forward<Args>(args)...);
}

template<bool IsLower, class... Args>
auto invoke_binary_search(Args&&... args) -> std::enable_if_t<!IsLower, hipError_t>
{
    return hipcub::DeviceFind::UpperBound(std::forward<Args>(args)...);
}

template<bool IsLower, typename RangeIteratorT, typename T, typename CompareOpT>
auto invoke_host_binary_search(RangeIteratorT first,
                               RangeIteratorT last,
                               const T&       value,
                               CompareOpT     comp) -> std::enable_if_t<IsLower, RangeIteratorT>
{
    return std::lower_bound(first, last, value, comp);
}

template<bool IsLower, typename RangeIteratorT, typename T, typename CompareOpT>
auto invoke_host_binary_search(RangeIteratorT first,
                               RangeIteratorT last,
                               const T&       value,
                               CompareOpT     comp) -> std::enable_if_t<!IsLower, RangeIteratorT>
{
    return std::upper_bound(first, last, value, comp);
}

template<typename TestFixture, bool IsLower, typename SizesF>
void test_binary_search(SizesF&& sizes_func)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using range_type      = typename TestFixture::range_type;
    using values_type     = typename TestFixture::values_type;
    using output_type     = typename TestFixture::output_type;
    using compare_op_type = typename TestFixture::compare_op_type;

    const compare_op_type compare_op{};

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : sizes_func(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const size_t             values_size = (size_t)std::sqrt(size);
            const size_t             d           = size / 100;
            std::vector<range_type>  range;
            std::vector<values_type> values;

            try
            {
                range.resize(size);
                test_utils::segmented_iota(range.begin(), range.end(), static_cast<range_type>(0));

                // Use a narrower range for values for checking out-of-haystack cases
                values = test_utils::get_random_data<values_type>(values_size,
                                                                  d,
                                                                  size + d,
                                                                  seed_value);
            }
            catch(const std::bad_alloc&)
            {
                GTEST_SKIP() << "Out of (host) memory. Skipping size = " << size;
            }

            range_type* d_range;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_range, range.size() * sizeof(range_type)));
            HIP_CHECK(hipMemcpy(d_range,
                                range.data(),
                                range.size() * sizeof(range_type),
                                hipMemcpyHostToDevice));
            values_type* d_values;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_values, values.size() * sizeof(values_type)));
            HIP_CHECK(hipMemcpy(d_values,
                                values.data(),
                                values.size() * sizeof(values_type),
                                hipMemcpyHostToDevice));
            output_type* d_output;
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_output, values.size() * sizeof(output_type)));

            // Compute expected results
            std::vector<output_type> expected(values.size());
            for(size_t i = 0; i < values.size(); ++i)
            {
                expected[i] = invoke_host_binary_search<IsLower>(range.begin(),
                                                                 range.end(),
                                                                 values[i],
                                                                 compare_op)
                              - range.begin();
            }

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(invoke_binary_search<IsLower>(nullptr,
                                                    temp_storage_size_bytes,
                                                    d_range,
                                                    range.size(),
                                                    d_values,
                                                    values.size(),
                                                    d_output,
                                                    compare_op,
                                                    stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void* d_temp_storage{};
            HIP_CHECK_MEMORY(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

            // Run
            HIP_CHECK(invoke_binary_search<IsLower>(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_range,
                                                    range.size(),
                                                    d_values,
                                                    values.size(),
                                                    d_output,
                                                    compare_op,
                                                    stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            std::vector<output_type> output(values.size());
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                values.size() * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_range));
            HIP_CHECK(hipFree(d_values));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));

            // Check
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }
}

TYPED_TEST(HipcubDeviceBinarySearchTests, LowerBound)
{
    test_binary_search<TestFixture, true /*IsLower*/>(
        [](unsigned int seed_value) { return test_utils::get_sizes(seed_value); });
}

TYPED_TEST(HipcubDeviceBinarySearchTests, UpperBound)
{
    test_binary_search<TestFixture, false /*IsLower*/>(
        [](unsigned int seed_value) { return test_utils::get_sizes(seed_value); });
}

template<class Params>
class HipcubDeviceBinarySearchLargeSizesTests : public ::testing::Test
{
public:
    using range_type      = typename Params::range_type;
    using values_type     = typename Params::values_type;
    using output_type     = typename Params::output_type;
    using compare_op_type = typename Params::compare_op_type;
};

using HipcubDeviceBinarySearchLargeSizesTestsParams
    = ::testing::Types<DeviceBinarySearchParams<char>>;

TYPED_TEST_SUITE(HipcubDeviceBinarySearchLargeSizesTests,
                 HipcubDeviceBinarySearchLargeSizesTestsParams);

TYPED_TEST(HipcubDeviceBinarySearchLargeSizesTests, LowerBoundLargeSizes)
{
    test_binary_search<TestFixture, true /*IsLower*/>(
        [](unsigned int seed_value) { return test_utils::get_large_sizes(seed_value); });
}

TYPED_TEST(HipcubDeviceBinarySearchLargeSizesTests, UpperBoundLargeSizes)
{
    test_binary_search<TestFixture, false /*IsLower*/>(
        [](unsigned int seed_value) { return test_utils::get_large_sizes(seed_value); });
}
