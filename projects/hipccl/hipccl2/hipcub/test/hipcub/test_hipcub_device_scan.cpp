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

// hipcub API
#include "hipcub/device/device_scan.hpp"
#include "hipcub/iterator/counting_input_iterator.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class ScanOp = hipcub::Sum,
    class KeyType = int
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using scan_op_type = ScanOp;

    static_assert(std::is_integral<KeyType>::value,
        "Keys must be integral");
    using key_type = KeyType;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceScanTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using scan_op_type = typename Params::scan_op_type;
    using key_type = typename Params::key_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceScanParams<int, long>,
    DeviceScanParams<unsigned long long, unsigned long long, hipcub::Min>,
    DeviceScanParams<unsigned long>,
    DeviceScanParams<short, float, hipcub::Max>,
    DeviceScanParams<int, double>
> HipcubDeviceScanTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 18) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(HipcubDeviceScanTests, HipcubDeviceScanTestsParams);

namespace
{
template<typename T>
std::vector<T> generate_segments(const size_t size,
                                 const size_t max_segment_length,
                                 const int seed_value)
{
    static_assert(std::is_integral<T>::value, "Key type must be integral");

    std::default_random_engine prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    std::uniform_int_distribution<T> key_distribution(std::numeric_limits<T>::max());
    std::vector<T> keys(size);

    size_t keys_start_index = 0;
    while (keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end = std::min(size, keys_start_index + new_segment_length);
        const T key = key_distribution(prng);
        std::fill(
            std::next(keys.begin(), keys_start_index),
            std::next(keys.begin(), new_segment_end),
            key
        );
        keys_start_index += new_segment_length;
    }
    return keys;
}
}

TYPED_TEST(HipcubDeviceScanTests, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
      for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
      {
          unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
          SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
          SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size(), 0);

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

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if(std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveSum(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveScan(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, scan_op, input.size(),
                        stream, debug_synchronous
                    )
                );
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            if(std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveSum(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveScan(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, scan_op, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
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
            for(size_t i = 0; i < output.size(); i++)
            {
                auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
                if(std::is_integral<U>::value) diff = 0;
                ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
            }

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
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
    constexpr size_t max_segment_length = 100;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for (auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            const unsigned int seed_value = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const hipStream_t stream = 0; // default

            // Generate data
            const std::vector<K> keys = generate_segments<K>(size, max_segment_length, seed_value);
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size(), 0);

            T *d_input;
            U *d_output;
            K *d_keys;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
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

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan_by_key(
                input.begin(), input.end(), keys.begin(),
                expected.begin(), scan_op, hipcub::Equality()
            );

            // temp storage
            size_t temp_storage_size_bytes{};
            void *d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if (std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveSumByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveScanByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, scan_op, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            if (std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveSumByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::InclusiveScanByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, scan_op, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }
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
            for (size_t i = 0; i < output.size(); i++)
            {
                auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
                if (std::is_integral<U>::value)
                {
                    diff = 0;
                }
                ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
            }

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}

TYPED_TEST(HipcubDeviceScanTests, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

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
            const T initial_value =
                std::is_same<scan_op_type, hipcub::Sum>::value
                ? T(0)
                : test_utils::get_random_value<T>(1, 100, seed_value + seed_value_addition);
            test_utils::host_exclusive_scan(
                input.begin(), input.end(),
                initial_value, expected.begin(),
                scan_op
            );

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if(std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveSum(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveScan(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, scan_op, initial_value, input.size(),
                        stream, debug_synchronous
                    )
                );
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            if(std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveSum(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveScan(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, scan_op, initial_value, input.size(),
                        stream, debug_synchronous
                    )
                );
            }
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
            for(size_t i = 0; i < output.size(); i++)
            {
                auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
                if(std::is_integral<U>::value) diff = 0;
                ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
            }

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
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
    constexpr size_t max_segment_length = 100;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for (auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            const unsigned int seed_value = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const hipStream_t stream = 0; // default

            // Generate data
            const std::vector<K> keys = generate_segments<K>(size, max_segment_length, seed_value);
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size(), 0);

            std::vector<T> initial_value_vector = test_utils::get_random_data<T>(1, 1, 10, seed_value);
            T initial_value = initial_value_vector.front();
            if (std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                initial_value = static_cast<T>(0);
            }

            T *d_input;
            U *d_output;
            K *d_keys;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(K)));
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

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan_by_key(
                input.begin(), input.end(), keys.begin(), initial_value,
                expected.begin(), scan_op, hipcub::Equality()
            );

            // temp storage
            size_t temp_storage_size_bytes;
            void *d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if (std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveSumByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveScanByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, scan_op, initial_value,
                        static_cast<int>(input.size()), hipcub::Equality(),
                        stream, debug_synchronous
                    )
                );
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            if (std::is_same<scan_op_type, hipcub::Sum>::value)
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveSumByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, static_cast<int>(input.size()),
                        hipcub::Equality(), stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    hipcub::DeviceScan::ExclusiveScanByKey(
                        d_temp_storage, temp_storage_size_bytes,
                        d_keys, d_input, d_output, scan_op, initial_value,
                        static_cast<int>(input.size()), hipcub::Equality(),
                        stream, debug_synchronous
                    )
                );
            }
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
            for (size_t i = 0; i < output.size(); i++)
            {
                auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
                if (std::is_integral<U>::value)
                {
                    diff = 0;
                }
                ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
            }

            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}

// CUB does not support large indices in inclusive and exclusive scans
#ifndef __HIP_PLATFORM_NVIDIA__

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

TEST(HipcubDeviceScanTests, LargeIndicesInclusiveScan)
{
    using T = unsigned int;
    using InputIterator = typename hipcub::CountingInputIterator<T>;
    using OutputIterator = single_index_iterator<T>;

    const bool debug_synchronous = false;

    const size_t size = (1ul << 31) + 1ul;

    hipStream_t stream = 0; // default

    unsigned int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    // Create CountingInputIterator<U> with random starting point
    InputIterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value));

    T * d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());
    OutputIterator output_it(d_output, size - 1);

    // temp storage
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;

    // Get temporary array size
    HIP_CHECK(
        hipcub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_size_bytes,
            input_begin, output_it,
            ::hipcub::Sum(), size,
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
        hipcub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_size_bytes,
            input_begin, output_it,
            ::hipcub::Sum(), size,
            stream, debug_synchronous
        )
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    T actual_output;
    HIP_CHECK(
        hipMemcpy(
            &actual_output, d_output,
            sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Validating results
    // Sum of 'size' increasing numbers starting at 'n' is size * (2n + size - 1) 
    // The division is not integer division but either (size) or (2n + size - 1) has to be even.
    const T multiplicand_1 = size;
    const T multiplicand_2 = 2 * (*input_begin) + size - 1;
    const T expected_output = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                        : multiplicand_1 * (multiplicand_2 / 2);
    ASSERT_EQ(expected_output, actual_output);

    hipFree(d_output);
    hipFree(d_temp_storage);
}

TEST(HipcubDeviceScanTests, LargeIndicesExclusiveScan)
{
    using T = unsigned int;
    using InputIterator = typename hipcub::CountingInputIterator<T>;
    using OutputIterator = single_index_iterator<T>;
    const bool debug_synchronous = false;

    const size_t size = (1ul << 31) + 1ul;

    hipStream_t stream = 0; // default

    unsigned int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    // Create CountingInputIterator<U> with random starting point
    InputIterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value));
    T initial_value = test_utils::get_random_value<T>(1, 10, seed_value);

    T * d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());
    OutputIterator output_it(d_output, size - 1);

    // temp storage
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;

    // Get temporary array size
    HIP_CHECK(
        hipcub::DeviceScan::ExclusiveScan(
            d_temp_storage, temp_storage_size_bytes,
            input_begin, output_it,
            ::hipcub::Sum(),
            initial_value, size,
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
        hipcub::DeviceScan::ExclusiveScan(
            d_temp_storage, temp_storage_size_bytes,
            input_begin, output_it,
            ::hipcub::Sum(),
            initial_value, size,
            stream, debug_synchronous
        )
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    T actual_output;
    HIP_CHECK(
        hipMemcpy(
            &actual_output, d_output,
            sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
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

    hipFree(d_output);
    hipFree(d_temp_storage);
}

#endif

template <typename T>
static __global__ void fill_initial_value(T* ptr, const T initial_value)
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
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            const unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const hipStream_t stream = 0; // default

            // Generate data
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);
            std::vector<U> output(input.size());

            T* d_input;
            U* d_output;
            U* d_initial_value;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_initial_value, sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // scan function
            const scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            const U initial_value = test_utils::get_random_value<T>(1, 100, seed_value + seed_value_addition);
            test_utils::host_exclusive_scan(
                input.begin(), input.end(),
                initial_value, expected.begin(),
                scan_op
            );

            const auto future_initial_value = hipcub::FutureValue<U>{d_initial_value};

            // Check the provided aliases to be correct at compile-time 
            static_assert(
                std::is_same<
                    typename decltype(future_initial_value)::value_type,
                    U>::value,
                "The futures value type is expected to be U");

            static_assert(
                std::is_same<
                    typename decltype(future_initial_value)::iterator_type,
                    U*>::value,
                "The futures iterator type is expected to be U*");

            // temp storage
            size_t temp_storage_size_bytes;
            void* d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, future_initial_value, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Fill initial value
            hipLaunchKernelGGL(
                fill_initial_value, dim3(1), dim3(1), 0, stream, d_initial_value, initial_value);
            HIP_CHECK(hipGetLastError());

            // Run
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, future_initial_value, input.size(),
                    stream, debug_synchronous
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
            for(size_t i = 0; i < output.size(); i++)
            {
                auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
                if(std::is_integral<U>::value) diff = 0;
                ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
            }

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_initial_value);
            hipFree(d_temp_storage);
        }
    }
}
