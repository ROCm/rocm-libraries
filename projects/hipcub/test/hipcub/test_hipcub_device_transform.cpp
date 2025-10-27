// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipcub/device/device_transform.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

template<typename InputType, typename OutputType = InputType>
struct DoubleIt
{
    __device__ __host__
    constexpr OutputType
        operator()(const InputType& value) const
    {
        return value * 2;
    }
};

template<typename InputType>
struct PointerDiff
{
    InputType* base_pointer;

    __device__ __host__
    constexpr size_t
        operator()(const InputType& value) const
    {
        return &value - base_pointer;
    }
};

template<typename T>
struct NonCopyable
{
    T data;

    __device__ __host__ NonCopyable() noexcept : data(T{}) {}

    __device__ __host__ NonCopyable(const T& data) noexcept : data(data) {}

    __device__ __host__ NonCopyable(NonCopyable&& other) noexcept : data(std::move(other.data)) {}

    __device__ __host__
    NonCopyable&
        operator=(NonCopyable&& other) noexcept
    {
        data = std::move(other.data);
        return *this;
    }

    __device__ __host__ NonCopyable(const NonCopyable&) = delete;

    __device__ __host__
    NonCopyable&
        operator=(const NonCopyable&)
        = delete;

    __device__ __host__
    constexpr
        operator T() const noexcept
    {
        return data;
    }
};

static_assert(std::is_same_v<std::invoke_result_t<DoubleIt<int32_t>, int32_t>, int32_t>);

// Params for tests
template<class InputType, class UnaryOp = DoubleIt<InputType>, bool UseGraphs = false>
struct HipcubDeviceTransformParams
{
    using input_type                 = InputType;
    using output_type                = std::invoke_result_t<UnaryOp, input_type>;
    using unary_op                   = UnaryOp;
};

template<class Params>
struct HipcubDeviceTransformTests : public ::testing::Test
{
    using input_type                        = typename Params::input_type;
    using output_type                       = typename Params::output_type;
    using unary_op                          = typename Params::unary_op;
};

using HipcubDeviceTransformTestsParams
    = ::testing::Types<HipcubDeviceTransformParams<int32_t>, HipcubDeviceTransformParams<float>>;

TYPED_TEST_SUITE(HipcubDeviceTransformTests, HipcubDeviceTransformTestsParams);

TYPED_TEST(HipcubDeviceTransformTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using unary_op    = typename TestFixture::unary_op;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = hipStreamDefault;
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> h_input
                = test_utils::get_random_data<input_type>(size, 1, 100, seed_value);
            std::vector<output_type> h_output(size);

            // Device pointers
            input_type*  d_input;
            output_type* d_output;

            // Allocate memory
            size_t num_input_bytes  = size * sizeof(input_type);
            size_t num_output_bytes = size * sizeof(output_type);
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, num_input_bytes));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, num_output_bytes));

            // Prepare input
            HIP_CHECK(hipMemcpy(d_input, h_input.data(), num_input_bytes, hipMemcpyHostToDevice));

            // Do the thing
            HIP_CHECK(
                hipcub::DeviceTransform::Transform(d_input, d_output, size, unary_op{}, stream));

            // Fetch output
            HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_output_bytes, hipMemcpyDeviceToHost))

            // Check output
            for(size_t i = 0; i < size; ++i)
            {
                ASSERT_EQ(unary_op{}(h_input[i]), h_output[i]);
            }
        }
    }
}

TYPED_TEST(HipcubDeviceTransformTests, TransformAddrStableNonCopyable)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using unary_op    = DoubleIt<NonCopyable<input_type>>;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = hipStreamDefault;
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> h_input
                = test_utils::get_random_data<input_type>(size, 1, 100, seed_value);
            std::vector<output_type> h_output(size);

            // Device pointers
            NonCopyable<input_type>*  d_input;
            NonCopyable<output_type>* d_output;

            // Allocate memory
            size_t num_input_bytes  = size * sizeof(input_type);
            size_t num_output_bytes = size * sizeof(output_type);
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, num_input_bytes));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, num_output_bytes));

            // Prepare input
            HIP_CHECK(hipMemcpy(d_input, h_input.data(), num_input_bytes, hipMemcpyHostToDevice));

            // Do the thing
            HIP_CHECK(hipcub::DeviceTransform::TransformStableArgumentAddresses(d_input,
                                                                                d_output,
                                                                                size,
                                                                                unary_op{},
                                                                                stream));

            // Fetch output
            HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_output_bytes, hipMemcpyDeviceToHost))

            // Check output
            for(size_t i = 0; i < size; ++i)
            {
                ASSERT_EQ(unary_op{}(h_input[i]), h_output[i]);
            }
        }
    }
}

TYPED_TEST(HipcubDeviceTransformTests, TransformAddrStablePointerDiff)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using unary_op    = PointerDiff<NonCopyable<input_type>>;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = hipStreamDefault;
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type>  h_input(size);
            std::vector<output_type> h_output(size);

            // Device pointers
            NonCopyable<input_type>*  d_input;
            NonCopyable<output_type>* d_output;

            // Allocate memory
            size_t num_input_bytes  = size * sizeof(input_type);
            size_t num_output_bytes = size * sizeof(output_type);
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, num_input_bytes));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, num_output_bytes));

            // Prepare input
            HIP_CHECK(hipMemcpy(d_input, h_input.data(), num_input_bytes, hipMemcpyHostToDevice));

            // Do the thing
            HIP_CHECK(hipcub::DeviceTransform::TransformStableArgumentAddresses(d_input,
                                                                                d_output,
                                                                                size,
                                                                                unary_op{d_input},
                                                                                stream));

            // Fetch output
            HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_output_bytes, hipMemcpyDeviceToHost))

            // Check output
            for(size_t i = 0; i < size; ++i)
            {
                ASSERT_EQ(i, h_output[i]);
            }
        }
    }
}
