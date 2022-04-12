// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipcub/device/device_adjacent_difference.hpp"

template <class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::true_type /*left*/,
                                        std::true_type /*copy*/,
                                        void* d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT d_output,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes, d_input, d_output,
        std::forward<Args>(args)...
    );
}

template <class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::true_type /*left*/,
                                        std::false_type /*copy*/,
                                        void* d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT /*d_output*/,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeft(
        d_temp_storage, temp_storage_bytes, d_input,
        std::forward<Args>(args)...
    );
}

template <class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::false_type /*left*/,
                                        std::true_type /*copy*/,
                                        void* d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT d_output,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRightCopy(
        d_temp_storage, temp_storage_bytes, d_input, d_output,
        std::forward<Args>(args)...
    );
}

template <class InputIteratorT, class OutputIteratorT, class... Args>
hipError_t dispatch_adjacent_difference(std::false_type /*left*/,
                                        std::false_type /*copy*/,
                                        void* d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        InputIteratorT d_input,
                                        OutputIteratorT /*d_output*/,
                                        Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRight(
        d_temp_storage, temp_storage_bytes, d_input,
        std::forward<Args>(args)...
    );
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction op,
                         std::true_type /*left*/)
{
    std::vector<Output> result(input.size());
    std::adjacent_difference(input.cbegin(), input.cend(), result.begin(), op);
    return result;
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction op,
                         std::false_type /*left*/)
{
    std::vector<Output> result(input.size());
    // "right" adjacent difference is just adjacent difference backwards
    std::adjacent_difference(input.crbegin(), input.crend(), result.rbegin(), op);
    return result;
}

template<
    class InputT,
    class OutputT = InputT,
    bool Left = true,
    bool Copy = true
>
struct params
{
    using input_type = InputT;
    using output_type = OutputT;
    static constexpr bool left = Left;
    static constexpr bool copy = Copy;
};

template<class Params>
class HipcubDeviceAdjacentDifference : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<int>,
    params<int, double>,
    params<int8_t, int8_t, true, false>,
    params<float, float, false, true>,
    params<double, double, true, true>
> Params;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = { 1, 10, 53, 211, 1024, 2345, 4096, 34567, (1 << 16) - 1220, (1 << 23) - 76543 };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(10, 1, 100000, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(HipcubDeviceAdjacentDifference, Params);

TYPED_TEST(HipcubDeviceAdjacentDifference, SubtractLeftCopy)
{
    using input_type = typename TestFixture::params::input_type;
    static constexpr std::integral_constant<bool, TestFixture::params::left> left_constant{};
    static constexpr std::integral_constant<bool, TestFixture::params::copy> copy_constant{};
    using output_type = std::conditional_t<copy_constant, input_type, typename TestFixture::params::output_type>;
    static constexpr hipStream_t stream = 0;
    static constexpr bool debug_synchronous = false;
    static constexpr ::hipcub::Difference op;

    const auto sizes = get_sizes();
    for (size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            const unsigned int seed_value = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            const auto input = test_utils::get_random_data<input_type>(
                size,
                static_cast<input_type>(-50),
                static_cast<input_type>(50),
                seed_value
            );

            input_type * d_input{};
            output_type * d_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(d_input[0])));
            if (copy_constant)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(d_output[0])));
            }
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    size * sizeof(input_type),
                    hipMemcpyHostToDevice
                )
            );

            const auto expected = get_expected_result<output_type>(input, op, left_constant);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(
                dispatch_adjacent_difference(
                    left_constant, copy_constant,
                    nullptr, temporary_storage_bytes,
                    d_input, d_output, size, op, stream, debug_synchronous
                )
            );

#ifdef __HIP_PLATFORM_AMD__
            ASSERT_GT(temporary_storage_bytes, 0U);
#endif

            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            HIP_CHECK(
                dispatch_adjacent_difference(
                    left_constant, copy_constant,
                    d_temporary_storage, temporary_storage_bytes,
                    d_input, d_output, size, op, stream, debug_synchronous
                )
            );

            std::vector<output_type> output(size);
            HIP_CHECK(
                hipMemcpy(
                    output.data(), copy_constant ? d_output : d_input,
                    size * sizeof(output[0]),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_input));
            if (copy_constant)
            {
                HIP_CHECK(hipFree(d_output));
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(output, expected));
        }
    }
}
