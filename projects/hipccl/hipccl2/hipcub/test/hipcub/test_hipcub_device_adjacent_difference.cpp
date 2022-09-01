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
#include "hipcub/iterator/counting_input_iterator.hpp"
#include "hipcub/iterator/transform_input_iterator.hpp"

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

std::vector<size_t> get_large_sizes(int seed_value)
{
    // clang-format off
    std::vector<size_t> sizes = {
        (size_t{1} << 32) - 1, size_t{1} << 32,
        (size_t{1} << 35) - 1, size_t{1} << 35
    };
    // clang-format on
    const std::vector<size_t> random_sizes
        = test_utils::get_random_data<size_t>(2,
                                              (size_t{1} << 30) + 1,
                                              (size_t{1} << 35) - 2,
                                              seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(HipcubDeviceAdjacentDifference, Params);

TYPED_TEST(HipcubDeviceAdjacentDifference, SubtractLeftCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

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

// Params for tests
template<bool Left = true, bool Copy = false>
struct DeviceAdjacentDifferenceLargeParams
{
    static constexpr bool left = Left;
    static constexpr bool copy = Copy;
};

template<class Params>
class HipcubDeviceAdjacentDifferenceLargeTests : public ::testing::Test
{
public:
    static constexpr bool left              = Params::left;
    static constexpr bool copy              = Params::copy;
    static constexpr bool debug_synchronous = false;
};

using HipcubDeviceAdjacentDifferenceLargeTestsParams
    = ::testing::Types<DeviceAdjacentDifferenceLargeParams<true, false>,
                       DeviceAdjacentDifferenceLargeParams<true, true>,
                       DeviceAdjacentDifferenceLargeParams<false, false>,
                       DeviceAdjacentDifferenceLargeParams<false, true>>;

TYPED_TEST_SUITE(HipcubDeviceAdjacentDifferenceLargeTests,
                 HipcubDeviceAdjacentDifferenceLargeTestsParams);

template<class T>
struct discard_write
{
    T value;

    __device__ operator T() const
    {
        return value;
    }
    __device__ discard_write& operator=(T)
    {
        return *this;
    }
};

class discard_iterator
{
public:
    struct discard_value
    {
        inline discard_value() = default;

        template<class T>
        HIPCUB_HOST_DEVICE inline discard_value(T){};

        inline ~discard_value() = default;

        template<class T>
        HIPCUB_HOST_DEVICE inline discard_value& operator=(const T&)
        {
            return *this;
        }
    };

    typedef discard_iterator self_type; ///< My own type
    typedef discard_value    value_type; ///< The type of the element the iterator can point to
    typedef discard_value*
        pointer; ///< The type of a pointer to an element the iterator can point to
    typedef discard_value
        reference; ///< The type of a reference to an element the iterator can point to
    typedef ptrdiff_t                       difference_type;
    typedef std::random_access_iterator_tag iterator_category; ///< The iterator category
    /// \brief Creates a new discard_iterator.
    ///
    /// \param index - optional index of discard iterator (default = 0).
    HIPCUB_HOST_DEVICE inline discard_iterator(size_t index = 0) : index_(index) {}

    inline ~discard_iterator() = default;

    HIPCUB_HOST_DEVICE inline discard_value operator*() const
    {
        return discard_value();
    }

    HIPCUB_HOST_DEVICE inline discard_value operator[](difference_type distance) const
    {
        discard_iterator i = (*this) + distance;
        return *i;
    }

    HIPCUB_HOST_DEVICE inline discard_iterator operator+(difference_type distance) const
    {
        auto i = static_cast<size_t>(static_cast<difference_type>(index_) + distance);
        return discard_iterator(i);
    }

private:
    mutable size_t index_;
};

template<class T, class InputIterator, class UnaryFunction>
HIPCUB_HOST_DEVICE inline auto make_transform_iterator(InputIterator iterator,
                                                       UnaryFunction transform)
{
    return ::hipcub::TransformInputIterator<T, UnaryFunction, InputIterator>(iterator, transform);
}

template<class T>
struct conversion_op : public std::unary_function<T, discard_write<T>>
{
    HIPCUB_HOST_DEVICE auto operator()(const T i) const
    {
        return discard_write<T>{i};
    }
};

template<class T>
struct flag_expected_op : public std::binary_function<T, T, discard_write<T>>
{
    bool left;
    T    expected;
    T    expected_above_limit;
    int* d_flags;
    flag_expected_op(bool left, T expected, T expected_above_limit, int* d_flags)
        : left(left)
        , expected(expected)
        , expected_above_limit(expected_above_limit)
        , d_flags(d_flags)
    {}

    HIPCUB_HOST_DEVICE T operator()(const discard_write<T>& minuend,
                                    const discard_write<T>& subtrahend)
    {
        if(left)
        {
            if(minuend == expected && subtrahend == expected - 1)
            {
                d_flags[0] = 1;
            }
            if(minuend == expected_above_limit && subtrahend == expected_above_limit - 1)
            {
                d_flags[1] = 1;
            }
        }
        else
        {
            if(minuend == expected && subtrahend == expected + 1)
            {
                d_flags[0] = 1;
            }
            if(minuend == expected_above_limit && subtrahend == expected_above_limit + 1)
            {
                d_flags[1] = 1;
            }
        }
        return 0;
    }
};

TYPED_TEST(HipcubDeviceAdjacentDifferenceLargeTests, LargeIndicesAndOpOnce)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                 = size_t;
    using OutputIterator                    = discard_iterator;
    static constexpr bool left              = TestFixture::left;
    static constexpr bool copy              = TestFixture::copy;
    const bool            debug_synchronous = TestFixture::debug_synchronous;

    SCOPED_TRACE(testing::Message() << "left = " << left << ", copy = " << copy);

    static constexpr hipStream_t stream = 0; // default

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_large_sizes(seed_value);

        for(const auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const OutputIterator output;

            // A transform iterator that can be written to (because in-place adjacent diff writes
            // to the input).
            // The conversion to T is used by flag_expected to actually perform the test
            const auto input = make_transform_iterator<discard_write<T>>(
                ::hipcub::CountingInputIterator<T>(T{0}),
                conversion_op<T>{});

            int  flags[2] = {0, 0};
            int* d_flags  = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, sizeof(flags)));
            HIP_CHECK(hipMemcpy(d_flags, flags, sizeof(flags), hipMemcpyHostToDevice));

            const T expected            = test_utils::get_random_value<T>(1, size - 2, seed_value);
            static constexpr auto limit = std::numeric_limits<unsigned int>::max();

            const T expected_above_limit
                = size - 2 > limit ? test_utils::get_random_value<T>(limit, size - 2, seed_value)
                                   : size - 2;

            SCOPED_TRACE(testing::Message() << "expected = " << expected);
            SCOPED_TRACE(testing::Message() << "expected_above_limit = " << expected_above_limit);
            flag_expected_op<T> flag_expected(left, expected, expected_above_limit, d_flags);

            static constexpr auto left_tag = std::integral_constant<bool, left>{};

            static constexpr auto copy_tag = std::integral_constant<bool, copy>{};

            // Allocate temporary storage
            std::size_t temp_storage_size = 0;
            void*       d_temp_storage    = nullptr;
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   copy_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   flag_expected,
                                                   stream,
                                                   debug_synchronous));

#ifdef __HIP_PLATFORM_AMD__
            ASSERT_GT(temp_storage_size, 0U);
#endif
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            // Run
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   copy_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   flag_expected,
                                                   stream,
                                                   debug_synchronous));

            // Copy output to host
            HIP_CHECK(hipMemcpy(flags, d_flags, sizeof(flags), hipMemcpyDeviceToHost));

            ASSERT_EQ(flags[0], 1);
            ASSERT_EQ(flags[1], 1);
            hipFree(d_temp_storage);
            hipFree(d_flags);
        }
    }
}