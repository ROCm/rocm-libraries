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
#include <rocprim/device/device_partition.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class FlagType = unsigned int,
    bool UseIdentityIterator = false
>
struct DevicePartitionParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDevicePartitionTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DevicePartitionParams<int, int, unsigned char, true>,
    DevicePartitionParams<unsigned int, unsigned long>,
    DevicePartitionParams<unsigned char, float>,
    DevicePartitionParams<int8_t, int8_t>,
    DevicePartitionParams<uint8_t, uint8_t>,
    DevicePartitionParams<rocprim::half, rocprim::half>,
    DevicePartitionParams<rocprim::bfloat16, rocprim::bfloat16>,
    DevicePartitionParams<test_utils::custom_test_type<long long>>
> RocprimDevicePartitionTestsParams;

TYPED_TEST_SUITE(RocprimDevicePartitionTests, RocprimDevicePartitionTestsParams);

TYPED_TEST(RocprimDevicePartitionTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<F> flags = test_utils::get_random_data01<F>(size, 0.25, seed_value);

            T * d_input;
            F * d_flags;
            U * d_output;
            unsigned int * d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(F)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_flags, flags.data(),
                    flags.size() * sizeof(F),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected_selected and expected_rejected results on host
            std::vector<U> expected_selected;
            std::vector<U> expected_rejected;
            expected_selected.reserve(input.size()/2);
            expected_rejected.reserve(input.size()/2);
            for(size_t i = 0; i < input.size(); i++)
            {
                if(flags[i] != 0)
                {
                    expected_selected.push_back(input[i]);
                }
                else
                {
                    expected_rejected.push_back(input[i]);
                }
            }
            std::reverse(expected_rejected.begin(), expected_rejected.end());

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::partition(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::partition(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected_selected
            unsigned int selected_count_output = 0;
            HIP_CHECK(
                hipMemcpy(
                    &selected_count_output, d_selected_count_output,
                    sizeof(unsigned int),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected_selected.size());

            // Check if output values are as expected_selected
            std::vector<U> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<U> output_rejected;
            for(size_t i = 0; i < expected_rejected.size(); i++)
            {
                auto j = i + expected_selected.size();
                output_rejected.push_back(output[j]);
            }
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected_selected, expected_selected.size()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_rejected, expected_rejected, expected_rejected.size()));

            hipFree(d_input);
            hipFree(d_flags);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }

}

TYPED_TEST(RocprimDevicePartitionTests, PredicateEmptyInput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    auto select_op = [] __host__ __device__ (const T& value) -> bool
    {
        if(value == T(50)) return true;
        return false;
    };

    U * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(U)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
    unsigned int selected_count_output = 123;
    HIP_CHECK(
        hipMemcpy(
            d_selected_count_output, &selected_count_output,
            sizeof(unsigned int),
            hipMemcpyHostToDevice
        )
    );

    test_utils::out_of_bounds_flag out_of_bounds;
    test_utils::bounds_checking_iterator<U> d_checking_output(
        d_output,
        out_of_bounds.device_pointer(),
        0
    );

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::partition(
            nullptr,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_checking_output,
            d_selected_count_output,
            0,
            select_op,
            stream,
            debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::partition(
            d_temp_storage,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_checking_output,
            d_selected_count_output,
            0,
            select_op,
            stream,
            debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_FALSE(out_of_bounds.get());

    // Check if number of selected value is 0
    HIP_CHECK(
        hipMemcpy(
            &selected_count_output, d_selected_count_output,
            sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    ASSERT_EQ(selected_count_output, 0);

    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

TYPED_TEST(RocprimDevicePartitionTests, Predicate)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    auto select_op = [] __host__ __device__ (const T& value) -> bool
    {
        if(value == T(50)) return true;
        return false;
    };

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

            T * d_input;
            U * d_output;
            unsigned int * d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected_selected and expected_rejected results on host
            std::vector<U> expected_selected;
            std::vector<U> expected_rejected;
            expected_selected.reserve(input.size()/2);
            expected_rejected.reserve(input.size()/2);
            for(size_t i = 0; i < input.size(); i++)
            {
                if(select_op(input[i]))
                {
                    expected_selected.push_back((U)input[i]);
                }
                else
                {
                    expected_rejected.push_back((U)input[i]);
                }
            }
            std::reverse(expected_rejected.begin(), expected_rejected.end());

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::partition(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    select_op,
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::partition(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    select_op,
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected_selected
            unsigned int selected_count_output = 0;
            HIP_CHECK(
                hipMemcpy(
                    &selected_count_output, d_selected_count_output,
                    sizeof(unsigned int),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected_selected.size());

            // Check if output values are as expected_selected
            std::vector<U> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<U> output_rejected;
            for(size_t i = 0; i < expected_rejected.size(); i++)
            {
                auto j = i + expected_selected.size();
                output_rejected.push_back(output[j]);
            }
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected_selected, expected_selected.size()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_rejected, expected_rejected, expected_rejected.size()));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }
}

namespace {
template <typename T>
struct LessOp {
    ROCPRIM_HOST_DEVICE LessOp(const T& pivot)
        : pivot_{pivot}
    {
    }

    ROCPRIM_HOST_DEVICE bool operator()(const T& val) const {
        return val < pivot_; 
    }
private:
    T pivot_;
};
}

TYPED_TEST(RocprimDevicePartitionTests, PredicateThreeWay)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const hipStream_t stream = 0; // default stream
    const std::vector<std::array<T,2>> limit_pairs{
        { static_cast<T>(30), static_cast<T>(60) }, // all sections may contain items
        { static_cast<T>(0), static_cast<T>(60) },  // first section is empty
        { static_cast<T>(30), static_cast<T>(30) }, // second section is empty
        { static_cast<T>(30), static_cast<T>(101) } // unselected is empty
    };

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value = seed_index < random_seeds_count
            ? static_cast<unsigned int>(rand()) : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(const auto& limits : limit_pairs)
            {
                SCOPED_TRACE(testing::Message() << "with limits = "
                    << std::get<0>(limits) << ", " << std::get<1>(limits));
                // Generate data
                const auto input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

                // Output
                auto selected_counts = std::array<unsigned int, 2>{};

                T* d_input                      = nullptr;
                U* d_first_output               = nullptr;
                U* d_second_output              = nullptr;
                U* d_unselected_output          = nullptr;
                unsigned int* d_selected_counts = nullptr;

                HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
                HIP_CHECK(hipMalloc(&d_first_output, input.size() * sizeof(U)));
                HIP_CHECK(hipMalloc(&d_second_output, input.size() * sizeof(U)));
                HIP_CHECK(hipMalloc(&d_unselected_output, input.size() * sizeof(U)));
                HIP_CHECK(hipMalloc(&d_selected_counts, sizeof(selected_counts)));
                HIP_CHECK(
                    hipMemcpy(
                        d_input, input.data(),
                        input.size() * sizeof(T),
                        hipMemcpyHostToDevice
                    )
                );


                const auto first_op = LessOp<T>{std::get<0>(limits)};
                const auto second_op = LessOp<T>{std::get<1>(limits)};

                auto copy = input;
                const auto partion_point =
                    std::stable_partition(copy.begin(), copy.end(), first_op);
                const auto second_partiton_point =
                    std::stable_partition(partion_point, copy.end(), second_op);

                const auto expected_counts = std::array<unsigned int, 2>{
                    static_cast<unsigned int>(partion_point - copy.begin()),
                    static_cast<unsigned int>(second_partiton_point - partion_point)
                };

                const auto expected = [&]{
                    auto result = std::vector<U>(copy.size());
                    std::copy(copy.cbegin(), copy.cend(), result.begin());
                    return result;
                }();

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(
                    rocprim::partition_three_way(
                        nullptr,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
                        d_selected_counts,
                        input.size(),
                        first_op,
                        second_op,
                        stream,
                        debug_synchronous
                    )
                );

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                void* d_temp_storage = nullptr;
                HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

                // Run
                HIP_CHECK(
                    rocprim::partition_three_way(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
                        d_selected_counts,
                        input.size(),
                        first_op,
                        second_op,
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected_selected
                HIP_CHECK(
                    hipMemcpy(
                        selected_counts.data(), d_selected_counts,
                        sizeof(selected_counts),
                        hipMemcpyDeviceToHost
                    )
                );
                ASSERT_EQ(selected_counts, expected_counts);

                // Check if output values are as expected_selected
                const auto output = [&]{
                    auto result = std::vector<U>(input.size());
                    HIP_CHECK(
                        hipMemcpy(
                            result.data(), d_first_output,
                            expected_counts[0] * sizeof(result[0]),
                            hipMemcpyDeviceToHost
                        )
                    );
                    HIP_CHECK(
                        hipMemcpy(
                            result.data() + expected_counts[0], d_second_output,
                            expected_counts[1] * sizeof(result[0]),
                            hipMemcpyDeviceToHost
                        )
                    );
                    HIP_CHECK(
                        hipMemcpy(
                            result.data() + expected_counts[0] + expected_counts[1],
                            d_unselected_output,
                            (input.size() - expected_counts[0] - expected_counts[1]) * sizeof(result[0]),
                            hipMemcpyDeviceToHost
                        )
                    );
                    return result;
                }();

                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                hipFree(d_input);
                hipFree(d_first_output);
                hipFree(d_second_output);
                hipFree(d_unselected_output);
                hipFree(d_selected_counts);
                hipFree(d_temp_storage);
            }
        }
    }
}

namespace
{

/// \brief An output iterator which checks the values written to it.
/// The expected output values should be partitioned with regards to the \p modulo parameter.
/// The check algorithm depends on \p CheckValue.
template<class CheckValue>
class check_modulo_iterator
{
public:
    using value_type        = CheckValue;
    using reference         = CheckValue;
    using pointer           = CheckValue*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    ROCPRIM_HOST_DEVICE
    check_modulo_iterator(const size_t        modulo,
                          const size_t        size,
                          unsigned int* const incorrect_flag)
        : current_index_(0), modulo_(modulo), size_(size), incorrect_flag_(incorrect_flag)
    {}

    ROCPRIM_HOST_DEVICE bool operator==(const check_modulo_iterator& rhs)
    {
        return current_index_ == rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE bool operator!=(const check_modulo_iterator& rhs)
    {
        return !(*this == rhs);
    }
    ROCPRIM_HOST_DEVICE reference operator*()
    {
        return value_type(current_index_, modulo_, size_, incorrect_flag_);
    }
    ROCPRIM_HOST_DEVICE reference operator[](const difference_type distance)
    {
        return *(*this + distance);
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator& operator+=(const difference_type rhs)
    {
        current_index_ += rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator& operator-=(const difference_type rhs)
    {
        current_index_ -= rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE difference_type operator-(const check_modulo_iterator& rhs) const
    {
        return current_index_ - rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator operator+(const difference_type rhs) const
    {
        return check_modulo_iterator(*this) += rhs;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator operator-(const difference_type rhs) const
    {
        return check_modulo_iterator(*this) -= rhs;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator& operator++()
    {
        ++current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator& operator--()
    {
        --current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator operator++(int)
    {
        return ++check_modulo_iterator{*this};
    }
    ROCPRIM_HOST_DEVICE check_modulo_iterator operator--(int)
    {
        return --check_modulo_iterator{*this};
    }

private:
    size_t        current_index_;
    size_t        modulo_;
    size_t        size_;
    unsigned int* incorrect_flag_;
};

/// \brief Checks if the selected values (multiples of \p modulo ) are written to the beginning of the
/// range and unselected values are written to the end of range in reverse order.
/// For example, in case of \p modulo 3, and a \p size of 10, the following output is expected:
/// [0, 3, 6, 9, 8, 7, 5, 4, 2, 1]
/// If an unexpected value is set, \p *incorrect_flag is atomically set to 1.
class check_two_way_modulo
{
public:
    ROCPRIM_HOST_DEVICE
    check_two_way_modulo(const size_t        current_index,
                         const size_t        modulo,
                         const size_t        size,
                         unsigned int* const incorrect_flag)
        : current_index_(current_index)
        , modulo_(modulo)
        , size_(size)
        , incorrect_flag_(incorrect_flag)
    {}

    ROCPRIM_DEVICE
    check_two_way_modulo& operator=(const size_t value)
    {
        const bool   is_mod         = (value % modulo_) == 0;
        const size_t expected_index = is_mod ? value / modulo_ : size_ - value + value / modulo_;
        if(current_index_ != expected_index)
        {
            rocprim::detail::atomic_exch(incorrect_flag_, 1);
        }
        return *this;
    }

private:
    size_t        current_index_;
    size_t        modulo_;
    size_t        size_;
    unsigned int* incorrect_flag_;
};

/// \brief Checks if multiples of \p modulo are written to the iterator.
/// If an unexpected value is set, \p *incorrect_flag is atomically set to 1.
class check_modulo
{
public:
    ROCPRIM_HOST_DEVICE
    check_modulo(size_t current_index, size_t modulo, size_t /*size*/, unsigned int* incorrect_flag)
        : current_index_(current_index), modulo_(modulo), incorrect_flag_(incorrect_flag)
    {}

    ROCPRIM_DEVICE
    check_modulo& operator=(size_t value)
    {
        const bool   is_mod         = (value % modulo_) == 0;
        const size_t expected_index = value / modulo_;
        if(!is_mod || current_index_ != expected_index)
        {
            rocprim::detail::atomic_exch(incorrect_flag_, 1);
        }
        return *this;
    }

private:
    size_t        current_index_;
    size_t        modulo_;
    unsigned int* incorrect_flag_;
};

/// \brief Checks if multiples of \p modulo are not written to the iterator,
/// but multiples of \p modulo-1 are written.
/// If an unexpected value is set, \p *incorrect_flag is atomically set to 1.
class check_modulo_exclude
{
public:
    ROCPRIM_HOST_DEVICE
    check_modulo_exclude(const size_t current_index,
                         const size_t modulo,
                         const size_t /*size*/,
                         unsigned int* const incorrect_flag)
        : current_index_(current_index)
        , modulo_(modulo)
        , modulo_exclude_(modulo_ - 1)
        , incorrect_flag_(incorrect_flag)
    {}

    ROCPRIM_DEVICE
    check_modulo_exclude& operator=(const size_t value)
    {
        const bool   is_mod         = value % modulo_ == 0 && value % modulo_exclude_ != 0;
        const size_t expected_index = value / modulo_ - value / (modulo_exclude_ * modulo_) - 1;
        if(!is_mod || current_index_ != expected_index)
        {
            rocprim::detail::atomic_exch(incorrect_flag_, 1);
        }
        return *this;
    }

private:
    size_t        current_index_;
    size_t        modulo_;
    size_t        modulo_exclude_;
    unsigned int* incorrect_flag_;
};

struct modulo_predicate
{
    size_t modulo_;

    ROCPRIM_DEVICE
    bool operator()(const size_t value) const
    {
        return value % modulo_ == 0;
    }
};

} // namespace

struct RocprimDevicePartitionLargeInputTests : public ::testing::TestWithParam<size_t>
{};

INSTANTIATE_TEST_SUITE_P(RocprimDevicePartitionLargeInputTest,
                         RocprimDevicePartitionLargeInputTests,
                         ::testing::Values(2, 2048, 38713));

TEST_P(RocprimDevicePartitionLargeInputTests, LargeInputPartition)
{
    static constexpr bool        debug_synchronous = false;
    static constexpr hipStream_t stream            = 0;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const auto modulo = GetParam();

    for(const auto size : test_utils::get_large_sizes(std::random_device{}()))
    {
        // limit the running time of the test
        if(size > (size_t{1} << 35))
        {
            break;
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);
        const auto input_iterator = rocprim::make_counting_iterator(static_cast<size_t>(0));
        const modulo_predicate predicate{modulo};

        unsigned int* d_incorrect_flag{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_incorrect_flag, sizeof(*d_incorrect_flag)));
        HIP_CHECK(hipMemsetAsync(d_incorrect_flag, 0, sizeof(*d_incorrect_flag), stream));
        const auto output_checker_it
            = check_modulo_iterator<check_two_way_modulo>(modulo, size, d_incorrect_flag);

        size_t* d_count_output{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_count_output, sizeof(*d_count_output)));

        void*  d_temporary_storage{};
        size_t temporary_storage_size{};
        HIP_CHECK(rocprim::partition(d_temporary_storage,
                                     temporary_storage_size,
                                     input_iterator,
                                     output_checker_it,
                                     d_count_output,
                                     size,
                                     predicate,
                                     stream,
                                     debug_synchronous));

        ASSERT_NE(0, temporary_storage_size);
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_size));

        HIP_CHECK(rocprim::partition(d_temporary_storage,
                                     temporary_storage_size,
                                     input_iterator,
                                     output_checker_it,
                                     d_count_output,
                                     size,
                                     predicate,
                                     stream,
                                     debug_synchronous));

        size_t count_output{};
        HIP_CHECK(hipMemcpyWithStream(&count_output,
                                      d_count_output,
                                      sizeof(count_output),
                                      hipMemcpyDeviceToHost,
                                      stream));

        const size_t expected_output = rocprim::detail::ceiling_div(size, modulo);
        ASSERT_EQ(count_output, expected_output);

        unsigned int incorrect_flag{};
        HIP_CHECK(hipMemcpyWithStream(&incorrect_flag,
                                      d_incorrect_flag,
                                      sizeof(incorrect_flag),
                                      hipMemcpyDeviceToHost,
                                      stream));

        ASSERT_EQ(incorrect_flag, 0);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_count_output));
        HIP_CHECK(hipFree(d_incorrect_flag));
    }
}

TEST_P(RocprimDevicePartitionLargeInputTests, LargeInputPartitionThreeWay)
{
    static constexpr bool        debug_synchronous = false;
    static constexpr hipStream_t stream            = 0;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const auto modulo_a = GetParam();
    const auto modulo_b = modulo_a + 1;

    for(const auto size : test_utils::get_large_sizes(std::random_device{}()))
    {
        // limit the running time of the test
        if(size > (size_t{1} << 35))
        {
            break;
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);
        const auto input_iterator = rocprim::make_counting_iterator(static_cast<size_t>(0));
        const auto predicate_a    = modulo_predicate{modulo_a};
        const auto predicate_b    = modulo_predicate{modulo_b};

        unsigned int* d_incorrect_flag{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_incorrect_flag, sizeof(*d_incorrect_flag)));
        HIP_CHECK(hipMemsetAsync(d_incorrect_flag, 0, sizeof(*d_incorrect_flag), stream));
        const auto output_checker_a
            = check_modulo_iterator<check_modulo>(modulo_a, size, d_incorrect_flag);
        const auto output_checker_b
            = check_modulo_iterator<check_modulo_exclude>(modulo_b, size, d_incorrect_flag);
        const auto unselected_output = rocprim::make_discard_iterator();

        size_t* d_count_output{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_count_output, 2 * sizeof(*d_count_output)));

        void*  d_temporary_storage{};
        size_t temporary_storage_size{};
        HIP_CHECK(rocprim::partition_three_way(d_temporary_storage,
                                               temporary_storage_size,
                                               input_iterator,
                                               output_checker_a,
                                               output_checker_b,
                                               unselected_output,
                                               d_count_output,
                                               size,
                                               predicate_a,
                                               predicate_b,
                                               stream,
                                               debug_synchronous));

        ASSERT_NE(0, temporary_storage_size);
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_size));

        HIP_CHECK(rocprim::partition_three_way(d_temporary_storage,
                                               temporary_storage_size,
                                               input_iterator,
                                               output_checker_a,
                                               output_checker_b,
                                               unselected_output,
                                               d_count_output,
                                               size,
                                               predicate_a,
                                               predicate_b,
                                               stream,
                                               debug_synchronous));

        size_t count_output[2]{};
        HIP_CHECK(hipMemcpyWithStream(&count_output,
                                      d_count_output,
                                      sizeof(count_output),
                                      hipMemcpyDeviceToHost,
                                      stream));

        const size_t expected_output_a = rocprim::detail::ceiling_div(size, modulo_a);
        // Beware: this only works when modulo_a and modulo_b are coprimes (e.g. when the difference is 1)
        const size_t expected_output_b = rocprim::detail::ceiling_div(size, modulo_b)
                                         - rocprim::detail::ceiling_div(size, modulo_a * modulo_b);
        ASSERT_EQ(count_output[0], expected_output_a);
        ASSERT_EQ(count_output[1], expected_output_b);

        unsigned int incorrect_flag{};
        HIP_CHECK(hipMemcpyWithStream(&incorrect_flag,
                                      d_incorrect_flag,
                                      sizeof(incorrect_flag),
                                      hipMemcpyDeviceToHost,
                                      stream));

        ASSERT_EQ(incorrect_flag, 0);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_count_output));
        HIP_CHECK(hipFree(d_incorrect_flag));
    }
}
