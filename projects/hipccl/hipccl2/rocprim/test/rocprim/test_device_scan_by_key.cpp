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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

// Required test headers
#include "bounds_checking_iterator.hpp"
#include "identity_iterator.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// Required rocprim headers
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/detail/device_scan_common.hpp>
#include <rocprim/device/detail/lookback_scan_state.hpp>
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>
#include <rocprim/device/device_scan_config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/atomic.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/future_value.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

template<unsigned int SizeLimit>
using size_limit_config
    = rocprim::scan_by_key_config<256,
                                  16,
                                  rocprim::block_load_method::block_load_transpose,
                                  rocprim::block_store_method::block_store_transpose,
                                  rocprim::block_scan_algorithm::using_warp_scan,
                                  SizeLimit>;

// Params for tests
template<class InputType,
         class OutputType = InputType,
         class ScanOp     = ::rocprim::plus<InputType>,
         // Tests output iterator with void value_type (OutputIterator concept)
         // scan-by-key primitives don't support output iterator with void value_type
         bool UseIdentityIteratorIfSupported = false,
         typename ConfigHelper               = rocprim::default_config,
         bool UseGraphs                      = false,
         bool Deterministic                  = false,
         bool UseInitialValue                = false>
struct DeviceScanByKeyParams
{
    using input_type                            = InputType;
    using output_type                           = OutputType;
    using scan_op_type                          = ScanOp;
    static constexpr bool use_identity_iterator = UseIdentityIteratorIfSupported;
    using config_helper                         = ConfigHelper;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool deterministic         = Deterministic;
    static constexpr bool use_initial_value     = UseInitialValue;
};

template<bool Deterministic, typename Config = rocprim::default_config, typename... Args>
constexpr hipError_t invoke_inclusive_scan_by_key(Args&&... args)
{
    if(Deterministic)
    {
        return rocprim::deterministic_inclusive_scan_by_key<Config>(std::forward<Args>(args)...);
    }
    else
    {
        return rocprim::inclusive_scan_by_key<Config>(std::forward<Args>(args)...);
    }
}

template<bool Deterministic, typename Config = rocprim::default_config, typename... Args>
constexpr hipError_t invoke_exclusive_scan_by_key(Args&&... args)
{
    if(Deterministic)
    {
        return rocprim::deterministic_exclusive_scan_by_key<Config>(std::forward<Args>(args)...);
    }
    else
    {
        return rocprim::exclusive_scan_by_key<Config>(std::forward<Args>(args)...);
    }
}

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceScanTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using scan_op_type                          = typename Params::scan_op_type;
    const bool            debug_synchronous     = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    using config_helper                         = typename Params::config_helper;
    bool                  use_graphs            = Params::use_graphs;
    static constexpr bool deterministic         = Params::deterministic;
    static constexpr bool use_initial_value     = Params::use_initial_value;
};

using RocprimDeviceScanTestsParams = ::testing::Types<
    // Small
    DeviceScanByKeyParams<char>,
    DeviceScanByKeyParams<unsigned short>,
    DeviceScanByKeyParams<short, int>,
    DeviceScanByKeyParams<int>,
    DeviceScanByKeyParams<int, int, rocprim::plus<int>, false, size_limit_config<512>>,
    DeviceScanByKeyParams<float, float, rocprim::maximum<float>>,
    DeviceScanByKeyParams<float, float, rocprim::plus<float>, false, size_limit_config<1024>>,
    DeviceScanByKeyParams<float,
                          float,
                          rocprim::plus<float>,
                          false,
                          size_limit_config<1024>,
                          false,
                          true>,
    DeviceScanByKeyParams<int, int, rocprim::plus<int>, false, size_limit_config<524288>>,
    DeviceScanByKeyParams<int, int, rocprim::plus<int>, false, size_limit_config<1048576>>,
    DeviceScanByKeyParams<int8_t, int8_t, rocprim::maximum<int8_t>>,
    DeviceScanByKeyParams<uint8_t, uint8_t, rocprim::maximum<uint8_t>, false>,
    DeviceScanByKeyParams<rocprim::half, rocprim::half, rocprim::maximum<rocprim::half>>,
    DeviceScanByKeyParams<rocprim::half,
                          float,
                          rocprim::plus<float>,
                          false,
                          rocprim::default_config,
                          false,
                          true>,
    DeviceScanByKeyParams<rocprim::bfloat16,
                          rocprim::bfloat16,
                          rocprim::maximum<rocprim::bfloat16>>,
    DeviceScanByKeyParams<rocprim::bfloat16,
                          float,
                          rocprim::plus<float>,
                          false,
                          rocprim::default_config,
                          false,
                          true>,
    // Large
    DeviceScanByKeyParams<int, double, rocprim::plus<int>>,
    DeviceScanByKeyParams<int, double, rocprim::plus<double>, false>,
    DeviceScanByKeyParams<int, long long, rocprim::plus<long long>>,
    DeviceScanByKeyParams<unsigned int, unsigned long long, rocprim::plus<unsigned long long>>,
    DeviceScanByKeyParams<long long, long long, rocprim::maximum<long long>>,
    DeviceScanByKeyParams<double, double, rocprim::plus<double>, true>,
    DeviceScanByKeyParams<double,
                          double,
                          rocprim::plus<double>,
                          false,
                          rocprim::default_config,
                          true,
                          true>,
    DeviceScanByKeyParams<signed char, long, rocprim::plus<long>>,
    DeviceScanByKeyParams<float, double, rocprim::minimum<double>>,
    DeviceScanByKeyParams<common::custom_type<int, int, true>>,
    DeviceScanByKeyParams<common::custom_type<double, double, true>,
                          common::custom_type<double, double, true>,
                          rocprim::plus<common::custom_type<double, double, true>>,
                          true>,
    DeviceScanByKeyParams<common::custom_type<double, double, true>,
                          common::custom_type<double, double, true>,
                          rocprim::plus<common::custom_type<double, double, true>>,
                          false,
                          rocprim::default_config,
                          true>,
    DeviceScanByKeyParams<common::custom_type<int, int, true>>,
    DeviceScanByKeyParams<test_utils::custom_test_array_type<long long, 5>>,
    DeviceScanByKeyParams<test_utils::custom_test_array_type<int, 10>>,
    // With graphs
    DeviceScanByKeyParams<int, int, rocprim::plus<int>, false, rocprim::default_config, true>,
    // With initial values
    DeviceScanByKeyParams<int,
                          int,
                          rocprim::plus<int>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<int,
                          int,
                          rocprim::maximum<int>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<int,
                          int,
                          rocprim::minimum<int>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<float,
                          float,
                          rocprim::plus<float>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<rocprim::half,
                          rocprim::half,
                          rocprim::minimum<rocprim::half>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<rocprim::half,
                          float,
                          rocprim::plus<float>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<rocprim::bfloat16,
                          rocprim::bfloat16,
                          rocprim::minimum<rocprim::bfloat16>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>,
    DeviceScanByKeyParams<rocprim::bfloat16,
                          float,
                          rocprim::plus<float>,
                          false,
                          size_limit_config<524288>,
                          false,
                          true,
                          true>>;

// Use float for accumulation of bfloat16 and half inputs if operator is plus
template<typename input_type, typename input_op_type>
struct accum_type
{
    static constexpr bool is_low_precision
        = std::is_same<input_type, ::rocprim::half>::value
          || std::is_same<input_type, ::rocprim::bfloat16>::value;
    static constexpr bool is_plus = test_utils::is_plus_operator<input_op_type>::value;
    using type = typename std::conditional_t<is_low_precision && is_plus, float, input_type>;
};

TYPED_TEST_SUITE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // Key type
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    const bool deterministic     = TestFixture::deterministic;

    using Config = typename TestFixture::config_helper;

    // Default
    hipStream_t stream = 0;
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * (size - 1) > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

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

            common::device_ptr<T> d_input(input);
            common::device_ptr<K> d_keys(keys);
            common::device_ptr<U> d_output(input.size());

            // Scan function
            scan_op_type scan_op;
            // Key compare function
            rocprim::equal_to<K> keys_compare_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_inclusive_scan_by_key(input.begin(),
                                                   input.end(),
                                                   keys.begin(),
                                                   expected.begin(),
                                                   scan_op,
                                                   keys_compare_op);

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    return invoke_inclusive_scan_by_key<deterministic, Config>(temp_storage,
                                                                               storage_bytes,
                                                                               d_keys.get(),
                                                                               input_iterator,
                                                                               d_output.get(),
                                                                               input.size(),
                                                                               scan_op,
                                                                               keys_compare_op,
                                                                               stream,
                                                                               debug_synchronous);
                },
                stream,
                TestFixture::use_graphs);

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * (size - 1)));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScanByKey)
{
    // scan-by-key does not support output iterator with void value_type
    using T = typename TestFixture::input_type;
    using K = unsigned int; // Key type
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    acc_type        initial_value;
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    const bool deterministic     = TestFixture::deterministic;

    using Config = typename TestFixture::config_helper;

    // Default
    hipStream_t stream = 0;
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(single_op_precision * (size - 1) > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

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

            common::device_ptr<T> d_input(input);
            common::device_ptr<K> d_keys(keys);
            common::device_ptr<U> d_output(input.size());

            // Scan function
            scan_op_type scan_op;

            // Key compare function
            rocprim::equal_to<K> keys_compare_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan_by_key(input.begin(),
                                                   input.end(),
                                                   keys.begin(),
                                                   initial_value,
                                                   expected.begin(),
                                                   scan_op,
                                                   keys_compare_op);

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    return invoke_exclusive_scan_by_key<deterministic, Config>(temp_storage,
                                                                               storage_bytes,
                                                                               d_keys.get(),
                                                                               input_iterator,
                                                                               d_output.get(),
                                                                               initial_value,
                                                                               input.size(),
                                                                               scan_op,
                                                                               keys_compare_op,
                                                                               stream,
                                                                               debug_synchronous);
                },
                stream,
                TestFixture::use_graphs);

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * (size - 1)));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
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
    using difference_type = std::ptrdiff_t;

    ROCPRIM_HOST_DEVICE check_run_iterator(const args_t args) : current_index_(0), args_(args) {}

    ROCPRIM_HOST_DEVICE
    bool operator==(const check_run_iterator& rhs) const
    {
        return current_index_ == rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE
    bool operator!=(const check_run_iterator& rhs) const
    {
        return !(*this == rhs);
    }
    ROCPRIM_HOST_DEVICE
    reference
        operator*()
    {
        return value_type{current_index_, args_};
    }
    ROCPRIM_HOST_DEVICE
    reference
        operator[](const difference_type distance) const
    {
        return *(*this + distance);
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator&
        operator+=(const difference_type rhs)
    {
        current_index_ += rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator&
        operator-=(const difference_type rhs)
    {
        current_index_ -= rhs;
        return *this;
    }
    ROCPRIM_HOST_DEVICE
    difference_type
        operator-(const check_run_iterator& rhs) const
    {
        return current_index_ - rhs.current_index_;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator
        operator+(const difference_type rhs) const
    {
        return check_run_iterator(*this) += rhs;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator
        operator-(const difference_type rhs) const
    {
        return check_run_iterator(*this) -= rhs;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator&
        operator++()
    {
        ++current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator&
        operator--()
    {
        --current_index_;
        return *this;
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator
        operator++(int)
    {
        return ++check_run_iterator{*this};
    }
    ROCPRIM_HOST_DEVICE
    check_run_iterator
        operator--(int)
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
    size_t
        operator=(const size_t value)
    {
        const size_t run_start    = current_index_ - (current_index_ % rocprim::get<0>(args_));
        const size_t index_in_run = current_index_ - run_start + 1;
        const size_t expected_sum = (run_start + current_index_) * index_in_run / 2;
        if(value != expected_sum)
        {
            rocprim::detail::atomic_store(rocprim::get<1>(args_), 1);
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
    size_t
        operator=(const size_t value)
    {
        const size_t run_start    = current_index_ - (current_index_ % rocprim::get<0>(args_));
        const size_t index_in_run = current_index_ - run_start;
        const size_t expected_sum
            = rocprim::get<1>(args_) + (run_start + current_index_ - 1) * index_in_run / 2;
        if(value != expected_sum)
        {
            rocprim::detail::atomic_store(rocprim::get<2>(args_), 1);
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
template<class ScanByKeyFun, bool UseGraphs = false>
void large_indices_scan_by_key_test(ScanByKeyFun scan_by_key_fun)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    constexpr bool debug_synchronous = false;
    hipStream_t    stream            = 0;
    if(UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const int seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    const auto size = test_utils::get_large_sizes(seed_value).back();
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    common::device_ptr<unsigned int> d_incorrect_flag(1);
    HIP_CHECK(hipMemset(d_incorrect_flag.get(), 0, sizeof(unsigned int)));

    const size_t run_length = test_utils::get_random_value<size_t>(1, 10000, seed_value);
    SCOPED_TRACE(testing::Message() << "with run_length = " << run_length);

    const auto keys_input = rocprim::make_transform_iterator(rocprim::counting_iterator<size_t>(0),
                                                             [run_length](const auto value)
                                                             { return value / run_length; });
    const auto values_input = rocprim::counting_iterator<size_t>(0);

    test_utils::test_kernel_wrapper(
        [&](void* temp_storage, size_t& storage_bytes)
        {
            return scan_by_key_fun(temp_storage,
                                   storage_bytes,
                                   keys_input,
                                   values_input,
                                   run_length,
                                   d_incorrect_flag.get(),
                                   size,
                                   stream,
                                   debug_synchronous,
                                   seed_value);
        },
        stream,
        UseGraphs);

    const auto incorrect_flag = d_incorrect_flag.load()[0];

    ASSERT_EQ(0, incorrect_flag);

    if(UseGraphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<bool UseGraphs = false>
void testLargeIndicesInclusiveScanByKey()
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
    large_indices_scan_by_key_test<decltype(inclusive_scan_by_key), UseGraphs>(
        inclusive_scan_by_key);
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanByKey)
{
    testLargeIndicesInclusiveScanByKey();
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanByKeyWithGraphs)
{
    testLargeIndicesInclusiveScanByKey<true>();
}

template<bool UseGraphs = false>
void testLargeIndicesExclusiveScanByKey()
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
    large_indices_scan_by_key_test<decltype(exclusive_scan_by_key), UseGraphs>(
        exclusive_scan_by_key);
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScanByKey)
{
    testLargeIndicesExclusiveScanByKey();
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScanByKeyWithGraphs)
{
    testLargeIndicesExclusiveScanByKey<true>();
}