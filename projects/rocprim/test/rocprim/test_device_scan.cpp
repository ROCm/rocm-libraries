// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
using size_limit_config = rocprim::scan_config<256,
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
struct DeviceScanParams
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
constexpr hipError_t invoke_inclusive_scan(Args&&... args)
{
    if(Deterministic)
    {
        return rocprim::deterministic_inclusive_scan<Config>(std::forward<Args>(args)...);
    }
    else
    {
        return rocprim::inclusive_scan<Config>(std::forward<Args>(args)...);
    }
}

template<bool Deterministic, typename Config = rocprim::default_config, typename... Args>
constexpr hipError_t invoke_exclusive_scan(Args&&... args)
{
    if(Deterministic)
    {
        return rocprim::deterministic_exclusive_scan<Config>(std::forward<Args>(args)...);
    }
    else
    {
        return rocprim::exclusive_scan<Config>(std::forward<Args>(args)...);
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
    DeviceScanParams<char>,
    DeviceScanParams<unsigned short>,
    DeviceScanParams<short, int>,
    DeviceScanParams<int>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, size_limit_config<512>>,
    DeviceScanParams<float, float, rocprim::maximum<float>>,
    DeviceScanParams<float, float, rocprim::plus<float>, false, size_limit_config<1024>>,
    DeviceScanParams<float,
                     float,
                     rocprim::plus<float>,
                     false,
                     size_limit_config<1024>,
                     false,
                     true>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, size_limit_config<524288>>,
    DeviceScanParams<int, int, rocprim::plus<int>, false, size_limit_config<1048576>>,
    DeviceScanParams<int8_t, int8_t, rocprim::maximum<int8_t>>,
    DeviceScanParams<uint8_t, uint8_t, rocprim::maximum<uint8_t>, false>,
    DeviceScanParams<rocprim::half, rocprim::half, rocprim::maximum<rocprim::half>>,
    DeviceScanParams<rocprim::half,
                     float,
                     rocprim::plus<float>,
                     false,
                     rocprim::default_config,
                     false,
                     true>,
    DeviceScanParams<rocprim::bfloat16, rocprim::bfloat16, rocprim::maximum<rocprim::bfloat16>>,
    DeviceScanParams<rocprim::bfloat16,
                     float,
                     rocprim::plus<float>,
                     false,
                     rocprim::default_config,
                     false,
                     true>,
    // Large
    DeviceScanParams<int, double, rocprim::plus<int>>,
    DeviceScanParams<int, double, rocprim::plus<double>, false>,
    DeviceScanParams<int, long long, rocprim::plus<long long>>,
    DeviceScanParams<unsigned int, unsigned long long, rocprim::plus<unsigned long long>>,
    DeviceScanParams<long long, long long, rocprim::maximum<long long>>,
    DeviceScanParams<double, double, rocprim::plus<double>, true>,
    DeviceScanParams<double,
                     double,
                     rocprim::plus<double>,
                     false,
                     rocprim::default_config,
                     true,
                     true>,
    DeviceScanParams<signed char, long, rocprim::plus<long>>,
    DeviceScanParams<float, double, rocprim::minimum<double>>,
    DeviceScanParams<common::custom_type<int, int, true>>,
    DeviceScanParams<common::custom_type<double, double, true>,
                     common::custom_type<double, double, true>,
                     rocprim::plus<common::custom_type<double, double, true>>,
                     true>,
    DeviceScanParams<common::custom_type<double, double, true>,
                     common::custom_type<double, double, true>,
                     rocprim::plus<common::custom_type<double, double, true>>,
                     false,
                     rocprim::default_config,
                     true>,
    DeviceScanParams<common::custom_type<int, int, true>>,
    DeviceScanParams<test_utils::custom_test_array_type<long long, 5>>,
    DeviceScanParams<test_utils::custom_test_array_type<int, 10>>,
    // With graphs
    DeviceScanParams<int, int, rocprim::plus<int>, false, rocprim::default_config, true>,
    // With initial values
    DeviceScanParams<int,
                     int,
                     rocprim::plus<int>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<int,
                     int,
                     rocprim::maximum<int>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<int,
                     int,
                     rocprim::minimum<int>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<float,
                     float,
                     rocprim::plus<float>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<rocprim::half,
                     rocprim::half,
                     rocprim::minimum<rocprim::half>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<rocprim::half,
                     float,
                     rocprim::plus<float>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<rocprim::bfloat16,
                     rocprim::bfloat16,
                     rocprim::minimum<rocprim::bfloat16>,
                     false,
                     size_limit_config<524288>,
                     false,
                     true,
                     true>,
    DeviceScanParams<rocprim::bfloat16,
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

TYPED_TEST(RocprimDeviceScanTests, LookBackScan)
{
    using T            = typename TestFixture::input_type;
    using U            = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op                 = test_utils::is_plus_operator<scan_op_type>;
    using acc_type                   = typename accum_type<U, scan_op_type>::type;
    using scan_state_type            = rocprim::detail::lookback_scan_state<acc_type>;
    using scan_state_with_sleep_type = rocprim::detail::lookback_scan_state<acc_type, true>;

    const bool deterministic     = TestFixture::deterministic;
    const bool use_initial_value = TestFixture::use_initial_value;

    using Config = typename TestFixture::config_helper;
    using config = rocprim::detail::wrapped_scan_config<Config, acc_type>;

    hipStream_t stream = hipStreamDefault;

    rocprim::detail::target_arch target_arch;
    HIP_CHECK(host_target_arch(stream, target_arch));
    const rocprim::detail::scan_config_params params
        = rocprim::detail::dispatch_target_arch<config, false>(target_arch);

    // For non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

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
            if(size == 0)
            {
                continue;
            }
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const auto         items_per_block  = block_size * items_per_thread;

            unsigned int number_of_blocks = (size + items_per_block - 1) / items_per_block;

            if(single_op_precision * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }
            hipStream_t stream = hipStreamDefault;
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 10, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(input.size());

            // Scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            acc_type       initial_value;
            if(use_initial_value)
            {
                initial_value = test_utils::get_random_value<acc_type>(1, 10, seed_value);
                test_utils::host_inclusive_scan(input.begin(),
                                                input.end(),
                                                expected.begin(),
                                                scan_op,
                                                initial_value);
            }
            else
            {
                test_utils::host_inclusive_scan(input.begin(),
                                                input.end(),
                                                expected.begin(),
                                                scan_op);
            }
            SCOPED_TRACE(use_initial_value
                             ? (testing::Message() << "with initial_value = " << initial_value)
                             : (testing::Message() << "without initial_value"));

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            // Pointer to array with block_prefixes
            acc_type* previous_last_element;
            acc_type* new_last_element;

            rocprim::detail::temp_storage::layout layout{};
            HIP_CHECK(scan_state_type::get_temp_storage_layout(number_of_blocks, stream, layout));

            size_t storage_size;
            HIP_CHECK(scan_state_type::get_storage_size(number_of_blocks, stream, storage_size));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(storage_size, 0);

            // Allocate temporary storage
            common::device_ptr<void> d_temp_storage(storage_size);

            scan_state_type scan_state{};
            HIP_CHECK(scan_state_type::create(scan_state,
                                              d_temp_storage.get(),
                                              number_of_blocks,
                                              stream));
            scan_state_with_sleep_type scan_state_with_sleep{};
            HIP_CHECK(scan_state_with_sleep_type::create(scan_state_with_sleep,
                                                         d_temp_storage.get(),
                                                         number_of_blocks,
                                                         stream));

            // Call the provided function with either scan_state or scan_state_with_sleep based on
            // the value of use_sleep
            bool use_sleep;
            HIP_CHECK(rocprim::detail::is_sleep_scan_state_used(stream, use_sleep));
            auto with_scan_state = [use_sleep, scan_state, scan_state_with_sleep](
                                       auto&& func) mutable -> decltype(auto)
            {
                if(use_sleep)
                {
                    return func(scan_state_with_sleep);
                }
                else
                {
                    return func(scan_state);
                }
            };
            auto grid_size = (number_of_blocks + block_size - 1) / block_size;
            with_scan_state(
                [&](const auto scan_state)
                {
                    rocprim::detail::init_lookback_scan_state_kernel<<<dim3(grid_size),
                                                                       dim3(block_size),
                                                                       0,
                                                                       stream>>>(scan_state,
                                                                                 number_of_blocks);
                });

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            static constexpr bool Exclusive = false;

            grid_size = number_of_blocks;

            const auto launch_err = with_scan_state(
                [&](const auto scan_state)
                {
                    return rocprim::detail::launch_lookback_scan < config,
                           deterministic
                               ? rocprim::detail::lookback_scan_determinism::deterministic
                               : rocprim::detail::lookback_scan_determinism::nondeterministic,
                           Exclusive, use_initial_value, decltype(input_iterator), U*, scan_op_type,
                           acc_type,
                           acc_type > (target_arch,
                                       input_iterator,
                                       d_output.get(),
                                       size,
                                       initial_value,
                                       scan_op,
                                       scan_state,
                                       number_of_blocks,
                                       dim3(grid_size),
                                       dim3(block_size),
                                       0,
                                       stream,
                                       previous_last_element,
                                       new_last_element,
                                       false,
                                       false);
                });

            ASSERT_EQ(hipSuccess, launch_err);

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, single_op_precision * size));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<class T, class scan_state_type>
__global__
void complete_value(T* values, scan_state_type scan_state)
{
    values[blockIdx.x] = scan_state.get_complete_value(blockIdx.x);
}

TYPED_TEST(RocprimDeviceScanTests, LookBackScanGetCompleteValue)
{
    using T            = typename TestFixture::input_type;
    using U            = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op                 = test_utils::is_plus_operator<scan_op_type>;
    using acc_type                   = typename accum_type<U, scan_op_type>::type;
    using scan_state_type            = rocprim::detail::lookback_scan_state<acc_type>;
    using scan_state_with_sleep_type = rocprim::detail::lookback_scan_state<acc_type, true>;

    const bool deterministic     = TestFixture::deterministic;
    const bool use_initial_value = TestFixture::use_initial_value;

    using Config = typename TestFixture::config_helper;
    using config = rocprim::detail::wrapped_scan_config<Config, acc_type>;

    hipStream_t stream = hipStreamDefault;

    rocprim::detail::target_arch target_arch;
    HIP_CHECK(host_target_arch(stream, target_arch));
    const rocprim::detail::scan_config_params params
        = rocprim::detail::dispatch_target_arch<config, false>(target_arch);

    // For non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const unsigned int block_size       = params.kernel_config.block_size;
        const unsigned int items_per_thread = params.kernel_config.items_per_thread;
        const auto         items_per_block  = block_size * items_per_thread;
        const auto         size             = items_per_block;

        unsigned int number_of_blocks = (size + items_per_block - 1) / items_per_block;

        if(single_op_precision * size > 0.5)
        {
            std::cout << "Test is skipped from size " << size
                      << " on, potential error of summation is more than 0.5 of the result "
                         "with current or larger size"
                      << std::endl;
            break;
        }
        hipStream_t stream = hipStreamDefault;
        if(TestFixture::use_graphs)
        {
            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 10, seed_value);

        common::device_ptr<T> d_input(input);
        common::device_ptr<U> d_output(input.size());

        // Scan function
        scan_op_type scan_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        acc_type       initial_value;
        if(use_initial_value)
        {
            initial_value = test_utils::get_random_value<acc_type>(1, 10, seed_value);
            test_utils::host_inclusive_scan(input.begin(),
                                            input.end(),
                                            expected.begin(),
                                            scan_op,
                                            initial_value);
        }
        else
        {
            test_utils::host_inclusive_scan(input.begin(), input.end(), expected.begin(), scan_op);
        }
        SCOPED_TRACE(use_initial_value
                         ? (testing::Message() << "with initial_value = " << initial_value)
                         : (testing::Message() << "without initial_value"));

        auto input_iterator
            = rocprim::make_transform_iterator(d_input.get(),
                                               [](T in) { return static_cast<acc_type>(in); });

        // Pointer to array with block_prefixes
        acc_type* previous_last_element;
        acc_type* new_last_element;

        rocprim::detail::temp_storage::layout layout{};
        HIP_CHECK(scan_state_type::get_temp_storage_layout(number_of_blocks, stream, layout));

        size_t storage_size;
        HIP_CHECK(scan_state_type::get_storage_size(number_of_blocks, stream, storage_size));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(storage_size, 0);

        // Allocate temporary storage
        common::device_ptr<void> d_temp_storage(storage_size);
        common::device_ptr<U>    d_save_dest(std::vector<U>(1));

        scan_state_type scan_state{};
        HIP_CHECK(
            scan_state_type::create(scan_state, d_temp_storage.get(), number_of_blocks, stream));
        scan_state_with_sleep_type scan_state_with_sleep{};
        HIP_CHECK(scan_state_with_sleep_type::create(scan_state_with_sleep,
                                                     d_temp_storage.get(),
                                                     number_of_blocks,
                                                     stream));

        // Call the provided function with either scan_state or scan_state_with_sleep based on
        // the value of use_sleep
        bool use_sleep;
        HIP_CHECK(rocprim::detail::is_sleep_scan_state_used(stream, use_sleep));
        auto with_scan_state
            = [use_sleep, scan_state, scan_state_with_sleep](auto&& func) mutable -> decltype(auto)
        {
            if(use_sleep)
            {
                return func(scan_state_with_sleep);
            }
            else
            {
                return func(scan_state);
            }
        };
        auto grid_size = (number_of_blocks + block_size - 1) / block_size;
        with_scan_state(
            [&](const auto scan_state)
            {
                rocprim::detail::init_lookback_scan_state_kernel<<<dim3(grid_size),
                                                                   dim3(block_size),
                                                                   0,
                                                                   stream>>>(scan_state,
                                                                             number_of_blocks);
            });

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        test_utils::GraphHelper gHelper;
        if(TestFixture::use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        static constexpr bool Exclusive = false;

        grid_size = number_of_blocks;

        const auto launch_err = with_scan_state(
            [&](const auto scan_state)
            {
                return rocprim::detail::launch_lookback_scan < config,
                       deterministic ? rocprim::detail::lookback_scan_determinism::deterministic
                                     : rocprim::detail::lookback_scan_determinism::nondeterministic,
                       Exclusive, use_initial_value, decltype(input_iterator), U*, scan_op_type,
                       acc_type,
                       acc_type > (target_arch,
                                   input_iterator,
                                   d_output.get(),
                                   size,
                                   initial_value,
                                   scan_op,
                                   scan_state,
                                   number_of_blocks,
                                   dim3(grid_size),
                                   dim3(block_size),
                                   0,
                                   stream,
                                   previous_last_element,
                                   new_last_element,
                                   false,
                                   false);
            });

        ASSERT_EQ(hipSuccess, launch_err);

        if(TestFixture::use_graphs)
        {
            gHelper.createAndLaunchGraph(stream, true, false);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        const auto output = d_output.load();

        // Check if output values are as expected
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output, expected, single_op_precision * size));

        common::device_ptr<U> d_output_complete(grid_size);
        with_scan_state(
            [&](const auto scan_state) {
                complete_value<<<dim3(grid_size), dim3(1), 0, stream>>>(d_output_complete.get(),
                                                                        scan_state);
            });

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        const auto output_complete = d_output_complete.load();

        test_utils::assert_near(output_complete[0], output[size - 1], single_op_precision * size);

        if(TestFixture::use_graphs)
        {
            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

TYPED_TEST_SUITE(RocprimDeviceScanTests, RocprimDeviceScanTestsParams);

TYPED_TEST(RocprimDeviceScanTests, InclusiveScanEmptyInput)
{
    using T            = typename TestFixture::input_type;
    using U            = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using acc_type               = typename accum_type<T, scan_op_type>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    const bool deterministic     = TestFixture::deterministic;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Default
    hipStream_t stream = 0;
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    common::device_ptr<U> d_output(1);

    test_utils::out_of_bounds_flag          out_of_bounds;
    test_utils::bounds_checking_iterator<U> d_checking_output(d_output.get(),
                                                              out_of_bounds.device_pointer(),
                                                              0);

    // Scan function
    scan_op_type scan_op;

    auto input_iterator
        = rocprim::make_transform_iterator(rocprim::make_constant_iterator<T>(T(345)),
                                           [](T in) { return static_cast<acc_type>(in); });

    test_utils::test_kernel_wrapper(
        [&](void* temp_storage, size_t& storage_bytes)
        {
            return invoke_inclusive_scan<deterministic>(temp_storage,
                                                        storage_bytes,
                                                        input_iterator,
                                                        d_checking_output,
                                                        0,
                                                        scan_op,
                                                        stream,
                                                        debug_synchronous);
        },
        stream,
        TestFixture::use_graphs);

    ASSERT_FALSE(out_of_bounds.get());

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceScanTests, InclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    // For non-associative operations in inclusive scan
    // intermediate results use the type of input iterator, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    static constexpr bool deterministic         = TestFixture::deterministic;
    static constexpr bool use_initial_value     = TestFixture::use_initial_value;

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

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(input.size());

            // Scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            acc_type       initial_value;
            if(use_initial_value)
            {
                initial_value = test_utils::get_random_value<acc_type>(1, 10, seed_value);
                test_utils::host_inclusive_scan(input.begin(),
                                                input.end(),
                                                expected.begin(),
                                                scan_op,
                                                initial_value);
            }
            else
            {
                test_utils::host_inclusive_scan(input.begin(),
                                                input.end(),
                                                expected.begin(),
                                                scan_op);
            }
            SCOPED_TRACE(use_initial_value
                             ? (testing::Message() << "with initial_value = " << initial_value)
                             : (testing::Message() << "without initial_value"));

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if constexpr(use_initial_value)
                    {
                        return invoke_inclusive_scan<deterministic, Config>(
                            temp_storage,
                            storage_bytes,
                            input_iterator,
                            test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                                d_output.get()),
                            initial_value,
                            input.size(),
                            scan_op,
                            stream,
                            TestFixture::debug_synchronous);
                    }
                    else
                    {
                        return invoke_inclusive_scan<deterministic, Config>(
                            temp_storage,
                            storage_bytes,
                            input_iterator,
                            test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                                d_output.get()),
                            input.size(),
                            scan_op,
                            stream,
                            TestFixture::debug_synchronous);
                    }
                },
                stream,
                TestFixture::use_graphs);

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            if(size > 0)
            {
                for(size_t i = 0; i < output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(output[i], expected[i], single_op_precision * i));
                }
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceScanTests, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    // If scan_op_type is rocprim::plus and input_type is bfloat16 or half,
    // use float as device-side accumulator and double as host-side accumulator
    using is_plus_op = test_utils::is_plus_operator<scan_op_type>;
    using acc_type   = typename accum_type<T, scan_op_type>::type;

    // For non-associative operations in exclusive scan
    // intermediate results use the type of initial value, then
    // as all conversions in the tests are to more precise types,
    // intermediate results use the same or more precise acc_type,
    // all scan operations use the same acc_type,
    // and all output types are the same acc_type,
    // therefore the only source of error is precision of operation itself
    acc_type        initial_value;
    constexpr float single_op_precision = is_plus_op::value ? test_utils::precision<acc_type> : 0;

    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    static constexpr bool deterministic         = TestFixture::deterministic;

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

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(input.size());

            // Scan function
            scan_op_type scan_op;

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            initial_value = test_utils::get_random_value<acc_type>(1, 10, seed_value);
            test_utils::host_exclusive_scan(input.begin(),
                                            input.end(),
                                            initial_value,
                                            expected.begin(),
                                            scan_op);

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    return invoke_exclusive_scan<deterministic, Config>(
                        temp_storage,
                        storage_bytes,
                        input_iterator,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                            d_output.get()),
                        initial_value,
                        input.size(),
                        scan_op,
                        stream,
                        debug_synchronous);
                },
                stream,
                TestFixture::use_graphs);

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            if(size > 0)
            {
                for(size_t i = 0; i < output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(output[i], expected[i], single_op_precision * i));
                }
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<typename T>
class single_index_iterator
{
public:
    class conditional_discard_value
    {
    public:
        __host__ __device__ explicit conditional_discard_value(T* const value, bool keep)
            : value_{value}, keep_{keep}
        {}

        __host__ __device__
        conditional_discard_value&
            operator=(T value)
        {
            if(keep_)
            {
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
    using difference_type = std::ptrdiff_t;

    __host__ __device__ single_index_iterator(T* value, size_t expected_index, size_t index = 0)
        : value_{value}, expected_index_{expected_index}, index_{index}
    {}

    __host__ __device__ single_index_iterator(const single_index_iterator&) = default;
    __host__ __device__
    single_index_iterator&
        operator=(const single_index_iterator&)
        = default;

    // clang-format off
    __host__ __device__ bool operator==(const single_index_iterator& rhs) const { return index_ == rhs.index_; }
    __host__ __device__ bool operator!=(const single_index_iterator& rhs) const { return !(this == rhs);       }

    __host__ __device__ reference operator*() { return value_type{value_, index_ == expected_index_}; }

    __host__ __device__ reference operator[](const difference_type distance) const { return *(*this + distance); }

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

template<bool UseGraphs = false, bool UseInitialValue = false>
void testLargeIndicesInclusiveScan()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using Iterator               = typename rocprim::counting_iterator<T>;
    using OutputIterator         = single_index_iterator<T>;
    const bool debug_synchronous = false;

    // Default
    hipStream_t stream = 0;
    if(UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Create counting_iterator<U> with random starting point
            Iterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value ^ size));

            SCOPED_TRACE(testing::Message() << "with starting point = " << *input_begin);

            common::device_ptr<T> d_output(1);

            OutputIterator output_it{d_output.get(), size - 1};

            size_t initial_value = 0;

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    if constexpr(UseInitialValue)
                    {
                        initial_value = test_utils::get_random_value<size_t>(0, 10000, seed_value);
                        return rocprim::inclusive_scan(temp_storage,
                                                       storage_bytes,
                                                       input_begin,
                                                       output_it,
                                                       initial_value,
                                                       size,
                                                       ::rocprim::plus<T>(),
                                                       stream,
                                                       debug_synchronous);
                    }
                    else
                    {
                        return rocprim::inclusive_scan(temp_storage,
                                                       storage_bytes,
                                                       input_begin,
                                                       output_it,
                                                       size,
                                                       ::rocprim::plus<T>(),
                                                       stream,
                                                       debug_synchronous);
                    }
                },
                stream,
                UseGraphs);

            SCOPED_TRACE(UseInitialValue
                             ? (testing::Message() << "with initial_value = " << initial_value)
                             : (testing::Message() << "without initial_value"));

            // Copy output to host
            const auto output = d_output.load()[0];

            // Sum of 'size' increasing numbers starting at 'n' is size * (2n + size - 1)
            // The division is not integer division but either (size) or (2n + size - 1) has to be even.
            const T multiplicand_1 = size;
            const T multiplicand_2 = 2 * (*input_begin) + size - 1;
            const T expected_output
                = ((multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                             : multiplicand_1 * (multiplicand_2 / 2))
                  + initial_value;

            ASSERT_EQ(output, expected_output);
        }
    }

    if(UseGraphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScan)
{
    testLargeIndicesInclusiveScan();
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanWithGraphs)
{
    testLargeIndicesInclusiveScan<true>();
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanWithInitialValue)
{
    testLargeIndicesInclusiveScan<false, true>();
}

TEST(RocprimDeviceScanTests, LargeIndicesInclusiveScanWithInitialValueAndGraphs)
{
    testLargeIndicesInclusiveScan<true, true>();
}

template<bool UseGraphs = false>
void testLargeIndicesExclusiveScan()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using Iterator               = typename rocprim::counting_iterator<T>;
    using OutputIterator         = single_index_iterator<T>;
    const bool debug_synchronous = false;

    // Default
    hipStream_t stream = 0;
    if(UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Create counting_iterator<U> with random starting point
            Iterator input_begin(test_utils::get_random_value<T>(0, 200, seed_value ^ size));
            T initial_value = test_utils::get_random_value<T>(1, 10, seed_value ^ *input_begin);

            SCOPED_TRACE(testing::Message() << "with starting point = " << *input_begin);
            SCOPED_TRACE(testing::Message() << "with initial value = " << initial_value);

            common::device_ptr<T> d_output(1);

            OutputIterator output_it{d_output.get(), size - 1};

            test_utils::test_kernel_wrapper(
                [&](void* temp_storage, size_t& storage_bytes)
                {
                    return rocprim::exclusive_scan(temp_storage,
                                                   storage_bytes,
                                                   input_begin,
                                                   output_it,
                                                   initial_value,
                                                   size,
                                                   ::rocprim::plus<T>(),
                                                   stream,
                                                   debug_synchronous);
                },
                stream,
                UseGraphs);

            // Copy output to host
            const auto output = d_output.load()[0];

            // Sum of 'size' - 1 increasing numbers starting at 'n' is (size - 1) * (2n + size - 2)
            // The division is not integer division but either (size - 1) or (2n + size - 2) has to be even.
            const T multiplicand_1 = size - 1;
            const T multiplicand_2 = 2 * (*input_begin) + size - 2;

            const T product = (multiplicand_1 % 2 == 0) ? multiplicand_1 / 2 * multiplicand_2
                                                        : multiplicand_1 * (multiplicand_2 / 2);

            const T expected_output = initial_value + product;

            ASSERT_EQ(output, expected_output);
        }
    }

    if(UseGraphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScan)
{
    testLargeIndicesExclusiveScan();
}

TEST(RocprimDeviceScanTests, LargeIndicesExclusiveScanWithGraphs)
{
    testLargeIndicesExclusiveScan<true>();
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

using RocprimDeviceScanFutureTestsParams = ::testing::Types<
    DeviceScanParams<char>,
    DeviceScanParams<int>,
    DeviceScanParams<float, double, rocprim::minimum<double>>,
    DeviceScanParams<double, double, rocprim::plus<double>, true>,
    DeviceScanParams<common::custom_type<int, int, true>>,
    DeviceScanParams<test_utils::custom_test_array_type<long long, 5>>,
    DeviceScanParams<int, int, ::rocprim::plus<int>, false, rocprim::default_config, true>>;

template<typename Params>
class RocprimDeviceScanFutureTests : public RocprimDeviceScanTests<Params>
{};

TYPED_TEST_SUITE(RocprimDeviceScanFutureTests, RocprimDeviceScanFutureTestsParams);

TYPED_TEST(RocprimDeviceScanFutureTests, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using scan_op_type = typename TestFixture::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using acc_type     = typename accum_type<T, scan_op_type>::type;

    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            deterministic         = TestFixture::deterministic;
    using Config                                = typename TestFixture::config_helper;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            constexpr int future_size = 2048;
            const float   precision
                = test_utils::precision<T> * future_size
                  + (is_plus_op::value ? test_utils::precision<acc_type> * size : 0);

            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Default
            hipStream_t stream = 0;
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const std::vector<T> future_input
                = test_utils::get_random_data<T>(future_size, 1, 10, ~seed_value);
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(input.size());
            common::device_ptr<T> d_future_input(future_input);
            common::device_ptr<T> d_initial_value(1);

            // Scan function
            scan_op_type scan_op;

            const acc_type initial_value
                = std::accumulate(future_input.begin(), future_input.end(), T(0));

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            test_utils::host_exclusive_scan(input.begin(),
                                            input.end(),
                                            initial_value,
                                            expected.begin(),
                                            scan_op);

            const auto future_iter = test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                d_initial_value.get());
            const auto future_initial_value
                = rocprim::future_value<acc_type, std::remove_const_t<decltype(future_iter)>>{
                    future_iter};

            auto input_iterator
                = rocprim::make_transform_iterator(d_input.get(),
                                                   [](T in) { return static_cast<acc_type>(in); });

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK((invoke_exclusive_scan<deterministic, Config>(
                nullptr,
                temp_storage_size_bytes,
                input_iterator,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                future_initial_value,
                input.size(),
                scan_op,
                stream,
                debug_synchronous)));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            size_t temp_storage_reduce = 0;
            HIP_CHECK(rocprim::reduce(nullptr,
                                      temp_storage_reduce,
                                      d_future_input.get(),
                                      d_initial_value.get(),
                                      2048,
                                      rocprim::plus<T>(),
                                      stream));

            // Allocate temporary storage,
            // we use a char pointer as we need to offset it
            common::device_ptr<char> d_temp_storage(temp_storage_size_bytes + temp_storage_reduce);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Fill initial value on the device
            HIP_CHECK(rocprim::reduce(d_temp_storage.get() + temp_storage_size_bytes,
                                      temp_storage_reduce,
                                      d_future_input.get(),
                                      d_initial_value.get(),
                                      2048,
                                      rocprim::plus<T>(),
                                      stream));

            // Run
            HIP_CHECK((invoke_exclusive_scan<deterministic, Config>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                input_iterator,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                future_initial_value,
                input.size(),
                scan_op,
                stream,
                debug_synchronous)));
            HIP_CHECK(hipGetLastError());

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, precision));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
