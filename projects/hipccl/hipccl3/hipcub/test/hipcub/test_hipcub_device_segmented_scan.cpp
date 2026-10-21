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

// Thread operators fixes for extended float types
#include "test_utils_data_generation.hpp"
#include "test_utils_thread_operators.hpp"

// hipcub API
#include <hipcub/device/device_segmented_scan.hpp>

template<class Input,
         class Output,
         class ScanOp = test_utils::plus,
         int Init
         = 0, // as only integral types supported, int is used here even for floating point inputs
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000>
struct DeviceSegmentedScanParams
{
    using input_type                                 = Input;
    using output_type                                = Output;
    using scan_op_type                               = ScanOp;
    static constexpr int          init               = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedScanTests : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubDeviceSegmentedScanTestsParams = ::testing::Types<
    DeviceSegmentedScanParams<unsigned int, unsigned int, test_utils::plus>,
    DeviceSegmentedScanParams<int, int, test_utils::plus, -100, 0, 10000>,
    DeviceSegmentedScanParams<double, double, test_utils::minimum, 1000, 0, 10000>,
    DeviceSegmentedScanParams<int, short, test_utils::maximum, 10, 1000, 10000>,
    DeviceSegmentedScanParams<float, double, test_utils::maximum, 50, 2, 10>,
    DeviceSegmentedScanParams<float, float, test_utils::plus, 123, 100, 200>,
    DeviceSegmentedScanParams<test_utils::half, test_utils::half, test_utils::maximum, 50, 2, 10>,
    DeviceSegmentedScanParams<test_utils::bfloat16,
                              test_utils::bfloat16,
                              test_utils::maximum,
                              50,
                              2,
                              10>
#ifdef __HIP_PLATFORM_AMD__
    ,
    DeviceSegmentedScanParams<short,
                              double,
                              test_utils::plus,
                              5,
                              1,
                              1000> // Doesn't work on NVIDIA / CUB
#endif
    >;

TYPED_TEST_SUITE(HipcubDeviceSegmentedScanTests, HipcubDeviceSegmentedScanTestsParams);

template<bool Exclusive,
         bool WithInit,
         class InputIteratorT,
         class OutputIteratorT,
         class ScanOpT,
         class InitValueT,
         class... Offsets>
auto invoke_segmented_scan(void*                temporary_storage,
                           size_t&              storage_size,
                           InputIteratorT       input,
                           OutputIteratorT      output,
                           _HIPCUB_STD::int64_t num_segments,
                           ScanOpT              scan_op,
                           const InitValueT /*initial_value*/,
                           hipStream_t stream,
                           Offsets... offsets) -> std::enable_if_t<!WithInit && !Exclusive, void>
{
    HIP_CHECK(hipcub::DeviceSegmentedScan::InclusiveSegmentedScan(temporary_storage,
                                                                  storage_size,
                                                                  input,
                                                                  output,
                                                                  offsets...,
                                                                  num_segments,
                                                                  scan_op,
                                                                  stream));
}

template<bool Exclusive,
         bool WithInit,
         class InputIteratorT,
         class OutputIteratorT,
         class ScanOpT,
         class InitValueT,
         class... Offsets>
auto invoke_segmented_scan(void*                temporary_storage,
                           size_t&              storage_size,
                           InputIteratorT       input,
                           OutputIteratorT      output,
                           _HIPCUB_STD::int64_t num_segments,
                           ScanOpT              scan_op,
                           const InitValueT     initial_value,
                           hipStream_t          stream,
                           Offsets... offsets) -> std::enable_if_t<WithInit && !Exclusive, void>
{
    HIP_CHECK(hipcub::DeviceSegmentedScan::InclusiveSegmentedScanInit(temporary_storage,
                                                                      storage_size,
                                                                      input,
                                                                      output,
                                                                      offsets...,
                                                                      num_segments,
                                                                      scan_op,
                                                                      initial_value,
                                                                      stream));
}

template<bool Exclusive,
         bool WithInit,
         class InputIteratorT,
         class OutputIteratorT,
         class ScanOpT,
         class InitValueT,
         class... Offsets>
auto invoke_segmented_scan(void*                temporary_storage,
                           size_t&              storage_size,
                           InputIteratorT       input,
                           OutputIteratorT      output,
                           _HIPCUB_STD::int64_t num_segments,
                           ScanOpT              scan_op,
                           const InitValueT     initial_value,
                           hipStream_t          stream,
                           Offsets... offsets) -> std::enable_if_t<WithInit && Exclusive, void>
{
    HIP_CHECK(hipcub::DeviceSegmentedScan::ExclusiveSegmentedScan(temporary_storage,
                                                                  storage_size,
                                                                  input,
                                                                  output,
                                                                  offsets...,
                                                                  num_segments,
                                                                  scan_op,
                                                                  initial_value,
                                                                  stream));
}

template<typename TestFixture, bool WithInit = false, bool WithOutOffsets = false>
void test_inclusive_scan()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using offset_type  = unsigned int;

    const input_type init = input_type{TestFixture::params::init};
    SCOPED_TRACE(testing::Message() << "with init = " << init);

    scan_op_type scan_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default stream

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const offset_type output_pad
            = WithOutOffsets ? test_utils::get_random_value<offset_type>(1, 100, seed_value) : 0;
        SCOPED_TRACE(testing::Message() << "with output_pad = " << output_pad);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            const size_t output_size = size + output_pad;

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(output_size);
            std::vector<input_type>  values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<offset_type> output_offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = std::min(segment_length_dis(gen), size - offset);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const auto output_offset = offset + output_pad;
                output_offsets.push_back(output_offset);

                const size_t end   = std::min(output_size, output_offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - output_offset);

                input_type aggregate = values_input[offset];
                if constexpr(WithInit)
                {
                    aggregate = scan_op(init, aggregate);
                }
                values_expected[output_offset] = aggregate;
                for(size_t i = output_offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i - output_pad]);
                    values_expected[i] = aggregate;
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);
            output_offsets.push_back(output_size);
            const unsigned int offsets_count = segments_count + 1;

            const float precision
                = test_utils::precision<decltype(init)>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                         offsets_count * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                offsets_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            offset_type* d_output_offsets;
            if constexpr(WithOutOffsets)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output_offsets,
                                                             offsets_count * sizeof(offset_type)));
                HIP_CHECK(hipMemcpy(d_output_offsets,
                                    output_offsets.data(),
                                    offsets_count * sizeof(offset_type),
                                    hipMemcpyHostToDevice));
            }

            output_type* d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output,
                                                         output_size * sizeof(output_type)));
            HIP_CHECK(hipMemset(d_values_output, 0, output_size * sizeof(output_type)));

            auto call = [&](void* d_temp_storage, size_t& temp_storage_bytes)
            {
                if constexpr(WithOutOffsets)
                {
                    invoke_segmented_scan<false /*Exclusive*/, WithInit>(d_temp_storage,
                                                                         temp_storage_bytes,
                                                                         d_values_input,
                                                                         d_values_output,
                                                                         segments_count,
                                                                         scan_op,
                                                                         init,
                                                                         stream,
                                                                         d_offsets,
                                                                         d_offsets + 1,
                                                                         d_output_offsets);
                }
                else
                {
                    invoke_segmented_scan<false /*Exclusive*/, WithInit>(d_temp_storage,
                                                                         temp_storage_bytes,
                                                                         d_values_input,
                                                                         d_values_output,
                                                                         segments_count,
                                                                         scan_op,
                                                                         init,
                                                                         stream,
                                                                         d_offsets,
                                                                         d_offsets + 1);
                }
            };

            size_t temporary_storage_bytes;
            call(nullptr, temporary_storage_bytes);

            ASSERT_GT(temporary_storage_bytes, 0);
            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            call(d_temporary_storage, temporary_storage_bytes);

            HIP_CHECK(hipDeviceSynchronize());

            std::vector<output_type> values_output(output_size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                output_size * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_temporary_storage));
            if constexpr(WithOutOffsets)
            {
                HIP_CHECK(hipFree(d_output_offsets));
            }

            if(size > 0)
            {
                for(size_t i = 0; i < output_offsets[0]; ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output[i], output_type{0}))
                        << " with i = " << i;
                }

                const float single_op_precision = precision / max_segment_length;
                size_t      current_offset      = output_offsets[0];
                size_t      current_size_index  = 0;
                for(size_t i = output_offsets[0]; i < output_size; ++i)
                {
                    if((i - current_offset) == sizes[current_size_index])
                    {
                        ++current_size_index;
                        current_offset = output_offsets[current_size_index];
                    }
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(values_output[i],
                                                values_expected[i],
                                                single_op_precision * (i - current_offset)))
                        << " with i = " << i;
                }
            }
        }
    }
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, InclusiveScan)
{
    test_inclusive_scan<TestFixture>();
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, InclusiveScanInit)
{
    test_inclusive_scan<TestFixture, true /*WithInit*/>();
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, InclusiveScanOutputOffsets)
{
    test_inclusive_scan<TestFixture, false /*WithInit*/, true /*WithOutputOffsets*/>();
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, InclusiveScanInitOutputOffsets)
{
    test_inclusive_scan<TestFixture, true /*WithInit*/, true /*WithOutputOffsets*/>();
}

template<typename TestFixture, bool WithOutOffsets = false>
void test_exclusive_scan()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using offset_type  = unsigned int;

    const input_type init = input_type{TestFixture::params::init};
    SCOPED_TRACE(testing::Message() << "with init = " << init);

    scan_op_type scan_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default stream

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const offset_type output_pad
            = WithOutOffsets ? test_utils::get_random_value<offset_type>(1, 100, seed_value) : 0;
        SCOPED_TRACE(testing::Message() << "with output_pad = " << output_pad);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            const size_t output_size = size + output_pad;

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(output_size);
            std::vector<input_type>  values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<offset_type> output_offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const auto output_offset = offset + output_pad;
                output_offsets.push_back(output_offset);

                const size_t end   = std::min(output_size, output_offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - output_offset);

                input_type aggregate           = init;
                values_expected[output_offset] = aggregate;
                for(size_t i = output_offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i - output_pad - 1]);
                    values_expected[i] = output_type(aggregate);
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);
            output_offsets.push_back(output_size);
            const unsigned int offsets_count = segments_count + 1;

            const float precision
                = test_utils::precision<decltype(init)>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                         offsets_count * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                offsets_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            offset_type* d_output_offsets;
            if constexpr(WithOutOffsets)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output_offsets,
                                                             offsets_count * sizeof(offset_type)));
                HIP_CHECK(hipMemcpy(d_output_offsets,
                                    output_offsets.data(),
                                    offsets_count * sizeof(offset_type),
                                    hipMemcpyHostToDevice));
            }

            output_type* d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output,
                                                         output_size * sizeof(output_type)));
            HIP_CHECK(hipMemset(d_values_output, 0, output_size * sizeof(output_type)));

            auto call = [&](void* d_temp_storage, size_t& temp_storage_bytes)
            {
                if constexpr(WithOutOffsets)
                {
                    invoke_segmented_scan<true /*Exclusive*/, true /*WithInit*/>(d_temp_storage,
                                                                                 temp_storage_bytes,
                                                                                 d_values_input,
                                                                                 d_values_output,
                                                                                 segments_count,
                                                                                 scan_op,
                                                                                 init,
                                                                                 stream,
                                                                                 d_offsets,
                                                                                 d_offsets + 1,
                                                                                 d_output_offsets);
                }
                else
                {
                    invoke_segmented_scan<true /*Exclusive*/, true /*WithInit*/>(d_temp_storage,
                                                                                 temp_storage_bytes,
                                                                                 d_values_input,
                                                                                 d_values_output,
                                                                                 segments_count,
                                                                                 scan_op,
                                                                                 init,
                                                                                 stream,
                                                                                 d_offsets,
                                                                                 d_offsets + 1);
                }
            };

            size_t temporary_storage_bytes;
            call(nullptr, temporary_storage_bytes);

            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_GT(temporary_storage_bytes, 0);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            call(d_temporary_storage, temporary_storage_bytes);

            HIP_CHECK(hipDeviceSynchronize());

            std::vector<output_type> values_output(output_size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                output_size * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));
            HIP_CHECK(hipFree(d_temporary_storage));
            if constexpr(WithOutOffsets)
            {
                HIP_CHECK(hipFree(d_output_offsets));
            }

            if(size > 0)
            {
                for(size_t i = 0; i < output_offsets[0]; ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output[i], output_type{0}))
                        << " with i = " << i;
                }

                const float single_op_precision = precision / max_segment_length;
                size_t      current_offset      = output_offsets[0];
                size_t      current_size_index  = 0;
                for(size_t i = output_offsets[0]; i < output_size; ++i)
                {
                    if((i - current_offset) == sizes[current_size_index])
                    {
                        ++current_size_index;
                        current_offset = output_offsets[current_size_index];
                    }
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(values_output[i],
                                                values_expected[i],
                                                single_op_precision * (i - current_offset)))
                        << " with i = " << i;
                }
            }
        }
    }
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, ExclusiveScan)
{
    test_exclusive_scan<TestFixture>();
}

TYPED_TEST(HipcubDeviceSegmentedScanTests, ExclusiveScanOutputOffsets)
{
    test_exclusive_scan<TestFixture, true /*WithOutputOffsets*/>();
}

template<class Input,
         class Output,
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000>
struct DeviceSegmentedOpParams
{
    using input_type                                 = Input;
    using output_type                                = Output;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedOpTests : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubDeviceSegmentedOpTestsParams = ::testing::Types<
    DeviceSegmentedOpParams<unsigned int, unsigned int>,
    DeviceSegmentedOpParams<int, int, 0, 10000>,
    DeviceSegmentedOpParams<double, double, 0, 10000>,
    DeviceSegmentedOpParams<int, long long, 1000, 10000>,
    DeviceSegmentedOpParams<float, double, 2, 10>,
    DeviceSegmentedOpParams<float, float, 100, 200>
    // #ifdef __HIP_PLATFORM_AMD__
    ,
    DeviceSegmentedOpParams<test_utils::half, float>, // Doesn't work on NVIDIA / CUB
    DeviceSegmentedOpParams<test_utils::bfloat16, float> // Doesn't work on NVIDIA / CUB
    // #endif
    >;

TYPED_TEST_SUITE(HipcubDeviceSegmentedOpTests, HipcubDeviceSegmentedOpTestsParams);

template<bool Exclusive>
struct SumDispatch
{
    template<typename... Args>
    auto operator()(Args&&... args) const
    {
        return hipcub::DeviceSegmentedScan::ExclusiveSegmentedSum(std::forward<Args>(args)...);
    }
};

template<>
struct SumDispatch<false>
{
    template<typename... Args>
    auto operator()(Args&&... args) const
    {
        return hipcub::DeviceSegmentedScan::InclusiveSegmentedSum(std::forward<Args>(args)...);
    }
};

template<typename DispatchFunction, typename... Args>
void invoke_segmented_scan_op(Args&&... args)
{
    DispatchFunction function{};
    HIP_CHECK(function(std::forward<Args>(args)...));
}

template<typename TestFixture, typename DispatchFunction, typename ScanOpType, bool Exclusive>
void test_sum(typename TestFixture::params::input_type init)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = ScanOpType;
    using offset_type  = unsigned int;

    SCOPED_TRACE(testing::Message() << "with init = " << init);

    scan_op_type scan_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default stream

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(size);
            std::vector<input_type>  values_input
                = test_utils::get_random_data<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                input_type aggregate    = Exclusive ? init : values_input[offset];
                values_expected[offset] = aggregate;
                for(size_t i = offset + 1; i < end; i++)
                {
                    const auto value   = Exclusive ? values_input[i - 1] : values_input[i];
                    aggregate          = scan_op(aggregate, value);
                    values_expected[i] = output_type(aggregate);
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);
            const unsigned int offsets_count = segments_count + 1;

            const float precision
                = test_utils::precision<decltype(init)>::value * max_segment_length;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            input_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(input_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(input_type),
                                hipMemcpyHostToDevice));

            offset_type* d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets,
                                                         offsets_count * sizeof(offset_type)));
            HIP_CHECK(hipMemcpy(d_offsets,
                                offsets.data(),
                                offsets_count * sizeof(offset_type),
                                hipMemcpyHostToDevice));

            output_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(output_type)));

            size_t temporary_storage_bytes;

            invoke_segmented_scan_op<DispatchFunction>(nullptr,
                                                       temporary_storage_bytes,
                                                       d_values_input,
                                                       d_values_output,
                                                       d_offsets,
                                                       d_offsets + 1,
                                                       segments_count,
                                                       stream);

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            invoke_segmented_scan_op<DispatchFunction>(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_values_input,
                                                       d_values_output,
                                                       d_offsets,
                                                       d_offsets + 1,
                                                       segments_count,
                                                       stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<output_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                size * sizeof(output_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(values_output, values_expected, precision));
        }
    }
}

TYPED_TEST(HipcubDeviceSegmentedOpTests, InclusiveSegmentedSum)
{
    using input_type           = typename TestFixture::params::input_type;
    using scan_op_type         = typename test_utils::plus;
    constexpr bool   Exclusive = false;
    const input_type init      = input_type(0);
    test_sum<TestFixture, SumDispatch<Exclusive>, scan_op_type, Exclusive>(init);
}

TYPED_TEST(HipcubDeviceSegmentedOpTests, ExclusiveSegmentedSum)
{
    using input_type           = typename TestFixture::params::input_type;
    using scan_op_type         = typename test_utils::plus;
    constexpr bool   Exclusive = true;
    const input_type init      = input_type(0);
    test_sum<TestFixture, SumDispatch<Exclusive>, scan_op_type, Exclusive>(init);
}
