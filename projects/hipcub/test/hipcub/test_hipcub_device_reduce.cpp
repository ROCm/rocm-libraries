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

#include "common_test_header.hpp"

// Thread operators fixes for extended float types
#include "hipcub/config.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_thread_operators.hpp"

// hipcub API
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/iterator/constant_input_iterator.hpp>

// Params for tests
template<class InputType, class OutputType = InputType, bool UseGraphs = false>
struct DeviceReduceParams
{
    using input_type                 = InputType;
    using output_type                = OutputType;
    static constexpr bool use_graphs = UseGraphs;
};

// ---------------------------------------------------------
// Test for reduction ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceReduceTests : public ::testing::Test
{
public:
    using input_type                 = typename Params::input_type;
    using output_type                = typename Params::output_type;
    static constexpr bool use_graphs = Params::use_graphs;
};

using HipcubDeviceReduceTestsParams = ::testing::Types<
    DeviceReduceParams<int, long>,
    DeviceReduceParams<unsigned long>,
    DeviceReduceParams<short>,
    DeviceReduceParams<float>,
    DeviceReduceParams<short, float>,
    DeviceReduceParams<int, double>,
    DeviceReduceParams<test_utils::half, test_utils::half>,
    DeviceReduceParams<test_utils::bfloat16, test_utils::bfloat16>,
    DeviceReduceParams<int, long, true>
#ifdef __HIP_PLATFORM_AMD__
    ,
    DeviceReduceParams<test_utils::half,
                       float>, // Doesn't work on NVIDIA / CUB
    DeviceReduceParams<test_utils::bfloat16,
                       float>, // Doesn't work on NVIDIA / CUB
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
#endif
    >;

TYPED_TEST_SUITE(HipcubDeviceReduceTests, HipcubDeviceReduceTestsParams);

TYPED_TEST(HipcubDeviceReduceTests, ReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

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
            if(test_utils::precision<U>::value * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1.0f, 100.0f, seed_value);
            std::vector<U> output(1, (U)0.0f);

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host using the same accumulator type than on device
            using Sum =
                typename AlgebraicSelector<hipcub::Sum, T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<Sum, T, U>;
            Sum    sum_op;
            AccumT tmp_result = AccumT(0.0f); // hipcub::Sum uses as initial type the output type
            for(unsigned int i = 0; i < input.size(); i++)
            {
                tmp_result = sum_op(tmp_result, input[i]);
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if constexpr(std::is_same<T, test_utils::half>::value
                         || std::is_same<T, test_utils::bfloat16>::value)
            {
                HIP_CHECK(hipcub::DeviceReduce::Reduce(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       input.size(),
                                                       ExtendedFloatBinOp<hipcub::Sum>(),
                                                       U(0.f),
                                                       stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceReduce::Sum(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_input,
                                                    d_output,
                                                    input.size(),
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
            if constexpr(std::is_same<T, test_utils::half>::value
                         || std::is_same<T, test_utils::bfloat16>::value)
            {
                HIP_CHECK(hipcub::DeviceReduce::Reduce(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       input.size(),
                                                       ExtendedFloatBinOp<hipcub::Sum>(),
                                                       U(0.f),
                                                       stream));
            }
            else
            {
                HIP_CHECK(hipcub::DeviceReduce::Sum(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    d_input,
                                                    d_output,
                                                    input.size(),
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
                test_utils::assert_near(output[0],
                                        expected,
                                        test_utils::precision<U>::value * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

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

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1.0f, 100.0f, seed_value);
            std::vector<U> output(1, U(0.0f));

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host using the same accumulator type than on device
            using Min    = typename MinSelector<T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<hipcub::Min, T, U>;
            Min    min_op;
            AccumT tmp_result = test_utils::numeric_limits<
                AccumT>::max(); // hipcub::Min uses as initial type the input type
            for(unsigned int i = 0; i < input.size(); i++)
            {
                tmp_result = min_op(tmp_result, input[i]);
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceReduce::Min(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                input.size(),
                                                stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DeviceReduce::Min(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
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
                test_utils::assert_near(output[0], expected, test_utils::precision<U>::value));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceMaximum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

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

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1.0f, 100.0f, seed_value);
            std::vector<U> output(1, U(0.0f));

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host using the same accumulator type than on device
            using Max    = typename MaxSelector<T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<hipcub::Max, T, U>;
            Max    max_op;
            AccumT tmp_result = test_utils::numeric_limits<AccumT>::min();
            for(unsigned int i = 0; i < input.size(); i++)
            {
                tmp_result = max_op(tmp_result, input[i]);
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceReduce::Max(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                input.size(),
                                                stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceReduce::Max(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                input.size(),
                                                stream));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

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
                test_utils::assert_near(output[0], expected, test_utils::precision<U>::value));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<class NumItemsT = int>
struct ArgMinDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    NumItemsT       num_items,
                    hipStream_t     stream) const
    {
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        return hipcub::DeviceReduce::ArgMin(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            num_items,
                                            stream);
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
    }
};

template<class NumItemsT = int>
struct ArgMaxDispatch
{
    template<typename InputIteratorT, typename OutputIteratorT>
    auto operator()(void*           d_temp_storage,
                    size_t&         temp_storage_bytes,
                    InputIteratorT  d_in,
                    OutputIteratorT d_out,
                    NumItemsT       num_items,
                    hipStream_t     stream) const
    {
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        return hipcub::DeviceReduce::ArgMax(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            num_items,
                                            stream);
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
    }
};

template<class NumItemsT = int>
struct ArgMin2Dispatch
{
    template<typename InputIteratorT, typename OutputIteratorT, typename OutputIndexIteratorT>
    auto operator()(void*                d_temp_storage,
                    size_t&              temp_storage_bytes,
                    InputIteratorT       d_in,
                    OutputIteratorT      d_out,
                    OutputIndexIteratorT d_index_out,
                    NumItemsT            num_items,
                    hipStream_t          stream) const
    {
        return hipcub::DeviceReduce::ArgMin(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            d_index_out,
                                            num_items,
                                            stream);
    }
};

template<class NumItemsT = int>
struct ArgMax2Dispatch
{
    template<typename InputIteratorT, typename OutputIteratorT, typename OutputIndexIteratorT>
    auto operator()(void*                d_temp_storage,
                    size_t&              temp_storage_bytes,
                    InputIteratorT       d_in,
                    OutputIteratorT      d_out,
                    OutputIndexIteratorT d_index_out,
                    NumItemsT            num_items,
                    hipStream_t          stream) const
    {
        return hipcub::DeviceReduce::ArgMax(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            d_index_out,
                                            num_items,
                                            stream);
    }
};

template<typename TestFixture, typename DispatchFunction, typename HostOp>
void test_argminmax(typename TestFixture::input_type empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T         = typename TestFixture::input_type;
    using Iterator  = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;

    DispatchFunction function;

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

        std::vector<size_t> sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(0);

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<T>         input = test_utils::get_random_data<T>(size, 0, 200, seed_value);
            std::vector<key_value> output(1);

            T*         d_input;
            key_value* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            key_value expected;
            if(size > 0)
            {
                // Calculate expected results on host
                Iterator        x(input.data());
                const key_value max = x[0];
                expected            = std::accumulate(x, x + size, max, HostOp());
            }
            else
            {
                // Empty inputs result in a special value
                expected = key_value(1, empty_value);
            }

            size_t temp_storage_size_bytes{};
            void*  d_temp_storage{};
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_output,
                               input.size(),
                               stream));

            // temp_storage_size_bytes must be > 0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_output,
                               input.size(),
                               stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(key_value),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, expected.key));
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output[0].value,
                                        expected.value,
                                        test_utils::precision<T>::value * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMinSelector<T>::type;
    test_argminmax<TestFixture, ArgMinDispatch<>, HostOp>(test_utils::numeric_limits<T>::max());
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMaximum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMaxSelector<T>::type;
    test_argminmax<TestFixture, ArgMaxDispatch<>, HostOp>(test_utils::numeric_limits<T>::lowest());
}

template<typename TestFixture, typename DispatchFunction, typename HostOp>
void test_argminmax2(typename TestFixture::input_type empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T             = typename TestFixture::input_type;
    using Iterator      = typename hipcub::ArgIndexInputIterator<T*, int>;
    using argidx_type   = typename Iterator::value_type;
    using extremum_type = typename argidx_type::value_type;
    using index_type    = typename argidx_type::key_type;

    DispatchFunction function;

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

        std::vector<size_t> sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(0);

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 200, seed_value);
            std::vector<extremum_type> extremum(1);
            std::vector<index_type>    index(1);

            T*             d_input;
            extremum_type* d_extremum;
            index_type*    d_index;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_extremum,
                                                         extremum.size() * sizeof(extremum_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_index, index.size() * sizeof(index_type)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            argidx_type expected;
            if(size > 0)
            {
                // Calculate expected results on host
                Iterator          x(input.data());
                const argidx_type max = x[0];
                expected              = std::accumulate(x, x + size, max, HostOp());
            }
            else
            {
                // Empty inputs result in a special value
                expected = argidx_type(1, empty_value);
            }

            size_t temp_storage_size_bytes{};
            void*  d_temp_storage{};
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_extremum,
                               d_index,
                               input.size(),
                               stream));

            // temp_storage_size_bytes must be > 0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(function(d_temp_storage,
                               temp_storage_size_bytes,
                               d_input,
                               d_extremum,
                               d_index,
                               input.size(),
                               stream));

            if(TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(extremum.data(),
                                d_extremum,
                                extremum.size() * sizeof(extremum_type),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(index.data(),
                                d_index,
                                index.size() * sizeof(index_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_extremum));
            HIP_CHECK(hipFree(d_temp_storage));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(index[0], expected.key));
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(extremum[0],
                                        expected.value,
                                        test_utils::precision<T>::value * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArg2Minimum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMinSelector<T>::type;
    test_argminmax2<TestFixture, ArgMin2Dispatch<>, HostOp>(test_utils::numeric_limits<T>::max());
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArg2Maximum)
{
    using T = typename TestFixture::input_type;
    // Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
    using HostOp = typename ArgMaxSelector<T>::type;
    test_argminmax2<TestFixture, ArgMax2Dispatch<>, HostOp>(
        test_utils::numeric_limits<T>::lowest());
}

template<class T>
class HipcubDeviceReduceArgMinMaxSpecialTests : public testing::Test
{};

using HipcubDeviceReduceArgMinMaxSpecialTestsParams
    = ::testing::Types<float, test_utils::half, test_utils::bfloat16>;
TYPED_TEST_SUITE(HipcubDeviceReduceArgMinMaxSpecialTests,
                 HipcubDeviceReduceArgMinMaxSpecialTestsParams);

template<typename TypeParam, typename DispatchFunction>
void test_argminmax_allinf(TypeParam value, TypeParam empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T         = TypeParam;
    using Iterator  = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;

    hipStream_t      stream = 0; // default
    DispatchFunction function;
    constexpr size_t size = 100'000;

    // Generate data
    std::vector<T>         input(size, value);
    std::vector<key_value> output(1);

    T*         d_input;
    key_value* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    size_t temp_storage_size_bytes{};
    void*  d_temp_storage{};

    HIP_CHECK(
        function(d_temp_storage, temp_storage_size_bytes, d_input, d_output, input.size(), stream));

    // temp_storage_size_bytes must be > 0
    ASSERT_GT(temp_storage_size_bytes, 0U);

    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
        function(d_temp_storage, temp_storage_size_bytes, d_input, d_output, input.size(), stream));
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(),
                        d_output,
                        output.size() * sizeof(key_value),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));

    if(size > 0)
    {
        // all +/- infinity should produce +/- infinity
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 0));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, value));
    }
    else
    {
        // empty input should produce a special value
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 1));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, empty_value));
    }
}

template<typename TypeParam, typename DispatchFunction>
void test_argminmax_extremum(TypeParam value, TypeParam empty_value)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T         = TypeParam;
    using Iterator  = typename hipcub::ArgIndexInputIterator<T*, int64_t>;
    using key_value = typename Iterator::value_type;

    hipStream_t      stream = 0; // default
    DispatchFunction function;
    constexpr size_t size = 200'000;

    // Generate data
    std::vector<T>         input(size, value);
    std::vector<key_value> output(1);

    T*         d_input;
    key_value* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    size_t temp_storage_size_bytes{};
    void*  d_temp_storage{};

    HIP_CHECK(
        function(d_temp_storage, temp_storage_size_bytes, d_input, d_output, input.size(), stream));

    // temp_storage_size_bytes must be > 0
    ASSERT_GT(temp_storage_size_bytes, 0U);

    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
        function(d_temp_storage, temp_storage_size_bytes, d_input, d_output, input.size(), stream));
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(),
                        d_output,
                        output.size() * sizeof(key_value),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));

    if(size > 0)
    {
        // all +/- infinity should produce +/- infinity
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 0));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, value));
    }
    else
    {
        // empty input should produce a special value
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].key, 1));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[0].value, empty_value));
    }
}

// ArgMin with all +Inf should result in +Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMinInf)
{
    test_argminmax_allinf<TypeParam, ArgMinDispatch<>>(
        test_utils::numeric_limits<TypeParam>::infinity(),
        test_utils::numeric_limits<TypeParam>::max());
}

// ArgMin with all +Inf should result in +Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMinExtream)
{
    test_argminmax_extremum<TypeParam, ArgMinDispatch<int64_t>>(
        test_utils::numeric_limits<TypeParam>::infinity(),
        test_utils::numeric_limits<TypeParam>::min());
}

// ArgMax with all -Inf should result in -Inf.
TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMaxInf)
{
    test_argminmax_allinf<TypeParam, ArgMaxDispatch<>>(
        test_utils::numeric_limits<TypeParam>::infinity_neg(),
        test_utils::numeric_limits<TypeParam>::lowest());
}

TYPED_TEST(HipcubDeviceReduceArgMinMaxSpecialTests, ReduceArgMaxExtream)
{
    test_argminmax_extremum<TypeParam, ArgMaxDispatch<int64_t>>(
        test_utils::numeric_limits<TypeParam>::infinity_neg(),
        test_utils::numeric_limits<TypeParam>::lowest());
}

struct TestTransformOp
{
    template<class T>
    HIPCUB_HOST_DEVICE
    T operator()(const T& x) const
    {
        return x + T(5);
    }
};

TYPED_TEST(HipcubDeviceReduceTests, TransformReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

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
            if(test_utils::precision<U>::value * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1.0f, 100.0f, seed_value);
            std::vector<U> output(1, U(0));

            T* d_input;
            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

            // Calculate expected results on host using the same accumulator type than on device
            using Sum =
                typename AlgebraicSelector<hipcub::Sum, T, U>::type; // For custom_type_test tests
            using AccumT = hipcub::detail::accumulator_t<Sum, T, U>;

            Sum             reduction_op;
            TestTransformOp transform_op;
            const U         init(10);
            AccumT          tmp_result = init; // hipcub::Sum uses as initial type the output type
            for(size_t i = 0; i < input.size(); ++i)
            {
                tmp_result = reduction_op(tmp_result, transform_op(input[i]));
            }
            const U expected = static_cast<U>(tmp_result);

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceReduce::TransformReduce(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            d_input,
                                                            d_output,
                                                            input.size(),
                                                            reduction_op,
                                                            transform_op,
                                                            init,
                                                            stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DeviceReduce::TransformReduce(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            d_input,
                                                            d_output,
                                                            input.size(),
                                                            reduction_op,
                                                            transform_op,
                                                            init,
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

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output[0],
                                        expected,
                                        test_utils::precision<U>::value * size));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

// ---------------------------------------------------------
// Test for large indices
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceReduceLargeIndicesTests : public ::testing::Test
{
public:
    using input_type  = typename Params::input_type;
    using output_type = typename Params::output_type;
};

using HipcubDeviceReduceLargeIndicesTestsParams
    = ::testing::Types<DeviceReduceParams<short, size_t>,
                       DeviceReduceParams<int, size_t>,
                       DeviceReduceParams<unsigned int, size_t>,
                       DeviceReduceParams<unsigned long, size_t>,
                       DeviceReduceParams<float, size_t>,
                       DeviceReduceParams<double, size_t>>;

TYPED_TEST_SUITE(HipcubDeviceReduceLargeIndicesTests, HipcubDeviceReduceLargeIndicesTestsParams);

TYPED_TEST(HipcubDeviceReduceLargeIndicesTests, LargeIndices)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T            = typename TestFixture::input_type;
    using U            = typename TestFixture::output_type;
    using IteratorType                  = rocprim::constant_iterator<T>;
    const std::vector<size_t> exponents = {30, 31, 32, 33, 34};
    for(auto exponent : exponents)
    {
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            const size_t size = 1ll << exponent;
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            hipStream_t stream = 0; // default

            // Generate data
            IteratorType   d_input(T{1});
            std::vector<U> output(1, (U)0.0f);

            U* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(hipDeviceSynchronize());

            // Temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;

            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DeviceReduce::Sum(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                size,
                                                stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // Allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(hipcub::DeviceReduce::Sum(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_output,
                                                size,
                                                stream));
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            const std::size_t result = output[0];
            ASSERT_EQ(result, size);

            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}
