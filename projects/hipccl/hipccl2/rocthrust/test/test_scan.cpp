/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scan.h>

#include <tuple>

#include "test_header.hpp"

TESTS_DEFINE(ScanTests, FullTestsParams);

TESTS_DEFINE(ScanVariablesTests, NumericalTestsParams);

TESTS_DEFINE(ScanVectorTests, VectorSignedIntegerTestsParams);

typedef ::testing::Types<
    Params<std::tuple<thrust::host_vector<int>, thrust::host_vector<float>>>,
    Params<std::tuple<thrust::device_vector<int>, thrust::device_vector<float>>>>
    MixedParams;

TESTS_DEFINE(ScanMixedTests, MixedParams);

template <typename T>
struct max_functor
{
    __host__ __device__ T operator()(T rhs, T lhs) const
    {
        return thrust::max(rhs, lhs);
    }
};

TYPED_TEST(ScanVectorTests, TestScanSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    typename Vector::iterator iter;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1;
    input[1] = 3;
    input[2] = -2;
    input[3] = 4;
    input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    iter      = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
    result[0] = 1;
    result[1] = 4;
    result[2] = 2;
    result[3] = 6;
    result[4] = 1;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input, input_copy);
    ASSERT_EQ(output, result);

    // exclusive scan
    iter      = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 0);
    result[0] = 0;
    result[1] = 1;
    result[2] = 4;
    result[3] = 2;
    result[4] = 6;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input, input_copy);
    ASSERT_EQ(output, result);

    // exclusive scan with init
    iter      = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3);
    result[0] = 3;
    result[1] = 4;
    result[2] = 7;
    result[3] = 5;
    result[4] = 9;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input, input_copy);
    ASSERT_EQ(output, result);

    // inclusive scan with op
    iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
    result[0] = 1;
    result[1] = 4;
    result[2] = 2;
    result[3] = 6;
    result[4] = 1;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input, input_copy);
    ASSERT_EQ(output, result);

    // exclusive scan with init and op
    iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), 3, thrust::plus<T>());
    result[0] = 3;
    result[1] = 4;
    result[2] = 7;
    result[3] = 5;
    result[4] = 9;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input, input_copy);
    ASSERT_EQ(output, result);

    // inplace inclusive scan
    input     = input_copy;
    iter      = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 1;
    result[1] = 4;
    result[2] = 2;
    result[3] = 6;
    result[4] = 1;
    ASSERT_EQ(iter - input.begin(), input.size());
    ASSERT_EQ(input, result);

    // inplace exclusive scan with init
    input     = input_copy;
    iter      = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), 3);
    result[0] = 3;
    result[1] = 4;
    result[2] = 7;
    result[3] = 5;
    result[4] = 9;
    ASSERT_EQ(iter - input.begin(), input.size());
    ASSERT_EQ(input, result);

    // inplace exclusive scan with implicit init=0
    input     = input_copy;
    iter      = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
    result[0] = 0;
    result[1] = 1;
    result[2] = 4;
    result[3] = 2;
    result[4] = 6;
    ASSERT_EQ(iter - input.begin(), input.size());
    ASSERT_EQ(input, result);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator
    inclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(ScanTests, TestInclusiveScanDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(ScanTests, TestInclusiveScanDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::inclusive_scan(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator
    exclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(ScanTests, TestExclusiveScanDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(ScanTests, TestExclusiveScanDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::exclusive_scan(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TEST(ScanTests, TestInclusiveScan32)
{
    using T  = int;
    size_t n = 32;

    thrust::host_vector<T> h_input
        = get_random_data<T>(n, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    thrust::host_vector<T> h_output_d(d_output);
    ASSERT_EQ(h_output_d, h_output);
}

TEST(ScanTests, TestExclusiveScan32)
{
    using T     = int;
    size_t n    = 32;
    T      init = 13;

    thrust::host_vector<T> h_input
        = get_random_data<T>(n, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

    ASSERT_EQ(d_output, h_output);
}

template <class IntVector, class FloatVector>
void TestScanMixedTypes(void)
{
    // make sure we get types for default args and operators correct
    IntVector int_input(4);
    int_input[0] = 1;
    int_input[1] = 2;
    int_input[2] = 3;
    int_input[3] = 4;

    FloatVector float_input(4);
    float_input[0] = 1.5;
    float_input[1] = 2.5;
    float_input[2] = 3.5;
    float_input[3] = 4.5;

    IntVector   int_output(4);
    FloatVector float_output(4);

    // float -> int should use using plus<int> operator by default
    thrust::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQ(int_output[0], 1);
    ASSERT_EQ(int_output[1], 3);
    ASSERT_EQ(int_output[2], 6);
    ASSERT_EQ(int_output[3], 10);

    // float -> float with plus<int> operator (int accumulator)
    thrust::inclusive_scan(
        float_input.begin(), float_input.end(), float_output.begin(), thrust::plus<int>());
    ASSERT_FLOAT_EQ(float_output[0], 1.0);
    ASSERT_FLOAT_EQ(float_output[1], 3.0);
    ASSERT_FLOAT_EQ(float_output[2], 6.0);
    ASSERT_FLOAT_EQ(float_output[3], 10.0);

    // float -> int should use using plus<int> operator by default
    thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
    ASSERT_EQ(int_output[0], 0);
    ASSERT_EQ(int_output[1], 1);
    ASSERT_EQ(int_output[2], 3);
    ASSERT_EQ(int_output[3], 6);

    // float -> int should use using plus<int> operator by default
    thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float)5.5);
    ASSERT_EQ(int_output[0], 5);
    ASSERT_EQ(int_output[1], 7);
    ASSERT_EQ(int_output[2], 9);
    ASSERT_EQ(int_output[3], 13);

    // int -> float should use using plus<float> operator by default
    thrust::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
    ASSERT_EQ(float_output[0], 1.0);
    ASSERT_EQ(float_output[1], 3.0);
    ASSERT_EQ(float_output[2], 6.0);
    ASSERT_EQ(float_output[3], 10.0);

    // int -> float should use using plus<float> operator by default
    thrust::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float)5.5);
    ASSERT_EQ(float_output[0], 5.5);
    ASSERT_EQ(float_output[1], 6.5);
    ASSERT_EQ(float_output[2], 8.5);
    ASSERT_EQ(float_output[3], 11.5);
}

TEST(ScanTests, TestScanMixedTypesHost)
{
    TestScanMixedTypes<thrust::host_vector<int>, thrust::host_vector<float>>();
}

TEST(ScanTests, TestScanMixedTypesDevice)
{
    TestScanMixedTypes<thrust::device_vector<int>, thrust::device_vector<float>>();
}

TYPED_TEST(ScanVariablesTests, TestScanWithOperator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), max_functor<T>());
        thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), max_functor<T>());
        ASSERT_EQ(d_output, h_output);

        thrust::exclusive_scan(
            h_input.begin(), h_input.end(), h_output.begin(), T(13), max_functor<T>());
        thrust::exclusive_scan(
            d_input.begin(), d_input.end(), d_output.begin(), T(13), max_functor<T>());
        ASSERT_EQ(d_output, h_output);
    }
}

TYPED_TEST(ScanVariablesTests, TestScanWithOperatorToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> reference(size);

        thrust::discard_iterator<> h_result = thrust::inclusive_scan(
            h_input.begin(), h_input.end(), thrust::make_discard_iterator(), max_functor<T>());

        thrust::discard_iterator<> d_result = thrust::inclusive_scan(
            d_input.begin(), d_input.end(), thrust::make_discard_iterator(), max_functor<T>());

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);

        h_result = thrust::exclusive_scan(h_input.begin(),
                                          h_input.end(),
                                          thrust::make_discard_iterator(),
                                          T(13),
                                          max_functor<T>());

        d_result = thrust::exclusive_scan(d_input.begin(),
                                          d_input.end(),
                                          thrust::make_discard_iterator(),
                                          T(13),
                                          max_functor<T>());

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(ScanVariablesTests, TestScan)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
        thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

        thrust::host_vector<T> h_output_d(d_output);
        ASSERT_EQ(h_output_d, h_output);

        thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
        thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
        h_output_d = d_output;
        ASSERT_EQ(h_output_d, h_output);

        thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T)11);
        thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T)11);
        h_output_d = d_output;
        ASSERT_EQ(h_output_d, h_output);

        // in-place scans
        h_output = h_input;
        d_output = d_input;
        thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
        thrust::inclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
        h_output_d = d_output;
        ASSERT_EQ(h_output_d, h_output);

        h_output = h_input;
        d_output = d_input;
        thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
        thrust::exclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
        h_output_d = d_output;
        ASSERT_EQ(h_output_d, h_output);
    }
}

TYPED_TEST(ScanVariablesTests, TestScanToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> h_result = thrust::inclusive_scan(
            h_input.begin(), h_input.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> d_result = thrust::inclusive_scan(
            d_input.begin(), d_input.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);

        h_result = thrust::exclusive_scan(
            h_input.begin(), h_input.end(), thrust::make_discard_iterator(), (T)11);

        d_result = thrust::exclusive_scan(
            d_input.begin(), d_input.end(), thrust::make_discard_iterator(), (T)11);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TEST(ScanTests, TestScanMixedTypes)
{
    const unsigned int n = 113;

    thrust::host_vector<unsigned int> h_input = get_random_data<unsigned int>(
        n, std::numeric_limits<unsigned int>::min(), std::numeric_limits<unsigned int>::max());
    for(size_t i = 0; i < n; i++)
        h_input[i] %= 10;
    thrust::device_vector<unsigned int> d_input = h_input;

    thrust::host_vector<float>   h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int>     h_int_output(n);
    thrust::device_vector<int>   d_int_output(n);

    //mixed input/output types
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float)3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float)3.5);
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int)3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int)3);
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int)3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int)3);
    ASSERT_EQ(d_int_output, h_int_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float)3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float)3.5);
    ASSERT_EQ(d_int_output, h_int_output);
}

template <typename T, unsigned int N>
void _TestScanWithLargeTypes(void)
{
    size_t n = (1024 * 1024) / sizeof(FixedVector<T, N>);

    thrust::host_vector<FixedVector<T, N>> h_input(n);
    thrust::host_vector<FixedVector<T, N>> h_output(n);

    for(size_t i = 0; i < h_input.size(); i++)
        h_input[i] = FixedVector<T, N>(i);

    thrust::device_vector<FixedVector<T, N>> d_input = h_input;
    thrust::device_vector<FixedVector<T, N>> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQ_QUIET(h_output, d_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), FixedVector<T, N>(0));
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), FixedVector<T, N>(0));

    ASSERT_EQ_QUIET(h_output, d_output);
}

TEST(ScanTests, TestScanWithLargeTypes)
{
    _TestScanWithLargeTypes<int, 1>();

    // XXX these are too big for sm_1x
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_HIP && !defined(__QNX__)
    _TestScanWithLargeTypes<int, 8>();
    _TestScanWithLargeTypes<int, 64>();
#else
//  KNOWN_FAILURE;
#endif
}

template <typename T>
struct plus_mod3
{
    T* table;

    plus_mod3(T* table)
        : table(table)
    {
    }

    __host__ __device__ T operator()(T a, T b)
    {
        return table[(int)(a + b)];
    }
};

TYPED_TEST(ScanVectorTests, TestInclusiveScanWithIndirection)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(7);
    data[0] = 0;
    data[1] = 1;
    data[2] = 2;
    data[3] = 1;
    data[4] = 2;
    data[5] = 0;
    data[6] = 1;

    Vector table(6);
    table[0] = 0;
    table[1] = 1;
    table[2] = 2;
    table[3] = 0;
    table[4] = 1;
    table[5] = 2;

    thrust::inclusive_scan(
        data.begin(), data.end(), data.begin(), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

    ASSERT_EQ(data[0], T(0));
    ASSERT_EQ(data[1], T(1));
    ASSERT_EQ(data[2], T(0));
    ASSERT_EQ(data[3], T(1));
    ASSERT_EQ(data[4], T(0));
    ASSERT_EQ(data[5], T(0));
    ASSERT_EQ(data[6], T(1));
}
