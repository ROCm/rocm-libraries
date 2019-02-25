// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

// Google Test
#include <gtest/gtest.h>
#include <iostream>

// Thrust
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class ReduceByKeysTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> ReduceByKeysTestsParams;

TYPED_TEST_CASE(ReduceByKeysTests, ReduceByKeysTestsParams);


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<typename T>
struct is_equal_div_10_reduce
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};

template <typename Vector>
void initialize_keys(Vector& keys)
{
	keys.resize(9);
	keys[0] = 11;
	keys[1] = 11;
	keys[2] = 21;
	keys[3] = 20;
	keys[4] = 21;
	keys[5] = 21;
	keys[6] = 21;
	keys[7] = 37;
	keys[8] = 37;
}

template <typename Vector>
void initialize_values(Vector& values)
{
	values.resize(9);
	values[0] = 0;
	values[1] = 1;
	values[2] = 2;
	values[3] = 3;
	values[4] = 4;
	values[5] = 5;
	values[6] = 6;
	values[7] = 7;
	values[8] = 8;
}

TYPED_TEST(ReduceByKeysTests, TestReduceByKeySimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

	Vector keys;
	Vector values;

	typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

	// basic test
    initialize_keys(keys);
    initialize_values(values);

	Vector output_keys(keys.size());
	Vector output_values(values.size());

	new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

    ASSERT_EQ(new_last.first  - output_keys.begin(),   5);
    ASSERT_EQ(new_last.second - output_values.begin(), 5);
    ASSERT_EQ(output_keys[0], 11);
    ASSERT_EQ(output_keys[1], 21);
    ASSERT_EQ(output_keys[2], 20);
    ASSERT_EQ(output_keys[3], 21);
    ASSERT_EQ(output_keys[4], 37);

    ASSERT_EQ(output_values[0],  1);
    ASSERT_EQ(output_values[1],  2);
    ASSERT_EQ(output_values[2],  3);
    ASSERT_EQ(output_values[3], 15);
    ASSERT_EQ(output_values[4], 15);

    // test BinaryPredicate
    initialize_keys(keys);
    initialize_values(values);

    new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>());

    ASSERT_EQ(new_last.first  - output_keys.begin(),   3);
    ASSERT_EQ(new_last.second - output_values.begin(), 3);
    ASSERT_EQ(output_keys[0], 11);
    ASSERT_EQ(output_keys[1], 21);
    ASSERT_EQ(output_keys[2], 37);

    ASSERT_EQ(output_values[0],  1);
    ASSERT_EQ(output_values[1], 20);
    ASSERT_EQ(output_values[2], 15);

    // test BinaryFunction
    initialize_keys(keys);
    initialize_values(values);

    new_last = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>());

    ASSERT_EQ(new_last.first  - output_keys.begin(),   5);
    ASSERT_EQ(new_last.second - output_values.begin(), 5);

    ASSERT_EQ(output_keys[0], 11);
    ASSERT_EQ(output_keys[1], 21);
    ASSERT_EQ(output_keys[2], 20);
    ASSERT_EQ(output_keys[3], 21);
    ASSERT_EQ(output_keys[4], 37);

    ASSERT_EQ(output_values[0],  1);
    ASSERT_EQ(output_values[1],  2);
    ASSERT_EQ(output_values[2],  3);
    ASSERT_EQ(output_values[3], 15);
    ASSERT_EQ(output_values[4], 15);
}


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
