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

// Thrust
#include <thrust/count.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

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
class CountTests : public ::testing::Test
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
> CountTestsParams;

TYPED_TEST_CASE(CountTests, CountTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(CountTests, TestCountSimple)
{
    using Vector = typename TestFixture::input_type;

    Vector data(5);
    data[0] = 1; data[1] = 1; data[2] = 0; data[3] = 0; data[4] = 1;

    ASSERT_EQ(thrust::count(data.begin(), data.end(), 0), 2);
    ASSERT_EQ(thrust::count(data.begin(), data.end(), 1), 3);
    ASSERT_EQ(thrust::count(data.begin(), data.end(), 2), 0);
}

TYPED_TEST(CountTests, TestCount)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(size,
                                                           std::numeric_limits<T>::min(),
                                                           std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;
        
        size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), T(5));
        size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), T(5));
        
        ASSERT_EQ(cpu_result, gpu_result);
    }
}

template <typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};

TYPED_TEST(CountTests, TestCountIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector data(5);
    data[0] = 1; data[1] = 6; data[2] = 1; data[3] = 9; data[4] = 2;

    ASSERT_EQ(thrust::count_if(data.begin(), data.end(), greater_than_five<T>()), 2);

}

TYPED_TEST(CountTests, TestCountIf)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(size,
                                                           std::numeric_limits<T>::min(),
                                                           std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;
        size_t cpu_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
        size_t gpu_result = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

        ASSERT_EQ(cpu_result, gpu_result);
    }
}

TYPED_TEST(CountTests, TestCountFromConstIteratorSimple)
{
    using Vector = typename TestFixture::input_type;

    Vector data(5);
    data[0] = 1; data[1] = 1; data[2] = 0; data[3] = 0; data[4] = 1;

    ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 0), 2);
    ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 1), 3);
    ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 2), 0);
}

template<typename InputIterator, typename EqualityComparable>
int count(my_system &system, InputIterator, InputIterator, EqualityComparable x)
{
    system.validate_dispatch();
    return x;
}

TEST(CountTests, TestCountDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::count(sys,
                  vec.begin(),
                  vec.end(),
                  13);

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator, typename EqualityComparable>
int count(my_tag, InputIterator, InputIterator, EqualityComparable x)
{
    return x;
}

TEST(CountTests, TestCountDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    int result = thrust::count(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.end()),
                               13);

    ASSERT_EQ(13, result);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
