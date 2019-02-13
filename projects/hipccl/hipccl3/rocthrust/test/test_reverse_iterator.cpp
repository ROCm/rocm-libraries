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

#include <vector>
#include <list>
#include <limits>
#include <utility>

// Google Test
#include <gtest/gtest.h>
#include "test_utils.hpp"

// Thrust
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

template< class InputType >
struct Params
{
    using input_type = InputType;
};

template<class Params>
class ReverseIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params>
class PrimitiveReverseIteratorTests : public ::testing::Test
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
> ReverseIteratorTestsParams;

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> ReverseIteratorTestsPrimitiveParams;

TYPED_TEST_CASE(ReverseIteratorTests, ReverseIteratorTestsParams);
TYPED_TEST_CASE(PrimitiveReverseIteratorTests, ReverseIteratorTestsPrimitiveParams);

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TEST(ReverseIteratorTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(ReverseIteratorTests, ReverseIteratorCopyConstructor)
{
  thrust::host_vector<int> h_v(1,13);

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter0(h_v.end());
  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter1(h_iter0);

  ASSERT_EQ(h_iter0, h_iter1);
  ASSERT_EQ(*h_iter0, *h_iter1);

  thrust::device_vector<int> d_v(1,13);

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter2(d_v.end());
  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter3(d_iter2);

  ASSERT_EQ(d_iter2, d_iter3);
  ASSERT_EQ(*d_iter2, *d_iter3);
}

TEST(ReverseIteratorTests, ReverseIteratorIncrement)
{
  thrust::host_vector<int> h_v(4);
  thrust::sequence(h_v.begin(), h_v.end());

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter(h_v.end());

  ASSERT_EQ(*h_iter, 3);

  h_iter++;
  ASSERT_EQ(*h_iter, 2);

  h_iter++;
  ASSERT_EQ(*h_iter, 1);

  h_iter++;
  ASSERT_EQ(*h_iter, 0);


  thrust::device_vector<int> d_v(4);
  thrust::sequence(d_v.begin(), d_v.end());

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter(d_v.end());

  ASSERT_EQ(*d_iter, 3);

  d_iter++;
  ASSERT_EQ(*d_iter, 2);

  d_iter++;
  ASSERT_EQ(*d_iter, 1);

  d_iter++;
  ASSERT_EQ(*d_iter, 0);
}


TYPED_TEST(ReverseIteratorTests, ReverseIteratorCopy)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector source(4);
    source[0] = (T)10;
    source[1] = (T)20;
    source[2] = (T)30;
    source[3] = (T)40;

    Vector destination(4,0);
    
    thrust::copy(thrust::make_reverse_iterator(source.end()),
                thrust::make_reverse_iterator(source.begin()),
                destination.begin());

    ASSERT_EQ(destination[0], (T)40);
    ASSERT_EQ(destination[1], (T)30);
    ASSERT_EQ(destination[2], (T)20);
    ASSERT_EQ(destination[3], (T)10);
}

//TODO: Un-comment these tests once the scan is implemented 
/*
TYPED_TEST(PrimitiveReverseIteratorTests, ReverseIteratorExclusiveScanSimple)
{
    using T = typename TestFixture::input_type;
    
    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        T error_margin = (T) 0.01 * size;
        thrust::host_vector<T> h_data(size);
        thrust::sequence(h_data.begin(), h_data.end());

        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(h_data.size());
        thrust::device_vector<T> d_result(d_data.size());
        
        thrust::exclusive_scan(thrust::make_reverse_iterator(h_data.end()),
                                thrust::make_reverse_iterator(h_data.begin()),
                                h_result.begin());

        thrust::exclusive_scan(thrust::make_reverse_iterator(d_data.end()),
                                thrust::make_reverse_iterator(d_data.begin()),
                                d_result.begin());
        
        for (size_t i = 0; i < size; i++)
            ASSERT_NEAR(h_result[i], d_result[i], error_margin);   
    }
}


TYPED_TEST(PrimitiveReverseIteratorTests, ReverseIteratorExclusiveScan)
{
    using T = typename TestFixture::input_type;
    
    const std::vector<size_t> sizes = get_sizes_smaller();
    for(auto size : sizes)
    {
        T error_margin = (T) 0.01 * size;
        thrust::host_vector<T> h_data = get_random_data<T>(size, 0, 10);

        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        thrust::exclusive_scan(thrust::make_reverse_iterator(h_data.end()),
                            thrust::make_reverse_iterator(h_data.begin()),
                            h_result.begin());

        thrust::exclusive_scan(thrust::make_reverse_iterator(d_data.end()),
                            thrust::make_reverse_iterator(d_data.begin()),
                            d_result.begin());

        for (size_t i = 0; i < size; i++)
            ASSERT_NEAR(h_result[i], d_result[i], error_margin);
    }
};*/

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
