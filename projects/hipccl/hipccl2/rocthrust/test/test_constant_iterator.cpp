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
///////////////////// This includes come from test_sort.cpp
#include <thrust/memory.h>
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
/////////////////////

///////////////////////////These ones are from constant_iterator.cu
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
////////////////////////

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include<iostream>


template< class InputType >
struct Params
{
    using input_type = InputType;
};

template<class Params>
class ConstantIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};


typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> ConstantIteratorTestsParams;


TYPED_TEST_CASE(ConstantIteratorTests, ConstantIteratorTestsParams);


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TEST(ConstantIteratorTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(ConstantIteratorTests, ConstantIteratorConstructFromConvertibleSystem)
{
  using namespace thrust;

  constant_iterator<int> default_system(13);

  constant_iterator<int, use_default, host_system_tag> host_system = default_system;
  ASSERT_EQ(*default_system, *host_system);

  constant_iterator<int, use_default, device_system_tag> device_system = default_system;
  ASSERT_EQ(*default_system, *device_system);
}


TEST(ConstantIteratorTests, ConstantIteratorIncrement)
{
    using namespace thrust;

    constant_iterator<int> lhs(0,0);
    constant_iterator<int> rhs(0,0);

    ASSERT_EQ(0, lhs - rhs);

    lhs++;
    ASSERT_EQ(1, lhs - rhs);
    
    lhs++;
    lhs++;
    ASSERT_EQ(3, lhs - rhs);

    lhs += 5;
    ASSERT_EQ(8, lhs - rhs);

    lhs -= 10;
    ASSERT_EQ(-2, lhs - rhs);
}

TEST(ConstantIteratorTests, ConstantIteratorComparison)
{
    using namespace thrust;

    constant_iterator<int> iter1(0);
    constant_iterator<int> iter2(0);

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);

    iter1++;
    
    ASSERT_EQ(1, iter1 - iter2);
    ASSERT_EQ(false, iter1 == iter2);
   
    iter2++;

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);
}


TEST(ConstantIteratorTests, TestMakeConstantIterator)
{
    using namespace thrust;

    // test one argument version
    constant_iterator<int> iter0 = make_constant_iterator<int>(13);

    ASSERT_EQ(13, *iter0);

    // test two argument version
    constant_iterator<int,int> iter1 = make_constant_iterator<int,int>(13, 7);

    ASSERT_EQ(13, *iter1);
    ASSERT_EQ(7, iter1 - iter0);
}


TYPED_TEST(ConstantIteratorTests, MakeConstantIterator)
{
    using namespace thrust;
    using Vector = typename TestFixture::input_type;

    typedef constant_iterator<int> ConstIter;

    Vector result(4);

    ConstIter first = make_constant_iterator<int>(7);
    ConstIter last  = first + result.size();
    thrust::copy(first, last, result.begin());

    ASSERT_EQ(7, result[0]);
    ASSERT_EQ(7, result[1]);
    ASSERT_EQ(7, result[2]);
    ASSERT_EQ(7, result[3]);
};


TYPED_TEST(ConstantIteratorTests, ConstantIteratorTransform)
{
    using namespace thrust;
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef constant_iterator<T> ConstIter;

    Vector result(4);

    ConstIter first1 = make_constant_iterator<T>(7);
    ConstIter last1  = first1 + result.size();
    ConstIter first2 = make_constant_iterator<T>(3);

    thrust::transform(first1, last1, result.begin(), thrust::negate<T>());

    ASSERT_EQ(-7, result[0]);
    ASSERT_EQ(-7, result[1]);
    ASSERT_EQ(-7, result[2]);
    ASSERT_EQ(-7, result[3]);
    
    thrust::transform(first1, last1, first2, result.begin(), thrust::plus<T>());

    ASSERT_EQ(10, result[0]);
    ASSERT_EQ(10, result[1]);
    ASSERT_EQ(10, result[2]);
    ASSERT_EQ(10, result[3]);
};

/*
TYPED_TEST(ConstantIteratorTests, ConstantIteratorReduce)
{
    using namespace thrust;
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;
  
    typedef constant_iterator<T> ConstIter;

    ConstIter first = make_constant_iterator<T>(7);
    ConstIter last  = first + 4;

    T sum = thrust::reduce(first, last);

    ASSERT_EQ(sum, 4 * 7);
};
*/
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC