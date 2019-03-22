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

#include <thrust/sort.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include "test_utils.hpp"
#include "test_assertions.hpp"

template<class InputType> struct Params
{
    using input_type = InputType;
};

template<class Params> class IsSortedUntilTests : public ::testing::Test
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
> TestParams;

TYPED_TEST_CASE(IsSortedUntilTests, TestParams);

template<class Params> class IsSortedUntilVectorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<thrust::device_vector<short>>,
        Params<thrust::device_vector<int>>,
        Params<thrust::host_vector<short>>,
        Params<thrust::host_vector<int>>
> VectorParams;

TYPED_TEST_CASE(IsSortedUntilVectorTests, VectorParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef typename Vector::iterator Iterator;

    Vector v(4);
    v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

    Iterator first = v.begin();

    Iterator last  = v.begin() + 0;
    Iterator ref = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 1;
    ref = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 2;
    ref = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 3;
    ref = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 4;
    ref = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 3;
    ref = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

    last = v.begin() + 4;
    ref = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

    last = v.begin() + 1;
    ref = v.begin() + 1;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

    last = v.begin() + 4;
    ref = v.begin() + 1;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

    first = v.begin() + 2;
    last = v.begin() + 4;
    ref = v.begin() + 4;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));
}

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntilRepeatedElements)
{
  using Vector = typename TestFixture::input_type;
  Vector v(10);

  v[0] = 0;
  v[1] = 1;
  v[2] = 1;
  v[3] = 2;
  v[4] = 3;
  v[5] = 4;
  v[6] = 5;
  v[7] = 5;
  v[8] = 5;
  v[9] = 6;

  ASSERT_EQ_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntil)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    const size_t n = (1 << 16) + 13;

    Vector v = get_random_data<T>(n,
                                  std::numeric_limits<T>::min(),
                                  std::numeric_limits<T>::max());

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ_QUIET(v.begin() + 1, thrust::is_sorted_until(v.begin(), v.end()));

    thrust::sort(v.begin(), v.end());

    ASSERT_EQ_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}


template<typename ForwardIterator>
ForwardIterator is_sorted_until(my_system &system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

TEST(IsSortedUntilTests, TestIsSortedUntilExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::is_sorted_until(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}


template<typename ForwardIterator>
ForwardIterator is_sorted_until(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

TEST(IsSortedUntilTests, TestIsSortedUntilImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::is_sorted_until(thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
