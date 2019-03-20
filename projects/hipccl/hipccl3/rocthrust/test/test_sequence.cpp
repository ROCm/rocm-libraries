// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(SequenceTests, FullTestsParams);
TESTS_DEFINE(PrimitiveSequenceTests, NumericalTestsParams);

TEST(SequenceTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

template<typename ForwardIterator>
void sequence(my_system &system, ForwardIterator, ForwardIterator)
{
    system.validate_dispatch();
}

TEST(SequenceTests, SequenceDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::sequence(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator>
void sequence(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
}

TEST(SequenceTests, SequenceDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::sequence(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SequenceTests, SequenceSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(5);

    thrust::sequence(v.begin(), v.end());

    ASSERT_EQ(v[0], 0);
    ASSERT_EQ(v[1], 1);
    ASSERT_EQ(v[2], 2);
    ASSERT_EQ(v[3], 3);
    ASSERT_EQ(v[4], 4);

    thrust::sequence(v.begin(), v.end(), (T)10);

    ASSERT_EQ(v[0], 10);
    ASSERT_EQ(v[1], 11);
    ASSERT_EQ(v[2], 12);
    ASSERT_EQ(v[3], 13);
    ASSERT_EQ(v[4], 14);
    
    thrust::sequence(v.begin(), v.end(), (T)10, (T)2);

    ASSERT_EQ(v[0], 10);
    ASSERT_EQ(v[1], 12);
    ASSERT_EQ(v[2], 14);
    ASSERT_EQ(v[3], 16);
    ASSERT_EQ(v[4], 18);
}

TYPED_TEST(PrimitiveSequenceTests, SequencesWithVariableLength)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    T error_margin = (T) 0.01;
    for(auto size : sizes)
    {
	  size_t step_size = (size * 0.01) + 1;

      thrust::host_vector<T>   h_data(size);
      thrust::device_vector<T> d_data(size);

      thrust::sequence(h_data.begin(), h_data.end());
      thrust::sequence(d_data.begin(), d_data.end());
      
      thrust::host_vector<T>   h_data_d = d_data;
      for (size_t i = 0; i < size; i += step_size)
        ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
	
      thrust::sequence(h_data.begin(), h_data.end(), T(10));
      thrust::sequence(d_data.begin(), d_data.end(), T(10));
      
      h_data_d = d_data;
      for (size_t i = 0; i < size; i += step_size)
        ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);

      thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
      thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));
      
      h_data_d = d_data;
      for (size_t i = 0; i < size; i += step_size)
        ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
      
      thrust::sequence(h_data.begin(), h_data.end(), size_t(10), size_t(2));
      thrust::sequence(d_data.begin(), d_data.end(), size_t(10), size_t(2));
      
      h_data_d = d_data;
      for (size_t i = 0; i < size; i += step_size)
        ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
    }
}

TYPED_TEST(PrimitiveSequenceTests, SequenceToDiscardIterator)
{
	using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data(size);
        thrust::device_vector<T> d_data(size);

        thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                        thrust::discard_iterator<thrust::device_system_tag>(13),
                        T(10),
                        T(2));
    }
    // nothing to check -- just make sure it compiles
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
