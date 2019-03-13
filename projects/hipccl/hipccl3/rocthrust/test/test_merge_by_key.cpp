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
#include "test_utils.hpp"

// Thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template< class InputType >
struct Params
{
    using input_type = InputType;
};

template<class Params>
class MergeByKeyTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params>
class PrimitiveMergeByKeyTests : public ::testing::Test
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
> MergeByKeyTestsParams;

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> MergeByKeyTestsPrimitiveParams;

TYPED_TEST_CASE(MergeByKeyTests, MergeByKeyTestsParams);
TYPED_TEST_CASE(PrimitiveMergeByKeyTests, MergeByKeyTestsPrimitiveParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TEST(MergeByKeyTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(MergeByKeyTests, MergeByKeySimple)
{
  using Vector = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  Vector a_key(3), a_val(3), b_key(4), b_val(4);

  a_key[0] = 0;  a_key[1] = 2; a_key[2] = 4;
  a_val[0] = 13; a_val[1] = 7; a_val[2] = 42;

  b_key[0] = 0 ; b_key[1] = 3;  b_key[2] = 3; b_key[3] = 4;
  b_val[0] = 42; b_val[1] = 42; b_val[2] = 7; b_val[3] = 13;

  Vector ref_key(7), ref_val(7);
  ref_key[0] = 0; ref_val[0] = 13;
  ref_key[1] = 0; ref_val[1] = 42;
  ref_key[2] = 2; ref_val[2] = 7;
  ref_key[3] = 3; ref_val[3] = 42;
  ref_key[4] = 3; ref_val[4] = 7;
  ref_key[5] = 4; ref_val[5] = 42;
  ref_key[6] = 4; ref_val[6] = 13;

  Vector result_key(7), result_val(7);

  thrust::pair<Iterator,Iterator> ends =
    thrust::merge_by_key(a_key.begin(), a_key.end(),
                         b_key.begin(), b_key.end(),
                         a_val.begin(), b_val.begin(),
                         result_key.begin(),
                         result_val.begin());
  
  EXPECT_EQ(result_key.end(), ends.first);
  EXPECT_EQ(result_val.end(), ends.second);
  ASSERT_EQ(ref_key, result_key);
  ASSERT_EQ(ref_val, result_val);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(my_system &system,
                 InputIterator1,
                 InputIterator1,
                 InputIterator2,
                 InputIterator2,
                 InputIterator3,
                 InputIterator4,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_result, values_result);
}

TEST(MergeByKeyTests, MergeByKeyDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::merge_by_key(sys,
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(my_tag,
                 InputIterator1,
                 InputIterator1,
                 InputIterator2,
                 InputIterator2,
                 InputIterator3,
                 InputIterator4,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  *keys_result = 13;
  return thrust::make_pair(keys_result, values_result);
}

TEST(MergeByKeyTests, MergeByKeyDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::merge_by_key(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveMergeByKeyTests, TestMergeByKeyWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
      thrust::host_vector<T> random_keys = get_random_data<unsigned short int>(size, 0, 255);
      thrust::host_vector<T> random_vals = get_random_data<unsigned short int>(size, 0, 255);

      size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      size_t num_denominators = sizeof(denominators) / sizeof(size_t);

      for(size_t i = 0; i < num_denominators; ++i)
      {
        size_t size_a = size / denominators[i];

        thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
        thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

        thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);
        thrust::host_vector<T> h_b_vals(random_vals.begin() + size_a, random_vals.end());

        thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
        thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

        thrust::device_vector<T> d_a_keys = h_a_keys;
        thrust::device_vector<T> d_b_keys = h_b_keys;

        thrust::device_vector<T> d_a_vals = h_a_vals;
        thrust::device_vector<T> d_b_vals = h_b_vals;

        thrust::host_vector<T> h_result_keys(size);
        thrust::host_vector<T> h_result_vals(size);

        thrust::device_vector<T> d_result_keys(size);
        thrust::device_vector<T> d_result_vals(size);


        thrust::pair<
          typename thrust::host_vector<T>::iterator,
          typename thrust::host_vector<T>::iterator
        > h_end;

        thrust::pair<
          typename thrust::device_vector<T>::iterator,
          typename thrust::device_vector<T>::iterator
        > d_end;


        h_end = thrust::merge_by_key(h_a_keys.begin(), h_a_keys.end(),
                                    h_b_keys.begin(), h_b_keys.end(),
                                    h_a_vals.begin(),
                                    h_b_vals.begin(),
                                    h_result_keys.begin(),
                                    h_result_vals.begin());
        h_result_keys.erase(h_end.first, h_result_keys.end());
        h_result_vals.erase(h_end.second, h_result_vals.end());

        d_end = thrust::merge_by_key(d_a_keys.begin(), d_a_keys.end(),
                                    d_b_keys.begin(), d_b_keys.end(),
                                    d_a_vals.begin(),
                                    d_b_vals.begin(),
                                    d_result_keys.begin(),
                                    d_result_vals.begin());
        d_result_keys.erase(d_end.first, d_result_keys.end());
        d_result_vals.erase(d_end.second, d_result_vals.end());

        thrust::host_vector<T> d_result_keys_h = d_result_keys;
        thrust::host_vector<T> d_result_vals_h = d_result_vals;

        ASSERT_EQ(h_result_keys, d_result_keys_h);
        ASSERT_EQ(h_result_vals, d_result_vals_h);
      }
    }
}

TYPED_TEST(PrimitiveMergeByKeyTests, MergeByKeyToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
		thrust::host_vector<T> h_a_keys = get_random_data<T>( size, 
															  std::numeric_limits<T>::min(), 
															  std::numeric_limits<T>::max());
        thrust::host_vector<T> h_b_keys = get_random_data<T>( size, 
															  std::numeric_limits<T>::min(), 
															  std::numeric_limits<T>::max());

        thrust::host_vector<T> h_a_vals = get_random_data<T>( size, 
															  std::numeric_limits<T>::min(), 
															  std::numeric_limits<T>::max());
        thrust::host_vector<T> h_b_vals = get_random_data<T>( size, 
															  std::numeric_limits<T>::min(), 
															  std::numeric_limits<T>::max());

        thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
        thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

        thrust::device_vector<T> d_a_keys = h_a_keys;
        thrust::device_vector<T> d_b_keys = h_b_keys;

        thrust::device_vector<T> d_a_vals = h_a_vals;
        thrust::device_vector<T> d_b_vals = h_b_vals;

        typedef thrust::pair<
          thrust::discard_iterator<>,
          thrust::discard_iterator<>
        > discard_pair;

        discard_pair h_result = 
          thrust::merge_by_key(h_a_keys.begin(), h_a_keys.end(),
                              h_b_keys.begin(), h_b_keys.end(),
                              h_a_vals.begin(),
                              h_b_vals.begin(),
                              thrust::make_discard_iterator(),
                              thrust::make_discard_iterator());

        discard_pair d_result = 
          thrust::merge_by_key(d_a_keys.begin(), d_a_keys.end(),
                              d_b_keys.begin(), d_b_keys.end(),
                              d_a_vals.begin(),
                              d_b_vals.begin(),
                              thrust::make_discard_iterator(),
                              thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(2 * size);

        ASSERT_EQ(reference, h_result.first);
        ASSERT_EQ(reference, h_result.second);
        ASSERT_EQ(reference, d_result.first);
        ASSERT_EQ(reference, d_result.second);
    }
}

TYPED_TEST(PrimitiveMergeByKeyTests, MergeByKeyDescending)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
		thrust::host_vector<T> random_keys = get_random_data<unsigned short int>(size, 0, 255);
		thrust::host_vector<T> random_vals = get_random_data<unsigned short int>(size, 0, 255);

		size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
		size_t num_denominators = sizeof(denominators) / sizeof(size_t);

		for(size_t i = 0; i < num_denominators; ++i)
		{
			size_t size_a = size / denominators[i];

			thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
			thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

			thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);
			thrust::host_vector<T> h_b_vals(random_vals.begin() + size_a, random_vals.end());

			thrust::stable_sort(h_a_keys.begin(), h_a_keys.end(), thrust::greater<T>());
			thrust::stable_sort(h_b_keys.begin(), h_b_keys.end(), thrust::greater<T>());

			thrust::device_vector<T> d_a_keys = h_a_keys;
			thrust::device_vector<T> d_b_keys = h_b_keys;

			thrust::device_vector<T> d_a_vals = h_a_vals;
			thrust::device_vector<T> d_b_vals = h_b_vals;

			thrust::host_vector<T> h_result_keys(size);
			thrust::host_vector<T> h_result_vals(size);

			thrust::device_vector<T> d_result_keys(size);
			thrust::device_vector<T> d_result_vals(size);


			thrust::pair<
			typename thrust::host_vector<T>::iterator,
			typename thrust::host_vector<T>::iterator
			> h_end;

			thrust::pair<
			typename thrust::device_vector<T>::iterator,
			typename thrust::device_vector<T>::iterator
			> d_end;


			h_end = thrust::merge_by_key(h_a_keys.begin(), h_a_keys.end(),
										h_b_keys.begin(), h_b_keys.end(),
										h_a_vals.begin(),
										h_b_vals.begin(),
										h_result_keys.begin(),
										h_result_vals.begin(),
										thrust::greater<T>());
			h_result_keys.erase(h_end.first, h_result_keys.end());
			h_result_vals.erase(h_end.second, h_result_vals.end());

			d_end = thrust::merge_by_key(d_a_keys.begin(), d_a_keys.end(),
										d_b_keys.begin(), d_b_keys.end(),
										d_a_vals.begin(),
										d_b_vals.begin(),
										d_result_keys.begin(),
										d_result_vals.begin(),
										thrust::greater<T>());
			d_result_keys.erase(d_end.first, d_result_keys.end());
			d_result_vals.erase(d_end.second, d_result_vals.end());

			ASSERT_EQ(h_result_keys, d_result_keys);
			ASSERT_EQ(h_result_vals, d_result_vals);
		}
	}
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
