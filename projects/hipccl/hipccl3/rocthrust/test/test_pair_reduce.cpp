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
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"
#include "test_assertations.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class PairReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
  Params<short>,
  Params<int>,
  Params<long long>,
  Params<unsigned short>,
  Params<unsigned int>,
  Params<unsigned long long>,
  Params<float>,
  Params<double>
> PairReduceTestsParams;

TYPED_TEST_CASE(PairReduceTests, PairReduceTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


struct add_pairs
{
  template <typename Pair1, typename Pair2>
  __host__ __device__
    Pair1 operator()(const Pair1 &x, const Pair2 &y)
  {
    return thrust::make_pair(x.first + y.first, x.second + y.second);
  } // end operator()
}; // end add_pairs

TYPED_TEST(PairReduceTests, TestPairReduce)
{
  using T = typename TestFixture::input_type;
  using P = thrust::pair<T,T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> h_p1 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());;

    thrust::host_vector<T> h_p2 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());;

    thrust::host_vector<P>   h_pairs(size);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;

    P init = thrust::make_pair(13,13);

    // reduce on the host
    P h_result = thrust::reduce(h_pairs.begin(), h_pairs.end(), init, add_pairs());

    // reduce on the device
    P d_result = thrust::reduce(d_pairs.begin(), d_pairs.end(), init, add_pairs());

    ASSERT_EQ_QUIET(h_result, d_result);
  }
}


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
