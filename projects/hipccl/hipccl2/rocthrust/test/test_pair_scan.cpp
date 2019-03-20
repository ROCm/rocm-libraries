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
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(PairScanVariablesTests, NumericalTestsParams);

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


TYPED_TEST(PairScanVariablesTests, TestPairScan)
{
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T> h_p1 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
    thrust::host_vector<T> h_p2 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
    thrust::host_vector<P>   h_pairs(size);
    thrust::host_vector<P>   h_output(size);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;
    thrust::device_vector<P> d_output(size);

    P init = thrust::make_pair(13,13);

    // scan with plus
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), add_pairs());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), add_pairs());
    ASSERT_EQ_QUIET(h_output, d_output);

    // scan with maximum (thrust issue #69)
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), thrust::maximum<P>());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), thrust::maximum<P>());
    ASSERT_EQ_QUIET(h_output, d_output);

    // scan with plus
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, add_pairs());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, add_pairs());
    ASSERT_EQ_QUIET(h_output, d_output);
    
    // scan with maximum (thrust issue #69)
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, thrust::maximum<P>());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, thrust::maximum<P>());
    ASSERT_EQ_QUIET(h_output, d_output);
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
