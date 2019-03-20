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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(ZipIteratorScanVariablesTests, NumericalTestsParams);

template<typename Tuple>
struct TuplePlus
{
  __host__ __device__
  Tuple operator()(Tuple x, Tuple y) const
  {
    using namespace thrust;
    return make_tuple(get<0>(x) + get<0>(y),
                      get<1>(x) + get<1>(y));
  }
}; // end SumTuple


TYPED_TEST(ZipIteratorScanVariablesTests, TestZipIteratorScan)
{
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    using namespace thrust;

    thrust::host_vector<T> h_data0 = get_random_data<T>(size,
                                                        std::numeric_limits<T>::min(),
                                                        std::numeric_limits<T>::max());
    thrust::host_vector<T> h_data1 = get_random_data<T>(size,
                                                        std::numeric_limits<T>::min(),
                                                        std::numeric_limits<T>::max());

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;

    typedef tuple<T,T> Tuple;

    host_vector<Tuple>   h_result(size);
    device_vector<Tuple> d_result(size);

    // inclusive_scan (tuple output)
    inclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                    make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                    h_result.begin(),
                    TuplePlus<Tuple>());
    inclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                    make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                    d_result.begin(),
                    TuplePlus<Tuple>());
    ASSERT_EQ_QUIET(h_result, d_result);
   
    // exclusive_scan (tuple output)
    exclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                    make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                    h_result.begin(),
                    make_tuple<T,T>(0,0),
                    TuplePlus<Tuple>());
    exclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                    make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                    d_result.begin(),
                    make_tuple<T,T>(0,0),
                    TuplePlus<Tuple>());
    ASSERT_EQ_QUIET(h_result, d_result);

    host_vector<T>   h_result0(size);
    host_vector<T>   h_result1(size);
    device_vector<T> d_result0(size);
    device_vector<T> d_result1(size);
    
    // inclusive_scan (zip_iterator output)
    inclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                    make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                    make_zip_iterator(make_tuple(h_result0.begin(), h_result1.begin())),
                    TuplePlus<Tuple>());
    inclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                    make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                    make_zip_iterator(make_tuple(d_result0.begin(), d_result1.begin())),
                    TuplePlus<Tuple>());
    ASSERT_EQ_QUIET(h_result0, d_result0);
    ASSERT_EQ_QUIET(h_result1, d_result1);
    
    // exclusive_scan (zip_iterator output)
    exclusive_scan( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                    make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                    make_zip_iterator(make_tuple(h_result0.begin(), h_result1.begin())),
                    make_tuple<T,T>(0,0),
                    TuplePlus<Tuple>());
    exclusive_scan( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                    make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                    make_zip_iterator(make_tuple(d_result0.begin(), d_result1.begin())),
                    make_tuple<T,T>(0,0),
                    TuplePlus<Tuple>());
    ASSERT_EQ_QUIET(h_result0, d_result0);
    ASSERT_EQ_QUIET(h_result1, d_result1);
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
