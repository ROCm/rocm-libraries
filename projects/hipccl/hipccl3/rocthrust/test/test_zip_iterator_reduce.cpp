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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

// Google Test
#include <gtest/gtest.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"
#include "test_assertions.hpp"

template<class InputType>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class ZipIteratorReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<char>,
        Params<signed char>,
        Params<unsigned char>,
        Params<short>,
        Params<unsigned short>,
        Params<int>,
        Params<unsigned int>,
        Params<long>,
        Params<unsigned long>,
        Params<long long>,
        Params<unsigned long long>
> IntegralTypesParams;

TYPED_TEST_CASE(ZipIteratorReduceTests, IntegralTypesParams);

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

TYPED_TEST(ZipIteratorReduceTests, TestZipIteratorReduce)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
      {
        using namespace thrust;

        thrust::host_vector<T> h_data0 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_data1 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        device_vector<T> d_data0 = h_data0;
        device_vector<T> d_data1 = h_data1;

        using Tuple = tuple<T,T>;

        // run on host
        Tuple h_result = reduce( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                                 make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                                 make_tuple<T,T>(0,0),
                                 TuplePlus<Tuple>());

        // run on device
        Tuple d_result = reduce( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                                 make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                                 make_tuple<T,T>(0,0),
                                 TuplePlus<Tuple>());

        ASSERT_EQ(get<0>(h_result), get<0>(d_result));
        ASSERT_EQ(get<1>(h_result), get<1>(d_result));
      }
}



