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

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <backend/cuda/testframework.h>
#endif

template<class InputType> struct Params
{
    using input_type = InputType;
};

template<class Params> class ZipIteratorReduceByKeyTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<unsigned short>,
        Params<unsigned int>,
        Params<unsigned long>,
        Params<unsigned long long>
> UnsignedIntegralTypesParams;

TYPED_TEST_CASE(ZipIteratorReduceByKeyTests, UnsignedIntegralTypesParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<typename Tuple> struct TuplePlus
{
    __host__ __device__
    Tuple operator()(Tuple x, Tuple y) const
    {
        using namespace thrust;
        return make_tuple(
                get<0>(x) + get<0>(y), get<1>(x) + get<1>(y));
    }
}; // end TuplePlus


TYPED_TEST(ZipIteratorReduceByKeyTests, TestZipIteratorReduceByKey)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for (auto size : sizes)
    {
        using namespace thrust;

        thrust::host_vector<T> h_data0 = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_data1 = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_data2 = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        device_vector<T> d_data0 = h_data0;
        device_vector<T> d_data1 = h_data1;
        device_vector<T> d_data2 = h_data2;

        typedef tuple<T, T> Tuple;

        // integer key, tuple value
        {
            host_vector<T> h_data3(size, 0);
            host_vector<T> h_data4(size, 0);
            host_vector<T> h_data5(size, 0);
            device_vector<T> d_data3(size, 0);
            device_vector<T> d_data4(size, 0);
            device_vector<T> d_data5(size, 0);

            // run on host
            reduce_by_key(
                    h_data0.begin(), h_data0.end(),
                    make_zip_iterator(make_tuple(h_data1.begin(), h_data2.begin())),
                    h_data3.begin(),
                    make_zip_iterator(make_tuple(h_data4.begin(), h_data5.begin())),
                    equal_to<T>(), TuplePlus<Tuple>());

            // run on device
            reduce_by_key(
                    d_data0.begin(), d_data0.end(),
                    make_zip_iterator(make_tuple(d_data1.begin(), d_data2.begin())),
                    d_data3.begin(),
                    make_zip_iterator(make_tuple(d_data4.begin(), d_data5.begin())),
                    equal_to<T>(), TuplePlus<Tuple>());

            ASSERT_EQ(h_data3, d_data3);
            ASSERT_EQ(h_data4, d_data4);
            ASSERT_EQ(h_data5, d_data5);
        }
        // The tests below get miscompiled on Tesla hw for 8b types

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        if(const CUDATestDriver *driver = dynamic_cast<const CUDATestDriver*>(&UnitTestDriver::s_driver()))
        {
          if(typeid(T) == typeid(unittest::uint8_t) && driver->current_device_architecture() < 200)
          {
            KNOWN_FAILURE;
          } // end if
        } // end if
#endif

        // tuple key, tuple value
        {
            host_vector<T> h_data3(size, 0);
            host_vector<T> h_data4(size, 0);
            host_vector<T> h_data5(size, 0);
            host_vector<T> h_data6(size, 0);
            device_vector<T> d_data3(size, 0);
            device_vector<T> d_data4(size, 0);
            device_vector<T> d_data5(size, 0);
            device_vector<T> d_data6(size, 0);

            // run on host
            reduce_by_key(
                    make_zip_iterator(make_tuple(h_data0.begin(), h_data0.begin())),
                    make_zip_iterator(make_tuple(h_data0.end(), h_data0.end())),
                    make_zip_iterator(make_tuple(h_data1.begin(), h_data2.begin())),
                    make_zip_iterator(make_tuple(h_data3.begin(), h_data4.begin())),
                    make_zip_iterator(make_tuple(h_data5.begin(), h_data6.begin())),
                    equal_to<Tuple>(), TuplePlus<Tuple>());

            // run on device
            reduce_by_key(
                    make_zip_iterator(make_tuple(d_data0.begin(), d_data0.begin())),
                    make_zip_iterator(make_tuple(d_data0.end(), d_data0.end())),
                    make_zip_iterator(make_tuple(d_data1.begin(), d_data2.begin())),
                    make_zip_iterator(make_tuple(d_data3.begin(), d_data4.begin())),
                    make_zip_iterator(make_tuple(d_data5.begin(), d_data6.begin())),
                    equal_to<Tuple>(), TuplePlus<Tuple>());

            ASSERT_EQ(h_data3, d_data3);
            ASSERT_EQ(h_data4, d_data4);
            ASSERT_EQ(h_data5, d_data5);
            ASSERT_EQ(h_data6, d_data6);
        }
    }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
