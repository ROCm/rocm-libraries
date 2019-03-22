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
#include <thrust/sort.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

typedef ::testing::Types<
    Params<int8_t>,
    // TODO: enable when we solved: issue 122
    //Params<int16_t>,
    Params<int32_t>
> TestParams;

TESTS_DEFINE(ZipIteratorStableSortByKeyTests, TestParams);

TYPED_TEST(ZipIteratorStableSortByKeyTests, TestZipIteratorStableSort)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        using namespace thrust;

        thrust::host_vector<T> h1 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h2 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h3 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h4 = get_random_data<T>(
                  size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        device_vector<T> d1 = h1;
        device_vector<T> d2 = h2;
        device_vector<T> d3 = h3;
        device_vector<T> d4 = h4;

          // sort with (tuple, scalar)
        stable_sort_by_key( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                              make_zip_iterator(make_tuple(h1.end(),   h2.end())),
                              h3.begin() );
        stable_sort_by_key( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                              make_zip_iterator(make_tuple(d1.end(),   d2.end())),
                              d3.begin() );

        ASSERT_EQ_QUIET(h1, d1);
        ASSERT_EQ_QUIET(h2, d2);
        ASSERT_EQ_QUIET(h3, d3);
        ASSERT_EQ_QUIET(h4, d4);

        // sort with (scalar, tuple)
        stable_sort_by_key( h1.begin(),
                              h1.end(),
                              make_zip_iterator(make_tuple(h3.begin(), h4.begin())) );
        stable_sort_by_key( d1.begin(),
                              d1.end(),
                              make_zip_iterator(make_tuple(d3.begin(), d4.begin())) );

        // sort with (tuple, tuple)
        stable_sort_by_key( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                            make_zip_iterator(make_tuple(h1.end(),   h2.end())),
                            make_zip_iterator(make_tuple(h3.begin(), h4.begin())) );
        stable_sort_by_key( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                            make_zip_iterator(make_tuple(d1.end(),   d2.end())),
                            make_zip_iterator(make_tuple(d3.begin(), d4.begin())) );

        ASSERT_EQ_QUIET(h1, d1);
        ASSERT_EQ_QUIET(h2, d2);
        ASSERT_EQ_QUIET(h3, d3);
        ASSERT_EQ_QUIET(h4, d4);
    }
};

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
