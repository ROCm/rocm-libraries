/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "test_header.hpp"

#if defined(WIN32) && defined(__HIP__)
typedef ::testing::Types<Params<int16_t>, Params<int32_t>> TestParams;
#else
typedef ::testing::Types<Params<int8_t>, Params<int16_t>, Params<int32_t>> TestParams;
#endif

TESTS_DEFINE(ZipIteratorStableSortByKeyTests, TestParams);

TYPED_TEST(ZipIteratorStableSortByKeyTests, TestZipIteratorStableSort)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h1 = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h2 = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );
            thrust::host_vector<T> h3 = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + 2 * seed_value_addition
            );
            thrust::host_vector<T> h4 = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + 3 *  + seed_value_addition
            );

            thrust::device_vector<T> d1 = h1;
            thrust::device_vector<T> d2 = h2;
            thrust::device_vector<T> d3 = h3;
            thrust::device_vector<T> d4 = h4;

            // sort with (tuple, scalar)
            thrust::stable_sort_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(h1.begin(), h2.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(h1.end(), h2.end())),
                h3.begin());
            thrust::stable_sort_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(d1.begin(), d2.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(d1.end(), d2.end())),
                d3.begin());

            ASSERT_EQ_QUIET(h1, d1);
            ASSERT_EQ_QUIET(h2, d2);
            ASSERT_EQ_QUIET(h3, d3);
            ASSERT_EQ_QUIET(h4, d4);

            // sort with (scalar, tuple)
            thrust::stable_sort_by_key(
                h1.begin(), h1.end(), thrust::make_zip_iterator(thrust::make_tuple(h3.begin(), h4.begin())));
            thrust::stable_sort_by_key(
                d1.begin(), d1.end(), thrust::make_zip_iterator(thrust::make_tuple(d3.begin(), d4.begin())));

            // sort with (tuple, tuple)
            thrust::stable_sort_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(h1.begin(), h2.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(h1.end(), h2.end())),
                thrust::make_zip_iterator(thrust::make_tuple(h3.begin(), h4.begin())));
            thrust::stable_sort_by_key(
                thrust::make_zip_iterator(thrust::make_tuple(d1.begin(), d2.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(d1.end(), d2.end())),
                thrust::make_zip_iterator(thrust::make_tuple(d3.begin(), d4.begin())));

            ASSERT_EQ_QUIET(h1, d1);
            ASSERT_EQ_QUIET(h2, d2);
            ASSERT_EQ_QUIET(h3, d3);
            ASSERT_EQ_QUIET(h4, d4);
        }
    }
}
