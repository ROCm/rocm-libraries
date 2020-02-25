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

#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetUnionByKeyDescendingTests, FullTestsParams);
TESTS_DEFINE(SetUnionByKeyDescendingPrimitiveTests, NumericalTestsParams);

TYPED_TEST(SetUnionByKeyDescendingTests, TestSetUnionByKeyDescendingSimple)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;
    using T        = typename Vector::value_type;

    Vector a_key(3), b_key(4);
    Vector a_val(3), b_val(4);

    a_key[0] = 4;
    a_key[1] = 2;
    a_key[2] = 0;
    a_val[0] = 0;
    a_val[1] = 0;
    a_val[2] = 0;

    b_key[0] = 4;
    b_key[1] = 3;
    b_key[2] = 3;
    b_key[3] = 0;
    b_val[0] = 1;
    b_val[1] = 1;
    b_val[2] = 1;
    b_val[3] = 1;

    Vector ref_key(5), ref_val(5);
    ref_key[0] = 4;
    ref_key[1] = 3;
    ref_key[2] = 3;
    ref_key[3] = 2;
    ref_key[4] = 0;
    ref_val[0] = 0;
    ref_val[1] = 1;
    ref_val[2] = 1;
    ref_val[3] = 0;
    ref_val[4] = 0;

    Vector result_key(5), result_val(5);

    thrust::pair<Iterator, Iterator> end = thrust::set_union_by_key(a_key.begin(),
                                                                    a_key.end(),
                                                                    b_key.begin(),
                                                                    b_key.end(),
                                                                    a_val.begin(),
                                                                    b_val.begin(),
                                                                    result_key.begin(),
                                                                    result_val.begin(),
                                                                    thrust::greater<T>());

    EXPECT_EQ(result_key.end(), end.first);
    EXPECT_EQ(result_val.end(), end.second);
    ASSERT_EQ(ref_key, result_key);
    ASSERT_EQ(ref_val, result_val);
}

TYPED_TEST(SetUnionByKeyDescendingPrimitiveTests, TestSetUnionByKeyDescendingEquivalentRanges)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<T> temp = get_random_data<T>(
                2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

            thrust::host_vector<T> h_a_key(temp.begin(), temp.begin() + size);
            thrust::host_vector<T> h_b_key(temp.begin() + size, temp.end());

            thrust::sort(h_a_key.begin(), h_a_key.end(), thrust::greater<T>());
            thrust::sort(h_b_key.begin(), h_b_key.end(), thrust::greater<T>());

            thrust::host_vector<T> h_a_val = get_random_data<T>(
                h_a_key.size(),
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed_value + seed_value_addition
            );
            thrust::host_vector<T> h_b_val = get_random_data<T>(
                h_b_key.size(),
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed_value + 2 * seed_value_addition
            );

            thrust::device_vector<T> d_a_key = h_a_key;
            thrust::device_vector<T> d_b_key = h_b_key;

            thrust::device_vector<T> d_a_val = h_a_val;
            thrust::device_vector<T> d_b_val = h_b_val;

            size_t                   max_size = h_a_key.size() + h_b_key.size();
            thrust::host_vector<T>   h_result_key(max_size), h_result_val(max_size);
            thrust::device_vector<T> d_result_key(max_size), d_result_val(max_size);

            thrust::pair<typename thrust::host_vector<T>::iterator,
                         typename thrust::host_vector<T>::iterator>
                h_end;

            thrust::pair<typename thrust::device_vector<T>::iterator,
                         typename thrust::device_vector<T>::iterator>
                d_end;

            h_end = thrust::set_union_by_key(h_a_key.begin(),
                                             h_a_key.end(),
                                             h_b_key.begin(),
                                             h_b_key.end(),
                                             h_a_val.begin(),
                                             h_b_val.begin(),
                                             h_result_key.begin(),
                                             h_result_val.begin(),
                                             thrust::greater<T>());
            h_result_key.erase(h_end.first, h_result_key.end());
            h_result_val.erase(h_end.second, h_result_val.end());

            d_end = thrust::set_union_by_key(d_a_key.begin(),
                                             d_a_key.end(),
                                             d_b_key.begin(),
                                             d_b_key.end(),
                                             d_a_val.begin(),
                                             d_b_val.begin(),
                                             d_result_key.begin(),
                                             d_result_val.begin(),
                                             thrust::greater<T>());
            d_result_key.erase(d_end.first, d_result_key.end());
            d_result_val.erase(d_end.second, d_result_val.end());

            ASSERT_EQ(h_result_key, d_result_key);
            ASSERT_EQ(h_result_val, d_result_val);
        }
    }
}

/*
TYPED_TEST(SetUnionByKeyDescendingTests, TestSetUnionByKeyDescendingSimple)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

  Vector a_key(3), b_key(4);
  Vector a_val(3), b_val(4);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4;
  b_val[0] = 1; b_val[1] = 1; b_val[2] = 1; b_val[3] = 1;

  Vector ref_key(5), ref_val(5);
  ref_key[0] = 0; ref_key[1] = 2; ref_key[2] = 3; ref_key[3] = 3; ref_key[4] = 4;
  ref_val[0] = 0; ref_val[1] = 0; ref_val[2] = 1; ref_val[3] = 1; ref_val[4] = 0;

  Vector result_key(5), result_val(5);

  thrust::pair<Iterator,Iterator> end =
    thrust::set_union_by_key(a_key.begin(), a_key.end(),
                             b_key.begin(), b_key.end(),
                             a_val.begin(),
                             b_val.begin(),
                             result_key.begin(),
                             result_val.begin());

  EXPECT_EQ(result_key.end(), end.first);
  EXPECT_EQ(result_val.end(), end.second);
  ASSERT_EQ(ref_key, result_key);
  ASSERT_EQ(ref_val, result_val);
}

TYPED_TEST(SetUnionByKeyDescendingPrimitiveTests, TestSetUnionByKeyDescending)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
{
unsigned int seed_value  = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        thrust::host_vector<T> temp = get_random_data<unsigned short int>(
            size,
            0,
            255, seed_value);

        thrust::host_vector<T> random_keys = get_random_data<unsigned short int>(
            size,
            0,
            255, seed_value);
        thrust::host_vector<T> random_vals = get_random_data<unsigned short int>(
            size,
            0,
            255, seed_value);

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

            size_t max_size = h_a_keys.size() + h_b_keys.size();

            thrust::host_vector<T> h_result_keys(max_size);
            thrust::host_vector<T> h_result_vals(max_size);

            thrust::device_vector<T> d_result_keys(max_size);
            thrust::device_vector<T> d_result_vals(max_size);


            thrust::pair<
            typename thrust::host_vector<T>::iterator,
            typename thrust::host_vector<T>::iterator
            > h_end;

            thrust::pair<
            typename thrust::device_vector<T>::iterator,
            typename thrust::device_vector<T>::iterator
            > d_end;


            h_end = thrust::set_union_by_key(h_a_keys.begin(), h_a_keys.end(),
                                            h_b_keys.begin(), h_b_keys.end(),
                                            h_a_vals.begin(),
                                            h_b_vals.begin(),
                                            h_result_keys.begin(),
                                            h_result_vals.begin());
            h_result_keys.erase(h_end.first, h_result_keys.end());
            h_result_vals.erase(h_end.second, h_result_vals.end());

            d_end = thrust::set_union_by_key(d_a_keys.begin(), d_a_keys.end(),
                                            d_b_keys.begin(), d_b_keys.end(),
                                            d_a_vals.begin(),
                                            d_b_vals.begin(),
                                            d_result_keys.begin(),
                                            d_result_vals.begin());
            d_result_keys.erase(d_end.first, d_result_keys.end());
            d_result_vals.erase(d_end.second, d_result_vals.end());

            ASSERT_EQ(h_result_keys, d_result_keys);
            ASSERT_EQ(h_result_vals, d_result_vals);
        }
    }
}
}


TYPED_TEST(SetUnionByKeyDescendingPrimitiveTests, TestSetUnionByKeyDescendingEquivalentRanges)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
{
unsigned int seed_value  = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        thrust::host_vector<T> temp = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);


        thrust::host_vector<T> h_a_key = temp;
        thrust::sort(h_a_key.begin(), h_a_key.end());
        thrust::host_vector<T> h_b_key = h_a_key;

        thrust::host_vector<T> h_a_val = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

        thrust::host_vector<T> h_b_val = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

        thrust::device_vector<T> d_a_key = h_a_key;
        thrust::device_vector<T> d_b_key = h_b_key;

        thrust::device_vector<T> d_a_val = h_a_val;
        thrust::device_vector<T> d_b_val = h_b_val;

        size_t max_size = h_a_key.size() + h_b_key.size();

        thrust::host_vector<T>   h_result_key(max_size), h_result_val(max_size);
        thrust::device_vector<T> d_result_key(max_size), d_result_val(max_size);

        thrust::pair<
            typename thrust::host_vector<T>::iterator,
            typename thrust::host_vector<T>::iterator
        > h_end;

        thrust::pair<
            typename thrust::device_vector<T>::iterator,
            typename thrust::device_vector<T>::iterator
        > d_end;

        h_end = thrust::set_union_by_key(h_a_key.begin(), h_a_key.end(),
                                        h_b_key.begin(), h_b_key.end(),
                                        h_a_val.begin(),
                                        h_b_val.begin(),
                                        h_result_key.begin(),
                                        h_result_val.begin());
        h_result_key.erase(h_end.first,  h_result_key.end());
        h_result_val.erase(h_end.second, h_result_val.end());

        d_end = thrust::set_union_by_key(d_a_key.begin(), d_a_key.end(),
                                        d_b_key.begin(), d_b_key.end(),
                                        d_a_val.begin(),
                                        d_b_val.begin(),
                                        d_result_key.begin(),
                                        d_result_val.begin());
        d_result_key.erase(d_end.first,  d_result_key.end());
        d_result_val.erase(d_end.second, d_result_val.end());

        ASSERT_EQ(h_result_key, d_result_key);
        ASSERT_EQ(h_result_val, d_result_val);
    }
}
}

TYPED_TEST(SetUnionByKeyDescendingPrimitiveTests, TestSetUnionByKeyDescendingMultiset)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
{
unsigned int seed_value  = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        thrust::host_vector<T> temp = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

        // restrict elements to [min,13)
        for(typename thrust::host_vector<T>::iterator i = temp.begin();
            i != temp.end();
            ++i)
        {
            int temp = static_cast<int>(*i);
            temp %= 13;
            *i = temp;
        }

        thrust::host_vector<T> h_a_key(temp.begin(), temp.begin() + size);
        thrust::host_vector<T> h_b_key(temp.begin() + size, temp.end());

        thrust::sort(h_a_key.begin(), h_a_key.end());
        thrust::sort(h_b_key.begin(), h_b_key.end());

        thrust::host_vector<T> h_a_val = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);
        thrust::host_vector<T> h_b_val = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

        thrust::device_vector<T> d_a_key = h_a_key;
        thrust::device_vector<T> d_b_key = h_b_key;

        thrust::device_vector<T> d_a_val = h_a_val;
        thrust::device_vector<T> d_b_val = h_b_val;

        size_t max_size = h_a_key.size() + h_b_key.size();
        thrust::host_vector<T>   h_result_key(max_size), h_result_val(max_size);
        thrust::device_vector<T> d_result_key(max_size), d_result_val(max_size);

        thrust::pair<
            typename thrust::host_vector<T>::iterator,
            typename thrust::host_vector<T>::iterator
        > h_end;

        thrust::pair<
            typename thrust::device_vector<T>::iterator,
            typename thrust::device_vector<T>::iterator
        > d_end;

        h_end = thrust::set_union_by_key(h_a_key.begin(), h_a_key.end(),
                                        h_b_key.begin(), h_b_key.end(),
                                        h_a_val.begin(),
                                        h_b_val.begin(),
                                        h_result_key.begin(),
                                        h_result_val.begin());
        h_result_key.erase(h_end.first,  h_result_key.end());
        h_result_val.erase(h_end.second, h_result_val.end());

        d_end = thrust::set_union_by_key(d_a_key.begin(), d_a_key.end(),
                                        d_b_key.begin(), d_b_key.end(),
                                        d_a_val.begin(),
                                        d_b_val.begin(),
                                        d_result_key.begin(),
                                        d_result_val.begin());
        d_result_key.erase(d_end.first,  d_result_key.end());
        d_result_val.erase(d_end.second, d_result_val.end());

        ASSERT_EQ(h_result_key, d_result_key);
        ASSERT_EQ(h_result_val, d_result_val);
    }
}
}*/
