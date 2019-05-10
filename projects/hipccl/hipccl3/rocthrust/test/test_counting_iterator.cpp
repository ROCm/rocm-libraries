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

#include <thrust/binary_search.h>
#include <thrust/detail/cstdint.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TEST(CountingIteratorTests, TestCountingIteratorCopyConstructor)
{
    thrust::counting_iterator<int> iter0(100);

    thrust::counting_iterator<int> iter1(iter0);

    ASSERT_EQ(iter0, iter1);
    ASSERT_EQ(*iter0, *iter1);

    // construct from related space
    thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
    ASSERT_EQ(*iter0, *h_iter);

    thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
    ASSERT_EQ(*iter0, *d_iter);
}

TEST(CountingIteratorTests, TestCountingIteratorIncrement)
{
    thrust::counting_iterator<int> iter(0);

    ASSERT_EQ(*iter, 0);

    iter++;

    ASSERT_EQ(*iter, 1);

    iter++;
    iter++;

    ASSERT_EQ(*iter, 3);

    iter += 5;

    ASSERT_EQ(*iter, 8);

    iter -= 10;

    ASSERT_EQ(*iter, -2);
}

TEST(CountingIteratorTests, TestCountingIteratorComparison)
{
    thrust::counting_iterator<int> iter1(0);
    thrust::counting_iterator<int> iter2(0);

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);

    iter1++;

    ASSERT_EQ(iter1 - iter2, 1);
    ASSERT_EQ(iter1 == iter2, false);

    iter2++;

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);

    iter1 += 100;
    iter2 += 100;

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);
}

TEST(CountingIteratorTests, TestCountingIteratorFloatComparison)
{
    thrust::counting_iterator<float> iter1(0);
    thrust::counting_iterator<float> iter2(0);

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 < iter2, false);
    ASSERT_EQ(iter2 < iter1, false);

    iter1++;

    ASSERT_EQ(iter1 - iter2, 1);
    ASSERT_EQ(iter1 == iter2, false);
    ASSERT_EQ(iter2 < iter1, true);
    ASSERT_EQ(iter1 < iter2, false);

    iter2++;

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 < iter2, false);
    ASSERT_EQ(iter2 < iter1, false);

    iter1 += 100;
    iter2 += 100;

    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 < iter2, false);
    ASSERT_EQ(iter2 < iter1, false);

    thrust::counting_iterator<float> iter3(0);
    thrust::counting_iterator<float> iter4(0.5);

    ASSERT_EQ(iter3 - iter4, 0);
    ASSERT_EQ(iter3 == iter4, true);
    ASSERT_EQ(iter3 < iter4, false);
    ASSERT_EQ(iter4 < iter3, false);

    iter3++; // iter3 = 1.0, iter4 = 0.5

    ASSERT_EQ(iter3 - iter4, 0);
    ASSERT_EQ(iter3 == iter4, true);
    ASSERT_EQ(iter3 < iter4, false);
    ASSERT_EQ(iter4 < iter3, false);

    iter4++; // iter3 = 1.0, iter4 = 1.5

    ASSERT_EQ(iter3 - iter4, 0);
    ASSERT_EQ(iter3 == iter4, true);
    ASSERT_EQ(iter3 < iter4, false);
    ASSERT_EQ(iter4 < iter3, false);

    iter4++; // iter3 = 1.0, iter4 = 2.5

    ASSERT_EQ(iter3 - iter4, -1);
    ASSERT_EQ(iter4 - iter3, 1);
    ASSERT_EQ(iter3 == iter4, false);
    ASSERT_EQ(iter3 < iter4, true);
    ASSERT_EQ(iter4 < iter3, false);
}

TEST(CountingIteratorTests, TestCountingIteratorDistance)
{
    thrust::counting_iterator<int> iter1(0);
    thrust::counting_iterator<int> iter2(5);

    ASSERT_EQ(thrust::distance(iter1, iter2), 5);

    iter1++;

    ASSERT_EQ(thrust::distance(iter1, iter2), 4);

    iter2 += 100;

    ASSERT_EQ(thrust::distance(iter1, iter2), 104);

    iter2 += 1000;

    ASSERT_EQ(thrust::distance(iter1, iter2), 1104);
}

TEST(CountingIteratorTests, TestCountingIteratorUnsignedType)
{
    thrust::counting_iterator<unsigned int> iter0(0);
    thrust::counting_iterator<unsigned int> iter1(5);

    ASSERT_EQ(iter1 - iter0, 5);
    ASSERT_EQ(iter0 - iter1, -5);
    ASSERT_EQ(iter0 != iter1, true);
    ASSERT_EQ(iter0 < iter1, true);
    ASSERT_EQ(iter1 < iter0, false);
}

TEST(CountingIteratorTests, TestCountingIteratorLowerBound)
{
    size_t       n = 10000;
    const size_t M = 100;

    thrust::host_vector<unsigned int> h_data = get_random_data<unsigned int>(
        n, std::numeric_limits<unsigned int>::min(), std::numeric_limits<unsigned int>::max());
    for(unsigned int i = 0; i < n; ++i)
        h_data[i] %= M;

    thrust::sort(h_data.begin(), h_data.end());

    thrust::device_vector<unsigned int> d_data = h_data;

    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::counting_iterator<unsigned int> search_end(M);

    thrust::host_vector<unsigned int>   h_result(M);
    thrust::device_vector<unsigned int> d_result(M);

    thrust::lower_bound(h_data.begin(), h_data.end(), search_begin, search_end, h_result.begin());

    thrust::lower_bound(d_data.begin(), d_data.end(), search_begin, search_end, d_result.begin());

    ASSERT_EQ(h_result, d_result);
}

TEST(CountingIteratorTests, TestCountingIteratorDifference)
{
    using Iterator   = typename thrust::counting_iterator<thrust::detail::uint64_t>;
    using Difference = typename thrust::iterator_difference<Iterator>::type;

    Difference diff = std::numeric_limits<thrust::detail::uint32_t>::max() + 1;

    Iterator first(0);
    Iterator last = first + diff;

    ASSERT_EQ(diff, last - first);
}
