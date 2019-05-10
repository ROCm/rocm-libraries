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

#include <thrust/iterator/discard_iterator.h>

#include "test_header.hpp"

using namespace thrust;

TEST(DiscardIteratorTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(DiscardIteratorTests, DiscardIteratorIncrement)
{
    discard_iterator<> lhs(0);
    discard_iterator<> rhs(0);

    ASSERT_EQ(0, lhs - rhs);

    lhs++;

    ASSERT_EQ(1, lhs - rhs);

    lhs++;
    lhs++;

    ASSERT_EQ(3, lhs - rhs);

    lhs += 5;

    ASSERT_EQ(8, lhs - rhs);

    lhs -= 10;

    ASSERT_EQ(-2, lhs - rhs);
}

TEST(DiscardIteratorTests, DiscardIteratorComparison)
{
    discard_iterator<> iter1(0);
    discard_iterator<> iter2(0);

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);

    iter1++;

    ASSERT_EQ(1, iter1 - iter2);
    ASSERT_EQ(false, iter1 == iter2);

    iter2++;

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);

    iter1 += 100;
    iter2 += 100;

    ASSERT_EQ(0, iter1 - iter2);
    ASSERT_EQ(true, iter1 == iter2);
}

TEST(DiscardIteratorTests, MakeDiscardIterator)
{
    discard_iterator<> iter0 = make_discard_iterator(13);

    *iter0 = 7;

    discard_iterator<> iter1 = make_discard_iterator(7);

    *iter1 = 13;

    ASSERT_EQ(6, iter0 - iter1);
}

TEST(DiscardIteratorTests, ZippedDiscardIterator)
{
    using IteratorTuple1 = tuple<discard_iterator<>>;
    using ZipIterator1   = zip_iterator<IteratorTuple1>;

    IteratorTuple1 t = make_tuple(make_discard_iterator());

    ZipIterator1 z_iter1_first = make_zip_iterator(t);
    ZipIterator1 z_iter1_last  = z_iter1_first + 10;
    for(; z_iter1_first != z_iter1_last; ++z_iter1_first)
    {
        ;
    }

    ASSERT_EQ(10, get<0>(z_iter1_first.get_iterator_tuple()) - make_discard_iterator());

    using IteratorTuple2 = tuple<int*, discard_iterator<>>;
    using ZipIterator2   = zip_iterator<IteratorTuple2>;

    ZipIterator2 z_iter_first = make_zip_iterator(make_tuple((int*)0, make_discard_iterator()));
    ZipIterator2 z_iter_last  = z_iter_first + 10;

    for(; z_iter_first != z_iter_last; ++z_iter_first)
    {
        ;
    }

    ASSERT_EQ(10, get<1>(z_iter_first.get_iterator_tuple()) - make_discard_iterator());
}
