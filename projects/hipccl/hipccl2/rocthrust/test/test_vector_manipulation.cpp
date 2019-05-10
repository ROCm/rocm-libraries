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

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(VectorManipulationTests, FullTestsParams);

TYPED_TEST(VectorManipulationTests, TestVectorManipulation)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> src = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        ASSERT_EQ(src.size(), size);

        // basic initialization
        Vector test0(size);
        Vector test1(size, T(3));
        ASSERT_EQ(test0.size(), size);
        ASSERT_EQ(test1.size(), size);
        ASSERT_EQ((test1 == std::vector<T>(size, T(3))), true);

        // initializing from other vector
        std::vector<T> stl_vector(src.begin(), src.end());
        Vector         cpy0 = src;
        Vector         cpy1(stl_vector);
        Vector         cpy2(stl_vector.begin(), stl_vector.end());

        ASSERT_EQ(cpy0, src);
        ASSERT_EQ(cpy1, src);
        ASSERT_EQ(cpy2, src);

        // resizing
        Vector vec1(src);
        vec1.resize(size + 3);
        ASSERT_EQ(vec1.size(), size + 3);
        vec1.resize(size);
        ASSERT_EQ(vec1.size(), size);
        ASSERT_EQ(vec1, src);

        vec1.resize(size + 20, T(11));
        Vector tail(vec1.begin() + size, vec1.end());
        ASSERT_EQ((tail == std::vector<T>(20, T(11))), true);

        // shrinking a vector should not invalidate iterators
        Iterator first = vec1.begin();
        vec1.resize(10);
        ASSERT_EQ(first, vec1.begin());

        vec1.resize(0);
        ASSERT_EQ(vec1.size(), 0);
        ASSERT_EQ(vec1.empty(), true);
        vec1.resize(10);
        ASSERT_EQ(vec1.size(), 10);
        vec1.clear();
        ASSERT_EQ(vec1.size(), 0);
        vec1.resize(5);
        ASSERT_EQ(vec1.size(), 5);

        // push_back
        Vector vec2;
        for(size_t i = 0; i < 10; ++i)
        {
            ASSERT_EQ(vec2.size(), i);
            vec2.push_back((T)i);
            ASSERT_EQ(vec2.size(), i + 1);
            for(size_t j = 0; j <= i; j++)
                ASSERT_EQ(vec2[j], j);
            ASSERT_EQ(vec2.back(), i);
        }

        // pop_back
        for(size_t i = 10; i > 0; --i)
        {
            ASSERT_EQ(vec2.size(), i);
            ASSERT_EQ(vec2.back(), i - 1);
            vec2.pop_back();
            ASSERT_EQ(vec2.size(), i - 1);
            for(size_t j = 0; j < i; j++)
                ASSERT_EQ(vec2[j], j);
        }
    }
}
