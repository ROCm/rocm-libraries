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

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(ScatterTests, FullTestsParams);
TESTS_DEFINE(ScatterPrimitiveTests, NumericalTestsParams);

TYPED_TEST(ScatterTests, TestScatterSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector map(5); // scatter indices
    Vector src(5); // source vector
    Vector dst(8); // destination vector

    map[0] = T(6);
    map[1] = T(3);
    map[2] = T(1);
    map[3] = T(7);
    map[4] = T(2);
    src[0] = T(0);
    src[1] = T(1);
    src[2] = T(2);
    src[3] = T(3);
    src[4] = T(4);
    dst[0] = T(0);
    dst[1] = T(0);
    dst[2] = T(0);
    dst[3] = T(0);
    dst[4] = T(0);
    dst[5] = T(0);
    dst[6] = T(0);
    dst[7] = T(0);

    thrust::scatter(src.begin(), src.end(), map.begin(), dst.begin());

    ASSERT_EQ(dst[0], T(0));
    ASSERT_EQ(dst[1], T(2));
    ASSERT_EQ(dst[2], T(4));
    ASSERT_EQ(dst[3], T(1));
    ASSERT_EQ(dst[4], T(0));
    ASSERT_EQ(dst[5], T(0));
    ASSERT_EQ(dst[6], T(0));
    ASSERT_EQ(dst[7], T(3));
}

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(
    my_system& system, InputIterator1, InputIterator1, InputIterator2, RandomAccessIterator)
{
    system.validate_dispatch();
}

TEST(ScatterTests, TestScatterDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::scatter(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(my_tag, InputIterator1, InputIterator1, InputIterator2, RandomAccessIterator output)
{
    *output = 13;
}

TEST(ScatterTests, TestScatterDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::scatter(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ScatterPrimitiveTests, TestScatter)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T>   h_input(size, (T)1);
        thrust::device_vector<T> d_input(size, (T)1);

        thrust::host_vector<unsigned int> h_map
            = get_random_data<unsigned int>(size,
                                            std::numeric_limits<unsigned int>::min(),
                                            std::numeric_limits<unsigned int>::max());

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % output_size;

        thrust::device_vector<unsigned int> d_map = h_map;

        thrust::host_vector<T>   h_output(output_size, T(0));
        thrust::device_vector<T> d_output(output_size, T(0));

        thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
        thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(ScatterPrimitiveTests, TestScatterToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T>   h_input(size, (T)1);
        thrust::device_vector<T> d_input(size, (T)1);

        thrust::host_vector<unsigned int> h_map
            = get_random_data<unsigned int>(size,
                                            std::numeric_limits<unsigned int>::min(),
                                            std::numeric_limits<unsigned int>::max());

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % output_size;

        thrust::device_vector<unsigned int> d_map = h_map;

        thrust::scatter(
            h_input.begin(), h_input.end(), h_map.begin(), thrust::make_discard_iterator());
        thrust::scatter(
            d_input.begin(), d_input.end(), d_map.begin(), thrust::make_discard_iterator());

        // there's nothing to check -- just make sure it compiles
    }
}

TYPED_TEST(ScatterTests, TestScatterIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector flg(5); // predicate array
    Vector map(5); // scatter indices
    Vector src(5); // source vector
    Vector dst(8); // destination vector

    flg[0] = T(0);
    flg[1] = T(1);
    flg[2] = T(0);
    flg[3] = T(1);
    flg[4] = T(0);
    map[0] = T(6);
    map[1] = T(3);
    map[2] = T(1);
    map[3] = T(7);
    map[4] = T(2);
    src[0] = T(0);
    src[1] = T(1);
    src[2] = T(2);
    src[3] = T(3);
    src[4] = T(4);
    dst[0] = T(0);
    dst[1] = T(0);
    dst[2] = T(0);
    dst[3] = T(0);
    dst[4] = T(0);
    dst[5] = T(0);
    dst[6] = T(0);
    dst[7] = T(0);

    thrust::scatter_if(src.begin(), src.end(), map.begin(), flg.begin(), dst.begin());

    ASSERT_EQ(dst[0], T(0));
    ASSERT_EQ(dst[1], T(0));
    ASSERT_EQ(dst[2], T(0));
    ASSERT_EQ(dst[3], T(1));
    ASSERT_EQ(dst[4], T(0));
    ASSERT_EQ(dst[5], T(0));
    ASSERT_EQ(dst[6], T(0));
    ASSERT_EQ(dst[7], T(3));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
void scatter_if(my_system& system,
                InputIterator1,
                InputIterator1,
                InputIterator2,
                InputIterator3,
                RandomAccessIterator)
{
    system.validate_dispatch();
}

TEST(ScatterTests, TestScatterIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::scatter_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
void scatter_if(my_tag,
                InputIterator1,
                InputIterator1,
                InputIterator2,
                InputIterator3,
                RandomAccessIterator output)
{
    *output = 13;
}

TEST(ScatterTests, TestScatterIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::scatter_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename T>
class is_even_scatter_if
{
public:
    __host__ __device__ bool operator()(const T i) const
    {
        return (i % 2) == 0;
    }
};

TYPED_TEST(ScatterPrimitiveTests, TestScatterIf)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T>   h_input(size, T(1));
        thrust::device_vector<T> d_input(size, T(1));

        thrust::host_vector<unsigned int> h_map
            = get_random_data<unsigned int>(size,
                                            std::numeric_limits<unsigned int>::min(),
                                            std::numeric_limits<unsigned int>::max());

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % output_size;

        thrust::device_vector<unsigned int> d_map = h_map;

        thrust::host_vector<T>   h_output(output_size, T(0));
        thrust::device_vector<T> d_output(output_size, T(0));

        thrust::scatter_if(h_input.begin(),
                           h_input.end(),
                           h_map.begin(),
                           h_map.begin(),
                           h_output.begin(),
                           is_even_scatter_if<unsigned int>());
        thrust::scatter_if(d_input.begin(),
                           d_input.end(),
                           d_map.begin(),
                           d_map.begin(),
                           d_output.begin(),
                           is_even_scatter_if<unsigned int>());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(ScatterPrimitiveTests, TestScatterIfToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T>   h_input(size, T(1));
        thrust::device_vector<T> d_input(size, T(1));

        thrust::host_vector<unsigned int> h_map
            = get_random_data<unsigned int>(size,
                                            std::numeric_limits<unsigned int>::min(),
                                            std::numeric_limits<unsigned int>::max());

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % output_size;

        thrust::device_vector<unsigned int> d_map = h_map;

        thrust::scatter_if(h_input.begin(),
                           h_input.end(),
                           h_map.begin(),
                           h_map.begin(),
                           thrust::make_discard_iterator(),
                           is_even_scatter_if<unsigned int>());
        thrust::scatter_if(d_input.begin(),
                           d_input.end(),
                           d_map.begin(),
                           d_map.begin(),
                           thrust::make_discard_iterator(),
                           is_even_scatter_if<unsigned int>());
    }
}

TYPED_TEST(ScatterTests, TestScatterCountingIterator)
{
    using Vector = typename TestFixture::input_type;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(10),
                    map.begin(),
                    output.begin());

    ASSERT_EQ(output, map);

    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(
        source.begin(), source.end(), thrust::make_counting_iterator(0), output.begin());

    ASSERT_EQ(output, map);

    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(10),
                    thrust::make_counting_iterator(0),
                    output.begin());

    ASSERT_EQ(output, map);
}

TYPED_TEST(ScatterTests, TestScatterIfCountingIterator)
{
    using Vector = typename TestFixture::input_type;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector stencil(10, 1);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(10),
                       map.begin(),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQ(output, map);

    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(source.begin(),
                       source.end(),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQ(output, map);

    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(10),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQ(output, map);
}