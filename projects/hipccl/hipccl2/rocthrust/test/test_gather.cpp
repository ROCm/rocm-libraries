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

// Google Test
#include <gtest/gtest.h>
#include "test_utils.hpp"

// Thrust
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <algorithm>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template< class InputType >
struct Params
{
    using input_type = InputType;
};

template<class Params>
class GatherTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params>
class PrimitiveGatherTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> GatherTestsParams;

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> GatherTestsPrimitiveParams;

TYPED_TEST_CASE(GatherTests, GatherTestsParams);
TYPED_TEST_CASE(PrimitiveGatherTests, GatherTestsPrimitiveParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TEST(GatherTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(GatherTests, GatherSimple)
{
    using Vector = typename TestFixture::input_type;

    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather(map.begin(), map.end(), src.begin(), dst.begin());

    ASSERT_EQ(dst[0], 6);
    ASSERT_EQ(dst[1], 2);
    ASSERT_EQ(dst[2], 1);
    ASSERT_EQ(dst[3], 7);
    ASSERT_EQ(dst[4], 2);
}

template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_system &system, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(GatherTests, GatherDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::gather(sys,
                   vec.begin(),
                   vec.end(),
                   vec.begin(),
                   vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_tag, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(GatherTests, GatherDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::gather(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveGatherTests, Gather)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    T min = (T) std::numeric_limits<T>::min();
    T max = (T) std::numeric_limits<T>::max();

    for(auto size : sizes)
    {
        const size_t source_size = std::min((size_t) 10, 2 * size);
        
        // source vectors to gather from
        thrust::host_vector<T>   h_source = get_random_data<T>(source_size, min, max);
        thrust::device_vector<T> d_source = h_source;
    
        // gather indices
        thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_map[i] =  h_map[i] % source_size;
        
        thrust::device_vector<unsigned int> d_map = h_map;

        // gather destination
        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());
        thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());

        thrust::host_vector<T> d_output_h = d_output;
        ASSERT_EQ(h_output, d_output_h);
    }
}

TYPED_TEST(PrimitiveGatherTests, GatherToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    T min = (T) std::numeric_limits<T>::min();
    T max = (T) std::numeric_limits<T>::max();

    for(auto size : sizes)
    {
        const size_t source_size = std::min((size_t) 10, 2 * size);

        // source vectors to gather from
        thrust::host_vector<T>   h_source = get_random_data<T>(source_size, min, max);
        thrust::device_vector<T> d_source = h_source;
    
        // gather indices
        thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_map[i] =  h_map[i] % source_size;
        
        thrust::device_vector<unsigned int> d_map = h_map;

        thrust::discard_iterator<> h_result = 
        thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), thrust::make_discard_iterator());

        thrust::discard_iterator<> d_result =
        thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}

TYPED_TEST(GatherTests, GatherIfSimple)
{
    using Vector = typename TestFixture::input_type;

    Vector flg(5);  // predicate array
    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather_if(map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());

    ASSERT_EQ(dst[0], 0);
    ASSERT_EQ(dst[1], 2);
    ASSERT_EQ(dst[2], 0);
    ASSERT_EQ(dst[3], 7);
    ASSERT_EQ(dst[4], 0);
}

template <typename T>
struct is_even_gather_if
{
    __host__ __device__
    bool operator()(const T i) const
    { 
        return (i % 2) == 0;
    }
};

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(my_system &system,
                         InputIterator1       map_first,
                         InputIterator1       map_last,
                         InputIterator2       stencil,
                         RandomAccessIterator input_first,
                         OutputIterator       result)
{
    system.validate_dispatch();
    return result;
}

TEST(GatherTests, GatherIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::gather_if(sys,
                      vec.begin(),
                      vec.end(),
                      vec.begin(),
                      vec.begin(),
                      vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(my_tag,
                         InputIterator1       map_first,
                         InputIterator1       map_last,
                         InputIterator2       stencil,
                         RandomAccessIterator input_first,
                         OutputIterator       result)
{
    *result = 13;
    return result;
}

TEST(GatherTests, GatherIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::gather_if(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.end()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveGatherTests, GatherIf)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    T min = (T) std::numeric_limits<T>::min();
    T max = (T) std::numeric_limits<T>::max();

    for(auto size : sizes)
    {
        const size_t source_size = std::min((size_t) 10, 2 * size);

        // source vectors to gather from
        thrust::host_vector<T>   h_source = get_random_data<T>(source_size, min, max);
        thrust::device_vector<T> d_source = h_source;
    
        // gather indices
        thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % source_size;
        
        thrust::device_vector<unsigned int> d_map = h_map;
        
        // gather stencil
        thrust::host_vector<unsigned int> h_stencil = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_stencil[i] = h_stencil[i] % 2;
        
        thrust::device_vector<unsigned int> d_stencil = h_stencil;

        // gather destination
        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), h_output.begin(), is_even_gather_if<unsigned int>());
        thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), d_output.begin(), is_even_gather_if<unsigned int>());

        thrust::host_vector<T> d_output_h = d_output;
        ASSERT_EQ(h_output, d_output_h);
    }
}

TYPED_TEST(PrimitiveGatherTests, GatherIfToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    T min = (T) std::numeric_limits<T>::min();
    T max = (T) std::numeric_limits<T>::max();

    for(auto size : sizes)
    {
        const size_t source_size = std::min((size_t) 10, 2 * size);

        // source vectors to gather from
        thrust::host_vector<T>   h_source = get_random_data<T>(source_size, min, max);
        thrust::device_vector<T> d_source = h_source;
    
        // gather indices
        thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_map[i] = h_map[i] % source_size;
        
        thrust::device_vector<unsigned int> d_map = h_map;
        
        // gather stencil
        thrust::host_vector<unsigned int> h_stencil = get_random_data<unsigned int>(size, min, max);

        for(size_t i = 0; i < size; i++)
            h_stencil[i] = h_stencil[i] % 2;
        
        thrust::device_vector<unsigned int> d_stencil = h_stencil;

        thrust::discard_iterator<> h_result =
        thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), thrust::make_discard_iterator(), is_even_gather_if<unsigned int>());

        thrust::discard_iterator<> d_result =
        thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), thrust::make_discard_iterator(), is_even_gather_if<unsigned int>());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}

TYPED_TEST(GatherTests, TestGatherCountingIterator)
{
    using Vector = typename TestFixture::input_type;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(map.begin(),
                   map.end(),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQ(output, map);
    
    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)source.size()),
                   source.begin(),
                   output.begin());

    ASSERT_EQ(output, map);
    
    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)output.size()),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQ(output, map);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
