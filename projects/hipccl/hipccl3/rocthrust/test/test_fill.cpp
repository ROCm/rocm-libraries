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

TESTS_DEFINE(FillTests, FullTestsParams);
TESTS_DEFINE(FillPrimitiveTests, NumericalTestsParams);

TYPED_TEST(FillTests, TestFillSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    thrust::fill(v.begin() + 1, v.begin() + 4, T(7));

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(7));
    ASSERT_EQ(v[2], T(7));
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));

    thrust::fill(v.begin() + 0, v.begin() + 3, T(8));

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], T(8));
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));

    thrust::fill(v.begin() + 2, v.end(), T(9));

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], T(9));
    ASSERT_EQ(v[3], T(9));
    ASSERT_EQ(v[4], T(9));

    thrust::fill(v.begin(), v.end(), T(1));

    ASSERT_EQ(v[0], T(1));
    ASSERT_EQ(v[1], T(1));
    ASSERT_EQ(v[2], T(1));
    ASSERT_EQ(v[3], T(1));
    ASSERT_EQ(v[4], T(1));
}

TEST(FillTests, TestFillDiscardIterator)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // there's no result to check because fill returns void
    thrust::fill(thrust::discard_iterator<thrust::host_system_tag>(),
                 thrust::discard_iterator<thrust::host_system_tag>(10),
                 13);

    thrust::fill(thrust::discard_iterator<thrust::device_system_tag>(),
                 thrust::discard_iterator<thrust::device_system_tag>(10),
                 13);
}

TYPED_TEST(FillTests, TestFillMixedTypes)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v(4);

    thrust::fill(v.begin(), v.end(), (long)10);

    ASSERT_EQ(v[0], T(10));
    ASSERT_EQ(v[1], T(10));
    ASSERT_EQ(v[2], T(10));
    ASSERT_EQ(v[3], T(10));

    thrust::fill(v.begin(), v.end(), (float)20);

    ASSERT_EQ(v[0], T(20));
    ASSERT_EQ(v[1], T(20));
    ASSERT_EQ(v[2], T(20));
    ASSERT_EQ(v[3], T(20));
}

TYPED_TEST(FillPrimitiveTests, TestFill)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);

            thrust::device_vector<T> d_data = h_data;

            thrust::fill(h_data.begin() + std::min((size_t)1, size),
                         h_data.begin() + std::min((size_t)3, size),
                         T(0));
            thrust::fill(d_data.begin() + std::min((size_t)1, size),
                         d_data.begin() + std::min((size_t)3, size),
                         T(0));

            ASSERT_EQ(h_data, d_data);

            thrust::fill(h_data.begin() + std::min((size_t)117, size),
                         h_data.begin() + std::min((size_t)367, size),
                         T(1));
            thrust::fill(d_data.begin() + std::min((size_t)117, size),
                         d_data.begin() + std::min((size_t)367, size),
                         T(1));

            ASSERT_EQ(h_data, d_data);

            thrust::fill(h_data.begin() + std::min((size_t)8, size),
                         h_data.begin() + std::min((size_t)259, size),
                         T(2));
            thrust::fill(d_data.begin() + std::min((size_t)8, size),
                         d_data.begin() + std::min((size_t)259, size),
                         T(2));

            ASSERT_EQ(h_data, d_data);

            thrust::fill(h_data.begin() + std::min((size_t)3, size), h_data.end(), T(3));
            thrust::fill(d_data.begin() + std::min((size_t)3, size), d_data.end(), T(3));

            ASSERT_EQ(h_data, d_data);

            thrust::fill(h_data.begin(), h_data.end(), T(4));
            thrust::fill(d_data.begin(), d_data.end(), T(4));

            ASSERT_EQ(h_data, d_data);
        }
    }
}

TYPED_TEST(FillTests, TestFillNSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    Iterator iter = thrust::fill_n(v.begin() + 1, 3, T(7));

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(7));
    ASSERT_EQ(v[2], T(7));
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));
    ASSERT_EQ_QUIET(v.begin() + 4, iter);

    iter = thrust::fill_n(v.begin() + 0, 3, T(8));

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], T(8));
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));
    ASSERT_EQ_QUIET(v.begin() + 3, iter);

    iter = thrust::fill_n(v.begin() + 2, 3, T(9));

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], T(9));
    ASSERT_EQ(v[3], T(9));
    ASSERT_EQ(v[4], T(9));
    ASSERT_EQ_QUIET(v.end(), iter);

    iter = thrust::fill_n(v.begin(), v.size(), T(1));

    ASSERT_EQ(v[0], T(1));
    ASSERT_EQ(v[1], T(1));
    ASSERT_EQ(v[2], T(1));
    ASSERT_EQ(v[3], T(1));
    ASSERT_EQ(v[4], T(1));
    ASSERT_EQ_QUIET(v.end(), iter);
}

TEST(FillTests, TestFillNDiscardIterator)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::discard_iterator<thrust::host_system_tag> h_result
        = thrust::fill_n(thrust::discard_iterator<thrust::host_system_tag>(), 10, 13);

    thrust::discard_iterator<thrust::device_system_tag> d_result
        = thrust::fill_n(thrust::discard_iterator<thrust::device_system_tag>(), 10, 13);

    thrust::discard_iterator<> reference(10);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
}

TYPED_TEST(FillTests, TestFillNMixedTypes)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v(4);

    Iterator iter = thrust::fill_n(v.begin(), v.size(), (long)10);

    ASSERT_EQ(v[0], T(10));
    ASSERT_EQ(v[1], T(10));
    ASSERT_EQ(v[2], T(10));
    ASSERT_EQ(v[3], T(10));
    ASSERT_EQ_QUIET(v.end(), iter);

    iter = thrust::fill_n(v.begin(), v.size(), (float)20);

    ASSERT_EQ(v[0], T(20));
    ASSERT_EQ(v[1], T(20));
    ASSERT_EQ(v[2], T(20));
    ASSERT_EQ(v[3], T(20));
    ASSERT_EQ_QUIET(v.end(), iter);
}

TYPED_TEST(FillPrimitiveTests, TestFillN)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);

            thrust::device_vector<T> d_data = h_data;

            size_t begin_offset = std::min<size_t>(1, size);
            thrust::fill_n(
                h_data.begin() + begin_offset, std::min((size_t)3, size) - begin_offset, T(0));
            thrust::fill_n(
                d_data.begin() + begin_offset, std::min((size_t)3, size) - begin_offset, T(0));

            ASSERT_EQ(h_data, d_data);

            begin_offset = std::min<size_t>(117, size);
            thrust::fill_n(
                h_data.begin() + begin_offset, std::min((size_t)367, size) - begin_offset, T(1));
            thrust::fill_n(
                d_data.begin() + begin_offset, std::min((size_t)367, size) - begin_offset, T(1));

            ASSERT_EQ(h_data, d_data);

            begin_offset = std::min<size_t>(8, size);
            thrust::fill_n(
                h_data.begin() + begin_offset, std::min((size_t)259, size) - begin_offset, T(2));
            thrust::fill_n(
                d_data.begin() + begin_offset, std::min((size_t)259, size) - begin_offset, T(2));

            ASSERT_EQ(h_data, d_data);

            begin_offset = std::min<size_t>(3, size);
            thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, T(3));
            thrust::fill_n(d_data.begin() + begin_offset, d_data.size() - begin_offset, T(3));

            ASSERT_EQ(h_data, d_data);

            thrust::fill_n(h_data.begin(), h_data.size(), T(4));
            thrust::fill_n(d_data.begin(), d_data.size(), T(4));

            ASSERT_EQ(h_data, d_data);
        }
    }
}

TYPED_TEST(FillTests, TestFillZipIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v1(3, T(0));
    Vector v2(3, T(0));
    Vector v3(3, T(0));

    thrust::fill(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin(), v3.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end(), v3.end())),
                 thrust::tuple<T, T, T>(4, 7, 13));

    ASSERT_EQ(T(4), v1[0]);
    ASSERT_EQ(T(4), v1[1]);
    ASSERT_EQ(T(4), v1[2]);
    ASSERT_EQ(T(7), v2[0]);
    ASSERT_EQ(T(7), v2[1]);
    ASSERT_EQ(T(7), v2[2]);
    ASSERT_EQ(T(13), v3[0]);
    ASSERT_EQ(T(13), v3[1]);
    ASSERT_EQ(T(13), v3[2]);
}

TYPED_TEST(FillPrimitiveTests, TestFillTuple)
{
    using T     = typename TestFixture::input_type;
    using Tuple = typename thrust::tuple<T, T>;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::host_vector<Tuple>   h(3, Tuple(0, 0));
    thrust::device_vector<Tuple> d(3, Tuple(0, 0));

    thrust::fill(h.begin(), h.end(), Tuple(4, 7));
    thrust::fill(d.begin(), d.end(), Tuple(4, 7));

    ASSERT_EQ_QUIET(h, d);
};

struct TypeWithTrivialAssigment
{
    int x, y, z;
};

TEST(FillTests, TestFillWithTrivialAssignment)
{
    using T = TypeWithTrivialAssigment;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::host_vector<T>   h(1);
    thrust::device_vector<T> d(1);

    ASSERT_EQ(h[0].x, 0);
    ASSERT_EQ(h[0].y, 0);
    ASSERT_EQ(h[0].z, 0);
    ASSERT_EQ(static_cast<T>(d[0]).x, 0);
    ASSERT_EQ(static_cast<T>(d[0]).y, 0);
    ASSERT_EQ(static_cast<T>(d[0]).z, 0);

    T val;
    val.x = 10;
    val.y = 20;
    val.z = -1;

    thrust::fill(h.begin(), h.end(), val);
    thrust::fill(d.begin(), d.end(), val);

    ASSERT_EQ(h[0].x, 10);
    ASSERT_EQ(h[0].y, 20);
    ASSERT_EQ(h[0].z, -1);
    ASSERT_EQ(static_cast<T>(d[0]).x, 10);
    ASSERT_EQ(static_cast<T>(d[0]).y, 20);
    ASSERT_EQ(static_cast<T>(d[0]).z, -1);
}

struct TypeWithNonTrivialAssigment
{
    int x, y, z;

    __host__ __device__ TypeWithNonTrivialAssigment()
        : x(0)
        , y(0)
        , z(0)
    {
    }

#if THRUST_CPP_DIALECT >= 2011
    TypeWithNonTrivialAssigment(const TypeWithNonTrivialAssigment &) = default;
#endif

    __host__ __device__ TypeWithNonTrivialAssigment& operator=(const TypeWithNonTrivialAssigment& t)
    {
        x = t.x;
        y = t.y;
        z = t.x + t.y;
        return *this;
    }

    __host__ __device__ bool operator==(const TypeWithNonTrivialAssigment& t) const
    {
        return x == t.x && y == t.y && z == t.z;
    }
};

TEST(FillTests, TestFillWithNonTrivialAssignment)
{
    using T = TypeWithNonTrivialAssigment;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::host_vector<T>   h(1);
    thrust::device_vector<T> d(1);

    ASSERT_EQ(h[0].x, 0);
    ASSERT_EQ(h[0].y, 0);
    ASSERT_EQ(h[0].z, 0);
    ASSERT_EQ(static_cast<T>(d[0]).x, 0);
    ASSERT_EQ(static_cast<T>(d[0]).y, 0);
    ASSERT_EQ(static_cast<T>(d[0]).z, 0);

    T val;
    val.x = 10;
    val.y = 20;
    val.z = -1;

    thrust::fill(h.begin(), h.end(), val);
    thrust::fill(d.begin(), d.end(), val);

    ASSERT_EQ(h[0].x, 10);
    ASSERT_EQ(h[0].y, 20);
    ASSERT_EQ(h[0].z, 30);
    ASSERT_EQ(static_cast<T>(d[0]).x, 10);
    ASSERT_EQ(static_cast<T>(d[0]).y, 20);
    ASSERT_EQ(static_cast<T>(d[0]).z, 30);
}

template <typename ForwardIterator, typename T>
void fill(my_system& system, ForwardIterator, ForwardIterator, const T&)
{
    system.validate_dispatch();
}

TEST(FillTests, TestFillDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::fill(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename T>
void fill(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
    *first = 13;
}

TEST(FillTests, TestFillDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::fill(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

template <typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_system& system, OutputIterator first, Size, const T&)
{
    system.validate_dispatch();
    return first;
}

TEST(FillTests, TestFillNDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::fill_n(sys, vec.begin(), vec.size(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_tag, OutputIterator first, Size, const T&)
{
    *first = 13;
    return first;
}

TEST(FillTests, TestFillNDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::fill_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

    ASSERT_EQ(13, vec.front());
}



__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void FillKernel(int const N, int* array, int fill_value)
{
    if(threadIdx.x == 0)
    {
        thrust::device_ptr<int> begin(array);
        thrust::device_ptr<int> end(array + N);
        thrust::fill(thrust::hip::par, begin, end,fill_value);
    }
}

TEST(FillTests, TestFillDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size: {0, 1, 2, 4, 6, 12, 16, 24, 32, 64, 84, 128, 160, 256} )
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
            thrust::device_vector<int> d_data = h_data;

            int fill_value = get_random_data<int>(1,0,size,seed)[0];
            SCOPED_TRACE(testing::Message() << "with fill_value= " <<fill_value);

            thrust::fill(h_data.begin(), h_data.end(),fill_value);
            hipLaunchKernelGGL(FillKernel,
                               dim3(1, 1, 1),
                               dim3(128, 1, 1),
                               0,
                               0,
                               size,
                               thrust::raw_pointer_cast(&d_data[0]),
                               fill_value);

            ASSERT_EQ(h_data, d_data);
        }
    }
}
