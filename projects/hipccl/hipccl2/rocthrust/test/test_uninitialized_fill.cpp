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
#include <thrust/iterator/retag.h>
#include <thrust/system/hip/config.h>
#include <thrust/uninitialized_fill.h>

#include "test_header.hpp"

TESTS_DEFINE(UninitializedFillTests, FullTestsParams);

template <typename ForwardIterator, typename T>
void uninitialized_fill(my_system& system, ForwardIterator, ForwardIterator, const T&)
{
    system.validate_dispatch();
}

TEST(UninitializedFillTests, TestUninitializedFillDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_fill(sys, vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename T>
void uninitialized_fill(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
    *first = 13;
}

TEST(UninitializedFillTests, TestUninitializedFillDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::uninitialized_fill(
        thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

    ASSERT_EQ(13, vec.front());
}

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(my_system& system, ForwardIterator first, Size, const T&)
{
    system.validate_dispatch();
    return first;
}

TEST(UninitializedFillTests, TestUninitializedFillNDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(my_tag, ForwardIterator first, Size, const T&)
{
    *first = 13;
    return first;
}

TEST(UninitializedFillTests, TestUninitializedFillNDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

TYPED_TEST(UninitializedFillTests, TestUninitializedFillPOD)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    T exemplar(7);

    thrust::uninitialized_fill(v.begin() + 1, v.begin() + 4, exemplar);

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], T(4));

    exemplar = T(8);

    thrust::uninitialized_fill(v.begin() + 0, v.begin() + 3, exemplar);

    ASSERT_EQ(v[0], exemplar);
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));

    exemplar = T(9);

    thrust::uninitialized_fill(v.begin() + 2, v.end(), exemplar);

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], T(9));

    exemplar = T(1);

    thrust::uninitialized_fill(v.begin(), v.end(), exemplar);

    ASSERT_EQ(v[0], exemplar);
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], exemplar);
}

struct CopyConstructTest
{
    CopyConstructTest(void)
        : copy_constructed_on_host(false)
        , copy_constructed_on_device(false)
    {
    }

    __host__ __device__ CopyConstructTest(const CopyConstructTest&)
    {
#if defined(THRUST_HIP_DEVICE_CODE)
        copy_constructed_on_device = true;
        copy_constructed_on_host   = false;
#else
        copy_constructed_on_device = false;
        copy_constructed_on_host   = true;
#endif
    }

    __host__ __device__ CopyConstructTest& operator=(const CopyConstructTest& x)
    {
        copy_constructed_on_host   = x.copy_constructed_on_host;
        copy_constructed_on_device = x.copy_constructed_on_device;
        return *this;
    }

    bool copy_constructed_on_host;
    bool copy_constructed_on_device;
};

TEST(UninitializedFillTests, TestUninitializedFillNonPOD)
{
    using T                 = CopyConstructTest;
    thrust::device_ptr<T> v = thrust::device_malloc<T>(5);

    T exemplar;
    ASSERT_EQ(false, exemplar.copy_constructed_on_device);
    ASSERT_EQ(false, exemplar.copy_constructed_on_host);

    T host_copy_of_exemplar(exemplar);
    ASSERT_EQ(false, host_copy_of_exemplar.copy_constructed_on_device);
    ASSERT_EQ(true, host_copy_of_exemplar.copy_constructed_on_host);

    // copy construct v from the exemplar
    thrust::uninitialized_fill(v, v + 1, exemplar);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::device_free(v);
}

TYPED_TEST(UninitializedFillTests, TestUninitializedFillNPOD)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    T exemplar(7);

    using Iterator = typename Vector::iterator;
    Iterator iter  = thrust::uninitialized_fill_n(v.begin() + 1, size_t(3), exemplar);

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], T(4));
    ASSERT_EQ(v.begin() + 4, iter);

    exemplar = T(8);

    iter = thrust::uninitialized_fill_n(v.begin() + 0, size_t(3), exemplar);

    ASSERT_EQ(v[0], exemplar);
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], T(7));
    ASSERT_EQ(v[4], T(4));
    ASSERT_EQ(v.begin() + 3, iter);

    exemplar = T(9);

    iter = thrust::uninitialized_fill_n(v.begin() + 2, size_t(3), exemplar);

    ASSERT_EQ(v[0], T(8));
    ASSERT_EQ(v[1], T(8));
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], T(9));
    ASSERT_EQ(v.end(), iter);

    exemplar = T(1);

    iter = thrust::uninitialized_fill_n(v.begin(), v.size(), exemplar);

    ASSERT_EQ(v[0], exemplar);
    ASSERT_EQ(v[1], exemplar);
    ASSERT_EQ(v[2], exemplar);
    ASSERT_EQ(v[3], exemplar);
    ASSERT_EQ(v[4], exemplar);
    ASSERT_EQ(v.end(), iter);
}

TEST(UninitializedFillTests, TestUninitializedFillNNonPOD)
{
    using T                 = CopyConstructTest;
    thrust::device_ptr<T> v = thrust::device_malloc<T>(5);

    T exemplar;
    ASSERT_EQ(false, exemplar.copy_constructed_on_device);
    ASSERT_EQ(false, exemplar.copy_constructed_on_host);

    T host_copy_of_exemplar(exemplar);
    ASSERT_EQ(false, host_copy_of_exemplar.copy_constructed_on_device);
    ASSERT_EQ(true, host_copy_of_exemplar.copy_constructed_on_host);

    // copy construct v from the exemplar
    thrust::uninitialized_fill_n(v, size_t(1), exemplar);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::device_free(v);
}
