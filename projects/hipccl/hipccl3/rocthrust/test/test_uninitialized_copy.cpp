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
#include <thrust/iterator/retag.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_copy.h>

#include "test_header.hpp"

TESTS_DEFINE(UninitializedCopyTests, FullTestsParams);

template <typename InputIterator, typename ForwardIterator>
ForwardIterator
    uninitialized_copy(my_system& system, InputIterator, InputIterator, ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_copy(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_tag, InputIterator, InputIterator, ForwardIterator result)
{
    *result = 13;
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::uninitialized_copy(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_system& system, InputIterator, Size, ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyNDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_copy_n(sys, vec.begin(), vec.size(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_tag, InputIterator, Size, ForwardIterator result)
{
    *result = 13;
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyNDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::uninitialized_copy_n(
        thrust::retag<my_tag>(vec.begin()), vec.size(), thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(UninitializedCopyTests, TestUninitializedCopySimplePOD)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(5);
    v1[0] = T(0);
    v1[1] = T(1);
    v1[2] = T(2);
    v1[3] = T(3);
    v1[4] = T(4);

    // copy to Vector
    Vector v2(5);
    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());
    ASSERT_EQ(v2[0], T(0));
    ASSERT_EQ(v2[1], T(1));
    ASSERT_EQ(v2[2], T(2));
    ASSERT_EQ(v2[3], T(3));
    ASSERT_EQ(v2[4], T(4));
}

TYPED_TEST(UninitializedCopyTests, TestUninitializedCopyNSimplePOD)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(5);
    v1[0] = T(0);
    v1[1] = T(1);
    v1[2] = T(2);
    v1[3] = T(3);
    v1[4] = T(4);

    // copy to Vector
    Vector v2(5);
    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());
    ASSERT_EQ(v2[0], T(0));
    ASSERT_EQ(v2[1], T(1));
    ASSERT_EQ(v2[2], T(2));
    ASSERT_EQ(v2[3], T(3));
    ASSERT_EQ(v2[4], T(4));
}

struct CopyConstructTest
{
    __host__ __device__
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

TEST(UninitializedCopyTests, TestUninitializedCopyNonPODDevice)
{
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    x = v2[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNNonPODDevice)
{
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    x = v2[0];
    ASSERT_EQ(true, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNonPODHost)
{
    using T = CopyConstructTest;

    thrust::host_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    x = v2[0];
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(true, x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNNonPODHost)
{
    using T = CopyConstructTest;

    thrust::host_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    x = v2[0];
    ASSERT_EQ(false, x.copy_constructed_on_device);
    ASSERT_EQ(true, x.copy_constructed_on_host);
}
