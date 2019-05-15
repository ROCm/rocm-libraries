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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/memory.h>

#include "test_header.hpp"

TESTS_DEFINE(ForEachTests, SignedIntegerTestsParams)
TESTS_DEFINE(ForEachVectorTests, FullTestsParams)
TESTS_DEFINE(ForEachPrimitiveTests, NumericalTestsParams);

template <typename T>
struct mark_processed_functor
{
    T*       ptr;
    __host__ __device__ void operator()(size_t x)
    {
        ptr[x] = 1;
    }
};

TYPED_TEST(ForEachTests, HostPathSimpleTest)
{
    thrust::device_system_tag tag;
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto ptr     = thrust::malloc<T>(tag, sizeof(T) * size);
        auto raw_ptr = thrust::raw_pointer_cast(ptr);
        if(size > 0)
            ASSERT_NE(raw_ptr, nullptr);

        // Zero input memory
        if(size > 0)
            HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T) * size));

        // Create unary function
        mark_processed_functor<T> func;
        func.ptr = raw_ptr;

        // Run for_each in [0; end] range
        auto end    = size < 2 ? size : size / 2;
        auto result = thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                                       thrust::make_counting_iterator<size_t>(end),
                                       func);
        ASSERT_EQ(result, thrust::make_counting_iterator<size_t>(end));

        std::vector<T> output(size);
        HIP_CHECK(hipMemcpy(output.data(), raw_ptr, size * sizeof(T), hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            if(i < end)
            {
                ASSERT_EQ(output[i], T(1)) << "where index = " << i;
            }
            else
            {
                ASSERT_EQ(output[i], T(0)) << "where index = " << i;
            }
        }

        // Free
        thrust::free(tag, ptr);
    }
}

template <class F>
__global__ void simple_test_kernel(F func, int size)
{
    // (void) func; (void) size;
    thrust::for_each(thrust::seq,
                     thrust::make_counting_iterator<int>(0),
                     thrust::make_counting_iterator<int>(size),
                     func);
}

TYPED_TEST(ForEachTests, DevicePathSimpleTest)
{
    thrust::device_system_tag tag;
    using T           = typename TestFixture::input_type;
    const size_t size = 1024;

    auto ptr     = thrust::malloc<T>(tag, sizeof(T) * size);
    auto raw_ptr = thrust::raw_pointer_cast(ptr);
    ASSERT_NE(raw_ptr, nullptr);

    // Zero input memory
    HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T) * size));

    // Create unary function
    mark_processed_functor<T> func;
    func.ptr = raw_ptr;

    // Run for_each in [0; end] range
    size_t end = 375;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(simple_test_kernel<mark_processed_functor<T>>),
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       func,
                       static_cast<int>(end));

    std::vector<T> output(size);
    HIP_CHECK(hipMemcpy(output.data(), raw_ptr, size * sizeof(T), hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
    {
        if(i < end)
        {
            ASSERT_EQ(output[i], T(1)) << "where index = " << i;
        }
        else
        {
            ASSERT_EQ(output[i], T(0)) << "where index = " << i;
        }
    }

    // Free
    thrust::free(tag, ptr);
}

template <typename T>
class mark_present_for_each
{
public:
    T*       ptr;
    __host__ __device__ void operator()(T x)
    {
        ptr[(int)x] = 1;
    }
};

TYPED_TEST(ForEachVectorTests, TestForEachSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector input(5);
    Vector output(7, (T)0);

    input[0] = T(3);
    input[1] = T(2);
    input[2] = T(3);
    input[3] = T(4);
    input[4] = T(6);

    mark_present_for_each<T> f;
    f.ptr = thrust::raw_pointer_cast(output.data());

    typename Vector::iterator result = thrust::for_each(input.begin(), input.end(), f);

    ASSERT_EQ(output[0], T(0));
    ASSERT_EQ(output[1], T(0));
    ASSERT_EQ(output[2], T(1));
    ASSERT_EQ(output[3], T(1));
    ASSERT_EQ(output[4], T(1));
    ASSERT_EQ(output[5], T(0));
    ASSERT_EQ(output[6], T(1));
    ASSERT_EQ_QUIET(result, input.end());
}

template <typename InputIterator, typename Function>
__host__ __device__ InputIterator
                    for_each(my_system& system, InputIterator first, InputIterator, Function)
{
    system.validate_dispatch();
    return first;
}

TEST(ForEachVectorTests, TestForEachDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::for_each(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Function>
__host__ __device__ InputIterator for_each(my_tag, InputIterator first, InputIterator, Function)
{
    *first = 13;
    return first;
}

TEST(ForEachVectorTests, TestForEachDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::for_each(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ForEachVectorTests, TestForEachNSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector input(5);
    Vector output(7, (T)0);

    input[0] = T(3);
    input[1] = T(2);
    input[2] = T(3);
    input[3] = T(4);
    input[4] = T(6);

    mark_present_for_each<T> f;
    f.ptr = thrust::raw_pointer_cast(output.data());

    typename Vector::iterator result = thrust::for_each_n(input.begin(), input.size(), f);

    ASSERT_EQ(output[0], T(0));
    ASSERT_EQ(output[1], T(0));
    ASSERT_EQ(output[2], T(1));
    ASSERT_EQ(output[3], T(1));
    ASSERT_EQ(output[4], T(1));
    ASSERT_EQ(output[5], T(0));
    ASSERT_EQ(output[6], T(1));
    ASSERT_EQ_QUIET(result, input.end());
}

template <typename InputIterator, typename Size, typename Function>
__host__ __device__ InputIterator for_each_n(my_system& system, InputIterator first, Size, Function)
{
    system.validate_dispatch();
    return first;
}

TEST(ForEachVectorTests, TestForEachNDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::for_each_n(sys, vec.begin(), vec.size(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Size, typename Function>
__host__ __device__ InputIterator for_each_n(my_tag, InputIterator first, Size, Function)
{
    *first = 13;
    return first;
}

TEST(ForEachVectorTests, TestForEachNDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::for_each_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

    ASSERT_EQ(13, vec.front());
}

TEST(ForEachVectorTests, TestForEachSimpleAnySystem)
{
    thrust::device_vector<int> output(7, 0);

    mark_present_for_each<int> f;
    f.ptr = thrust::raw_pointer_cast(output.data());

    thrust::counting_iterator<int> result
        = thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(5), f);

    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 1);
    ASSERT_EQ(output[3], 1);
    ASSERT_EQ(output[4], 1);
    ASSERT_EQ(output[5], 0);
    ASSERT_EQ(output[6], 0);
    ASSERT_EQ_QUIET(result, thrust::make_counting_iterator(5));
}

TEST(ForEachVectorTests, TestForEachNSimpleAnySystem)
{
    thrust::device_vector<int> output(7, 0);

    mark_present_for_each<int> f;
    f.ptr = thrust::raw_pointer_cast(output.data());

    thrust::counting_iterator<int> result
        = thrust::for_each_n(thrust::make_counting_iterator(0), 5, f);

    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 1);
    ASSERT_EQ(output[3], 1);
    ASSERT_EQ(output[4], 1);
    ASSERT_EQ(output[5], 0);
    ASSERT_EQ(output[6], 0);
    ASSERT_EQ_QUIET(result, thrust::make_counting_iterator(5));
}

TYPED_TEST(ForEachPrimitiveTests, TestForEach)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        for(size_t i = 0; i < size; i++)
            h_input[i] = ((size_t)h_input[i]) % output_size;

        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(output_size, (T)0);
        thrust::device_vector<T> d_output(output_size, (T)0);

        mark_present_for_each<T> h_f;
        mark_present_for_each<T> d_f;
        h_f.ptr = &h_output[0];
        d_f.ptr = (&d_output[0]).get();

        typename thrust::host_vector<T>::iterator h_result
            = thrust::for_each(h_input.begin(), h_input.end(), h_f);

        typename thrust::device_vector<T>::iterator d_result
            = thrust::for_each(d_input.begin(), d_input.end(), d_f);

        ASSERT_EQ(h_output, d_output);
        ASSERT_EQ_QUIET(h_result, h_input.end());
        ASSERT_EQ_QUIET(d_result, d_input.end());
    }
}

TYPED_TEST(ForEachPrimitiveTests, TestForEachN)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        const size_t output_size = std::min((size_t)10, 2 * size);

        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        for(size_t i = 0; i < size; i++)
            h_input[i] = ((size_t)h_input[i]) % output_size;

        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(output_size, (T)0);
        thrust::device_vector<T> d_output(output_size, (T)0);

        mark_present_for_each<T> h_f;
        mark_present_for_each<T> d_f;
        h_f.ptr = &h_output[0];
        d_f.ptr = (&d_output[0]).get();

        typename thrust::host_vector<T>::iterator h_result
            = thrust::for_each_n(h_input.begin(), h_input.size(), h_f);

        typename thrust::device_vector<T>::iterator d_result
            = thrust::for_each_n(d_input.begin(), d_input.size(), d_f);

        ASSERT_EQ(h_output, d_output);
        ASSERT_EQ_QUIET(h_result, h_input.end());
        ASSERT_EQ_QUIET(d_result, d_input.end());
    }
}

template <typename T, unsigned int N>
struct SetFixedVectorToConstant
{
    FixedVector<T, N> exemplar;

    SetFixedVectorToConstant(T scalar)
        : exemplar(scalar)
    {
    }

    __host__ __device__ void operator()(FixedVector<T, N>& t)
    {
        t = exemplar;
    }
};

template <typename T, unsigned int N>
void _TestForEachWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

    thrust::host_vector<FixedVector<T, N>> h_data(n);

    for(size_t i = 0; i < h_data.size(); i++)
        h_data[i] = FixedVector<T, N>(i);

    thrust::device_vector<FixedVector<T, N>> d_data = h_data;

    SetFixedVectorToConstant<T, N> func(123);

    thrust::for_each(h_data.begin(), h_data.end(), func);
    thrust::for_each(d_data.begin(), d_data.end(), func);

    ASSERT_EQ_QUIET(h_data, d_data);
}

TEST(ForEachVectorTests, TestForEachWithLargeTypes)
{
    _TestForEachWithLargeTypes<int, 1>();
    _TestForEachWithLargeTypes<int, 2>();
    _TestForEachWithLargeTypes<int, 4>();
    _TestForEachWithLargeTypes<int, 8>();
    _TestForEachWithLargeTypes<int, 16>();

    _TestForEachWithLargeTypes<int, 32>(); // fails on Linux 32 w/ gcc 4.1
    _TestForEachWithLargeTypes<int, 64>();
    _TestForEachWithLargeTypes<int, 128>();
    _TestForEachWithLargeTypes<int, 256>();
    _TestForEachWithLargeTypes<int, 512>();
    _TestForEachWithLargeTypes<int, 1024>(); // fails on Vista 64 w/ VS2008
}

template <typename T, unsigned int N>
void _TestForEachNWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

    thrust::host_vector<FixedVector<T, N>> h_data(n);

    for(size_t i = 0; i < h_data.size(); i++)
        h_data[i] = FixedVector<T, N>(i);

    thrust::device_vector<FixedVector<T, N>> d_data = h_data;

    SetFixedVectorToConstant<T, N> func(123);

    thrust::for_each_n(h_data.begin(), h_data.size(), func);
    thrust::for_each_n(d_data.begin(), d_data.size(), func);

    ASSERT_EQ_QUIET(h_data, d_data);
}

TEST(ForEachVectorTests, TestForEachNWithLargeTypes)
{
    _TestForEachNWithLargeTypes<int, 1>();
    _TestForEachNWithLargeTypes<int, 2>();
    _TestForEachNWithLargeTypes<int, 4>();
    _TestForEachNWithLargeTypes<int, 8>();
    _TestForEachNWithLargeTypes<int, 16>();

    _TestForEachNWithLargeTypes<int, 32>(); // fails on Linux 32 w/ gcc 4.1
    _TestForEachNWithLargeTypes<int, 64>();
    _TestForEachNWithLargeTypes<int, 128>();
    _TestForEachNWithLargeTypes<int, 256>();
    _TestForEachNWithLargeTypes<int, 512>();
    _TestForEachNWithLargeTypes<int, 1024>(); // fails on Vista 64 w/ VS2008
}