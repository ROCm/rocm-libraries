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
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(DevicePtrTests, FullTestsParams);
TESTS_DEFINE(DevicePtrPrimitiveTests, NumericalTestsParams);

template <typename T>
struct mark_processed_functor
{
    thrust::device_ptr<T> ptr;
    __host__ __device__ void operator()(size_t x)
    {
        ptr[x] = 1;
    }
};

TEST(DevicePtrTests, TestDevicePointerManipulation)
{
    thrust::device_vector<int> data(5);

    thrust::device_ptr<int> begin(&data[0]);
    thrust::device_ptr<int> end(&data[0] + 5);

    ASSERT_EQ(end - begin, 5);

    begin++;
    begin--;

    ASSERT_EQ(end - begin, 5);

    begin += 1;
    begin -= 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (int)1;
    begin = begin - (int)1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (unsigned int)1;
    begin = begin - (unsigned int)1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (size_t)1;
    begin = begin - (size_t)1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (ptrdiff_t)1;
    begin = begin - (ptrdiff_t)1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (thrust::device_ptr<int>::difference_type)1;
    begin = begin - (thrust::device_ptr<int>::difference_type)1;

    ASSERT_EQ(end - begin, 5);
}

TYPED_TEST(DevicePtrPrimitiveTests, MakeDevicePointer)
{
    using T = typename TestFixture::input_type;

    T*                    raw_ptr = 0;
    thrust::device_ptr<T> p0      = thrust::device_pointer_cast(raw_ptr);

    ASSERT_EQ(thrust::raw_pointer_cast(p0), raw_ptr);
    thrust::device_ptr<T> p1 = thrust::device_pointer_cast(p0);
    ASSERT_EQ(p0, p1);
}

TYPED_TEST(DevicePtrTests, TestRawPointerCast)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector vec(3);

    T* first;
    T* last;

    first = thrust::raw_pointer_cast(&vec[0]);
    last  = thrust::raw_pointer_cast(&vec[3]);
    ASSERT_EQ(last - first, 3);

    first = thrust::raw_pointer_cast(&vec.front());
    last  = thrust::raw_pointer_cast(&vec.back());
    ASSERT_EQ(last - first, 2);
}

TYPED_TEST(DevicePtrPrimitiveTests, TestDevicePointerValue)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::device_vector<T> d_data(size);

        thrust::device_ptr<T> begin(&d_data[0]);

        auto raw_ptr_begin = thrust::raw_pointer_cast(begin);
        if(size > 0)
            ASSERT_NE(raw_ptr_begin, nullptr);

        // Zero input memory
        if(size > 0)
            HIP_CHECK(hipMemset(raw_ptr_begin, 0, sizeof(T) * size));

        // Create unary function
        mark_processed_functor<T> func;
        func.ptr = begin;

        // Run for_each in [0; end] range
        auto end    = size < 2 ? size : size / 2;
        auto result = thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                                       thrust::make_counting_iterator<size_t>(end),
                                       func);
        ASSERT_EQ(result, thrust::make_counting_iterator<size_t>(end));

        thrust::host_vector<T> h_data = d_data;

        for(size_t i = 0; i < size; i++)
        {
            if(i < end)
            {
                ASSERT_EQ(h_data[i], T(1)) << "where index = " << i;
            }
            else
            {
                ASSERT_EQ(h_data[i], T(0)) << "where index = " << i;
            }
        }
    }
}
