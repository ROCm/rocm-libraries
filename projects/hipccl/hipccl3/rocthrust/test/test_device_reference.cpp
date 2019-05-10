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

#include <thrust/device_reference.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(DeviceReferenceTests, NumericalTestsParams);
TESTS_DEFINE(DeviceReferenceIntegerTests, IntegerTestsParams);

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceConstructorFromDeviceReference)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T>    v(1, T(0));
    thrust::device_reference<T> ref = v[0];

    // ref equals the object at v[0]
    ASSERT_EQ(v[0], ref);

    // the address of ref equals the address of v[0]
    ASSERT_EQ(&v[0], &ref);

    // modifying v[0] modifies ref
    v[0] = T(13);
    ASSERT_EQ(T(13), ref);
    ASSERT_EQ(v[0], ref);

    // modifying ref modifies v[0]
    ref = T(7);
    ASSERT_EQ(T(7), v[0]);
    ASSERT_EQ(v[0], ref);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceConstructorFromDevicePointer)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T>    v(1, T(0));
    thrust::device_ptr<T>       ptr = &v[0];
    thrust::device_reference<T> ref(ptr);

    // ref equals the object pointed to by ptr
    ASSERT_EQ(*ptr, ref);

    // the address of ref equals ptr
    ASSERT_EQ(ptr, &ref);

    // modifying *ptr modifies ref
    *ptr = T(13);
    ASSERT_EQ(T(13), ref);
    ASSERT_EQ(v[0], ref);

    // modifying ref modifies *ptr
    ref = T(7);
    ASSERT_EQ(T(7), *ptr);
    ASSERT_EQ(v[0], ref);
}

TEST(DeviceReferenceTests, TestDeviceReferenceAssignmentFromDeviceReference)
{
    // test same types
    using T0 = int;
    thrust::device_vector<T0>    v0(2, 0);
    thrust::device_reference<T0> ref0 = v0[0];
    thrust::device_reference<T0> ref1 = v0[1];

    ref0 = 13;

    ref1 = ref0;

    // ref1 equals 13
    ASSERT_EQ(13, ref1);
    ASSERT_EQ(ref0, ref1);

    // test different types
    using T1 = float;
    thrust::device_vector<T1>    v1(1, 0.0f);
    thrust::device_reference<T1> ref2 = v1[0];

    ref2 = ref1;

    // ref2 equals 13.0f
    ASSERT_EQ(13.0f, ref2);
    ASSERT_EQ(ref0, ref2);
    ASSERT_EQ(ref1, ref2);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceManipulation)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T>    v(1, T(0));
    thrust::device_ptr<T>       ptr = &v[0];
    thrust::device_reference<T> ref(ptr);

    // reset
    ref = T(0);

    // test prefix increment
    ++ref;
    ASSERT_EQ(T(1), ref);
    ASSERT_EQ(T(1), *ptr);
    ASSERT_EQ(T(1), v[0]);

    // reset
    ref = T(0);

    // test postfix increment
    T x1 = ref++;
    ASSERT_EQ(T(0), x1);
    ASSERT_EQ(T(1), ref);
    ASSERT_EQ(T(1), *ptr);
    ASSERT_EQ(T(1), v[0]);

    // reset
    ref = T(0);

    // test addition-assignment
    ref += T(5);
    ASSERT_EQ(T(5), ref);
    ASSERT_EQ(T(5), *ptr);
    ASSERT_EQ(T(5), v[0]);

    // reset
    ref = T(0);

    // test prefix decrement
    --ref;
    ASSERT_EQ(T(-1), ref);
    ASSERT_EQ(T(-1), *ptr);
    ASSERT_EQ(T(-1), v[0]);

    // reset
    ref = T(0);

    // test subtraction-assignment
    ref -= T(5);
    ASSERT_EQ(T(-5), ref);
    ASSERT_EQ(T(-5), *ptr);
    ASSERT_EQ(T(-5), v[0]);

    // reset
    ref = T(1);

    // test multiply-assignment
    ref *= T(5);
    ASSERT_EQ(T(5), ref);
    ASSERT_EQ(T(5), *ptr);
    ASSERT_EQ(T(5), v[0]);

    // reset
    ref = T(5);

    // test divide-assignment
    ref /= T(5);
    ASSERT_EQ(T(1), ref);
    ASSERT_EQ(T(1), *ptr);
    ASSERT_EQ(T(1), v[0]);

    // test equality of const references
    thrust::device_reference<const T> ref1 = v[0];
    ASSERT_EQ(true, ref1 == ref);
}

TYPED_TEST(DeviceReferenceIntegerTests, TestDeviceReferenceIntegerManipulation)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T>    v(1, T(0));
    thrust::device_ptr<T>       ptr = &v[0];
    thrust::device_reference<T> ref(ptr);

    // reset
    ref = T(5);

    // test modulus-assignment
    ref %= T(5);
    ASSERT_EQ(T(0), ref);
    ASSERT_EQ(T(0), *ptr);
    ASSERT_EQ(T(0), v[0]);

    // reset
    ref = T(1);

    // test left shift-assignment
    ref <<= T(1);
    ASSERT_EQ(T(2), ref);
    ASSERT_EQ(T(2), *ptr);
    ASSERT_EQ(T(2), v[0]);

    // reset
    ref = T(2);

    // test right shift-assignment
    ref >>= T(1);
    ASSERT_EQ(T(1), ref);
    ASSERT_EQ(T(1), *ptr);
    ASSERT_EQ(T(1), v[0]);

    // reset
    ref = T(0);

    // test OR-assignment
    ref |= T(1);
    ASSERT_EQ(T(1), ref);
    ASSERT_EQ(T(1), *ptr);
    ASSERT_EQ(T(1), v[0]);

    // reset
    ref = T(1);

    // test XOR-assignment
    ref ^= T(1);
    ASSERT_EQ(T(0), ref);
    ASSERT_EQ(T(0), *ptr);
    ASSERT_EQ(T(0), v[0]);

    // test equality of const references
    thrust::device_reference<const T> ref1 = v[0];
    ASSERT_EQ(true, ref1 == ref);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceSwap)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T>    v(T(2));
    thrust::device_reference<T> ref1 = v.front();
    thrust::device_reference<T> ref2 = v.back();

    ref1 = T(7);
    ref2 = T(13);

    // test thrust::swap()
    thrust::swap(ref1, ref2);
    ASSERT_EQ(T(13), ref1);
    ASSERT_EQ(T(7), ref2);

    // test .swap()
    ref1.swap(ref2);
    ASSERT_EQ(T(7), ref1);
    ASSERT_EQ(T(13), ref2);
}
