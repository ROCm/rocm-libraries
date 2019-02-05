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

#include <iostream>
#include <type_traits>
#include <cstdlib>
#include <vector>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/memory.h>
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class DeviceReferenceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> DeviceReferenceTestsParams;

TYPED_TEST_CASE(DeviceReferenceTests, DeviceReferenceTestsParams);

template<class Params>
class DeviceReferenceIntegerTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>
> DeviceReferenceIntegerTestsParams;

TYPED_TEST_CASE(DeviceReferenceIntegerTests, DeviceReferenceIntegerTestsParams);



#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceConstructorFromDeviceReference)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> v(1,T(0));
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

  thrust::device_vector<T> v(1,T(0));
  thrust::device_ptr<T> ptr = &v[0];
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

// TODO: No device code available for function: _ZN7rocprim6detail16transform_kernelILj256ELj16ERKjN6thrust10device_ptrIjEES6_NS4_8identityIjEEEEvT2_mT3_T4_
/*TEST(DeviceReferenceTests, TestDeviceReferenceAssignmentFromDeviceReference)
{
  // test same types
  using T0 = int;
  thrust::device_vector<T0> v0(2,0);
  thrust::device_reference<T0> ref0 = v0[0];
  thrust::device_reference<T0> ref1 = v0[1];

  ref0 = 13;

  ref1 = ref0;

  // ref1 equals 13
  ASSERT_EQ(13, ref1);
  ASSERT_EQ(ref0, ref1);

  // test different types
  using T1 = float;
  thrust::device_vector<T1> v1(1,0.0f);
  thrust::device_reference<T1> ref2 = v1[0];

  ref2 = ref1;

  // ref2 equals 13.0f
  ASSERT_EQ(13.0f, ref2);
  ASSERT_EQ(ref0, ref2);
  ASSERT_EQ(ref1, ref2);
}*/

TYPED_TEST(DeviceReferenceTests,TestDeviceReferenceManipulation)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> v(1,T(0));
  thrust::device_ptr<T> ptr = &v[0];
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

TYPED_TEST(DeviceReferenceIntegerTests,TestDeviceReferenceIntegerManipulation)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> v(1,T(0));
  thrust::device_ptr<T> ptr = &v[0];
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

// TODO: Implement swap_ranges
/*TYPED_TEST(DeviceReferenceTests,TestDeviceReferenceSwap)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T> v(T(2));
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
}*/

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
