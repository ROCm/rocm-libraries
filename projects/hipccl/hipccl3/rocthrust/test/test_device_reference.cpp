/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(DeviceReferenceTests, NumericalTestsParams);
TESTS_DEFINE(DeviceReferenceIntegerTests, IntegerTestsParams);

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceConstructorFromDeviceReference)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T> v(1, 0);
  thrust::device_reference<T> ref = v[0];

  // ref equals the object at v[0]
  ASSERT_EQ(v[0], ref);

  // the address of ref equals the address of v[0]
  ASSERT_EQ(&v[0], &ref);

  // modifying v[0] modifies ref
  v[0] = 13;
  ASSERT_EQ(13, ref);
  ASSERT_EQ(v[0], ref);

  // modifying ref modifies v[0]
  ref = 7;
  ASSERT_EQ(7, v[0]);
  ASSERT_EQ(v[0], ref);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceConstructorFromDevicePointer)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T> v(1, 0);
  thrust::device_ptr<T> ptr = &v[0];
  thrust::device_reference<T> ref(ptr);

  // ref equals the object pointed to by ptr
  ASSERT_EQ(*ptr, ref);

  // the address of ref equals ptr
  ASSERT_EQ(ptr, &ref);

  // modifying *ptr modifies ref
  *ptr = 13;
  ASSERT_EQ(13, ref);
  ASSERT_EQ(v[0], ref);

  // modifying ref modifies *ptr
  ref = 7;
  ASSERT_EQ(7, *ptr);
  ASSERT_EQ(v[0], ref);
}

TEST(DeviceReferenceTests, TestDeviceReferenceAssignmentFromDeviceReference)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // test same types
  using T0 = int;
  thrust::device_vector<T0> v0(2, 0);
  thrust::device_reference<T0> ref0 = v0[0];
  thrust::device_reference<T0> ref1 = v0[1];

  ref0 = 13;

  ref1 = ref0;

  // ref1 equals 13
  ASSERT_EQ(13, ref1);
  ASSERT_EQ(ref0, ref1);

  // test different types
  using T1 = float;
  thrust::device_vector<T1> v1(1, 0.0f);
  thrust::device_reference<T1> ref2 = v1[0];

  ref2 = ref1;

  // ref2 equals 13.0f
  ASSERT_EQ(13.0f, ref2);
  ASSERT_EQ(ref0, ref2);
  ASSERT_EQ(ref1, ref2);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceManipulation)
{
  using T1 = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T1> v(1, 0);
  thrust::device_ptr<T1> ptr = &v[0];
  thrust::device_reference<T1> ref(ptr);

  // reset
  ref = 0;

  // test prefix increment
  ++ref;
  ASSERT_EQ(1, ref);
  ASSERT_EQ(1, *ptr);
  ASSERT_EQ(1, v[0]);

  // reset
  ref = 0;

  // test postfix increment
  T1 x1 = ref++;
  ASSERT_EQ(0, x1);
  ASSERT_EQ(1, ref);
  ASSERT_EQ(1, *ptr);
  ASSERT_EQ(1, v[0]);

  // reset
  ref = 0;

  // test addition-assignment
  ref += 5;
  ASSERT_EQ(5, ref);
  ASSERT_EQ(5, *ptr);
  ASSERT_EQ(5, v[0]);

  // reset
  ref = 0;

  // test prefix decrement
  --ref;
  ASSERT_EQ(T1(-1), ref);
  ASSERT_EQ(T1(-1), *ptr);
  ASSERT_EQ(T1(-1), v[0]);

  // reset
  ref = 0;

  // test subtraction-assignment
  ref -= 5;
  ASSERT_EQ(T1(-5), ref);
  ASSERT_EQ(T1(-5), *ptr);
  ASSERT_EQ(T1(-5), v[0]);

  // reset
  ref = 1;

  // test multiply-assignment
  ref *= 5;
  ASSERT_EQ(5, ref);
  ASSERT_EQ(5, *ptr);
  ASSERT_EQ(5, v[0]);

  // reset
  ref = 5;

  // test divide-assignment
  ref /= 5;
  ASSERT_EQ(1, ref);
  ASSERT_EQ(1, *ptr);
  ASSERT_EQ(1, v[0]);

  // test equality of const references
  thrust::device_reference<const T1> ref1 = v[0];
  ASSERT_EQ(true, ref1 == ref);
}

TYPED_TEST(DeviceReferenceIntegerTests, TestDeviceReferenceIntegerManipulation)
{
  using T1 = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T1> v(1, 0);
  thrust::device_ptr<T1> ptr = &v[0];
  thrust::device_reference<T1> ref(ptr);

  // reset
  ref = 5;

  // test modulus-assignment
  ref %= 5;
  ASSERT_EQ(0, ref);
  ASSERT_EQ(0, *ptr);
  ASSERT_EQ(0, v[0]);

  // reset
  ref = 1;

  // test left shift-assignment
  ref <<= 1;
  ASSERT_EQ(2, ref);
  ASSERT_EQ(2, *ptr);
  ASSERT_EQ(2, v[0]);

  // reset
  ref = 2;

  // test right shift-assignment
  ref >>= 1;
  ASSERT_EQ(1, ref);
  ASSERT_EQ(1, *ptr);
  ASSERT_EQ(1, v[0]);

  // reset
  ref = 0;

  // test OR-assignment
  ref |= 1;
  ASSERT_EQ(1, ref);
  ASSERT_EQ(1, *ptr);
  ASSERT_EQ(1, v[0]);

  // reset
  ref = 1;

  // test XOR-assignment
  ref ^= 1;
  ASSERT_EQ(0, ref);
  ASSERT_EQ(0, *ptr);
  ASSERT_EQ(0, v[0]);

  // test equality of const references
  thrust::device_reference<const T1> ref1 = v[0];
  ASSERT_EQ(true, ref1 == ref);
}

TYPED_TEST(DeviceReferenceTests, TestDeviceReferenceSwap)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T> v(2);
  thrust::device_reference<T> ref1 = v.front();
  thrust::device_reference<T> ref2 = v.back();

  ref1 = 7;
  ref2 = 13;

  // test ADL two-step swap
  using _THRUST_STD::swap;
  swap(ref1, ref2);
  ASSERT_EQ(13, ref1);
  ASSERT_EQ(7, ref2);

  // test .swap()
  ref1.swap(ref2);
  ASSERT_EQ(7, ref1);
  ASSERT_EQ(13, ref2);
}

TEST(DeviceReferenceTests, TestDeviceReferenceCompare)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using device_ref = thrust::device_reference<T1>;

    device_ref ref1 = v1.front();
    device_ref ref2 = v1.back();

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  using T2                     = float;
  thrust::device_vector<T2> v2 = {42.0f, 1337.0f};
  { // test different element type
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::device_reference<T2>;

    device_ref ref1 = v1.front();
    other_ref ref2  = v2.back();

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  { // test different reference types
    using device_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!_THRUST_STD::is_same_v<device_ref, other_ref>);

    device_ref ref1 = v1.front();
    other_ref ref2{other_pointer{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  { // test different reference types with different element types
    using device_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T2, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!_THRUST_STD::is_same_v<device_ref, other_ref>);

    device_ref ref1 = v1.front();
    other_ref ref2{other_pointer{thrust::raw_pointer_cast(v2.data() + 1)}};

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  // For the non-cuda/hip backends host_system_tag and device_system_tag are comparable
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  { // ensure that references with different tags are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T1, thrust::host_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<device_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }

  { // ensure that references with different tags and types are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T2, thrust::host_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<device_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP

  { // ensure that references with incomparable element type are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<_THRUST_STD::pair<T1, T2>, thrust::device_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<device_ref, other_ref>);

    static_assert(
      thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<T1, _THRUST_STD::pair<T1, T2>>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }
}

TEST(DeviceReferenceTests, TestTaggedReferenceCompare)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using tagged_ref = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using tagged_ptr = typename tagged_ref::pointer;

    tagged_ref ref1{tagged_ptr{thrust::raw_pointer_cast(v1.data())}};
    tagged_ref ref2{tagged_ptr{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  using T2                     = float;
  thrust::device_vector<T2> v2 = {42.0f, 1337.0f};
  { // test different element type
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::device_reference<T2>;

    tagged_ref ref1 = v1.front();
    other_ref ref2  = v2.back();

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  { // test different reference types
    using tagged_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!_THRUST_STD::is_same_v<tagged_ref, other_ref>);

    tagged_ref ref1 = v1.front();
    other_ref ref2{other_pointer{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

  { // test different reference types with different element types
    using tagged_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T2, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!_THRUST_STD::is_same_v<tagged_ref, other_ref>);

    tagged_ref ref1 = v1.front();
    other_ref ref2{other_pointer{thrust::raw_pointer_cast(v2.data() + 1)}};

    // Equality
    ASSERT_EQ(true, (ref1 == ref1));
    ASSERT_EQ(false, (ref1 != ref1));

    ASSERT_EQ(false, (ref1 == ref2));
    ASSERT_EQ(true, (ref1 != ref2));

    // Relations
    ASSERT_EQ(true, (ref1 < ref2));
    ASSERT_EQ(true, (ref1 <= ref2));
    ASSERT_EQ(true, (ref2 > ref1));
    ASSERT_EQ(true, (ref2 >= ref1));

    ASSERT_EQ(false, (ref2 < ref1));
    ASSERT_EQ(false, (ref2 <= ref1));
    ASSERT_EQ(false, (ref1 > ref2));
    ASSERT_EQ(false, (ref1 >= ref2));
  }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  { // ensure that references with different tags are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T1, thrust::host_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<tagged_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }

  { // ensure that references with different tags and types are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T2, thrust::host_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<tagged_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP

  { // ensure that references with incomparable element type are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<_THRUST_STD::pair<T1, T2>, thrust::device_system_tag>;
    static_assert(!_THRUST_STD::is_same_v<tagged_ref, other_ref>);

    static_assert(
      thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }
}
