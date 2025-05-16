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

#include <thrust/detail/config.h>

#include <thrust/device_malloc_allocator.h>
#include <thrust/sequence.h>

#include <initializer_list>
#include <limits>
#include <list>
#include <utility>
#include <vector>

#include "test_real_assertions.hpp"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(VectorTests, FullTestsParams);

TYPED_TEST(VectorTests, TestVectorZeroSize)
{
  using Vector = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v;
  ASSERT_EQ(v.size(), 0lu);
  ASSERT_EQ((v.begin() == v.end()), true);
}

TEST(VectorTests, TestVectorBool)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<bool> h(3);
  thrust::device_vector<bool> d(3);

  h[0] = true;
  h[1] = false;
  h[2] = true;
  d[0] = true;
  d[1] = false;
  d[2] = true;

  ASSERT_EQ(h[0], true);
  ASSERT_EQ(h[1], false);
  ASSERT_EQ(h[2], true);

  ASSERT_EQ(d[0], true);
  ASSERT_EQ(d[1], false);
  ASSERT_EQ(d[2], true);
}

TYPED_TEST(VectorTests, TestVectorInitializerList)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{1, 2, 3};
  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
  ASSERT_EQ(v[2], 3);

  v = {1, 2, 3, 4};
  ASSERT_EQ(v.size(), 4lu);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
  ASSERT_EQ(v[2], 3);
  ASSERT_EQ(v[3], 4);

  const auto alloc = v.get_allocator();
  Vector v2{{1, 2, 3}, alloc};
  ASSERT_EQ(v2.size(), 3lu);
  ASSERT_EQ(v2[0], 1);
  ASSERT_EQ(v2[1], 2);
  ASSERT_EQ(v2[2], 3);
}

TYPED_TEST(VectorTests, TestVectorFrontBack)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  ASSERT_EQ(v.front(), T(0));
  ASSERT_EQ(v.back(), T(2));
}

TYPED_TEST(VectorTests, TestVectorData)
{
  using Vector        = typename TestFixture::input_type;
  using PointerT      = typename Vector::pointer;
  using PointerConstT = typename Vector::const_pointer;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  ASSERT_EQ(0, *v.data());
  ASSERT_EQ(1, *(v.data() + 1));
  ASSERT_EQ(2, *(v.data() + 2));
  ASSERT_EQ(PointerT(&v.front()), v.data());
  ASSERT_EQ(PointerT(&*v.begin()), v.data());
  ASSERT_EQ(PointerT(&v[0]), v.data());

  const Vector& c_v = v;

  ASSERT_EQ(0, *c_v.data());
  ASSERT_EQ(1, *(c_v.data() + 1));
  ASSERT_EQ(2, *(c_v.data() + 2));
  ASSERT_EQ(PointerConstT(&c_v.front()), c_v.data());
  ASSERT_EQ(PointerConstT(&*c_v.begin()), c_v.data());
  ASSERT_EQ(PointerConstT(&c_v[0]), c_v.data());
}

TYPED_TEST(VectorTests, TestVectorElementAssignment)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3);

  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);

  v[0] = 10;
  v[1] = 11;
  v[2] = 12;

  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(v[1], 11);
  ASSERT_EQ(v[2], 12);

  Vector w(3);
  w[0] = v[0];
  w[1] = v[1];
  w[2] = v[2];

  ASSERT_EQ(v, w);
}

TYPED_TEST(VectorTests, TestVectorFromSTLVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::vector<T> stl_vector(3);
  stl_vector[0] = 0;
  stl_vector[1] = 1;
  stl_vector[2] = 2;

  thrust::host_vector<T> v(stl_vector);

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);

  v = stl_vector;

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
}

TYPED_TEST(VectorTests, TestVectorFillAssign)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> v;
  v.assign(3, 13);

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 13);
  ASSERT_EQ(v[1], 13);
  ASSERT_EQ(v[2], 13);
}

TYPED_TEST(VectorTests, TestVectorAssignFromSTLVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::vector<T> stl_vector(3);
  stl_vector[0] = 0;
  stl_vector[1] = 1;
  stl_vector[2] = 2;

  thrust::host_vector<T> v;
  v.assign(stl_vector.begin(), stl_vector.end());

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
}

TYPED_TEST(VectorTests, TestVectorFromBiDirectionalIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v(stl_list.begin(), stl_list.end());

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
}

TYPED_TEST(VectorTests, TestVectorAssignFromBiDirectionalIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v;
  v.assign(stl_list.begin(), stl_list.end());

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
}

TYPED_TEST(VectorTests, TestVectorAssignFromHostVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h(3);
  h[0] = 0;
  h[1] = 1;
  h[2] = 2;

  Vector v;
  v.assign(h.begin(), h.end());

  ASSERT_EQ(v, h);
}

TYPED_TEST(VectorTests, TestVectorToAndFromHostVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h(3);
  h[0] = 0;
  h[1] = 1;
  h[2] = 2;

  Vector v(h);

  ASSERT_EQ(v, h);

  THRUST_DIAG_PUSH
  THRUST_DIAG_SUPPRESS_CLANG("-Wself-assign")
  v = v;
  THRUST_DIAG_POP

  ASSERT_EQ(v, h);

  v[0] = 10;
  v[1] = 11;
  v[2] = 12;

  ASSERT_EQ(h[0], 0);
  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(h[1], 1);
  ASSERT_EQ(v[1], 11);
  ASSERT_EQ(h[2], 2);
  ASSERT_EQ(v[2], 12);

  h = v;

  ASSERT_EQ(v, h);

  h[1] = 11;

  v = h;

  ASSERT_EQ(v, h);
}

TYPED_TEST(VectorTests, TestVectorAssignFromDeviceVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T> d(3);
  d[0] = 0;
  d[1] = 1;
  d[2] = 2;

  Vector v;
  v.assign(d.begin(), d.end());

  ASSERT_EQ(v, d);
}

TYPED_TEST(VectorTests, TestVectorToAndFromDeviceVector)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<T> h(3);
  h[0] = 0;
  h[1] = 1;
  h[2] = 2;

  Vector v(h);

  ASSERT_EQ(v, h);

  THRUST_DIAG_PUSH
  THRUST_DIAG_SUPPRESS_CLANG("-Wself-assign")
  v = v;
  THRUST_DIAG_POP

  ASSERT_EQ(v, h);

  v[0] = 10;
  v[1] = 11;
  v[2] = 12;

  ASSERT_EQ(h[0], 0);
  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(h[1], 1);
  ASSERT_EQ(v[1], 11);
  ASSERT_EQ(h[2], 2);
  ASSERT_EQ(v[2], 12);

  h = v;

  ASSERT_EQ(v, h);

  h[1] = 11;

  v = h;

  ASSERT_EQ(v, h);
}

TYPED_TEST(VectorTests, TestVectorWithInitialValue)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const T init = 17;

  Vector v(3, init);

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], init);
  ASSERT_EQ(v[1], init);
  ASSERT_EQ(v[2], init);
}

TYPED_TEST(VectorTests, TestVectorSwap)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  Vector u(3);
  u[0] = 10;
  u[1] = 11;
  u[2] = 12;

  v.swap(u);

  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(u[0], 0);
  ASSERT_EQ(v[1], 11);
  ASSERT_EQ(u[1], 1);
  ASSERT_EQ(v[2], 12);
  ASSERT_EQ(u[2], 2);
}

TYPED_TEST(VectorTests, TestVectorErasePosition)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(5);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;
  v[3] = 3;
  v[4] = 4;

  v.erase(v.begin() + 2);

  ASSERT_EQ(v.size(), 4lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 3);
  ASSERT_EQ(v[3], 4);

  v.erase(v.begin() + 0);

  ASSERT_EQ(v.size(), 3lu);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 3);
  ASSERT_EQ(v[2], 4);

  v.erase(v.begin() + 2);

  ASSERT_EQ(v.size(), 2lu);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 3);

  v.erase(v.begin() + 1);

  ASSERT_EQ(v.size(), 1lu);
  ASSERT_EQ(v[0], 1);

  v.erase(v.begin() + 0);

  ASSERT_EQ(v.size(), 0lu);
}

TYPED_TEST(VectorTests, TestVectorEraseRange)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(6);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;
  v[3] = 3;
  v[4] = 4;
  v[5] = 5;

  v.erase(v.begin() + 1, v.begin() + 3);

  ASSERT_EQ(v.size(), 4lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 3);
  ASSERT_EQ(v[2], 4);
  ASSERT_EQ(v[3], 5);

  v.erase(v.begin() + 2, v.end());

  ASSERT_EQ(v.size(), 2lu);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 3);

  v.erase(v.begin() + 0, v.begin() + 1);

  ASSERT_EQ(v.size(), 1lu);
  ASSERT_EQ(v[0], 3);

  v.erase(v.begin(), v.end());

  ASSERT_EQ(v.size(), 0lu);
}

TYPED_TEST(VectorTests, TestVectorEquality)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h_a(3);
  thrust::host_vector<T> h_b(3);
  thrust::host_vector<T> h_c(3);
  h_a[0] = 0;
  h_a[1] = 1;
  h_a[2] = 2;
  h_b[0] = 0;
  h_b[1] = 1;
  h_b[2] = 3;
  h_b[0] = 0;
  h_b[1] = 1;

  thrust::device_vector<T> d_a(3);
  thrust::device_vector<T> d_b(3);
  thrust::device_vector<T> d_c(3);
  d_a[0] = 0;
  d_a[1] = 1;
  d_a[2] = 2;
  d_b[0] = 0;
  d_b[1] = 1;
  d_b[2] = 3;
  d_b[0] = 0;
  d_b[1] = 1;

  std::vector<T> s_a(3);
  std::vector<T> s_b(3);
  std::vector<T> s_c(3);
  s_a[0] = 0;
  s_a[1] = 1;
  s_a[2] = 2;
  s_b[0] = 0;
  s_b[1] = 1;
  s_b[2] = 3;
  s_b[0] = 0;
  s_b[1] = 1;

  ASSERT_EQ((h_a == h_a), true);
  ASSERT_EQ((h_a == d_a), true);
  ASSERT_EQ((d_a == h_a), true);
  ASSERT_EQ((d_a == d_a), true);
  ASSERT_EQ((h_b == h_b), true);
  ASSERT_EQ((h_b == d_b), true);
  ASSERT_EQ((d_b == h_b), true);
  ASSERT_EQ((d_b == d_b), true);
  ASSERT_EQ((h_c == h_c), true);
  ASSERT_EQ((h_c == d_c), true);
  ASSERT_EQ((d_c == h_c), true);
  ASSERT_EQ((d_c == d_c), true);

  // test vector vs device_vector
  ASSERT_EQ((s_a == d_a), true);
  ASSERT_EQ((d_a == s_a), true);
  ASSERT_EQ((s_b == d_b), true);
  ASSERT_EQ((d_b == s_b), true);
  ASSERT_EQ((s_c == d_c), true);
  ASSERT_EQ((d_c == s_c), true);

  // test vector vs host_vector
  ASSERT_EQ((s_a == h_a), true);
  ASSERT_EQ((h_a == s_a), true);
  ASSERT_EQ((s_b == h_b), true);
  ASSERT_EQ((h_b == s_b), true);
  ASSERT_EQ((s_c == h_c), true);
  ASSERT_EQ((h_c == s_c), true);

  ASSERT_EQ((h_a == h_b), false);
  ASSERT_EQ((h_a == d_b), false);
  ASSERT_EQ((d_a == h_b), false);
  ASSERT_EQ((d_a == d_b), false);
  ASSERT_EQ((h_b == h_a), false);
  ASSERT_EQ((h_b == d_a), false);
  ASSERT_EQ((d_b == h_a), false);
  ASSERT_EQ((d_b == d_a), false);
  ASSERT_EQ((h_a == h_c), false);
  ASSERT_EQ((h_a == d_c), false);
  ASSERT_EQ((d_a == h_c), false);
  ASSERT_EQ((d_a == d_c), false);
  ASSERT_EQ((h_c == h_a), false);
  ASSERT_EQ((h_c == d_a), false);
  ASSERT_EQ((d_c == h_a), false);
  ASSERT_EQ((d_c == d_a), false);
  ASSERT_EQ((h_b == h_c), false);
  ASSERT_EQ((h_b == d_c), false);
  ASSERT_EQ((d_b == h_c), false);
  ASSERT_EQ((d_b == d_c), false);
  ASSERT_EQ((h_c == h_b), false);
  ASSERT_EQ((h_c == d_b), false);
  ASSERT_EQ((d_c == h_b), false);
  ASSERT_EQ((d_c == d_b), false);

  // test vector vs device_vector
  ASSERT_EQ((s_a == d_b), false);
  ASSERT_EQ((d_a == s_b), false);
  ASSERT_EQ((s_b == d_a), false);
  ASSERT_EQ((d_b == s_a), false);
  ASSERT_EQ((s_a == d_c), false);
  ASSERT_EQ((d_a == s_c), false);
  ASSERT_EQ((s_c == d_a), false);
  ASSERT_EQ((d_c == s_a), false);
  ASSERT_EQ((s_b == d_c), false);
  ASSERT_EQ((d_b == s_c), false);
  ASSERT_EQ((s_c == d_b), false);
  ASSERT_EQ((d_c == s_b), false);

  // test vector vs host_vector
  ASSERT_EQ((s_a == h_b), false);
  ASSERT_EQ((h_a == s_b), false);
  ASSERT_EQ((s_b == h_a), false);
  ASSERT_EQ((h_b == s_a), false);
  ASSERT_EQ((s_a == h_c), false);
  ASSERT_EQ((h_a == s_c), false);
  ASSERT_EQ((s_c == h_a), false);
  ASSERT_EQ((h_c == s_a), false);
  ASSERT_EQ((s_b == h_c), false);
  ASSERT_EQ((h_b == s_c), false);
  ASSERT_EQ((s_c == h_b), false);
  ASSERT_EQ((h_c == s_b), false);
}

TYPED_TEST(VectorTests, TestVectorInequality)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h_a(3);
  thrust::host_vector<T> h_b(3);
  thrust::host_vector<T> h_c(3);
  h_a[0] = 0;
  h_a[1] = 1;
  h_a[2] = 2;
  h_b[0] = 0;
  h_b[1] = 1;
  h_b[2] = 3;
  h_b[0] = 0;
  h_b[1] = 1;

  thrust::device_vector<T> d_a(3);
  thrust::device_vector<T> d_b(3);
  thrust::device_vector<T> d_c(3);
  d_a[0] = 0;
  d_a[1] = 1;
  d_a[2] = 2;
  d_b[0] = 0;
  d_b[1] = 1;
  d_b[2] = 3;
  d_b[0] = 0;
  d_b[1] = 1;

  std::vector<T> s_a(3);
  std::vector<T> s_b(3);
  std::vector<T> s_c(3);
  s_a[0] = 0;
  s_a[1] = 1;
  s_a[2] = 2;
  s_b[0] = 0;
  s_b[1] = 1;
  s_b[2] = 3;
  s_b[0] = 0;
  s_b[1] = 1;

  ASSERT_EQ((h_a != h_a), false);
  ASSERT_EQ((h_a != d_a), false);
  ASSERT_EQ((d_a != h_a), false);
  ASSERT_EQ((d_a != d_a), false);
  ASSERT_EQ((h_b != h_b), false);
  ASSERT_EQ((h_b != d_b), false);
  ASSERT_EQ((d_b != h_b), false);
  ASSERT_EQ((d_b != d_b), false);
  ASSERT_EQ((h_c != h_c), false);
  ASSERT_EQ((h_c != d_c), false);
  ASSERT_EQ((d_c != h_c), false);
  ASSERT_EQ((d_c != d_c), false);

  // test vector vs device_vector
  ASSERT_EQ((s_a != d_a), false);
  ASSERT_EQ((d_a != s_a), false);
  ASSERT_EQ((s_b != d_b), false);
  ASSERT_EQ((d_b != s_b), false);
  ASSERT_EQ((s_c != d_c), false);
  ASSERT_EQ((d_c != s_c), false);

  // test vector vs host_vector
  ASSERT_EQ((s_a != h_a), false);
  ASSERT_EQ((h_a != s_a), false);
  ASSERT_EQ((s_b != h_b), false);
  ASSERT_EQ((h_b != s_b), false);
  ASSERT_EQ((s_c != h_c), false);
  ASSERT_EQ((h_c != s_c), false);

  ASSERT_EQ((h_a != h_b), true);
  ASSERT_EQ((h_a != d_b), true);
  ASSERT_EQ((d_a != h_b), true);
  ASSERT_EQ((d_a != d_b), true);
  ASSERT_EQ((h_b != h_a), true);
  ASSERT_EQ((h_b != d_a), true);
  ASSERT_EQ((d_b != h_a), true);
  ASSERT_EQ((d_b != d_a), true);
  ASSERT_EQ((h_a != h_c), true);
  ASSERT_EQ((h_a != d_c), true);
  ASSERT_EQ((d_a != h_c), true);
  ASSERT_EQ((d_a != d_c), true);
  ASSERT_EQ((h_c != h_a), true);
  ASSERT_EQ((h_c != d_a), true);
  ASSERT_EQ((d_c != h_a), true);
  ASSERT_EQ((d_c != d_a), true);
  ASSERT_EQ((h_b != h_c), true);
  ASSERT_EQ((h_b != d_c), true);
  ASSERT_EQ((d_b != h_c), true);
  ASSERT_EQ((d_b != d_c), true);
  ASSERT_EQ((h_c != h_b), true);
  ASSERT_EQ((h_c != d_b), true);
  ASSERT_EQ((d_c != h_b), true);
  ASSERT_EQ((d_c != d_b), true);

  // test vector vs device_vector
  ASSERT_EQ((s_a != d_b), true);
  ASSERT_EQ((d_a != s_b), true);
  ASSERT_EQ((s_b != d_a), true);
  ASSERT_EQ((d_b != s_a), true);
  ASSERT_EQ((s_a != d_c), true);
  ASSERT_EQ((d_a != s_c), true);
  ASSERT_EQ((s_c != d_a), true);
  ASSERT_EQ((d_c != s_a), true);
  ASSERT_EQ((s_b != d_c), true);
  ASSERT_EQ((d_b != s_c), true);
  ASSERT_EQ((s_c != d_b), true);
  ASSERT_EQ((d_c != s_b), true);

  // test vector vs host_vector
  ASSERT_EQ((s_a != h_b), true);
  ASSERT_EQ((h_a != s_b), true);
  ASSERT_EQ((s_b != h_a), true);
  ASSERT_EQ((h_b != s_a), true);
  ASSERT_EQ((s_a != h_c), true);
  ASSERT_EQ((h_a != s_c), true);
  ASSERT_EQ((s_c != h_a), true);
  ASSERT_EQ((h_c != s_a), true);
  ASSERT_EQ((s_b != h_c), true);
  ASSERT_EQ((h_b != s_c), true);
  ASSERT_EQ((s_c != h_b), true);
  ASSERT_EQ((h_c != s_b), true);
}

TYPED_TEST(VectorTests, TestVectorResizing)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v;

  v.resize(3);

  ASSERT_EQ(v.size(), 3lu);

  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  v.resize(5);

  ASSERT_EQ(v.size(), 5lu);

  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);

  v[3] = 3;
  v[4] = 4;

  v.resize(4);

  ASSERT_EQ(v.size(), 4lu);

  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
  ASSERT_EQ(v[3], 3);

  v.resize(0);

  ASSERT_EQ(v.size(), 0lu);

// TODO remove this WAR
#if ((THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && CUDART_VERSION == 3000)
  || (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP)
  // depending on sizeof(T), we will receive one
  // of two possible exceptions
  try
  {
    v.resize(std::numeric_limits<size_t>::max());
  }
  catch (std::length_error e)
  {}
  catch (std::bad_alloc e)
  {
    // reset the CUDA or HIP error
#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    cudaGetLastError();
#  else
    hipGetLastError();
#  endif
  } // end catch
#endif // ((THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && CUDART_VERSION == 3000) || (THRUST_DEVICE_COMPILER
       // == THRUST_DEVICE_COMPILER_HIP)

  ASSERT_EQ(v.size(), 0lu);
}

TYPED_TEST(VectorTests, TestVectorReserving)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v;

  v.reserve(3);

  ASSERT_GE(v.capacity(), 3lu);

  size_t old_capacity = v.capacity();

  v.reserve(0);

  ASSERT_EQ(v.capacity(), old_capacity);

// TODO remove this WAR
#if ((THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && CUDART_VERSION == 3000)
  || (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP)
  try
  {
    v.reserve(std::numeric_limits<size_t>::max());
  }
  catch (std::length_error e)
  {}
  catch (std::bad_alloc e)
  {}
#endif // ((THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC) && CUDART_VERSION == 3000) || (THRUST_DEVICE_COMPILER
       // == THRUST_DEVICE_COMPILER_HIP)

  ASSERT_EQ(v.capacity(), old_capacity);
}

TEST(VectorTests, TestVectorUninitialisedCopy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> v;
  std::vector<int> std_vector;

  v = std_vector;

  ASSERT_EQ(v.size(), static_cast<size_t>(0));
}

TYPED_TEST(VectorTests, TestVectorShrinkToFit)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v;

  v.reserve(200);

  ASSERT_GE(v.capacity(), 200lu);

  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  v.shrink_to_fit();

  ASSERT_EQ(T(1), v[0]);
  ASSERT_EQ(T(2), v[1]);
  ASSERT_EQ(T(3), v[2]);
  ASSERT_EQ(3lu, v.size());
  ASSERT_EQ(3lu, v.capacity());
}

template <int N>
struct LargeStruct
{
  int data[N];

  THRUST_HOST_DEVICE bool operator==(const LargeStruct& ls) const
  {
    for (int i = 0; i < N; i++)
    {
      if (data[i] != ls.data[i])
      {
        return false;
      }
    }
    return true;
  }
};

TEST(VectorTests, TestVectorContainingLargeType)
{
  // Thrust issue #5
  // http://code.google.com/p/thrust/issues/detail?id=5
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const static int N = 100;
  using T            = LargeStruct<N>;

  thrust::device_vector<T> dv1;
  thrust::host_vector<T> hv1;

  ASSERT_EQ_QUIET(dv1, hv1);

  thrust::device_vector<T> dv2(20);
  thrust::host_vector<T> hv2(20);

  ASSERT_EQ_QUIET(dv2, hv2);

  // initialize tofirst element to something nonzero
  T ls;

  for (int i = 0; i < N; i++)
  {
    ls.data[i] = i;
  }

  thrust::device_vector<T> dv3(20, ls);
  thrust::host_vector<T> hv3(20, ls);

  ASSERT_EQ_QUIET(dv3, hv3);

  // change first element
  ls.data[0] = -13;

  dv3[2] = ls;
  hv3[2] = ls;

  ASSERT_EQ_QUIET(dv3, hv3);
}

TYPED_TEST(VectorTests, TestVectorReversed)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  ASSERT_EQ(3, v.rend() - v.rbegin());
  ASSERT_EQ(3, static_cast<const Vector&>(v).rend() - static_cast<const Vector&>(v).rbegin());
  ASSERT_EQ(3, v.crend() - v.crbegin());

  ASSERT_EQ(2, *v.rbegin());
  ASSERT_EQ(2, *static_cast<const Vector&>(v).rbegin());
  ASSERT_EQ(2, *v.crbegin());

  ASSERT_EQ(1, *(v.rbegin() + 1));
  ASSERT_EQ(0, *(v.rbegin() + 2));

  ASSERT_EQ(0, *(v.rend() - 1));
  ASSERT_EQ(1, *(v.rend() - 2));
}

TYPED_TEST(VectorTests, TestVectorMove)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // test move construction
  Vector v1(3);
  v1[0] = 0;
  v1[1] = 1;
  v1[2] = 2;

  const auto ptr1  = v1.data();
  const auto size1 = v1.size();

  Vector v2(std::move(v1));
  const auto ptr2  = v2.data();
  const auto size2 = v2.size();

  // ensure v1 was left empty
  ASSERT_EQ(true, v1.empty());

  // ensure v2 received the data from before
  ASSERT_EQ(v2[0], 0);
  ASSERT_EQ(v2[1], 1);
  ASSERT_EQ(v2[2], 2);
  ASSERT_EQ(size1, size2);

  // ensure v2 received the pointer from before
  ASSERT_EQ(ptr1, ptr2);

  // test move assignment
  Vector v3(3);
  v3[0] = 3;
  v3[1] = 4;
  v3[2] = 5;

  const auto ptr3  = v3.data();
  const auto size3 = v3.size();

  v2               = std::move(v3);
  const auto ptr4  = v2.data();
  const auto size4 = v2.size();

  // ensure v3 was left empty
  ASSERT_EQ(true, v3.empty());

  // ensure v2 received the data from before
  ASSERT_EQ(v2[0], 3);
  ASSERT_EQ(v2[1], 4);
  ASSERT_EQ(v2[2], 5);
  ASSERT_EQ(size3, size4);

  // ensure v2 received the pointer from before
  ASSERT_EQ(ptr3, ptr4);
}
