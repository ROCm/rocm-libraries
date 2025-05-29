/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/allocate_unique.h>
#include <thrust/sequence.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/universal_vector.h>

#include <numeric>
#include <vector>

#include "test_header.hpp"

TESTS_DEFINE(UniversalTests, NumericalTestsParams);

// The managed_memory_pointer class should be identified as a
// contiguous_iterator
THRUST_STATIC_ASSERT(thrust::is_contiguous_iterator<thrust::universal_allocator<int>::pointer>::value);

template <typename T>
struct some_object
{
  some_object(T data)
      : m_data(data)
  {}

  void setter(T data)
  {
    m_data = data;
  }
  T getter() const
  {
    return m_data;
  }

private:
  T m_data;
};

TYPED_TEST(UniversalTests, TestUniversalAllocateUnique)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // Simple test to ensure that pointers created with universal_memory_resource
  // can be dereferenced and used with STL code. This is necessary as some
  // STL implementations break when using fancy references that overload
  // operator&, so universal_memory_resource uses a special pointer type that
  // returns regular C++ references that can be safely used host-side.

  // These operations fail to compile with fancy references:
  auto raw = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);
  auto obj = thrust::allocate_unique<some_object<T>>(thrust::universal_allocator<some_object<T>>{}, 42);

  static_assert(std::is_same<decltype(raw.get()), thrust::universal_ptr<T>>::value,
                "Unexpected pointer type returned from std::unique_ptr::get.");
  static_assert(std::is_same<decltype(obj.get()), thrust::universal_ptr<some_object<T>>>::value,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  ASSERT_EQ(*raw, T(42));
  ASSERT_EQ(*raw.get(), T(42));
  ASSERT_EQ(obj->getter(), T(42));
  ASSERT_EQ((*obj).getter(), T(42));
  ASSERT_EQ(obj.get()->getter(), T(42));
  ASSERT_EQ((*obj.get()).getter(), T(42));
}

TYPED_TEST(UniversalTests, TestUniversalIterationRaw)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto array = thrust::allocate_unique_n<T>(thrust::universal_allocator<T>{}, 6, 42);

  static_assert(std::is_same<decltype(array.get()), thrust::universal_ptr<T>>::value,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQ(*iter, T(42));
    ASSERT_EQ(*iter.get(), T(42));
  }
}

TYPED_TEST(UniversalTests, TestUniversalIterationObj)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto array = thrust::allocate_unique_n<some_object<T>>(thrust::universal_allocator<some_object<T>>{}, 6, 42);

  static_assert(std::is_same<decltype(array.get()), thrust::universal_ptr<some_object<T>>>::value,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQ(iter->getter(), T(42));
    ASSERT_EQ((*iter).getter(), T(42));
    ASSERT_EQ(iter.get()->getter(), T(42));
    ASSERT_EQ((*iter.get()).getter(), T(42));
  }
}

TYPED_TEST(UniversalTests, TestUniversalRawPointerCast)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto obj = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);

  static_assert(std::is_same<decltype(obj.get()), thrust::universal_ptr<T>>::value,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  static_assert(std::is_same<decltype(thrust::raw_pointer_cast(obj.get())), T*>::value,
                "Unexpected pointer type returned from thrust::raw_pointer_cast.");

  *thrust::raw_pointer_cast(obj.get()) = T(17);

  ASSERT_EQ(*obj, T(17));
}

TYPED_TEST(UniversalTests, TestUniversalThrustVector)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> host(size);
    thrust::universal_vector<T> universal(size);

    static_assert(
      std::is_same<typename std::decay<decltype(universal)>::type::pointer, thrust::universal_ptr<T>>::value,
      "Unexpected thrust::universal_vector pointer type.");

    thrust::sequence(host.begin(), host.end(), 0);
    thrust::sequence(universal.begin(), universal.end(), 0);

    ASSERT_EQ(host.size(), size);
    ASSERT_EQ(universal.size(), size);

    for (unsigned int i = 0; i < size; i++)
    {
      ASSERT_EQ(host[i], universal[i]);
    }
  }
}

// Verify that a std::vector using the universal allocator will work with
// Standard Library algorithms.
TYPED_TEST(UniversalTests, TestUniversalStdVector)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    std::vector<T> host(size);
    std::vector<T, thrust::universal_allocator<T>> universal(size);

    static_assert(
      std::is_same<typename std::decay<decltype(universal)>::type::pointer, thrust::universal_ptr<T>>::value,
      "Unexpected std::vector pointer type.");

    std::iota(host.begin(), host.end(), 0);
    std::iota(universal.begin(), universal.end(), 0);

    ASSERT_EQ(host.size(), size);
    ASSERT_EQ(universal.size(), size);

    for (unsigned int i = 0; i < size; i++)
    {
      ASSERT_EQ(host[i], universal[i]);
    }
  }
}
