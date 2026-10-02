/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <thrust/detail/nv_target.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cpp/vector.h>

#include <memory>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using VariableParams =
  ::testing::Types<Params<signed char>,
                   Params<unsigned char>,
                   Params<short>,
                   Params<unsigned short>,
                   Params<int>,
                   Params<unsigned int>,
                   Params<float>,
                   Params<double>>;

TESTS_DEFINE(AllocatorTests, VariableParams)

// WAR NVIDIA/cccl#1731
// Some tests miscompile for non-CUDA backends on MSVC 2017 and 2019 (though 2022 is fine).
// This is due to a bug in the compiler that breaks __THRUST_DEFINE_HAS_MEMBER_FUNCTION.
#if defined(_MSC_VER) && _MSC_VER <= 1929
#  define WAR_BUG_1731
#endif

// The needs_copy_construct_via_allocator trait depends on has_member_function:
#ifndef WAR_BUG_1731

template <typename T>
struct my_allocator_with_custom_construct1 : thrust::device_malloc_allocator<T>
{
  THRUST_HOST_DEVICE my_allocator_with_custom_construct1() {}

  THRUST_HOST_DEVICE void construct(T* p)
  {
    *p = 13;
  }
};

TYPED_TEST(AllocatorTests, TestAllocatorCustomDefaultConstruct)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::device_vector<T> ref(size, 13);
    thrust::device_vector<T, my_allocator_with_custom_construct1<T>> vec(size);

    ASSERT_EQ_QUIET(ref, vec);
  }
}

template <typename T>
struct my_allocator_with_custom_construct2 : thrust::device_malloc_allocator<T>
{
  THRUST_HOST_DEVICE my_allocator_with_custom_construct2() {}

  template <typename Arg>
  THRUST_HOST_DEVICE void construct(T* p, const Arg&)
  {
    *p = 13;
  }
};

TYPED_TEST(AllocatorTests, TestAllocatorCustomCopyConstruct)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::device_vector<T> ref(size, 13);
    thrust::device_vector<T> copy_from(size, 7);
    thrust::device_vector<T, my_allocator_with_custom_construct2<T>> vec(copy_from.begin(), copy_from.end());

    ASSERT_EQ_QUIET(ref, vec);
  }
}

#endif // !WAR_BUG_1731

// The has_member_destroy trait depends on has_member_function:
#ifndef WAR_BUG_1731

template <typename T>
struct my_allocator_with_custom_destroy
{
  // This is only used with thrust::cpp::vector:
  using system_type = thrust::cpp::tag;

  using value_type      = T;
  using reference       = T&;
  using const_reference = const T&;

  static bool g_state;

  THRUST_HOST my_allocator_with_custom_destroy() {}

  THRUST_HOST my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy& other)
      : use_me_to_alloc(other.use_me_to_alloc)
  {}

  THRUST_HOST ~my_allocator_with_custom_destroy() {}

  THRUST_HOST_DEVICE void destroy(T*) noexcept
  {
    NV_IF_TARGET(NV_IS_HOST, (g_state = true;));
  }

  value_type* allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type* ptr, std::ptrdiff_t n) noexcept
  {
    use_me_to_alloc.deallocate(ptr, n);
  }

  bool operator==(const my_allocator_with_custom_destroy&) const
  {
    return true;
  }

  bool operator!=(const my_allocator_with_custom_destroy& other) const
  {
    return !(*this == other);
  }

  using is_always_equal = thrust::detail::true_type;

  // use composition rather than inheritance
  // to avoid inheriting std::allocator's member
  // function destroy
  std::allocator<T> use_me_to_alloc;
};

template <typename T>
bool my_allocator_with_custom_destroy<T>::g_state = false;

TYPED_TEST(AllocatorTests, TestAllocatorCustomDestroy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  my_allocator_with_custom_destroy<T>::g_state = false;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    {
      thrust::cpp::vector<T, my_allocator_with_custom_destroy<T>> vec(size);
    } // destroy everything

    // state should only be true when there are values to destroy:
    ASSERT_EQ(size > 0, my_allocator_with_custom_destroy<T>::g_state);
  }
}

#endif // !WAR_BUG_1731

template <typename T>
struct my_minimal_allocator
{
  using value_type = T;

  // XXX ideally, we shouldn't require these two aliases
  using reference       = T&;
  using const_reference = const T&;

  THRUST_HOST my_minimal_allocator() {}

  THRUST_HOST my_minimal_allocator(const my_minimal_allocator& other)
      : use_me_to_alloc(other.use_me_to_alloc)
  {}

  THRUST_HOST ~my_minimal_allocator() {}

  value_type* allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type* ptr, std::ptrdiff_t n) noexcept
  {
    use_me_to_alloc.deallocate(ptr, n);
  }

  std::allocator<T> use_me_to_alloc;
};

TYPED_TEST(AllocatorTests, TestAllocatorMinimal)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::cpp::vector<int, my_minimal_allocator<int>> vec(size, 13);

    // XXX copy to h_vec because ASSERT_EQ doesn't know about cpp::vector
    thrust::host_vector<int> h_vec(vec.begin(), vec.end());
    thrust::host_vector<int> ref(size, 13);

    ASSERT_EQ(ref, h_vec);
  }
}

TEST(AllocatorTests, TestAllocatorTraitsRebind)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(
    (_THRUST_STD::is_same<typename thrust::detail::allocator_traits<
                            thrust::device_malloc_allocator<int>>::template rebind_traits<float>::other,
                          typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<float>>>::value),
    true);

  ASSERT_EQ(
    (_THRUST_STD::is_same<
      typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_traits<float>::other,
      typename thrust::detail::allocator_traits<my_minimal_allocator<float>>>::value),
    true);
}

TEST(AllocatorTests, TestAllocatorTraitsRebindCpp11)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(
    (_THRUST_STD::is_same<
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<int>>::template rebind_alloc<float>,
      thrust::device_malloc_allocator<float>>::value),
    true);

  ASSERT_EQ((_THRUST_STD::is_same<
              typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_alloc<float>,
              my_minimal_allocator<float>>::value),
            true);

  ASSERT_EQ(
    (_THRUST_STD::is_same<
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<int>>::template rebind_traits<float>,
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<float>>>::value),
    true);

  ASSERT_EQ((_THRUST_STD::is_same<
              typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_traits<float>,
              typename thrust::detail::allocator_traits<my_minimal_allocator<float>>>::value),
            true);
}
