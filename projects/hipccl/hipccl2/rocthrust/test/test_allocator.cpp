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
#include <algorithm>
#include <memory>

// Thrust
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cpp/vector.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

struct my_allocator_with_custom_construct1
  : thrust::device_malloc_allocator<int>
{
  __host__ __device__
  my_allocator_with_custom_construct1()
  {}

  template<typename T>
  __host__ __device__
  void construct(T *p)
  {
    *p = 13;
  }
};

TEST(AllocatorTests, TestAllocatorCustomDefaultConstruct)
{
  thrust::device_vector<int> ref(10,13);
  thrust::device_vector<int, my_allocator_with_custom_construct1> vec(10);

  ASSERT_EQ(ref, vec);
}

struct my_allocator_with_custom_construct2
  : thrust::device_malloc_allocator<int>
{
  __host__ __device__
  my_allocator_with_custom_construct2()
  {}

  template<typename T, typename Arg>
  __host__ __device__
  void construct(T *p, const Arg &)
  {
    *p = 13;
  }
};

TEST(AllocatorTests, TestAllocatorCustomCopyConstruct)
{
  thrust::device_vector<int> ref(10,13);
  thrust::device_vector<int> copy_from(10,7);
  thrust::device_vector<int, my_allocator_with_custom_construct2> vec(copy_from.begin(), copy_from.end());

  ASSERT_EQ(ref, vec);
}

static int g_state;

struct my_allocator_with_custom_destroy
{
  typedef int         value_type;
  typedef int &       reference;
  typedef const int & const_reference;

  __host__ __device__
  my_allocator_with_custom_destroy(){}

  __host__ __device__
  my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__ __device__
  ~my_allocator_with_custom_destroy(){}

  template<typename T>
  __host__ __device__
  void destroy(T*)
  {
#if !defined(THRUST_HIP_DEVICE_CODE)
    g_state = 13;
#endif
  }

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }

  // use composition rather than inheritance
  // to avoid inheriting std::allocator's member
  // function construct
  std::allocator<int> use_me_to_alloc;
};

TEST(AllocatorTests, TestAllocatorCustomDestroy)
{
  thrust::cpp::vector<int, my_allocator_with_custom_destroy> vec(10);

  // destroy everything
  vec.shrink_to_fit();

  ASSERT_EQ(13, g_state);
}

struct my_minimal_allocator
{
  typedef int         value_type;

  // XXX ideally, we shouldn't require
  //     these two typedefs
  typedef int &       reference;
  typedef const int & const_reference;

  __host__ __device__
  my_minimal_allocator(){}

  __host__ __device__
  my_minimal_allocator(const my_minimal_allocator &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__ __device__
  ~my_minimal_allocator(){}

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }

  std::allocator<int> use_me_to_alloc;
};

TEST(AllocatorTests, TestAllocatorMinimal)
{
  thrust::cpp::vector<int, my_minimal_allocator> vec(10, 13);

  // XXX copy to h_vec because ASSERT_EQUAL doesn't know about cpp::vector
  thrust::host_vector<int> h_vec(vec.begin(), vec.end());
  thrust::host_vector<int> ref(10, 13);

  ASSERT_EQ(ref, h_vec);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
