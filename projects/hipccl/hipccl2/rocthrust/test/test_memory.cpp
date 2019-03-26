// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Thrust
#include <thrust/memory.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>

#include "test_header.hpp"

TEST(HipThrustMemory, VoidMalloc)
{
  const size_t size = 9001;
  thrust::device_system_tag dev_tag;

  using pointer = thrust::pointer<int, thrust::device_system_tag>;
  // Malloc on device
  auto void_ptr = thrust::malloc(dev_tag, sizeof(int) * size);
  pointer ptr = pointer(static_cast<int*>(void_ptr.get()));
  // Free
  thrust::free(dev_tag, ptr);
}

TEST(HipThrustMemory, TypeMalloc)
{
  const size_t size = 9001;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto ptr = thrust::malloc<int>(dev_tag, sizeof(int) * size);
  // Free
  thrust::free(dev_tag, ptr);
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TEST(HipThrustMemory, MallocUseMemory)
{
  const size_t size = 1024;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto ptr = thrust::malloc<int>(dev_tag, sizeof(int) * size);

  // Try allocated memory with HIP function
  HIP_CHECK(hipMemset(ptr.get(), 0, size * sizeof(int)));

  // Free
  thrust::free(dev_tag, ptr);
}
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

// Define a new system class, as the my_system one is already used with a thrust::sort template definition
// that calls back into sort.cu
class my_memory_system : public thrust::device_execution_policy<my_memory_system>
{
  public:
    my_memory_system(int)
      : correctly_dispatched(false),
        num_copies(0)
    {}

    my_memory_system(const my_memory_system &other)
      : correctly_dispatched(false),
        num_copies(other.num_copies + 1)
    {}

    void validate_dispatch()
    {
      correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
      return correctly_dispatched;
    }

  private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;


    // disallow default construction
    my_memory_system();
};

template<typename T1, typename T2>
bool are_same(const T1 &, const T2 &)
{
  return false;
}


template<typename T>
bool are_same(const T &, const T &)
{
  return true;
}

TEST(MemoryTests, TestSelectSystemDifferentTypes)
{
  using thrust::system::detail::generic::select_system;

  my_memory_system my_sys(0);
  thrust::device_system_tag device_sys;

  // select_system(my_system, device_system_tag) should return device_system_tag (the minimum tag)
  bool is_device_system_tag = are_same(device_sys, select_system(my_sys, device_sys));
  ASSERT_EQ(true, is_device_system_tag);

  // select_system(device_system_tag, my_tag) should return device_system_tag (the minimum tag)
  is_device_system_tag = are_same(device_sys, select_system(device_sys, my_sys));
  ASSERT_EQ(true, is_device_system_tag);
}

TEST(MemoryTests, TestSelectSystemSameTypes)
{
  using thrust::system::detail::generic::select_system;

  my_memory_system my_sys(0);
  thrust::device_system_tag device_sys;
  thrust::host_system_tag host_sys;

  // select_system(host_system_tag, host_system_tag) should return host_system_tag
  bool is_host_system_tag = are_same(host_sys, select_system(host_sys, host_sys));
  ASSERT_EQ(true, is_host_system_tag);

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = are_same(device_sys, select_system(device_sys, device_sys));
  ASSERT_EQ(true, is_device_system_tag);

  // select_system(my_system, my_system) should return my_system
  bool is_my_system = are_same(my_sys, select_system(my_sys, my_sys));
  ASSERT_EQ(true, is_my_system);
}

TEST(MemoryTests, TestGetTemporaryBuffer)
{
  const size_t size = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(dev_tag, size);

  ASSERT_EQ(ptr_and_sz.second, size);

  const int ref_val = 13;
  thrust::device_vector<int> ref(size, ref_val);

  thrust::fill_n(ptr_and_sz.first, size, ref_val);

  ASSERT_EQ(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + size, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(dev_tag, ptr_and_sz.first);
}

TEST(MemoryTests, TestMalloc)
{
  const size_t size = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  pointer ptr = pointer(static_cast<int*>(thrust::malloc(dev_tag, sizeof(int) * size).get()));

  const int ref_val = 13;
  thrust::device_vector<int> ref(size, ref_val);

  thrust::fill_n(ptr, size, ref_val);

  ASSERT_EQ(true, thrust::all_of(ptr, ptr + size, thrust::placeholders::_1 == ref_val));

  thrust::free(dev_tag, ptr);
}

thrust::pointer<void,my_memory_system>
  malloc(my_memory_system &system, std::size_t)
{
  system.validate_dispatch();

  return thrust::pointer<void,my_memory_system>();
}

TEST(MemoryTests, TestMallocDispatchExplicit)
{
  const size_t size = 0;

  my_memory_system sys(0);
  thrust::malloc(sys, size);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename Pointer>
void free(my_memory_system &system, Pointer)
{
  system.validate_dispatch();
}

TEST(MemoryTests, TestFreeDispatchExplicit)
{
  thrust::pointer<my_memory_system,void> ptr;

  my_memory_system sys(0);
  thrust::free(sys, ptr);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename T>
  thrust::pair<thrust::pointer<T,my_memory_system>, std::ptrdiff_t>
    get_temporary_buffer(my_memory_system &system, std::ptrdiff_t size)
{
  system.validate_dispatch();

  thrust::device_system_tag device_sys;
  thrust::pair<thrust::pointer<T, thrust::device_system_tag>, std::ptrdiff_t> result = thrust::get_temporary_buffer<T>(device_sys, size);
  return thrust::make_pair(thrust::pointer<T,my_memory_system>(result.first.get()), result.second);
}

TEST(MemoryTests, TestGetTemporaryBufferDispatchImplicit)
{
  const size_t size = 9001;

  my_memory_system sys(0);
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(sys, size);

  ASSERT_EQ(ptr_and_sz.second, size);
  ASSERT_EQ(true, sys.is_valid());

  const int ref_val = 13;
  thrust::device_vector<int> ref(size, ref_val);

  thrust::fill_n(ptr_and_sz.first, size, ref_val);

  ASSERT_EQ(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + size, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(sys, ptr_and_sz.first);
}

TEST(MemoryTests, TestGetTemporaryBufferDispatchExplicit)
{
  if(are_same(thrust::device_system_tag(), thrust::system::cpp::tag()))
  {
    // XXX cpp uses the internal scalar backend, which currently elides user tags
    // KNOWN_FAILURE;
  }
  else
  {
    thrust::device_vector<int> vec(9001);

    thrust::sequence(vec.begin(), vec.end());
    thrust::reverse(vec.begin(), vec.end());

    // call something we know will invoke get_temporary_buffer
    my_memory_system sys(0);
    thrust::sort(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, thrust::is_sorted(vec.begin(), vec.end()));
    ASSERT_EQ(true, sys.is_valid());
  }
}
