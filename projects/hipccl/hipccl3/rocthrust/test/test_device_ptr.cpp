/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include _THRUST_STD_INCLUDE(iterator)

#include <cstdlib>
#include <iterator>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#ifdef __cpp_lib_concepts
static_assert(std::indirectly_writable<thrust::device_ptr<uint8_t>, uint8_t>);
#endif // __cpp_lib_concepts
static_assert(_THRUST_STD::indirectly_writable<thrust::device_ptr<uint8_t>, uint8_t>);

#define HIP_CHECK_HMM(condition)                                                 \
  {                                                                              \
    hipError_t error = condition;                                                \
    if (error != hipSuccess)                                                     \
    {                                                                            \
      std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
      exit(error);                                                               \
    }                                                                            \
  }

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

bool supports_hmm()
{
  hipDeviceProp_t device_prop;
  int device_id;
  HIP_CHECK_HMM(hipGetDevice(&device_id));
  HIP_CHECK_HMM(hipGetDeviceProperties(&device_prop, device_id));
  if (device_prop.managedMemory == 1)
  {
    return true;
  }

  return false;
}

bool use_hmm()
{
  char* env      = test::get_env("ROCPRIM_USE_HMM");
  const bool hmm = (env != nullptr) && (strcmp(env, "1") == 0);
  test::clean_env(env);
  return hmm;
}

TEST(DevicePtrTests, TestDevicePointerManipulation)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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

  begin = begin + (int) 1;
  begin = begin - (int) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (unsigned int) 1;
  begin = begin - (unsigned int) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (size_t) 1;
  begin = begin - (size_t) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (ptrdiff_t) 1;
  begin = begin - (ptrdiff_t) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (thrust::device_ptr<int>::difference_type) 1;
  begin = begin - (thrust::device_ptr<int>::difference_type) 1;

  ASSERT_EQ(end - begin, 5);
}

TEST(DevicePtrTests, TestDevicePointerManipulationHmm)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  if (!(use_hmm()))
  {
    return;
  }

  int* data;
  HIP_CHECK(hipMallocManaged((void**) &data, 5 * sizeof(int)));

  thrust::device_ptr<int> begin(&data[0]);
  thrust::device_ptr<int> end(&data[0] + 5);

  ASSERT_EQ(end - begin, 5);

  begin++;
  begin--;

  ASSERT_EQ(end - begin, 5);

  begin += 1;
  begin -= 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (int) 1;
  begin = begin - (int) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (unsigned int) 1;
  begin = begin - (unsigned int) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (size_t) 1;
  begin = begin - (size_t) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (ptrdiff_t) 1;
  begin = begin - (ptrdiff_t) 1;

  ASSERT_EQ(end - begin, 5);

  begin = begin + (thrust::device_ptr<int>::difference_type) 1;
  begin = begin - (thrust::device_ptr<int>::difference_type) 1;

  ASSERT_EQ(end - begin, 5);

  HIP_CHECK(hipFree(data));
}

TYPED_TEST(DevicePtrPrimitiveTests, MakeDevicePointer)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  T* raw_ptr               = 0;
  thrust::device_ptr<T> p0 = thrust::device_pointer_cast(raw_ptr);

  ASSERT_EQ(thrust::raw_pointer_cast(p0), raw_ptr);
  thrust::device_ptr<T> p1 = thrust::device_pointer_cast(p0);
  ASSERT_EQ(p0, p1);
}

TYPED_TEST(DevicePtrTests, TestRawPointerCast)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const std::vector<size_t> sizes = get_sizes();
  for (auto size : sizes)
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::device_vector<T> d_data(size);

    thrust::device_ptr<T> begin(&d_data[0]);

    auto raw_ptr_begin = thrust::raw_pointer_cast(begin);
    if (size > 0)
    {
      ASSERT_NE(raw_ptr_begin, nullptr);
    }

    // Zero input memory
    if (size > 0)
    {
      HIP_CHECK(hipMemset(raw_ptr_begin, 0, sizeof(T) * size));
    }

    // Create unary function
    mark_processed_functor<T> func;
    func.ptr = begin;

    // Run for_each in [0; end] range
    auto end = size < 2 ? size : size / 2;
    auto result =
      thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(end), func);
    ASSERT_EQ(result, thrust::make_counting_iterator<size_t>(end));

    thrust::host_vector<T> h_data = d_data;

    for (size_t i = 0; i < size; i++)
    {
      if (i < end)
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

TEST(DevicePtrTests, TestDevicePointerCompare)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using device_ptr = thrust::device_ptr<T1>;

    device_ptr ptr1 = v1.data();
    device_ptr ptr2 = ptr1 + 1;

    // Equality
    ASSERT_EQ(true, (ptr1 == ptr1));
    ASSERT_EQ(false, (ptr1 != ptr1));

    ASSERT_EQ(false, (ptr1 == ptr2));
    ASSERT_EQ(true, (ptr1 != ptr2));

    // Relations
    ASSERT_EQ(true, (ptr1 < ptr2));
    ASSERT_EQ(true, (ptr1 <= ptr2));
    ASSERT_EQ(true, (ptr2 > ptr1));
    ASSERT_EQ(true, (ptr2 >= ptr1));

    ASSERT_EQ(false, (ptr2 < ptr1));
    ASSERT_EQ(false, (ptr2 <= ptr1));
    ASSERT_EQ(false, (ptr1 > ptr2));
    ASSERT_EQ(false, (ptr1 >= ptr2));
  }

  using T2 = float;
  { // Ensure different element types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr  = thrust::device_ptr<T2>;

    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }

  { // test different pointer types
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T1, thrust::device_system_tag, thrust::tagged_reference<T1, thrust::device_system_tag>>;
    static_assert(!_THRUST_STD::is_same_v<device_ptr, other_ptr>);

    device_ptr ptr1 = v1.data();
    other_ptr ptr2{other_ptr{thrust::raw_pointer_cast(ptr1 + 1)}};

    // Equality
    ASSERT_EQ(true, (ptr1 == ptr1));
    ASSERT_EQ(false, (ptr1 != ptr1));

    ASSERT_EQ(false, (ptr1 == ptr2));
    ASSERT_EQ(true, (ptr1 != ptr2));

    // Relations
    ASSERT_EQ(true, (ptr1 < ptr2));
    ASSERT_EQ(true, (ptr1 <= ptr2));
    ASSERT_EQ(true, (ptr2 > ptr1));
    ASSERT_EQ(true, (ptr2 >= ptr1));

    ASSERT_EQ(false, (ptr2 < ptr1));
    ASSERT_EQ(false, (ptr2 <= ptr1));
    ASSERT_EQ(false, (ptr1 > ptr2));
    ASSERT_EQ(false, (ptr1 >= ptr2));
  }

  { // ensure that different pointer types with different element types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T2, thrust::device_system_tag, thrust::tagged_reference<T1, thrust::device_system_tag>>;

    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }

  // For the non-cuda/hip backends host_system_tag and device_system_tag are comparable
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  { // ensure that pointers with different tags are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T1, thrust::host_system_tag, thrust::tagged_reference<T1, thrust::host_system_tag>>;

    static_assert(!thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP

  { // ensure that different pointers with different tags and types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T2, thrust::host_system_tag, thrust::tagged_reference<T1, thrust::host_system_tag>>;

    // For the non-cuda/hip backends host_system_tag and device_system_tag are comparable
    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>
                  == (THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
                      && THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_HIP));
    static_assert(!_THRUST_STD::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!_THRUST_STD::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }
}
