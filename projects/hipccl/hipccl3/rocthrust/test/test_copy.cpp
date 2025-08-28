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

#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <list>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using IntegralVariableParams =
  ::testing::Types<Params<signed char>,
                   Params<unsigned char>,
                   Params<short>,
                   Params<unsigned short>,
                   Params<int>,
                   Params<unsigned int>>;

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::device_vector<
    int,
    thrust::mr::stateless_resource_allocator<int, thrust::universal_host_pinned_memory_resource>>>>;

TESTS_DEFINE(CopyTests, FullWithLargeTypesTestsParams)
TESTS_DEFINE(CopyIntegerTests, IntegerTestsParams)
TESTS_DEFINE(CopyIfSequenceTest, IntegralVariableParams)
TESTS_DEFINE(CopyIfStencilSimpleTest, VectorTestsParams)

TEST(HipThrustCopy, HostToDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t size = 256;
  thrust::device_system_tag dev_tag;
  thrust::host_system_tag host_tag;

  // Malloc on host
  auto h_ptr = thrust::malloc<int>(host_tag, sizeof(int) * size);
  // Malloc on device
  auto d_ptr = thrust::malloc<int>(dev_tag, sizeof(int) * size);

  for (size_t i = 0; i < size; i++)
  {
    *h_ptr = i;
  }

  // Compiles thanks to a temporary fix in
  // thrust/system/detail/generic/for_each.h
  thrust::copy(h_ptr, h_ptr + 256, d_ptr);

  // Free
  thrust::free(host_tag, h_ptr);
  thrust::free(dev_tag, d_ptr);
}

TEST(HipThrustCopy, DeviceToDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t size = 256;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto d_ptr1 = thrust::malloc<int>(dev_tag, sizeof(int) * size);
  auto d_ptr2 = thrust::malloc<int>(dev_tag, sizeof(int) * size);

  // Zero d_ptr1 memory
  HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr1), 0, sizeof(int) * size));
  HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr2), 0xdead, sizeof(int) * size));

  // Copy device->device
  thrust::copy(d_ptr1, d_ptr1 + 256, d_ptr2);

  std::vector<int> output(size);
  HIP_CHECK(hipMemcpy(output.data(), thrust::raw_pointer_cast(d_ptr2), size * sizeof(int), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; i++)
  {
    ASSERT_EQ(output[i], int(0)) << "where index = " << i;
  }

  // Free
  thrust::free(dev_tag, d_ptr1);
  thrust::free(dev_tag, d_ptr2);
}

TEST(CopyTests, TestCopyFromConstIterator)
{
  using T = int;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::vector<T> v{0, 1, 2, 3, 4};

  std::vector<int>::const_iterator begin = v.begin();
  std::vector<int>::const_iterator end   = v.end();

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());

  thrust::host_vector<T> href{0, 1, 2, 3, 4};
  ASSERT_EQ(h, href);
  ASSERT_EQ_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQ(d, dref);
  ASSERT_EQ_QUIET(d_result, d.end());
}

TEST(CopyTests, TestCopyToDiscardIterator)
{
  using T = int;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h_input(5, 1);
  thrust::device_vector<T> d_input = h_input;

  thrust::discard_iterator<> reference(5);

  // copy from host_vector
  thrust::discard_iterator<> h_result = thrust::copy(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

  // copy from device_vector
  thrust::discard_iterator<> d_result = thrust::copy(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

  ASSERT_EQ_QUIET(reference, h_result);
  ASSERT_EQ_QUIET(reference, d_result);
}

TEST(CopyTests, TestCopyToDiscardIteratorZipped)
{
  using T = int;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> h_input(5, 1);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(5);
  thrust::device_vector<T> d_output(5);
  thrust::discard_iterator<> reference(5);

  using Tuple1 = thrust::tuple<thrust::discard_iterator<>, thrust::host_vector<T>::iterator>;
  using Tuple2 = thrust::tuple<thrust::discard_iterator<>, thrust::device_vector<T>::iterator>;

  using ZipIterator1 = thrust::zip_iterator<Tuple1>;
  using ZipIterator2 = thrust::zip_iterator<Tuple2>;

  // copy from host_vector
  ZipIterator1 h_result = thrust::copy(
    thrust::make_zip_iterator(thrust::make_tuple(h_input.begin(), h_input.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(h_input.end(), h_input.end())),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), h_output.begin())));

  // copy from device_vector
  ZipIterator2 d_result = thrust::copy(
    thrust::make_zip_iterator(thrust::make_tuple(d_input.begin(), d_input.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(d_input.end(), d_input.end())),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), d_output.begin())));

  ASSERT_EQ(h_output, h_input);
  ASSERT_EQ(d_output, d_input);
  ASSERT_EQ_QUIET(reference, thrust::get<0>(h_result.get_iterator_tuple()));
  ASSERT_EQ_QUIET(reference, thrust::get<0>(d_result.get_iterator_tuple()));
}

TYPED_TEST(CopyTests, TestCopyMatchingTypes)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  typename thrust::host_vector<T>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
  thrust::host_vector<T> href{0, 1, 2, 3, 4};
  ASSERT_EQ(h, href);
  ASSERT_EQ_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  typename thrust::device_vector<T>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());

  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQ(d, dref);
  ASSERT_EQ_QUIET(d_result, d.end());
}

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244) // '=': conversion from 'int' to '_Ty', possible loss of data

TYPED_TEST(CopyTests, TestCopyMixedTypes)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector with different type
  thrust::host_vector<float> h(5, (float) 10);
  typename thrust::host_vector<float>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
  thrust::host_vector<float> href{0, 1, 2, 3, 4};
  ASSERT_EQ(h, href);
  ASSERT_EQ_QUIET(h_result, h.end());

  // copy to device_vector with different type
  thrust::device_vector<float> d(5, (float) 10);
  typename thrust::device_vector<float>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
  thrust::device_vector<float> dref{0, 1, 2, 3, 4};
  ASSERT_EQ(d, dref);
  ASSERT_EQ_QUIET(d_result, d.end());
}

THRUST_DIAG_POP

TEST(CopyTests, TestCopyVectorBool)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  std::vector<bool> v{true, false, true};

  thrust::host_vector<bool> h(3);
  thrust::device_vector<bool> d(3);

  thrust::copy(v.begin(), v.end(), h.begin());
  thrust::copy(v.begin(), v.end(), d.begin());

  thrust::host_vector<bool> href{true, false, true};
  ASSERT_EQ(h, href);

  thrust::device_vector<bool> dref{true, false, true};
  ASSERT_EQ(d, dref);
}

TYPED_TEST(CopyTests, TestCopyListTo)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // copy from list to Vector
  std::list<T> l{0, 1, 2, 3, 4};

  Vector v(l.size());

  typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQ(v, ref);
  ASSERT_EQ_QUIET(v_result, v.end());

  l.clear();

  thrust::copy(v.begin(), v.end(), std::back_insert_iterator<std::list<T>>(l));

  ASSERT_EQ(l.size(), 5lu);

  typename std::list<T>::const_iterator iter = l.begin();
  ASSERT_EQ(*iter, T(0));
  iter++;
  ASSERT_EQ(*iter, T(1));
  iter++;
  ASSERT_EQ(*iter, T(2));
  iter++;
  ASSERT_EQ(*iter, T(3));
  iter++;
  ASSERT_EQ(*iter, T(4));
  iter++;
}

template <typename T>
struct is_even
{
  THRUST_HOST_DEVICE bool operator()(T x)
  {
    return (x & 1) == 0;
  }
};

template <typename T>
struct is_true
{
  THRUST_HOST_DEVICE bool operator()(T x) const
  {
    return x ? true : false;
  }
};

template <typename T>
struct mod_3
{
  THRUST_HOST_DEVICE unsigned int operator()(T x)
  {
    return x % 3;
  }
};

TYPED_TEST(CopyTests, TestCopyIfSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{0, 1, 2, 3, 4};

  Vector dest(4);

  typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), dest.begin(), is_true<T>());

  Vector ref{1, 2, 3, 4};
  ASSERT_EQ(ref, dest);
  ASSERT_EQ_QUIET(dest.end(), dest_end);
}

TYPED_TEST(CopyIntegerTests, TestCopyIf)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      typename thrust::host_vector<T>::iterator h_new_end;
      typename thrust::device_vector<T>::iterator d_new_end;

      // test with Predicate that returns a bool
      {
        thrust::host_vector<T> h_result(size);
        thrust::device_vector<T> d_result(size);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }

      // test with Predicate that returns a non-bool
      {
        thrust::host_vector<T> h_result(size);
        thrust::device_vector<T> d_result(size);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }
    }
  }
}

TYPED_TEST(CopyIntegerTests, TestCopyIfIntegral)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<T>(size);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator h_new_end;
    typename thrust::device_vector<T>::iterator d_new_end;

    // test with Predicate that returns a bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }

    // test with Predicate that returns a non-bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}

TYPED_TEST(CopyIfSequenceTest, TestCopyIfSequence)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data(size);
    thrust::sequence(h_data.begin(), h_data.end());
    thrust::device_vector<T> d_data(size);
    thrust::sequence(d_data.begin(), d_data.end());

    typename thrust::host_vector<T>::iterator h_new_end;
    typename thrust::device_vector<T>::iterator d_new_end;

    // test with Predicate that returns a bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }

    // test with Predicate that returns a non-bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}

TYPED_TEST(CopyIfStencilSimpleTest, TestCopyIfStencilSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{0, 1, 2, 3, 4};
  Vector s{1, 1, 0, 1, 0};

  Vector dest(3);

  typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), s.begin(), dest.begin(), is_true<T>());

  Vector ref{0, 1, 3};
  ASSERT_EQ(ref, dest);
  ASSERT_EQ_QUIET(dest.end(), dest_end);
}

TEST(CopyLargeTypesTests, TestCopyIfStencilLargeType)
{
  using T = large_data;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data(size);
    thrust::sequence(h_data.begin(), h_data.end());
    thrust::device_vector<T> d_data(size);
    thrust::sequence(d_data.begin(), d_data.end());

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_stencil =
        get_random_data<int>(size, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), seed);
      ;
      thrust::device_vector<T> d_stencil = h_stencil;

      typename thrust::host_vector<T>::iterator h_new_end;
      typename thrust::device_vector<T>::iterator d_new_end;

      // test with Predicate that returns a bool
      {
        thrust::host_vector<T> h_result(size);
        thrust::device_vector<T> d_result(size);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_even<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_even<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }

      // test with Predicate that returns a non-bool
      {
        thrust::host_vector<T> h_result(size);
        thrust::device_vector<T> d_result(size);

        h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), mod_3<T>());
        d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), mod_3<T>());

        h_result.resize(h_new_end - h_result.begin());
        d_result.resize(d_new_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }
    }
  }
}

TYPED_TEST(CopyIntegerTests, TestCopyIfStencil)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data(size);
    thrust::sequence(h_data.begin(), h_data.end());
    thrust::device_vector<T> d_data(size);
    thrust::sequence(d_data.begin(), d_data.end());

    thrust::host_vector<T> h_stencil   = random_integers<T>(size);
    thrust::device_vector<T> d_stencil = random_integers<T>(size);

    typename thrust::host_vector<T>::iterator h_new_end;
    typename thrust::device_vector<T>::iterator d_new_end;

    // test with Predicate that returns a bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_even<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_even<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }

    // test with Predicate that returns a non-bool
    {
      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), mod_3<T>());
      d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), mod_3<T>());

      h_result.resize(h_new_end - h_result.begin());
      d_result.resize(d_new_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}

namespace
{

struct object_with_non_trivial_ctor
{
  // This struct will only properly assign if its `magic` member is
  // set to this certain number.
  static constexpr int MAGIC = 923390;

  int field;
  int magic;

  THRUST_HOST_DEVICE object_with_non_trivial_ctor()
  {
    magic = MAGIC;
    field = 0;
  }
  THRUST_HOST_DEVICE object_with_non_trivial_ctor(int f)
  {
    magic = MAGIC;
    field = f;
  }

  object_with_non_trivial_ctor(const object_with_non_trivial_ctor& x) = default;

  // This non-trivial assignment requires that `this` points to initialized
  // memory
  THRUST_HOST_DEVICE object_with_non_trivial_ctor& operator=(const object_with_non_trivial_ctor& x)
  {
    // To really copy over x's field value, require we have magic value set.
    // If copy_if copies to uninitialized bits, the field will rarely be 923390.
    if (magic == MAGIC)
    {
      field = x.field;
    }
    return *this;
  }
};

struct always_true
{
  THRUST_HOST_DEVICE bool operator()(const object_with_non_trivial_ctor&)
  {
    return true;
  };
};

} // namespace

TEST(CopyTests, TestCopyIfNonTrivial)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // Attempting to copy an object_with_non_trivial_ctor into uninitialized
  // memory will fail:
  {
    static constexpr size_t BufferAlign = alignof(object_with_non_trivial_ctor);
    static constexpr size_t BufferSize  = sizeof(object_with_non_trivial_ctor);
    alignas(BufferAlign) std::array<unsigned char, BufferSize> buffer;

    // Fill buffer with 0s to prevent warnings about uninitialized reads while
    // ensure that the 'magic number' mechanism works as intended:
    std::fill(buffer.begin(), buffer.end(), static_cast<unsigned char>(0));

    object_with_non_trivial_ctor initialized;
    object_with_non_trivial_ctor* uninitialized = reinterpret_cast<object_with_non_trivial_ctor*>(buffer.data());

    object_with_non_trivial_ctor source(42);
    initialized    = source;
    *uninitialized = source;

    ASSERT_EQ(42, initialized.field);
    ASSERT_NE(42, uninitialized->field);
  }

  // This test ensures that we use placement new instead of assigning
  // to uninitialized memory. See Thrust Github issue #1153.
  thrust::device_vector<object_with_non_trivial_ctor> a(10, object_with_non_trivial_ctor(99));
  thrust::device_vector<object_with_non_trivial_ctor> b(10);

  thrust::copy_if(a.begin(), a.end(), b.begin(), always_true());

  for (int i = 0; i < 10; i++)
  {
    object_with_non_trivial_ctor ha(a[i]);
    object_with_non_trivial_ctor hb(b[i]);
    int ia = ha.field;
    int ib = hb.field;

    ASSERT_EQ(ia, ib);
  }
}

TYPED_TEST(CopyTests, TestCopyCountingIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<T> iter(1);

  Vector vec(4);

  thrust::copy(iter, iter + 4, vec.begin());

  ASSERT_EQ(vec[0], 1);
  ASSERT_EQ(vec[1], 2);
  ASSERT_EQ(vec[2], 3);
  ASSERT_EQ(vec[3], 4);
}

TYPED_TEST(CopyTests, TestCopyZipIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // initializer list doesn't work with GCC when
  // Vector = thrust::host_vector<signed char>
  // Vector v1{1, 2, 3};

  Vector v1(3);
  v1[0] = 1;
  v1[1] = 2;
  v1[2] = 3;
  Vector v2(3);
  v2[0] = 4;
  v2[1] = 5;
  v2[2] = 6;
  Vector v3(3, T(0));
  Vector v4(3, T(0));

  thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())),
               thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end())),
               thrust::make_zip_iterator(thrust::make_tuple(v3.begin(), v4.begin())));

  ASSERT_EQ(v1, v3);
  ASSERT_EQ(v2, v4);
}

TYPED_TEST(CopyTests, TestCopyConstantIteratorToZipIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1(3, T(0));
  Vector v2(3, T(0));

  thrust::copy(thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)),
               thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)) + v1.size(),
               thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())));

  Vector ref1{4, 4, 4};
  Vector ref2{7, 7, 7};
  ASSERT_EQ(v1, ref1);
  ASSERT_EQ(v2, ref2);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(CopyTests, TestCopyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy(sys, vec.begin(), vec.end(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(CopyTests, TestCopyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::copy(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate)
{
  system.validate_dispatch();
  return result;
}

TEST(CopyTests, TestCopyIfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy_if(sys, vec.begin(), vec.end(), vec.begin(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate)
{
  *result = 13;
  return result;
}

TEST(CopyTests, TestCopyIfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::copy_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()), 0);

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator
copy_if(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
  system.validate_dispatch();
  return result;
}

TEST(CopyTests, TestCopyIfStencilDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy_if(sys, vec.begin(), vec.end(), vec.begin(), vec.begin(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
  *result = 13;
  return result;
}

TEST(CopyTests, TestCopyIfStencilDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::copy_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.end()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0);

  ASSERT_EQ(13, vec.front());
}

#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE

struct only_set_when_expected_it
{
  long long expected;
  bool* flag;

  THRUST_HOST_DEVICE only_set_when_expected_it operator++() const
  {
    return *this;
  }
  THRUST_HOST_DEVICE only_set_when_expected_it operator*() const
  {
    return *this;
  }
  template <typename Difference>
  THRUST_HOST_DEVICE only_set_when_expected_it operator+(Difference) const
  {
    return *this;
  }
  template <typename Difference>
  THRUST_HOST_DEVICE only_set_when_expected_it operator+=(Difference) const
  {
    return *this;
  }
  template <typename Index>
  THRUST_HOST_DEVICE only_set_when_expected_it operator[](Index) const
  {
    return *this;
  }

  THRUST_DEVICE void operator=(long long value) const
  {
    if (value == expected)
    {
      *flag = true;
    }
  }
};

THRUST_NAMESPACE_BEGIN
namespace detail
{
// We need this type to pass as a non-const ref for unary_transform_functor
// to compile:
template <>
struct is_non_const_reference<only_set_when_expected_it> : thrust::true_type
{};
} // end namespace detail

template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type        = long long;
  using reference         = only_set_when_expected_it;
  using iterator_category = thrust::random_access_device_iterator_tag;
};
THRUST_NAMESPACE_END

void TestCopyWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(0);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::copy(thrust::device, begin, end, out);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQ(has_executed_h, true);
}

TEST(CopyTests, TestCopyWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestCopyWithBigIndexesHelper(30);
  TestCopyWithBigIndexesHelper(31);
  TestCopyWithBigIndexesHelper(32);
  TestCopyWithBigIndexesHelper(33);
}

#endif

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void CopyKernel(int const N, int* in_array, int* out_array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);
    thrust::device_ptr<int> out_begin(out_array);

    thrust::copy(thrust::hip::par, in_begin, in_end, out_begin);
  }
}

TEST(CopyTests, TestCopyDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);

      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output(size);
      hipLaunchKernelGGL(
        CopyKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output[0]));

      ASSERT_EQ(h_data, d_output);
    }
  }
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void CopyIfKernel(int const N, int* in_array, int* out_array, int* out_size)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);
    thrust::device_ptr<int> out_begin(out_array);

    thrust::device_vector<int>::iterator last =
      thrust::copy_if(thrust::hip::par, in_begin, in_end, out_begin, is_even<int>());
    out_size[0] = last - thrust::device_vector<int>::iterator(out_begin);
  }
}

TEST(CopyTests, TestCopyIfDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);

      thrust::host_vector<int> h_output(size, 0);
      auto host_output_last = thrust::copy_if(h_data.begin(), h_data.end(), h_output.begin(), is_even<int>());
      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output_size(1, 0);
      thrust::device_vector<int> d_output(size);

      hipLaunchKernelGGL(
        CopyIfKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output[0]),
        thrust::raw_pointer_cast(&d_output_size[0]));

      h_output.resize(host_output_last - h_output.begin());
      d_output.resize(d_output_size[0]);

      ASSERT_EQ(h_output, d_output);
    }
  }
}
