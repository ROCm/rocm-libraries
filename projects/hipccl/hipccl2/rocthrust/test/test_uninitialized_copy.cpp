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

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/scatter.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class UninitializedCopyTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> UninitializedCopyTestsParams;

TYPED_TEST_CASE(UninitializedCopyTests, UninitializedCopyTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_system &system,
                                   InputIterator,
                                   InputIterator,
                                   ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::uninitialized_copy(sys,
                             vec.begin(),
                             vec.begin(),
                             vec.begin());

  ASSERT_EQ(true, sys.is_valid());

}

template<typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_tag,
                                   InputIterator,
                                   InputIterator,
                                   ForwardIterator result)
{
    *result = 13;
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy(thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template<typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_system &system,
                                     InputIterator,
                                     Size,
                                     ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyNDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::uninitialized_copy_n(sys,
                               vec.begin(),
                               vec.size(),
                               vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_tag,
                                     InputIterator,
                                     Size,
                                     ForwardIterator result)
{
    *result = 13;
    return result;
}

TEST(UninitializedCopyTests, TestUninitializedCopyNDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy_n(thrust::retag<my_tag>(vec.begin()),
                               vec.size(),
                               thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(UninitializedCopyTests, TestUninitializedCopySimplePOD)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector v1(5);
  v1[0] = T(0); v1[1] = T(1); v1[2] = T(2); v1[3] = T(3); v1[4] = T(4);

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());
  ASSERT_EQ(v2[0], T(0));
  ASSERT_EQ(v2[1], T(1));
  ASSERT_EQ(v2[2], T(2));
  ASSERT_EQ(v2[3], T(3));
  ASSERT_EQ(v2[4], T(4));
}

TYPED_TEST(UninitializedCopyTests, TestUninitializedCopyNSimplePOD)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector v1(5);
  v1[0] = T(0); v1[1] = T(1); v1[2] = T(2); v1[3] = T(3); v1[4] = T(4);

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());
  ASSERT_EQ(v2[0], T(0));
  ASSERT_EQ(v2[1], T(1));
  ASSERT_EQ(v2[2], T(2));
  ASSERT_EQ(v2[3], T(3));
  ASSERT_EQ(v2[4], T(4));
}

struct CopyConstructTest
{
  CopyConstructTest(void)
    :copy_constructed_on_host(false),
     copy_constructed_on_device(false)
  {}

  __host__ __device__
  CopyConstructTest(const CopyConstructTest&)
  {
#if defined(THRUST_HIP_DEVICE_CODE)
    copy_constructed_on_device = true;
    copy_constructed_on_host   = false;
#else
    copy_constructed_on_device = false;
    copy_constructed_on_host   = true;
#endif
  }

  __host__ __device__
  CopyConstructTest &operator=(const CopyConstructTest &x)
  {
    copy_constructed_on_host   = x.copy_constructed_on_host;
    copy_constructed_on_device = x.copy_constructed_on_device;
    return *this;
  }

  bool copy_constructed_on_host;
  bool copy_constructed_on_device;
};

TEST(UninitializedCopyTests, TestUninitializedCopyNonPODDevice)
{
  using T = CopyConstructTest;

  thrust::device_vector<T> v1(5), v2(5);

  T x;
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  x = v1[0];
  ASSERT_EQ(true, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

  x = v2[0];
  ASSERT_EQ(true,  x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNNonPODDevice)
{
  using T = CopyConstructTest;

  thrust::device_vector<T> v1(5), v2(5);

  T x;
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  x = v1[0];
  ASSERT_EQ(true, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

  x = v2[0];
  ASSERT_EQ(true,  x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNonPODHost)
{
  using T = CopyConstructTest;

  thrust::host_vector<T> v1(5), v2(5);

  T x;
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  x = v1[0];
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

  x = v2[0];
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(true,  x.copy_constructed_on_host);
}

TEST(UninitializedCopyTests, TestUninitializedCopyNNonPODHost)
{
  using T = CopyConstructTest;

  thrust::host_vector<T> v1(5), v2(5);

  T x;
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  x = v1[0];
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(false, x.copy_constructed_on_host);

  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

  x = v2[0];
  ASSERT_EQ(false, x.copy_constructed_on_device);
  ASSERT_EQ(true,  x.copy_constructed_on_host);
}


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
