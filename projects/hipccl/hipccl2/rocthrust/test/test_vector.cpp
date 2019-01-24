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

#include <vector>
#include <list>
#include <limits>
#include <utility>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/memory.h>
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
class VectorTests : public ::testing::Test
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
> VectorTestsParams;

TYPED_TEST_CASE(VectorTests, VectorTestsParams);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        0, 1, 2, 12, 63, 64, 211, 256, 344,
        1024, 2048, 5096, 34567, (1 << 17) - 1220
    };
    return sizes;
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(VectorTests, TestVectorZeroSize)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> h_v;

  ASSERT_EQ(h_v.size(), 0);
  ASSERT_EQ((h_v.begin() == h_v.end()), true);

  thrust::device_vector<T> d_v;

  ASSERT_EQ(d_v.size(), 0);
  ASSERT_EQ((d_v.begin() == d_v.end()), true);
}

TEST(VectorTests, TestVectorBool)
{
  thrust::host_vector<bool> h(3);
  thrust::device_vector<bool> d(3);

  h[0] = true; h[1] = false; h[2] = true;
  d[0] = true; d[1] = false; d[2] = true;

  ASSERT_EQ(h[0], true);
  ASSERT_EQ(h[1], false);
  ASSERT_EQ(h[2], true);

  ASSERT_EQ(d[0], true);
  ASSERT_EQ(d[1], false);
  ASSERT_EQ(d[2], true);
}

TYPED_TEST(VectorTests, TestHostVectorFrontBack)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> v(3);
  v[0] = T(0); v[1] = T(1); v[2] = T(2);

  ASSERT_EQ(v.front(), T(0));
  ASSERT_EQ(v.back(),  T(2));
}

TYPED_TEST(VectorTests, TestDeviceVectorFrontBack)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> v(3);
  v[0] = T(0); v[1] = T(1); v[2] = T(2);

  ASSERT_EQ(v.front(), T(0));
  ASSERT_EQ(v.back(),  T(2));
}

TYPED_TEST(VectorTests, TestHostVectorData)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> v(3);
  v[0] = T(0); v[1] = T(1); v[2] = T(2);

  ASSERT_EQ(T(0),       *v.data());
  ASSERT_EQ(T(1),       *(v.data() + 1));
  ASSERT_EQ(T(2),       *(v.data() + 2));
  ASSERT_EQ(&v.front(),  v.data());
  ASSERT_EQ(&*v.begin(), v.data());
  ASSERT_EQ(&v[0],       v.data());

  const thrust::host_vector<T> &c_v = v;

  ASSERT_EQ(T(0),         *c_v.data());
  ASSERT_EQ(T(1),         *(c_v.data() + 1));
  ASSERT_EQ(T(2),         *(c_v.data() + 2));
  ASSERT_EQ(&c_v.front(),  c_v.data());
  ASSERT_EQ(&*c_v.begin(), c_v.data());
  ASSERT_EQ(&c_v[0],       c_v.data());
}

TYPED_TEST(VectorTests, TestDeviceVectorData)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> v(3);
  v[0] = T(0); v[1] = T(1); v[2] = T(2);

  ASSERT_EQ(T(0),       *v.data());
  ASSERT_EQ(T(1),       *(v.data() + 1));
  ASSERT_EQ(T(2),       *(v.data() + 2));
  ASSERT_EQ(&v.front(),  v.data());
  ASSERT_EQ(&*v.begin(), v.data());
  ASSERT_EQ(&v[0],       v.data());

  const thrust::device_vector<T> &c_v = v;

  ASSERT_EQ(T(0),         *c_v.data());
  ASSERT_EQ(T(1),         *(c_v.data() + 1));
  ASSERT_EQ(T(2),         *(c_v.data() + 2));
  ASSERT_EQ(&c_v.front(),  c_v.data());
  ASSERT_EQ(&*c_v.begin(), c_v.data());
  ASSERT_EQ(&c_v[0],       c_v.data());
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
