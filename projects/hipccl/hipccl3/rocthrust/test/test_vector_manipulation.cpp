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

// Google Test
#include <gtest/gtest.h>

// Thrust include
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <random>

template<class T>
inline auto get_random_data(size_t size, T min, T max)
  -> typename std::enable_if<rocprim::is_integral<T>::value, thrust::host_vector<T>>::type
{
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<T> distribution(min, max);
  thrust::host_vector<T> data(size);
  std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
  return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max)
  -> typename std::enable_if<rocprim::is_floating_point<T>::value, thrust::host_vector<T>>::type
{
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<T> distribution(min, max);
  thrust::host_vector<T> data(size);
  std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
  return data;
}

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
class VectorManipulationTests : public ::testing::Test
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
> VectorManipulationTestsParams;

TYPED_TEST_CASE(VectorManipulationTests, VectorManipulationTestsParams);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        0, 1, 2, 12, 63, 64, 211, 256, 344,
        1024, 2048, 5096, 34567, (1 << 17) - 1220
    };
    return sizes;
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(VectorManipulationTests, TestVectorManipulation)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> src;
    if (std::is_floating_point<T>::value)
    {
      src = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
      src = get_random_data<T>(
        size,
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max()
      );
    }

    ASSERT_EQ(src.size(), size);

    // basic initialization
    Vector test0(size);
    Vector test1(size, T(3));
    ASSERT_EQ(test0.size(), size);
    ASSERT_EQ(test1.size(), size);
    ASSERT_EQ((test1 == std::vector<T>(size, T(3))), true);

    #if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
    // XXX MSVC 2005's STL unintentionally uses adl to dispatch advance which
    //     produces an ambiguity between std::advance & thrust::advance
    //     don't produce a KNOWN_FAILURE, just ignore the issue
    #else
    // initializing from other vector
    std::vector<T> stl_vector(src.begin(), src.end());
    Vector cpy0 = src;
    Vector cpy1(stl_vector);
    Vector cpy2(stl_vector.begin(), stl_vector.end());
    // TODO: Implement reduce in system
    //ASSERT_EQ(cpy0, src);
    //ASSERT_EQ(cpy1, src);
    //ASSERT_EQ(cpy2, src);
    #endif

    // resizing
    Vector vec1(src);
    vec1.resize(size + 3);
    ASSERT_EQ(vec1.size(), size + 3);
    vec1.resize(size);
    ASSERT_EQ(vec1.size(), size);
    //ASSERT_EQ(vec1, src); // TODO: Implement reduce in system

    vec1.resize(size + 20, T(11));
    Vector tail(vec1.begin() + size, vec1.end());
    ASSERT_EQ( (tail == std::vector<T>(20, T(11))), true);

    // shrinking a vector should not invalidate iterators
    Iterator first = vec1.begin();
    vec1.resize(10);
    ASSERT_EQ(first, vec1.begin());

    vec1.resize(0);
    ASSERT_EQ(vec1.size(), 0);
    ASSERT_EQ(vec1.empty(), true);
    vec1.resize(10);
    ASSERT_EQ(vec1.size(), 10);
    vec1.clear();
    ASSERT_EQ(vec1.size(), 0);
    vec1.resize(5);
    ASSERT_EQ(vec1.size(), 5);

    // push_back
    Vector vec2;
    for(size_t i = 0; i < 10; ++i)
    {
        ASSERT_EQ(vec2.size(), i);
        vec2.push_back( (T) i );
        ASSERT_EQ(vec2.size(), i + 1);
        for(size_t j = 0; j <= i; j++)
            ASSERT_EQ(vec2[j],     j);
        ASSERT_EQ(vec2.back(), i);
    }

    // pop_back
    for(size_t i = 10; i > 0; --i)
    {
        ASSERT_EQ(vec2.size(), i);
        ASSERT_EQ(vec2.back(), i-1);
        vec2.pop_back();
        ASSERT_EQ(vec2.size(), i-1);
        for(size_t j = 0; j < i; j++)
            ASSERT_EQ(vec2[j], j);
    }
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
