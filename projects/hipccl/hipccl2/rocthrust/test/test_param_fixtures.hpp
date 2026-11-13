// Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <thrust/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <list>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "../testing/unittest/random.h"
#include "bitwise_repro/bwr_db.hpp"
#include <gtest/gtest.h>

/**
 * test_param_fixtures contains all parameters for typed_test suites
 */

// Input type parameter
template <class InputType, class ExecutionPolicy = std::decay_t<decltype(thrust::hip::par)>>
struct Params
{
  using input_type       = InputType;
  using execution_policy = thrust::detail::host_t;
};

template <class T, class ExecutionPolicy>
struct Params<thrust::device_vector<T>, ExecutionPolicy>
{
  using input_type       = thrust::device_vector<T>;
  using execution_policy = ExecutionPolicy;
};

// Definition of typed test cases with given parameter type
#define TESTS_DEFINE(x, y)                                      \
  template <class Params>                                       \
  class x : public ::testing::Test                              \
  {                                                             \
  public:                                                       \
    using input_type       = typename Params::input_type;       \
    using execution_policy = typename Params::execution_policy; \
  };                                                            \
                                                                \
  TYPED_TEST_SUITE(x, y);

// Set of test parameter types

class large_data
{
public:
  __host__ __device__ large_data()
  {
    data[0] = 0;
  }
  __host__ __device__ large_data(large_data const& val)
  {
    data[0] = val.data[0];
  }
  __host__ __device__ large_data(int n)
  {
    data[0] = static_cast<int8_t>(n);
  }
  large_data& __host__ __device__ operator=(large_data const& val)
  {
    data[0] = val.data[0];
    return *this;
  }
  bool __host__ __device__ operator==(large_data const& val) const
  {
    return data[0] == val.data[0];
  }
  large_data& __host__ __device__ operator++()
  {
    ++data[0];
    return *this;
  }
  __host__ __device__ operator int() const
  {
    return static_cast<int>(data[0]);
  }

  int8_t data[512];
};

template <class T>
bool __host__ __device__ operator==(T const& lhs, large_data const& rhs)
{
  return static_cast<large_data>(lhs).data[0] == rhs.data[0];
}

// Host and device vectors of all type as a test parameter
using FullTestsParams = ::testing::Types<
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
  Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_nosync)>>,
  Params<thrust::device_vector<long long>>,
  Params<thrust::device_vector<unsigned short>>,
  Params<thrust::device_vector<unsigned int>>,
  Params<thrust::device_vector<unsigned long long>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det)>>,
  Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>,
  Params<thrust::device_vector<double>>>;

using FullWithLargeTypesTestsParams = ::testing::Types<
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
  Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_nosync)>>,
  Params<thrust::device_vector<long long>>,
  Params<thrust::device_vector<unsigned short>>,
  Params<thrust::device_vector<unsigned int>>,
  Params<thrust::device_vector<unsigned long long>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det)>>,
  Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>,
  Params<thrust::device_vector<double>>,
  Params<thrust::device_vector<large_data>>>;

// Host and device vectors of signed type
using VectorSignedTestsParams =
  ::testing::Types<Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<long long>>,
                   Params<thrust::host_vector<float>>,
                   Params<thrust::host_vector<double>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::device_vector<long long>>,
                   Params<thrust::device_vector<float>>,
                   Params<thrust::device_vector<double>>>;

// Host and device vectors of integer types as a test parameter
using VectorIntegerTestsParams = ::testing::Types<
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<long long>>,
  Params<thrust::host_vector<unsigned short>>,
  Params<thrust::host_vector<unsigned int>>,
  Params<thrust::host_vector<unsigned long long>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<long long>>,
  Params<thrust::device_vector<unsigned short>>,
  Params<thrust::device_vector<unsigned int>>,
  Params<thrust::device_vector<unsigned long long>>>;

// Host and device vectors of signed integer types as a test parameter
using VectorSignedIntegerTestsParams =
  ::testing::Types<Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<long long>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::device_vector<long long>>>;

// Host vectors of numerical types as a test parameter
using HostVectorTestsParams =
  ::testing::Types<Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<long long>>,
                   Params<thrust::host_vector<unsigned short>>,
                   Params<thrust::host_vector<unsigned int>>,
                   Params<thrust::host_vector<unsigned long long>>,
                   Params<thrust::host_vector<float>>,
                   Params<thrust::host_vector<double>>>;

// Host vectors of integer types as a test parameter
using HostVectorIntegerTestsParams =
  ::testing::Types<Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<long long>>,
                   Params<thrust::host_vector<unsigned short>>,
                   Params<thrust::host_vector<unsigned int>>,
                   Params<thrust::host_vector<unsigned long long>>>;

// Scalar numerical types
using NumericalTestsParams =
  ::testing::Types<Params<short>,
                   Params<int>,
                   Params<long long>,
                   Params<unsigned short>,
                   Params<unsigned int>,
                   Params<unsigned long long>,
                   Params<float>,
                   Params<double>>;

// Scalar integer types
using IntegerTestsParams =
  ::testing::Types<Params<short>,
                   Params<int>,
                   Params<long long>,
                   Params<unsigned short>,
                   Params<unsigned int>,
                   Params<unsigned long long>>;

// Scalar signed integer types
using SignedIntegerTestsParams = ::testing::Types<Params<short>, Params<int>, Params<long long>>;

#if defined(_WIN32) && defined(__HIP__)
// Scalar unsigned integer types of all lengths
using UnsignedIntegerTestsParams =
  ::testing::Types<Params<std::uint16_t>, Params<std::uint32_t>, Params<std::uint64_t>>;

using AllIntegerTestsParams =
  ::testing::Types<Params<short>,
                   Params<int>,
                   Params<long long>,
                   Params<unsigned short>,
                   Params<unsigned int>,
                   Params<unsigned long long>,
                   Params<std::uint16_t>,
                   Params<std::uint32_t>,
                   Params<std::uint64_t>>;
#else
// Scalar unsigned integer types of all lengths
using UnsignedIntegerTestsParams =
  ::testing::Types<Params<std::uint8_t>, Params<std::uint16_t>, Params<std::uint32_t>, Params<std::uint64_t>>;

// Scalar all integer types
using AllIntegerTestsParams =
  ::testing::Types<Params<short>,
                   Params<int>,
                   Params<long long>,
                   Params<unsigned short>,
                   Params<unsigned int>,
                   Params<unsigned long long>,
                   Params<std::uint8_t>,
                   Params<std::uint16_t>,
                   Params<std::uint32_t>,
                   Params<std::uint64_t>>;
#endif

// Scalar float types
using FloatTestsParams = ::testing::Types<Params<float>, Params<double>>;

// --------------------Input Output test parameters--------
template <class Input, class Output = Input>
struct ParamsInOut
{
  using input_type  = Input;
  using output_type = Output;
};

// Definition of typed test cases with given parameter type
#define TESTS_INOUT_DEFINE(x, y)                           \
  template <class ParamsInOut>                             \
  class x : public ::testing::Test                         \
  {                                                        \
  public:                                                  \
    using input_type  = typename ParamsInOut::input_type;  \
    using output_type = typename ParamsInOut::output_type; \
  };                                                       \
                                                           \
  TYPED_TEST_SUITE(x, y);

using AllInOutTestsParams = ::testing::Types<
  ParamsInOut<short>,
  ParamsInOut<int>,
  ParamsInOut<long long>,
  ParamsInOut<unsigned short>,
  ParamsInOut<unsigned int>,
  ParamsInOut<unsigned long long>,
  ParamsInOut<float>,
  ParamsInOut<double>,
  ParamsInOut<int, long long>,
  ParamsInOut<unsigned int, unsigned long long>,
  ParamsInOut<float, double>>;

// --------------------Pairs test parameters--------
template <class T, class U>

struct ParamsPairs
{
  using first_type  = T;
  using second_type = U;
};

#define TESTS_PAIRS_DEFINE(x, y)                           \
  template <class ParamsPairs>                             \
  class x : public ::testing::Test                         \
  {                                                        \
  public:                                                  \
    using first_type  = typename ParamsPairs::first_type;  \
    using second_type = typename ParamsPairs::second_type; \
  };                                                       \
  TYPED_TEST_SUITE(x, y);

using PairsTestsParams = ::testing::
  Types<ParamsPairs<float, float>, ParamsPairs<double, double>, ParamsPairs<float, double>, ParamsPairs<double, float>>;
