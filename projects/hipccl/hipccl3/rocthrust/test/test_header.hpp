/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <list>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_assertions.hpp"
#include "test_utils.hpp"

// Input type parameter
template <class InputType>
struct Params
{
    using input_type = InputType;
};

// Definition of typed test cases with given parameter type
#define TESTS_DEFINE(x, y)                              \
    template <class Params>                             \
    class x : public ::testing::Test                    \
    {                                                   \
    public:                                             \
        using input_type = typename Params::input_type; \
    };                                                  \
                                                        \
    TYPED_TEST_CASE(x, y);

// Set of test parameter types

// Host and device vectors of all type as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
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
                         Params<thrust::device_vector<double>>>
    FullTestsParams;

// Host and device vectors of signed type
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<float>>,
                         Params<thrust::host_vector<double>>,
                         Params<thrust::device_vector<short>>,
                         Params<thrust::device_vector<int>>,
                         Params<thrust::device_vector<long long>>,
                         Params<thrust::device_vector<float>>,
                         Params<thrust::device_vector<double>>>
    VectorSignedTestsParams;

// Host and device vectors of integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
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
                         Params<thrust::device_vector<unsigned long long>>>
    VectorIntegerTestsParams;

// Host and device vectors of signed integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::device_vector<short>>,
                         Params<thrust::device_vector<int>>,
                         Params<thrust::device_vector<long long>>>
    VectorSignedIntegerTestsParams;

// Host vectors of numerical types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<unsigned short>>,
                         Params<thrust::host_vector<unsigned int>>,
                         Params<thrust::host_vector<unsigned long long>>,
                         Params<thrust::host_vector<float>>,
                         Params<thrust::host_vector<double>>>
    HostVectorTestsParams;

// Host vectors of integer types as a test parameter
typedef ::testing::Types<Params<thrust::host_vector<short>>,
                         Params<thrust::host_vector<int>>,
                         Params<thrust::host_vector<long long>>,
                         Params<thrust::host_vector<unsigned short>>,
                         Params<thrust::host_vector<unsigned int>>,
                         Params<thrust::host_vector<unsigned long long>>>
    HostVectorIntegerTestsParams;

// Scalar numerical types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>,
                         Params<float>,
                         Params<double>>
    NumericalTestsParams;

// Scalar interger types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>>
    IntegerTestsParams;

// Scalar signed interger types
typedef ::testing::Types<Params<short>, Params<int>, Params<long long>> SignedIntegerTestsParams;

// Scalar unsigned interger types of all lengths
typedef ::testing::Types<Params<thrust::detail::uint8_t>,
                         Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    UnsignedIntegerTestsParams;

// Scalar all interger types
typedef ::testing::Types<Params<short>,
                         Params<int>,
                         Params<long long>,
                         Params<unsigned short>,
                         Params<unsigned int>,
                         Params<unsigned long long>,
                         Params<thrust::detail::uint8_t>,
                         Params<thrust::detail::uint16_t>,
                         Params<thrust::detail::uint32_t>,
                         Params<thrust::detail::uint64_t>>
    AllIntegerTestsParams;

// Scalar float types
typedef ::testing::Types<Params<float>, Params<double>> FloatTestsParams;

// --------------------Input Output test parameters--------
template <class Input, class Output = Input>
struct ParamsInOut
{
    using input_type  = Input;
    using output_type = Output;
};

// Definition of typed test cases with given parameter type
#define TESTS_INOUT_DEFINE(x, y)                               \
    template <class ParamsInOut>                               \
    class x : public ::testing::Test                           \
    {                                                          \
    public:                                                    \
        using input_type  = typename ParamsInOut::input_type;  \
        using output_type = typename ParamsInOut::output_type; \
    };                                                         \
                                                               \
    TYPED_TEST_CASE(x, y);

typedef ::testing::Types<ParamsInOut<short>,
                         ParamsInOut<int>,
                         ParamsInOut<long long>,
                         ParamsInOut<unsigned short>,
                         ParamsInOut<unsigned int>,
                         ParamsInOut<unsigned long long>,
                         ParamsInOut<float>,
                         ParamsInOut<double>,
                         ParamsInOut<int, long long>,
                         ParamsInOut<unsigned int, unsigned long long>,
                         ParamsInOut<float, double>>
    AllInOutTestsParams;
