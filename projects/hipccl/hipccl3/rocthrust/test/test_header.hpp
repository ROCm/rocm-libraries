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

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_assertions.hpp"
#include "test_utils.hpp"

// Input type parameter
template<class InputType>
struct Params
{
    using input_type = InputType;
};

// Definition of typed test cases with given parameter type
#define TESTS_DEFINE(x, y) \
template<class Params> \
class x : public ::testing::Test \
{ \
public: \
    using input_type = typename Params::input_type; \
}; \
\
TYPED_TEST_CASE(x, y);

// Set of test parameter types

// Host and device vectors of all type as a test parameter
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
> FullTestsParams;


// Host and device vectors of signed type
typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> VectorSignedTestsParams;


// Host and device vectors of integer types as a test parameter
typedef ::testing::Types<
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
Params<thrust::device_vector<unsigned long long>>
> VectorIntegerTestsParams;


// Host and device vectors of signed integer types as a test parameter
typedef ::testing::Types<
Params<thrust::host_vector<short>>,
Params<thrust::host_vector<int>>,
Params<thrust::host_vector<long long>>,
Params<thrust::device_vector<short>>,
Params<thrust::device_vector<int>>,
Params<thrust::device_vector<long long>>
> VectorSignedIntegerTestsParams;


// Host vectors of numerical types as a test parameter
typedef ::testing::Types<
Params<thrust::host_vector<short>>,
Params<thrust::host_vector<int>>,
Params<thrust::host_vector<long long>>,
Params<thrust::host_vector<unsigned short>>,
Params<thrust::host_vector<unsigned int>>,
Params<thrust::host_vector<unsigned long long>>,
Params<thrust::host_vector<float>>,
Params<thrust::host_vector<double>>
> HostVectorTestsParams;


// Host vectors of integer types as a test parameter
typedef ::testing::Types<
Params<thrust::host_vector<short>>,
Params<thrust::host_vector<int>>,
Params<thrust::host_vector<long long>>,
Params<thrust::host_vector<unsigned short>>,
Params<thrust::host_vector<unsigned int>>,
Params<thrust::host_vector<unsigned long long>>
> HostVectorIntegerTestsParams;


// Scalar numerical types
typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> NumericalTestsParams;


// Scalar interger types
typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>
> IntegerTestsParams;


// Scalar signed interger types
typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>
> SignedIntegerTestsParams;


// Scalar unsigned interger types of all lengths
typedef ::testing::Types<
    Params<thrust::detail::uint8_t>,
    Params<thrust::detail::uint16_t>,
    Params<thrust::detail::uint32_t>,
    Params<thrust::detail::uint64_t>
> UnsignedIntegerTestsParams;


// Scalar all interger types
typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<thrust::detail::uint8_t>,
    Params<thrust::detail::uint16_t>,
    Params<thrust::detail::uint32_t>,
    Params<thrust::detail::uint64_t>
> AllIntegerTestsParams;


// Scalar float types
typedef ::testing::Types<
    Params<float>,
    Params<double>
> FloatTestsParams;

