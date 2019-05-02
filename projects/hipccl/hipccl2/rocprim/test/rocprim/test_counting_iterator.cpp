// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>
#include <algorithm>
#include <type_traits>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

namespace rp = rocprim;

// Params for tests
template<class InputType>
struct RocprimCountingIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimCountingIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    RocprimCountingIteratorParams<int>,
    RocprimCountingIteratorParams<unsigned int>,
    RocprimCountingIteratorParams<unsigned long>,
    RocprimCountingIteratorParams<size_t>
> RocprimCountingIteratorTestsParams;

TYPED_TEST_CASE(RocprimCountingIteratorTests, RocprimCountingIteratorTestsParams);

template<class T>
struct transform
{
    __device__ __host__
    constexpr T operator()(const T& a) const
    {
        return 5 + a;
    }
};

TYPED_TEST(RocprimCountingIteratorTests, Transform)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::counting_iterator<T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const size_t size = 1024;

    hipStream_t stream = 0; // default

    // Create counting_iterator<U> with random starting point
    Iterator input_begin(test_utils::get_random_value<T>(0, 200));

    std::vector<T> output(size);
    T * d_output;
    HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    // Calculate expected results on host
    std::vector<T> expected(size);
    std::transform(
        input_begin,
        input_begin + size,
        expected.begin(),
        transform<T>()
    );

    // Run
    HIP_CHECK(
        rocprim::transform(
            input_begin, d_output, size,
            transform<T>(), stream, debug_synchronous
        )
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    HIP_CHECK(
        hipMemcpy(
            output.data(), d_output,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Validating results
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
    }

    hipFree(d_output);
}
