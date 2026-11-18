// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Tests for the deprecated legacy API. It only exists when
// HIPCUB_USE_LEGACY_API is enabled and only on the rocPRIM backend.

#include <hipcub/config.hpp>

#if HIPCUB_USE_LEGACY_API && defined(__HIP_PLATFORM_AMD__)

#include <gtest/gtest.h>

#include <hipcub/hipcub.hpp>

#include <hipcub/iterator/constant_input_iterator.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>
#include <hipcub/iterator/discard_output_iterator.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>

#include "common_test_header.hpp"

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace
{

    struct plus_four
    {
        __host__ __device__ int operator()(int value) const
        {
            return value + 4;
        }
    };

    __global__ void legacy_operators_kernel(int* results)
    {
        results[0] = hipcub::Max()(3, 7);
        results[1] = hipcub::Min()(3, 7);
        results[2] = hipcub::Sum()(3, 7);
        results[3] = hipcub::Difference()(7, 3);
        results[4] = hipcub::Division()(21, 7);
        results[5] = hipcub::Equality()(5, 5) ? 1 : 0;
        results[6] = hipcub::Inequality()(5, 4) ? 1 : 0;

        hipcub::InequalityWrapper<hipcub::Equality> wrapper{hipcub::Equality{}};
        results[7] = wrapper(5, 4) ? 1 : 0;
    }

} // namespace

TEST(HipcubLegacyApiTests, ThreadOperators)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    int* results = nullptr;
    HIP_CHECK(hipMalloc(&results, 8 * sizeof(int)));
    legacy_operators_kernel<<<1, 1>>>(results);
    HIP_CHECK(hipDeviceSynchronize());

    int host_results[8];
    HIP_CHECK(hipMemcpy(host_results, results, sizeof(host_results), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(results));

    EXPECT_EQ(host_results[0], 7); // Max
    EXPECT_EQ(host_results[1], 3); // Min
    EXPECT_EQ(host_results[2], 10); // Sum
    EXPECT_EQ(host_results[3], 4); // Difference
    EXPECT_EQ(host_results[4], 3); // Division
    EXPECT_EQ(host_results[5], 1); // Equality
    EXPECT_EQ(host_results[6], 1); // Inequality
    EXPECT_EQ(host_results[7], 1); // InequalityWrapper
}

TEST(HipcubLegacyApiTests, Iterators)
{
    hipcub::ConstantInputIterator<int> constant(42);
    EXPECT_EQ(constant[0], 42);
    EXPECT_EQ(constant[5], 42);

    hipcub::CountingInputIterator<int> counting(10);
    EXPECT_EQ(counting[0], 10);
    EXPECT_EQ(counting[3], 13);

    int data[4] = {1, 2, 3, 4};
    hipcub::TransformInputIterator<int, plus_four, int*> transformed(data, plus_four());
    EXPECT_EQ(transformed[0], 5);
    EXPECT_EQ(transformed[3], 8);

    hipcub::DiscardOutputIterator<> discard;
    hipcub::DiscardOutputIterator<> start;
    *discard++ = 1;
    *discard++ = 2;
    EXPECT_EQ(discard - start, 2);
    EXPECT_TRUE(discard != start);
    discard[5] = 3; // Assigns to anything without storing
}

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif

#endif // HIPCUB_USE_LEGACY_API && __HIP_PLATFORM_AMD__
