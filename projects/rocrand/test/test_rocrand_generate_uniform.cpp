// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

class rocrand_generate_uniform_tests : public ::testing::TestWithParam<rocrand_rng_type> { };

TEST_P(rocrand_generate_uniform_tests, float_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    float * data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, 1));
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data + 1, 2));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, size));
    HIP_CHECK(hipDeviceSynchronize());

    // No output pointer
    ROCRAND_CHECK(rocrand_generate_uniform(generator, nullptr, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_uniform_tests, float_host_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator_host(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    std::vector<float> data(size);
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data.data(), 1));
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data.data() + 1, 2));
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data.data(), size));
    ROCRAND_CHECK(rocrand_generate_uniform(generator, nullptr, size));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_uniform_tests, double_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    double * data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(double)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data, 1));
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data + 1, 2));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data, size));
    HIP_CHECK(hipDeviceSynchronize());

    // No output pointer
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, nullptr, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}


TEST_P(rocrand_generate_uniform_tests, double_host_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator_host(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    std::vector<double> data(size);
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data.data(), 1));
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data.data() + 1, 2));
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data.data(), size));
    ROCRAND_CHECK(rocrand_generate_uniform_double(generator, nullptr, size));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_uniform_tests, half_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    half * data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(half)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data, 1));
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data + 1, 2));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data, size));
    HIP_CHECK(hipDeviceSynchronize());

    // No output pointer
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, nullptr, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_uniform_tests, half_host_test)

{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator_host(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    std::vector<half> data(size);
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data.data(), 1));
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data.data() + 1, 2));
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, data.data(), size));
    ROCRAND_CHECK(rocrand_generate_uniform_half(generator, nullptr, size));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST(rocrand_generate_uniform_tests, neg_test)
{
    const size_t size = 256;
    void*        data = nullptr;

    rocrand_generator generator = nullptr;

    EXPECT_EQ(rocrand_generate_uniform(generator, static_cast<float*>(data), size),
              ROCRAND_STATUS_NOT_CREATED);

    EXPECT_EQ(rocrand_generate_uniform_double(generator, static_cast<double*>(data), size),
              ROCRAND_STATUS_NOT_CREATED);

    EXPECT_EQ(rocrand_generate_uniform_half(generator, static_cast<half*>(data), size),
              ROCRAND_STATUS_NOT_CREATED);
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_uniform_tests,
                        rocrand_generate_uniform_tests,
                        ::testing::ValuesIn(rng_types));
