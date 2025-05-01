// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

class rocrand_basic_tests : public ::testing::TestWithParam<rocrand_rng_type> { };

TEST(rocrand_basic_tests, rocrand_get_version_test)
{
    EXPECT_EQ(rocrand_get_version(NULL), ROCRAND_STATUS_OUT_OF_RANGE);
    int version;
    ROCRAND_CHECK(rocrand_get_version(&version));
    EXPECT_EQ(version, ROCRAND_VERSION);
}

TEST(rocrand_basic_tests, rocrand_generator_test)
{
    rocrand_generator gen = 0;
    EXPECT_EQ(gen, static_cast<rocrand_generator>(0));
}

TEST_P(rocrand_basic_tests, rocrand_create_destroy_generator_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator g = NULL;
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    ROCRAND_CHECK(rocrand_destroy_generator(g));
}

TEST_P(rocrand_basic_tests, rocrand_set_stream_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator g = NULL;
    EXPECT_EQ(rocrand_set_stream(g, NULL), ROCRAND_STATUS_NOT_CREATED);
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ROCRAND_CHECK(rocrand_set_stream(g, stream));
    ROCRAND_CHECK(rocrand_set_stream(g, NULL));
    HIP_CHECK(hipStreamDestroy(stream));
    ROCRAND_CHECK(rocrand_destroy_generator(g));
}

TEST_P(rocrand_basic_tests, rocrand_initialize_generator_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator g = NULL;
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    ROCRAND_CHECK(rocrand_initialize_generator(g));
    ROCRAND_CHECK(rocrand_destroy_generator(g));

    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    ROCRAND_CHECK(rocrand_initialize_generator(g));
    ROCRAND_CHECK(rocrand_initialize_generator(g));
    ROCRAND_CHECK(rocrand_initialize_generator(g));
    ROCRAND_CHECK(rocrand_destroy_generator(g));
}

INSTANTIATE_TEST_SUITE_P(rocrand_basic_tests,
                        rocrand_basic_tests,
                        ::testing::ValuesIn(rng_types));
