// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/mtgp32.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

using rocrand_impl::host::mtgp32_generator;

// Generator API tests
using mtgp32_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<mtgp32_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<mtgp32_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(mtgp32_generator,
                               generator_prng_tests,
                               mtgp32_generator_prng_tests_types);

// Continuity cannot be implemented for MTGP32, as 'offset' is not supported for this
// generator. Therefore, continuity tests fail.
// INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mtgp32,
//                                generator_prng_continuity_tests,
//                                rocrand_mtgp32_generator_prng_tests_types);
