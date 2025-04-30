// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "../../shared/fft_params.h"

#include "../../shared/accuracy_test.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_against_fftw.h"

using ::testing::ValuesIn;

// Set parameters

// TODO: 512, 1024, 2048 make the tests take too long; re-enable when
// test speed is improved.
static std::vector<size_t> pow2_range      = {4, 8, 16, 32, 128, 256};
static std::vector<size_t> pow2_range_half = {4, 8, 16, 32};

// SBCC+SBRC as a sub-node of a 3D TRTRTR
std::vector<std::vector<size_t>> pow2_adhoc = {{4, 4, 8192}};

static std::vector<size_t> pow3_range = {3, 9, 27, 81, 243};

static std::vector<size_t> pow5_range = {5, 25, 125};

static std::vector<size_t> prime_range = {7, 11, 13, 17, 19, 23, 29};

static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(
    pow2_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow2_range, pow2_range, pow2_range}),
                                        precision_range_sp_dp,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(
    DISABLED_offset_pow2_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow2_range, pow2_range, pow2_range}),
                                        precision_range_full,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range,
                                        ooffset_range,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_3D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_half, pow2_range_half, pow2_range_half}),
                             {fft_precision_half},
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             false,
                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_3D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_half, pow2_range_half, pow2_range_half}),
                             {fft_precision_half},
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range,
                             ooffset_range,
                             place_range,
                             false,
                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow3_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow3_range, pow3_range, pow3_range}),
                                        precision_range_sp_dp,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(
    DISABLED_offset_pow3_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow3_range, pow3_range, pow3_range}),
                                        precision_range_full,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range,
                                        ooffset_range,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow5_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow5_range, pow5_range, pow5_range}),
                                        precision_range_sp_dp,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(
    DISABLED_offset_pow5_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow5_range, pow5_range, pow5_range}),
                                        precision_range_full,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range,
                                        ooffset_range,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    prime_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({prime_range, prime_range, prime_range}),
                                        precision_range_sp_dp,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(
    DISABLED_offset_prime_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({prime_range, prime_range, prime_range}),
                                        precision_range_full,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    mix_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow2_range, pow3_range, prime_range}),
                                        precision_range_sp_dp,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(
    DISABLED_offset_mix_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({pow2_range, pow3_range, prime_range}),
                                        precision_range_full,
                                        batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range,
                                        ooffset_range,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

// Test combinations of SBRC sizes, plus a non-SBRC size (10) to
// exercise fused SBRC+transpose kernels.
static std::vector<size_t> sbrc_range       = {50, 64, 81, 100, 200, 10, 128, 256};
static std::vector<size_t> sbrc_batch_range = {2, 1};
INSTANTIATE_TEST_SUITE_P(
    sbrc_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({sbrc_range, sbrc_range, sbrc_range}),
                                        precision_range_sp_dp,
                                        sbrc_batch_range,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        false)),
    accuracy_test::TestName);

// pick small sizes that will exercise 2D_SINGLE and a couple of sizes that won't
static std::vector<size_t> inner_batch_3D_range       = {4, 8, 16, 32, 20, 24, 64};
static std::vector<size_t> inner_batch_3D_batch_range = {3, 2, 1};

INSTANTIATE_TEST_SUITE_P(
    inner_batch_3D,
    accuracy_test,
    // TODO: enable for real as well, but currently real kernels have
    // trouble with weird strides
    ::testing::ValuesIn(param_generator_complex(
        test_prob,
        generate_lengths({inner_batch_3D_range, inner_batch_3D_range, inner_batch_3D_range}),
        precision_range_sp_dp,
        inner_batch_3D_batch_range,
        stride_generator_3D_inner_batch(stride_range),
        stride_generator_3D_inner_batch(stride_range),
        ioffset_range_zero,
        ooffset_range_zero,
        place_range,
        false,
        false)),
    accuracy_test::TestName);
