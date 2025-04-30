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

// TODO: enable 16384, 32768 when omp support is available (takes too
// long!)
const static std::vector<size_t> pow2_range
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

// For the current configuration, half-precision has a fft size limit of 65536
const static std::vector<size_t> pow2_range_half = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

const static std::vector<size_t> pow3_range = {3, 27, 81, 243, 729, 2187, 6561};

const static std::vector<size_t> pow5_range = {5, 25, 125, 625, 3125, 15625};

const static std::vector<size_t> prime_range = {7, 11, 13, 17, 19, 23, 29, 263, 269, 271, 277};

const static std::vector<size_t> mix_range = {56, 120, 336, 2160, 5000, 6000, 8000};

const static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(pow2_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range,
                                                                               pow2_range}),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range,
                                                                               pow2_range}),
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

INSTANTIATE_TEST_SUITE_P(pow2_2D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half,
                                                                               {2, 4, 8, 16, 32}}),
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

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_2D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half,
                                                                               {2, 4, 8, 16, 32}}),
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

INSTANTIATE_TEST_SUITE_P(pow3_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range,
                                                                               pow3_range}),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow3_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range,
                                                                               pow3_range}),
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

INSTANTIATE_TEST_SUITE_P(pow5_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range,
                                                                               pow5_range}),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow5_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range,
                                                                               pow5_range}),
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

INSTANTIATE_TEST_SUITE_P(prime_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range,
                                                                               prime_range}),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_prime_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range,
                                                                               prime_range}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range,
                                                                               mix_range}),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_mix_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range,
                                                                               mix_range}),
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

// test length-1 on one dimension against a variety of non-1 lengths
INSTANTIATE_TEST_SUITE_P(len1_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({{1}, {4, 8, 8192, 3, 27, 7, 11, 5000, 8000}}),
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

// length-1 on the other dimension
INSTANTIATE_TEST_SUITE_P(len1_swap_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({{4, 8, 8192, 3, 27, 7, 11, 5000, 8000}, {1}}),
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
