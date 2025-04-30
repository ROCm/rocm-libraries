// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/accuracy_test.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_against_fftw.h"
#include "accuracy_tests_range.h"

using ::testing::ValuesIn;

INSTANTIATE_TEST_SUITE_P(pow2_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_2D,
                                                                               pow2_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_2D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half_2D,
                                                                               {2, 4, 8, 16, 32}}),
                                                             {fft_precision_half},
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_2D,
                                                                               pow2_range_2D}),
                                                             precision_range_full,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range_2D,
                                                                               pow3_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow3_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range_2D,
                                                                               pow3_range_2D}),
                                                             precision_range_full,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range_2D,
                                                                               pow5_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow5_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range_2D,
                                                                               pow5_range_2D}),
                                                             precision_range_full,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range_2D,
                                                                               prime_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_prime_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range_2D,
                                                                               prime_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range_2D,
                                                                               mix_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_mix_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range_2D,
                                                                               mix_range_2D}),
                                                             precision_range_full,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

// test length-1 on one dimension against a variety of non-1 lengths
INSTANTIATE_TEST_SUITE_P(len1_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({{1}, {4, 8, 8192, 3, 27, 7, 11, 5000, 8000}}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);

// length-1 on the other dimension
INSTANTIATE_TEST_SUITE_P(len1_swap_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({{4, 8, 8192, 3, 27, 7, 11, 5000, 8000}, {1}}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);
