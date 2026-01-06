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

INSTANTIATE_TEST_SUITE_P(pow2_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_3D, pow2_range_3D, pow2_range_3D}),
                             precision_range_sp_dp,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_3D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half_3D,
                                                                               pow2_range_half_3D,
                                                                               pow2_range_half_3D}),
                                                             {fft_precision_half},
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_3D, pow2_range_3D, pow2_range_3D}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range,
                             ooffset_range,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow3_range_3D, pow3_range_3D, pow3_range_3D}),
                             precision_range_sp_dp,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow3_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow3_range_3D, pow3_range_3D, pow3_range_3D}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range,
                             ooffset_range,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow5_range_3D, pow5_range_3D, pow5_range_3D}),
                             precision_range_sp_dp,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow5_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow5_range_3D, pow5_range_3D, pow5_range_3D}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range,
                             ooffset_range,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({prime_range_3D, prime_range_3D, prime_range_3D}),
                             precision_range_sp_dp,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_prime_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({prime_range_3D, prime_range_3D, prime_range_3D}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_3D, pow3_range_3D, prime_range_3D}),
                             precision_range_sp_dp,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_mix_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({pow2_range_3D, pow3_range_3D, prime_range_3D}),
                             precision_range_full,
                             batch_range,
                             stride_range,
                             stride_range,
                             ioffset_range,
                             ooffset_range,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(sbrc_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(
                             test_prob,
                             generate_lengths({sbrc_range_3D, sbrc_range_3D, sbrc_range_3D}),
                             precision_range_sp_dp,
                             sbrc_batch_range_3D,
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             place_range,
                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    inner_batch_3D,
    accuracy_test,
    ::testing::ValuesIn(param_generator(
        test_prob,
        generate_lengths({inner_batch_3D_range, inner_batch_3D_range, inner_batch_3D_range}),
        precision_range_sp_dp,
        inner_batch_3D_batch_range,
        stride_generator_3D_inner_batch(stride_range),
        stride_generator_3D_inner_batch(stride_range),
        ioffset_range_zero,
        ooffset_range_zero,
        place_range,
        true)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    inner_batch_3D_half,
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        generate_lengths({inner_batch_3D_range_half,
                                                          inner_batch_3D_range_half,
                                                          inner_batch_3D_range_half}),
                                        {fft_precision_half},
                                        inner_batch_3D_batch_range,
                                        stride_generator_3D_inner_batch(stride_range),
                                        stride_generator_3D_inner_batch(stride_range),
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        true)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(partial_pass_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             partial_pass_adhoc_3D,
                                                             precision_range_sp_dp,
                                                             partial_pass_batch_range_3D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
#ifdef WIN32
    DISABLED_partial_pass_3D_callback,
#else
    partial_pass_3D_callback,
#endif
    accuracy_test,
    ::testing::ValuesIn(param_generator(test_prob,
                                        partial_pass_adhoc_3D,
                                        precision_range_sp_dp,
                                        partial_pass_batch_range_3D,
                                        stride_range,
                                        stride_range,
                                        ioffset_range_zero,
                                        ooffset_range_zero,
                                        place_range,
                                        false,
                                        true)),
    accuracy_test::TestName);
