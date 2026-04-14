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

INSTANTIATE_TEST_SUITE_P(pow2_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_1D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half_1D}),
                                                             {fft_precision_half},
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_1D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half_1D}),
                                                             {fft_precision_half},
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow3_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow5_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(radX_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({radX_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_radX_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({radX_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_prime_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_mix_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range_1D}),
                                                             precision_range_full,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

// small 1D sizes just need to make sure our factorization isn't
// completely broken, so we just check simple C2C outplace interleaved
INSTANTIATE_TEST_SUITE_P(
    small_1D,
    accuracy_test,
    ::testing::ValuesIn(param_generator_base(
        test_prob,
        {fft_transform_type_complex_forward, fft_transform_type_real_forward},
        generate_lengths({small_1D_sizes()}),
        {fft_precision_single},
        {1},
        [](fft_transform_type                       t,
           const std::vector<fft_result_placement>& place_range,
           const bool                               planar) {
            if(t == fft_transform_type_complex_forward)
                return std::vector<type_place_io_t>{
                    std::make_tuple(t,
                                    place_range[0],
                                    fft_array_type_complex_interleaved,
                                    fft_array_type_complex_interleaved)};
            else
                return std::vector<type_place_io_t>{std::make_tuple(
                    t, place_range[0], fft_array_type_real, fft_array_type_hermitian_interleaved)};
        },
        stride_range,
        stride_range,
        ioffset_range_zero,
        ooffset_range_zero,
        {fft_placement_inplace},
        true)),
    accuracy_test::TestName);

// NB:
// We have known non-unit strides issues for 1D:
// - C2C middle size(for instance, single precision, 8192)
// - C2C large size(for instance, single precision, 524288)
// We need to fix non-unit strides first, and then address non-unit strides + batch tests.
// Then check these problems of R2C and C2R. After that, we could open arbitrary permutations in the
// main tests.
//
// The below test covers non-unit strides, pow of 2, middle sizes, which has SBCC/SBRC kernels
// invloved.

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_complex,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex(test_prob,
                                                generate_lengths({pow2_range_for_stride_1D}),
                                                precision_range_sp_dp,
                                                batch_range_1D,
                                                stride_range_for_pow2_1D,
                                                stride_range_for_pow2_1D,
                                                ioffset_range_zero,
                                                ooffset_range_zero,
                                                place_range,
                                                true)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_complex_half,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex(test_prob,
                                                generate_lengths({pow2_range_for_stride_half_1D}),
                                                {fft_precision_half},
                                                batch_range_1D,
                                                stride_range_for_pow2_1D,
                                                stride_range_for_pow2_1D,
                                                ioffset_range_zero,
                                                ooffset_range_zero,
                                                place_range,
                                                true)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real,
    accuracy_test,
    ::testing::ValuesIn(param_generator_real(test_prob,
                                             generate_lengths({pow2_range_for_stride_1D}),
                                             precision_range_sp_dp,
                                             batch_range_1D,
                                             stride_range_for_pow2_1D,
                                             stride_range_for_pow2_1D,
                                             ioffset_range_zero,
                                             ooffset_range_zero,
                                             place_range,
                                             true)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real_half,
    accuracy_test,
    ::testing::ValuesIn(param_generator_real(test_prob,
                                             generate_lengths({pow2_range_for_stride_half_1D}),
                                             {fft_precision_half},
                                             batch_range_1D,
                                             stride_range_for_pow2_1D,
                                             stride_range_for_pow2_1D,
                                             ioffset_range_zero,
                                             ooffset_range_zero,
                                             place_range,
                                             true)),
    accuracy_test::TestName);

// Create an array parameters for strided 2D batched transforms.
inline auto
    param_generator_complex_1d_batched_2d(const double                             base_prob,
                                          const std::vector<std::vector<size_t>>&  v_lengths,
                                          const std::vector<fft_precision>&        precision_range,
                                          const std::vector<std::vector<size_t>>&  ioffset_range,
                                          const std::vector<std::vector<size_t>>&  ooffset_range,
                                          const std::vector<fft_result_placement>& place_range)
{

    std::vector<fft_params> params;

    for(auto& transform_type : trans_type_range_complex)
    {
        for(const auto& lengths : v_lengths)
        {
            // try to ensure that we are given literal lengths, not
            // something to be passed to generate_lengths
            if(lengths.empty() || lengths.size() > 3)
            {
                assert(false);
                continue;
            }
            for(const auto precision : precision_range)
            {
                for(const auto& types : generate_types(transform_type, place_range, true))
                {
                    for(const auto& ioffset : ioffset_range)
                    {
                        for(const auto& ooffset : ooffset_range)
                        {
                            fft_params param;

                            param.length         = lengths;
                            param.istride        = lengths;
                            param.ostride        = lengths;
                            param.nbatch         = lengths[0];
                            param.precision      = precision;
                            param.transform_type = std::get<0>(types);
                            param.placement      = std::get<1>(types);
                            param.idist          = 1;
                            param.odist          = 1;
                            param.itype          = std::get<2>(types);
                            param.otype          = std::get<3>(types);
                            param.ioffset        = ioffset;
                            param.ooffset        = ooffset;

                            param.validate();

                            const double roll = hash_prob(random_seed, param.token());
                            const double run_prob
                                = base_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                                  * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                                  * (param.is_real() ? real_prob_factor : 1.0);

                            if(roll > run_prob)
                            {
                                if(verbose > 4)
                                {
                                    std::cout << "Test skipped (probability " << run_prob << " > "
                                              << roll << ")\n";
                                }
                                continue;
                            }
                            if(param.valid(0))
                            {
                                params.push_back(param);
                            }
                        }
                    }
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_complex_batched_2D_strided,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex_1d_batched_2d(test_prob,
                                                              generate_lengths({pow2_range_2D}),
                                                              precision_range_sp_dp,
                                                              ioffset_range_zero,
                                                              ooffset_range_zero,
                                                              place_range)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow3_1D_complex_batched_2D_strided,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex_1d_batched_2d(test_prob,
                                                              generate_lengths({pow3_range_2D}),
                                                              precision_range_sp_dp,
                                                              ioffset_range_zero,
                                                              ooffset_range_zero,
                                                              place_range)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow5_1D_complex_batched_2D_strided,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex_1d_batched_2d(test_prob,
                                                              generate_lengths({pow5_range_2D}),
                                                              precision_range_sp_dp,
                                                              ioffset_range_zero,
                                                              ooffset_range_zero,
                                                              place_range)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    prime_1D_complex_batched_2D_strided,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex_1d_batched_2d(test_prob,
                                                              generate_lengths({prime_range_2D}),
                                                              precision_range_sp_dp,
                                                              ioffset_range_zero,
                                                              ooffset_range_zero,
                                                              place_range)),
    accuracy_test::TestName);
