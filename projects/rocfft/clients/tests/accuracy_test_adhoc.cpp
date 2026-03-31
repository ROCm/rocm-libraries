// Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/accuracy_test.h"
#include "../../shared/params_gen.h"

std::vector<std::vector<size_t>> adhoc_sizes = {
    // sizes that exercise L1D_TRTRT subplan of 2D_RTRT or 3D_TRTRTR
    {1, 220},
    {1, 330},
    {81, 220, 36},

    // L1D_CC subplan of 3D_TRTRTR
    {4, 4, 8192},

    // SBRC 192 with special param
    {192, 192, 192},
    {192, 84, 84},

    // Failure with build_CS_3D_BLOCK_RC
    {680, 128, 128},

    // Large 1D primes that fall above the block threshold (length 262144).
    // Bluestein requires two forwards and one inverse FFTs, and the plan
    // for these sizes breakdown these FFTs either as:
    // L1D_TRTRT (T + STOCKHAM + T + STOCKHAM + T) for lengthBlue <= 4096^2
    // or
    // L1D_TRTRT (T + L1D_CC + STOCKHAM_BL_CC + STOCHMAM_BL_RC + T + STOCKHAM + T)
    // for lengthBlue > 4096^2.
    {196597},
    {25165813},

    // 2D single-kernel bluestein size combined with multi-kernel bluestein
    {19, 2053},

    // TILE_UNALIGNED type of SBRC 3D ERC
    {98, 98, 98},

    // 3D_BLOCK_CR
    {336, 336, 56},
};

const static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             adhoc_sizes,
                                                             precision_range_sp_dp,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             adhoc_sizes,
                                                             precision_range_full,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

// Test that dist is ignored for batch-1 transforms.  Normally,
// in-place transforms require same dist, but for batch-1 dist isn't
// used for anything and differing dist should be allowed.
inline auto param_permissive_iodist()
{
    std::vector<std::vector<size_t>> lengths = adhoc_sizes;
    lengths.push_back({4});

    std::vector<fft_params> params;
    for(const auto precision : precision_range_sp_dp)
    {
        for(const auto trans_type : trans_type_range)
        {
            for(const auto& types : generate_types(trans_type, place_range, true))
            {
                if(std::get<1>(types) != fft_placement_inplace)
                    continue;
                for(const auto& len : lengths)
                {
                    fft_params param;

                    param.length         = len;
                    param.precision      = precision;
                    param.idist          = 2;
                    param.odist          = 3;
                    param.transform_type = std::get<0>(types);
                    param.placement      = std::get<1>(types);
                    param.itype          = std::get<2>(types);
                    param.otype          = std::get<3>(types);

                    param.validate();

                    const double roll = hash_prob(random_seed, param.token());
                    const double run_prob
                        = test_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                          * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                          * (param.is_real() ? real_prob_factor : 1.0);

                    if(roll > run_prob)
                    {
                        if(verbose > 4)
                        {
                            std::cout << "Test skipped (probability " << run_prob << " > " << roll
                                      << ")\n";
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

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_dist,
                         accuracy_test,
                         ::testing::ValuesIn(param_permissive_iodist()),
                         accuracy_test::TestName);

inline auto param_adhoc_colmajor()
{
    // generate basic FFTs of adhoc sizes
    auto params = param_generator(test_prob,
                                  adhoc_sizes,
                                  {fft_precision_single},
                                  {2},
                                  stride_range,
                                  stride_range,
                                  ioffset_range_zero,
                                  ooffset_range_zero,
                                  {fft_placement_notinplace},
                                  false);

    // remove any params that are:
    // - 1D (not enough dims to swap)
    // - real-complex 2D (we only get to play with higher dims, so
    //   again not enough dims to swap)
    params.erase(std::remove_if(params.begin(),
                                params.end(),
                                [](const fft_params& param) {
                                    if(param.length.size() == 1)
                                        return true;
                                    if(param.length.size() == 2)
                                    {
                                        if(param.transform_type == fft_transform_type_real_forward
                                           || param.transform_type
                                                  == fft_transform_type_real_inverse)
                                            return true;
                                    }
                                    return false;
                                }),
                 params.end());

    // reverse length/stride order on remaining params to make them
    // col-major
    std::for_each(params.begin(), params.end(), [](fft_params& param) {
        size_t start_dim = 0;
        // for real-complex we can't touch the fastest dim
        if(param.transform_type == fft_transform_type_real_forward
           || param.transform_type == fft_transform_type_real_inverse)
            ++start_dim;
        std::reverse(param.length.rbegin() + start_dim, param.length.rend());
        std::reverse(param.istride.rbegin() + start_dim, param.istride.rend());
        std::reverse(param.ostride.rbegin() + start_dim, param.ostride.rend());
    });
    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_colmajor,
                         accuracy_test,
                         ::testing::ValuesIn(param_adhoc_colmajor()),
                         accuracy_test::TestName);

inline auto param_adhoc_stride()
{
    std::vector<fft_params> params;

    for(const auto precision : precision_range_full)
    {
        for(const auto& types : generate_types(fft_transform_type_complex_forward,
                                               {fft_placement_inplace, fft_placement_notinplace},
                                               true))
        {
            // 2D with non-contiguous strides and dist
            fft_params param;
            param.length         = {2, 35};
            param.precision      = precision;
            param.idist          = 200;
            param.odist          = 200;
            param.transform_type = fft_transform_type_complex_forward;
            param.nbatch         = 2;
            param.placement      = std::get<1>(types);
            param.itype          = std::get<2>(types);
            param.otype          = std::get<3>(types);
            param.istride        = {90, 2};
            param.ostride        = {90, 2};
            params.push_back(param);
        }

        // test C2R/R2C with non-contiguous higher strides and dist - we
        // want unit stride for length0 so we do the even-length optimization
        for(const auto& types :
            generate_types(fft_transform_type_real_forward, {fft_placement_notinplace}, true))
        {

            fft_params param;
            param.length         = {4, 4, 4};
            param.precision      = precision;
            param.idist          = 0;
            param.odist          = 0;
            param.transform_type = fft_transform_type_real_forward;
            param.nbatch         = 2;
            param.placement      = std::get<1>(types);
            param.itype          = std::get<2>(types);
            param.otype          = std::get<3>(types);
            param.istride        = {16, 4, 1};
            param.ostride        = {16, 4, 1};

            param.validate();

            {
                const double roll = hash_prob(random_seed, param.token());
                const double run_prob
                    = test_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                      * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                      * (param.is_real() ? real_prob_factor : 1.0);

                if(roll > run_prob)
                {
                    if(verbose > 4)
                    {
                        std::cout << "Test skipped (probability " << run_prob << " > " << roll
                                  << ")\n";
                    }
                    continue;
                }
                else
                {
                    if(param.valid(0))
                    {
                        params.push_back(param);
                    }
                }
            }

            param.length         = {2, 2, 2};
            param.precision      = precision;
            param.idist          = 0;
            param.odist          = 0;
            param.transform_type = fft_transform_type_real_forward;
            param.nbatch         = 2;
            param.placement      = std::get<1>(types);
            param.itype          = std::get<2>(types);
            param.otype          = std::get<3>(types);
            param.istride        = {20, 6, 1};
            param.ostride        = {20, 6, 1};

            param.validate();

            {
                const double roll = hash_prob(random_seed, param.token());
                const double run_prob
                    = test_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                      * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                      * (param.is_real() ? real_prob_factor : 1.0);

                if(roll > run_prob)
                {
                    if(verbose > 4)
                    {
                        std::cout << "Test skipped (probability " << run_prob << " > " << roll
                                  << ")\n";
                    }
                    continue;
                }
                else
                {
                    if(param.valid(0))
                    {
                        params.push_back(param);
                    }
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_stride,
                         accuracy_test,
                         ::testing::ValuesIn(param_adhoc_stride()),
                         accuracy_test::TestName);

const auto adhoc_tokens = {
    // clang-format off
    "complex_forward_len_4_4_4_single_op_batch_2_istride_16_4_1_CI_ostride_4_16_1_CI_idist_64_odist_64_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_512_64_single_ip_batch_3_istride_192_3_CI_ostride_192_3_CI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_1024_1024_1024_single_op_batch_1_istride_1048576_1024_1_R_ostride_525312_513_1_HI_idist_1073741824_odist_537919488_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_6144_single_ip_batch_34_istride_35_CI_ostride_35_CI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_8192_single_ip_batch_65537_istride_1_R_ostride_1_HI_idist_8194_odist_4097_ioffset_0_0_ooffset_0_0",
    "real_forward_len_520_single_op_batch_270400_istride_1_R_ostride_1_HI_idist_520_odist_261_ioffset_0_0_ooffset_0_0",
    "real_forward_len_630_single_op_batch_396900_istride_1_R_ostride_1_HI_idist_630_odist_316_ioffset_0_0_ooffset_0_0",
    "real_forward_len_660_single_op_batch_435600_istride_1_R_ostride_1_HI_idist_660_odist_331_ioffset_0_0_ooffset_0_0",
    "real_forward_len_700_single_op_batch_490000_istride_1_R_ostride_1_HI_idist_700_odist_351_ioffset_0_0_ooffset_0_0",
    "real_forward_len_728_single_op_batch_529984_istride_1_R_ostride_1_HI_idist_728_odist_365_ioffset_0_0_ooffset_0_0",
    "real_forward_len_968_single_op_batch_937024_istride_1_R_ostride_1_HI_idist_968_odist_485_ioffset_0_0_ooffset_0_0",
    "real_forward_len_1020_single_op_batch_1040400_istride_1_R_ostride_1_HI_idist_1020_odist_511_ioffset_0_0_ooffset_0_0",
    "real_forward_len_378_42_single_ip_batch_66000_istride_44_1_R_ostride_22_1_HI_idist_16632_odist_8316_ioffset_0_0_ooffset_0_0",
    "real_forward_len_527_25_single_ip_batch_67500_istride_26_1_R_ostride_13_1_HI_idist_13702_odist_6851_ioffset_0_0_ooffset_0_0",
    "real_forward_len_630_38_single_ip_batch_65540_istride_40_1_R_ostride_20_1_HI_idist_25200_odist_12600_ioffset_0_0_ooffset_0_0",
    // degenerate length-1 test cases
    "complex_forward_len_1_single_ip_batch_2_istride_1_CI_ostride_1_CI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_1_single_op_batch_2_istride_1_CI_ostride_1_CI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_2_single_ip_batch_1_istride_1_R_ostride_1_HI_idist_4_odist_2_ioffset_0_0_ooffset_0_0",
    "real_forward_len_2_single_op_batch_1_istride_1_R_ostride_1_HI_idist_4_odist_2_ioffset_0_0_ooffset_0_0",
    "real_forward_len_1_2_single_ip_batch_2_istride_1_1_R_ostride_1_1_HI_idist_4_odist_2_ioffset_0_0_ooffset_0_0",
    "real_forward_len_1_2_single_op_batch_2_istride_1_1_R_ostride_1_1_HI_idist_2_odist_2_ioffset_0_0_ooffset_0_0",
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(adhoc_token,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_token(test_prob, adhoc_tokens)),
                         accuracy_test::TestName);

const auto adhoc_hermitian_input_tokens = {
    // clang-format off
    "real_inverse_len_1_8_double_ip_batch_1_istride_5_1_HI_ostride_10_1_R_idist_5_odist_10_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_8_double_ip_batch_10_istride_5_1_HI_ostride_10_1_R_idist_5_odist_10_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_8_1_double_ip_batch_1_istride_1_1_HI_ostride_2_1_R_idist_8_odist_16_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_8_1_double_ip_batch_10_istride_1_1_HI_ostride_2_1_R_idist_8_odist_16_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_2_7_double_ip_batch_1_istride_4_1_HI_ostride_8_1_R_idist_8_odist_16_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_7_2_double_ip_batch_1_istride_2_1_HI_ostride_4_1_R_idist_14_odist_28_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_16_512_double_ip_batch_1_istride_257_1_HI_ostride_514_1_R_idist_4112_odist_8224_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_16_512_double_ip_batch_10_istride_257_1_HI_ostride_514_1_R_idist_4112_odist_8224_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_512_16_double_ip_batch_1_istride_9_1_HI_ostride_18_1_R_idist_4608_odist_9216_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_512_16_double_ip_batch_10_istride_9_1_HI_ostride_18_1_R_idist_4608_odist_9216_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_1_1_double_ip_batch_1_istride_1_1_1_HI_ostride_2_2_1_R_idist_1_odist_2_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_1_1_double_ip_batch_10_istride_1_1_1_HI_ostride_2_2_1_R_idist_1_odist_2_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_2_1_double_ip_batch_1_istride_2_1_1_HI_ostride_4_2_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_2_1_double_ip_batch_10_istride_2_1_1_HI_ostride_4_2_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_2_1_1_double_ip_batch_1_istride_1_1_1_HI_ostride_2_2_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_2_1_1_double_ip_batch_10_istride_1_1_1_HI_ostride_2_2_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_1_2_double_ip_batch_1_istride_2_2_1_HI_ostride_4_4_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0"
    "real_inverse_len_1_1_2_double_ip_batch_10_istride_2_2_1_HI_ostride_4_4_1_R_idist_2_odist_4_ioffset_0_0_ooffset_0_0"
    "real_inverse_len_8_1_2_double_ip_batch_1_istride_2_2_1_HI_ostride_4_4_1_R_idist_16_odist_32_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_8_1_2_double_ip_batch_10_istride_2_2_1_HI_ostride_4_4_1_R_idist_16_odist_32_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_2_8_double_ip_batch_1_istride_10_5_1_HI_ostride_20_10_1_R_idist_10_odist_20_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_2_8_double_ip_batch_10_istride_10_5_1_HI_ostride_20_10_1_R_idist_10_odist_20_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_8_2_double_ip_batch_1_istride_16_2_1_HI_ostride_32_4_1_R_idist_16_odist_32_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_1_8_2_double_ip_batch_10_istride_16_2_1_HI_ostride_32_4_1_R_idist_16_odist_32_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_512_7_2_double_ip_batch_10_istride_14_2_1_HI_ostride_28_4_1_R_idist_7168_odist_14336_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_7_2_512_double_ip_batch_10_istride_514_257_1_HI_ostride_1028_514_1_R_idist_3598_odist_7196_ioffset_0_0_ooffset_0_0",
    "real_inverse_len_2_512_7_double_ip_batch_10_istride_2048_4_1_HI_ostride_4096_8_1_R_idist_4096_odist_8192_ioffset_0_0_ooffset_0_0"
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(adhoc_token_hermitian_input,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_token(test_prob,
                                                                   adhoc_hermitian_input_tokens)),
                         accuracy_test::TestName);

const auto adhoc_nondefault_layout_complex_tokens = {
    // clang-format off
    "complex_forward_len_29_4_16_double_op_batch_233_istride_540_135_5_CI_ostride_1540_140_4_CI_idist_20520_odist_55440_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_16_4_single_op_batch_253_istride_450_10_2_CI_ostride_1376_32_4_CI_idist_16650_odist_41280_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_16_4_double_ip_batch_376_istride_384_16_4_CI_ostride_384_16_4_CI_idist_10752_odist_10752_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_4_16_double_op_batch_1996_istride_1536_192_6_CI_ostride_552_24_1_CI_idist_41472_odist_17112_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_46_4_16_single_ip_batch_613_istride_441_49_1_CI_ostride_441_49_1_CI_idist_26460_odist_26460_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_25_16_single_op_batch_3261_istride_2210_85_5_CI_ostride_1134_42_2_CI_idist_46410_odist_22680_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_25_16_single_op_batch_3261_istride_2210_85_5_CI_ostride_1134_42_2_CI_idist_55250_odist_27216_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_29_25_16_single_op_batch_3261_istride_2210_85_5_CI_ostride_1134_42_2_CI_idist_68510_odist_34020_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_16_25_single_op_batch_3261_istride_2210_130_5_CI_ostride_1080_60_2_CI_idist_46410_odist_21600_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_16_25_single_op_batch_3261_istride_2210_130_5_CI_ostride_1080_60_2_CI_idist_55250_odist_25920_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_29_16_25_single_op_batch_3261_istride_2210_130_5_CI_ostride_1080_60_2_CI_idist_68510_odist_32400_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_41_9_8_double_ip_batch_160_istride_1680_56_4_CI_ostride_1680_56_4_CI_idist_70560_odist_70560_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_9_8_double_ip_batch_160_istride_1680_56_4_CI_ostride_1680_56_4_CI_idist_40320_odist_40320_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_19_9_8_double_ip_batch_160_istride_1680_56_4_CI_ostride_1680_56_4_CI_idist_33600_odist_33600_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_41_8_9_double_ip_batch_160_istride_1680_120_4_CI_ostride_1680_120_4_CI_idist_70560_odist_70560_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_8_9_double_ip_batch_160_istride_1680_120_4_CI_ostride_1680_120_4_CI_idist_40320_odist_40320_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_19_8_9_double_ip_batch_160_istride_1680_120_4_CI_ostride_1680_120_4_CI_idist_33600_odist_33600_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_8_4_single_ip_batch_1826_istride_60_6_1_CI_ostride_60_6_1_CI_idist_2340_odist_2340_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_4_8_single_ip_batch_1826_istride_60_10_1_CI_ostride_60_10_1_CI_idist_2340_odist_2340_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_4_9_single_ip_batch_2450_istride_140_14_1_CI_ostride_140_14_1_CI_idist_4900_odist_4900_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_9_4_single_ip_batch_2450_istride_140_10_1_CI_ostride_140_10_1_CI_idist_4900_odist_4900_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_20_10_single_ip_batch_701_istride_2100_100_5_CI_ostride_2100_100_5_CI_idist_52500_odist_52500_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_10_20_single_ip_batch_701_istride_2100_105_5_CI_ostride_2100_105_5_CI_idist_52500_odist_52500_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_25_4_double_ip_batch_74_istride_2058_42_6_CI_ostride_2058_42_6_CI_idist_94668_odist_94668_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_4_25_double_ip_batch_74_istride_2058_294_6_CI_ostride_2058_294_6_CI_idist_94668_odist_94668_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_8_8_single_ip_batch_3646_istride_136_8_1_CI_ostride_136_8_1_CI_idist_4352_odist_4352_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_4_4_single_ip_batch_2460_istride_112_14_2_CI_ostride_112_14_2_CI_idist_2688_odist_2688_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_25_25_single_op_batch_217_istride_2610_87_3_CI_ostride_3920_112_2_CI_idist_54810_odist_188160_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_29_27_4_double_op_batch_1358_istride_216_8_1_CI_ostride_672_24_2_CI_idist_6480_odist_22176_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_29_4_27_double_op_batch_1358_istride_216_27_1_CI_ostride_672_56_2_CI_idist_6480_odist_22176_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_16_16_single_ip_batch_3_istride_342_18_1_CI_ostride_342_18_1_CI_idist_6840_odist_6840_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_19_27_27_double_op_batch_600_istride_1624_58_2_CI_ostride_3780_140_4_CI_idist_34104_odist_128520_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_81_74_single_ip_batch_2_istride_74_1_CI_ostride_74_1_CI_idist_7696_odist_7696_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_50_129_single_op_batch_3189_istride_1432_8_CI_ostride_129_1_CI_idist_81624_odist_24252_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_9_4_29_single_op_batch_3186_istride_4_1_36_CI_ostride_1_9_36_CI_idist_1044_odist_1044_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_25_4_19_single_op_batch_6953_istride_1_25_100_CI_ostride_1_475_25_CI_idist_1900_odist_1900_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_9_29_32_double_op_batch_1584_istride_1_288_9_CI_ostride_928_1_29_CI_idist_8352_odist_8352_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_9_27_19_single_op_batch_4484_istride_1_9_243_CI_ostride_27_1_243_CI_idist_4617_odist_4617_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_4_32_57_single_op_batch_567_istride_1_4_128_CI_ostride_1_228_4_CI_idist_7296_odist_7296_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_4_25_47_single_op_batch_852_istride_1_4_100_CI_ostride_1175_47_1_CI_idist_4700_odist_4700_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_9_27_23_single_op_batch_2126_istride_27_1_243_CI_ostride_621_1_27_CI_idist_5589_odist_5589_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_47_32_8_double_op_batch_445_istride_256_8_1_CI_ostride_32_1_1504_CI_idist_12032_odist_12032_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_16_29_9_single_op_batch_4808_istride_9_144_1_CI_ostride_261_1_29_CI_idist_4176_odist_4176_ioffset_0_0_ooffset_0_0"
    "complex_forward_len_19_9_16_single_op_batch_4875_istride_144_16_1_CI_ostride_16_304_1_CI_idist_2736_odist_2736_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_32_38_4_double_op_batch_1202_istride_4_128_1_CI_ostride_1_32_1216_CI_idist_4864_odist_4864_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_16_9_double_op_batch_3154_istride_144_9_1_CI_ostride_1_171_19_CI_idist_2736_odist_2736_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_32_29_9_single_op_batch_3859_istride_9_288_1_CI_ostride_29_1_928_CI_idist_8352_odist_8352_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_4_23_4_single_op_batch_4948_istride_1_16_4_CI_ostride_92_4_1_CI_idist_368_odist_368_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_32_23_25_double_op_batch_2016_istride_25_800_1_CI_ostride_575_25_1_CI_idist_18400_odist_18400_ioffset_0_0",
    "complex_forward_len_16_50_4_single_op_batch_441_istride_1_64_16_CI_ostride_50_1_800_CI_idist_3200_odist_3200_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_9_29_9_single_op_batch_3750_istride_1_81_9_CI_ostride_29_1_261_CI_idist_2349_odist_2349_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_8_23_8_double_op_batch_5361_istride_8_64_1_CI_ostride_184_8_1_CI_idist_1472_odist_1472_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_31_9_32_single_op_batch_3955_istride_288_1_9_CI_ostride_9_1_279_CI_idist_8928_odist_8928_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_19_9_25_double_op_batch_3123_istride_225_1_9_CI_ostride_225_25_1_CI_idist_4275_odist_4275_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_32_16_23_double_op_batch_822_istride_16_1_512_CI_ostride_368_23_1_CI_idist_11776_odist_11776_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_31_16_9_single_op_batch_2599_istride_144_9_1_CI_ostride_9_279_1_CI_idist_4464_odist_4464_ioffset_0_0_ooffset_0_0",
    "complex_inverse_len_23_16_8_double_op_batch_912_istride_186_8_1_CI_ostride_173_8_1_CI_idist_5888_odist_5888_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_23_8_25_double_ip_batch_2056_istride_323_25_1_CI_ostride_323_25_1_CI_idist_9200_odist_9200_ioffset_0_0_ooffset_0_0",
    // clang-format on
};

const auto adhoc_nondefault_layout_real_tokens = {
    // clang-format off
    "real_forward_len_25_8_double_ip_batch_7851_istride_22_1_R_ostride_11_1_HI_idist_1650_odist_825_ioffset_0_0_ooffset_0_0",
    "real_forward_len_9_54_single_ip_batch_2976_istride_212_1_R_ostride_106_1_HI_idist_6572_odist_3286_ioffset_0_0_ooffset_0_0",
    "real_forward_len_81_18_double_ip_batch_2790_istride_24_1_R_ostride_12_1_HI_idist_3768_odist_1884_ioffset_0_0_ooffset_0_0",
    "real_forward_len_25_16_double_ip_batch_5802_istride_26_1_R_ostride_13_1_HI_idist_1300_odist_650_ioffset_0_0_ooffset_0_0",
    "real_forward_len_8_18_single_ip_batch_904_istride_54_1_R_ostride_27_1_HI_idist_4104_odist_2052_ioffset_0_0_ooffset_0_0",
    "real_forward_len_64_8_double_ip_batch_2663_istride_40_1_R_ostride_20_1_HI_idist_5560_odist_2780_ioffset_0_0_ooffset_0_0",
    "real_forward_len_27_16_double_ip_batch_5966_istride_94_1_R_ostride_47_1_HI_idist_4606_odist_2303_ioffset_0_0_ooffset_0_0",
    "real_forward_len_25_32_double_ip_batch_4876_istride_112_1_R_ostride_56_1_HI_idist_8512_odist_4256_ioffset_0_0_ooffset_0_0",
    "real_forward_len_9_54_single_ip_batch_2468_istride_286_1_R_ostride_143_1_HI_idist_10010_odist_5005_ioffset_0_0_ooffset_0_0",
    "real_forward_len_9_16_single_ip_batch_7730_istride_44_1_R_ostride_22_1_HI_idist_792_odist_396_ioffset_0_0_ooffset_0_0",
    "real_forward_len_8_8_double_ip_batch_4302_istride_62_1_R_ostride_31_1_HI_idist_8804_odist_4402_ioffset_0_0_ooffset_0_0",
    "real_forward_len_9_32_single_ip_batch_5591_istride_40_1_R_ostride_20_1_HI_idist_2320_odist_1160_ioffset_0_0_ooffset_0_0",
    "real_forward_len_4_1_18_double_ip_batch_2265_istride_56_28_1_R_ostride_28_14_1_HI_idist_1176_odist_588_ioffset_0_0_ooffset_0_0",
    "real_forward_len_20_20_single_ip_batch_4737_istride_226_1_R_ostride_113_1_HI_idist_7458_odist_3729_ioffset_0_0_ooffset_0_0",
    "real_forward_len_16_8_double_ip_batch_2683_istride_26_1_R_ostride_13_1_HI_idist_884_odist_442_ioffset_0_0_ooffset_0_0",
    "real_forward_len_16_1_18_double_ip_batch_2970_istride_196_28_1_R_ostride_98_14_1_HI_idist_3136_odist_1568_ioffset_0_0_ooffset_0_0",
    "real_forward_len_20_20_single_ip_batch_6342_istride_198_1_R_ostride_99_1_HI_idist_6732_odist_3366_ioffset_0_0_ooffset_0_0",
    "real_forward_len_8_1_8_double_ip_batch_5635_istride_250_10_1_R_ostride_125_5_1_HI_idist_2750_odist_1375_ioffset_0_0_ooffset_0_0",
    "real_forward_len_32_8_double_ip_batch_1752_istride_70_1_R_ostride_35_1_HI_idist_16450_odist_8225_ioffset_0_0_ooffset_0_0",
    "real_forward_len_81_64_single_ip_batch_7187_istride_96_1_R_ostride_48_1_HI_idist_10176_odist_5088_ioffset_0_0_ooffset_0_0",
    "real_forward_len_64_16_double_ip_batch_7289_istride_92_1_R_ostride_46_1_HI_idist_6256_odist_3128_ioffset_0_0_ooffset_0_0",
    "real_forward_len_16_16_single_ip_batch_6600_istride_38_1_R_ostride_19_1_HI_idist_1596_odist_798_ioffset_0_0_ooffset_0_0",
    "real_forward_len_4_18_double_ip_batch_7484_istride_128_1_R_ostride_64_1_HI_idist_16384_odist_8192_ioffset_0_0_ooffset_0_0",
    "real_forward_len_9_8_double_ip_batch_6908_istride_14_1_R_ostride_7_1_HI_idist_924_odist_462_ioffset_0_0_ooffset_0_0",
    "real_forward_len_32_18_double_ip_batch_7083_istride_30_1_R_ostride_15_1_HI_idist_1830_odist_915_ioffset_0_0_ooffset_0_0",
    "real_forward_len_486_double_op_batch_7014_istride_7014_R_ostride_7014_HI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_486_double_op_batch_910_istride_910_R_ostride_910_HI_idist_1_odist_1_ioffset_0_0_ooffset_0_0",
    "real_forward_len_15_12_76_single_ip_batch_125_istride_1280_80_1_R_ostride_640_40_1_HI_idist_32000_odist_16000_ioffset_0_0_ooffset_0_0",
    "real_forward_len_11_12_38_single_ip_batch_1253_istride_2128_56_1_R_ostride_1064_28_1_HI_idist_23408_odist_11704_ioffset_0_0_ooffset_0_0",
    "real_forward_len_36_13_46_single_ip_batch_91_istride_2340_78_1_R_ostride_1170_39_1_HI_idist_91260_odist_45630_ioffset_0_0_ooffset_0_0",
    "real_forward_len_20_8_38_double_ip_batch_846_istride_1392_48_1_R_ostride_696_24_1_HI_idist_66816_odist_33408_ioffset_0_0_ooffset_0_0",
    "real_forward_len_2_5_46_double_ip_batch_281_istride_4340_70_1_R_ostride_2170_35_1_HI_idist_30380_odist_15190_ioffset_0_0_ooffset_0_0",
    "real_forward_len_5_12_38_single_ip_batch_2725_istride_1012_44_1_R_ostride_506_22_1_HI_idist_17204_odist_8602_ioffset_0_0_ooffset_0_0",
    "real_forward_len_6_24_38_single_ip_batch_830_istride_2688_64_1_R_ostride_1344_32_1_HI_idist_48384_odist_24192_ioffset_0_0_ooffset_0_0",
    "real_forward_len_6_25_38_single_ip_batch_2006_istride_1360_40_1_R_ostride_680_20_1_HI_idist_13600_odist_6800_ioffset_0_0_ooffset_0_0",
    "real_forward_len_26_26_134_double_ip_batch_41_istride_5934_138_1_R_ostride_2967_69_1_HI_idist_231426_odist_115713_ioffset_0_0_ooffset_0_0",
    "real_forward_len_2_40_62_single_op_batch_922_istride_62_124_1_R_ostride_1280_32_1_HI_idist_4960_odist_2560_ioffset_0_0_ooffset_0_0",
    "real_forward_len_6_12_46_single_op_batch_2312_istride_46_276_1_R_ostride_288_24_1_HI_idist_3312_odist_1728_ioffset_0_0_ooffset_0_0",
    "real_forward_len_7_14_76_double_ip_batch_202_istride_3588_78_1_R_ostride_1794_39_1_HI_idist_150696_odist_75348_ioffset_0_0_ooffset_0_0",
    "real_forward_len_2_14_46_single_ip_batch_499_istride_2050_50_1_R_ostride_1025_25_1_HI_idist_71750_odist_35875_ioffset_0_0_ooffset_0_0",
    "real_forward_len_26_52_double_op_batch_3899_istride_240_4_R_ostride_224_7_HI_idist_15600_odist_8960_ioffset_0_0_ooffset_0_0",
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    adhoc_nondefault_layout_complex,
    accuracy_test,
    ::testing::ValuesIn(param_generator_token(test_prob, adhoc_nondefault_layout_complex_tokens)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    adhoc_nondefault_layout_real,
    accuracy_test,
    ::testing::ValuesIn(param_generator_token(test_prob, adhoc_nondefault_layout_real_tokens)),
    accuracy_test::TestName);

inline auto param_even_real_odd_base_index()
{
    std::vector<fft_params> params;
    // C2R/R2C of even (real) lengths with odd distances (1D)
    // or odd strides along first non-contiguous dimension (>1D)

    const std::vector<std::vector<size_t>> test_lengths = {
        // 1D
        {16},
        {64},
        {192},
        {194},
        {442},
        //2D
        {16, 16},
        {25, 16},
        {64, 64},
        {75, 64},
        {64, 192},
        {225, 192},
        {128, 194},
        {225, 194},
        {256, 442},
        {625, 442},
        // 3D
        {16, 16, 16},
        {16, 25, 16},
        {25, 16, 16},
        {25, 25, 16},
        {64, 64, 64},
        {64, 75, 64},
        {75, 64, 64},
        {75, 75, 64},
        {128, 128, 192},
        {128, 225, 192},
        {225, 128, 192},
        {225, 225, 192},
        {128, 128, 194},
        {128, 225, 194},
        {225, 128, 194},
        {225, 225, 194},
    };

    for(auto precision : precision_range_sp_dp)
    {
        for(const auto& len : test_lengths)
        {
            fft_params param;
            param.precision      = precision;
            param.transform_type = fft_transform_type_real_forward;
            param.itype          = fft_array_type_real;
            param.otype          = fft_array_type_hermitian_interleaved;
            param.placement      = fft_placement_notinplace; // cannot be in-place by definition
            param.length         = len;
            if(len.size() == 1)
            {
                param.nbatch  = 2;
                param.istride = {1};
                param.ostride = {1};
                param.idist   = len[0] + 1; // +1 for generating odd addresses for some rows
                param.odist   = len[0] / 2 + 1;
            }
            else if(len.size() > 1)
            {
                param.nbatch = 1;
                param.idist  = 0; // irrelevant for batch size 1
                param.odist  = 0; // irrelevant for batch size 1
                param.istride.resize(len.size());
                param.ostride.resize(len.size());
                // unit stride along fastest-varying dimension
                param.istride.back() = param.ostride.back() = 1;
                param.istride[len.size() - 2]
                    = len.back() + 1; // +1 for generating odd addresses for some rows
                param.ostride[len.size() - 2] = len.back() / 2 + 1;
                for(auto dim = len.size() - 2; dim-- > 0;)
                {
                    param.istride[dim] = param.istride[dim + 1] * len[dim + 1];
                    param.ostride[dim] = param.ostride[dim + 1] * len[dim + 1];
                }
            }
            param.validate();

            {
                const double roll = hash_prob(random_seed, param.token());
                const double run_prob
                    = test_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                      * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                      * (param.is_real() ? real_prob_factor : 1.0);

                if(roll > run_prob)
                {
                    if(verbose > 4)
                    {
                        std::cout << "Test skipped (probability " << run_prob << " > " << roll
                                  << ")\n";
                    }
                    continue;
                }
                else
                {
                    if(param.valid(0))
                    {
                        params.push_back(param);
                    }
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_real_even_length_odd_base_index,
                         accuracy_test,
                         ::testing::ValuesIn(param_even_real_odd_base_index()),
                         accuracy_test::TestName);
