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
