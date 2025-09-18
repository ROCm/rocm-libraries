// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../../shared/params_gen.h"
#include "../../../shared/rocfft_params.h"
#include "../accuracy_tests_range.h"
#include "bitwise_repro_test.h"

using ::testing::ValuesIn;

TEST(bitwise_repro_test, compare_precisions)
{

    rocfft_params params_1;
    // clang-format off
    params_1.from_token(std::string("complex_forward_len_192_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_192_odist_192_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_1.validate();

    rocfft_params params_2;
    // clang-format off
    params_2.from_token(std::string("complex_forward_len_192_double_ip_batch_1_istride_1_CI_ostride_1_CI_idist_192_odist_192_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_2.validate();

    if(!params_1.valid(verbose) || !params_2.valid(verbose))
    {
        if(verbose)
            std::cout << "Invalid parameters, skip this test." << std::endl;

        GTEST_SKIP();
    }

    try
    {
        bitwise_repro(params_1, params_2);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    SUCCEED();
}

TEST(bitwise_repro_test, compare_lengths)
{
    rocfft_params params_1;
    // clang-format off
    params_1.from_token(std::string("complex_forward_len_64_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_64_odist_64_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_1.validate();

    rocfft_params params_2;
    // clang-format off
    params_2.from_token(std::string("complex_forward_len_32_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_32_odist_32_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_2.validate();

    if(!params_1.valid(verbose) || !params_2.valid(verbose))
    {
        if(verbose)
            std::cout << "Invalid parameters, skip this test." << std::endl;

        GTEST_SKIP();
    }

    try
    {
        bitwise_repro(params_1, params_2);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    SUCCEED();
}

TEST(bitwise_repro_test, compare_transform_types)
{
    rocfft_params params_1;
    // clang-format off
    params_1.from_token(std::string("complex_forward_len_256_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_256_odist_256_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_1.validate();

    rocfft_params params_2;
    // clang-format off
    params_2.from_token(std::string("complex_inverse_len_256_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_256_odist_256_ioffset_0_0_ooffset_0_0"));
    // clang-format on
    params_2.validate();

    if(!params_1.valid(verbose) || !params_2.valid(verbose))
    {
        if(verbose)
            std::cout << "Invalid parameters, skip this test." << std::endl;

        GTEST_SKIP();
    }

    try
    {
        bitwise_repro(params_1, params_2);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    SUCCEED();
}

TEST_P(bitwise_repro_test, compare_to_reference)
{
    if(repro_db == nullptr)
        GTEST_SKIP() << "A database file is required for this test." << std::endl;

    rocfft_params params(GetParam());

    params.validate();

    // Test that the tokenization works as expected.
    auto       token = params.token();
    fft_params tokentest;
    tokentest.from_token(token);
    auto token1 = tokentest.token();
    EXPECT_EQ(token, token1);

    if(!params.valid(verbose))
    {
        if(verbose)
        {
            std::cout << "Invalid parameters, skip this test." << std::endl;
        }
        GTEST_SKIP();
    }

    try
    {
        bitwise_repro(params);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    SUCCEED();
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 1D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(pow2_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_1D_half,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(radX_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_1D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

// small 1D sizes just need to make sure our factorization isn't
// completely broken, so we just check simple C2C outplace interleaved
INSTANTIATE_TEST_SUITE_P(small_1D,
                         bitwise_repro_test,
                         ::testing::ValuesIn(param_generator_base(
                             test_prob,
                             {fft_transform_type_complex_forward},
                             generate_lengths({small_1D_sizes()}),
                             {fft_precision_single},
                             {1},
                             [](fft_transform_type                       t,
                                const std::vector<fft_result_placement>& place_range,
                                const bool                               planar) {
                                 return std::vector<type_place_io_t>{
                                     std::make_tuple(t,
                                                     place_range[0],
                                                     fft_array_type_complex_interleaved,
                                                     fft_array_type_complex_interleaved)};
                             },
                             stride_range,
                             stride_range,
                             ioffset_range_zero,
                             ooffset_range_zero,
                             {fft_placement_notinplace},
                             true)),
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_complex,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_complex_half,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real_half,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 2D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(pow2_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_2D_half,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

// test length-1 on one dimension against a variety of non-1 lengths
INSTANTIATE_TEST_SUITE_P(len1_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

// length-1 on the other dimension
INSTANTIATE_TEST_SUITE_P(len1_swap_2D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// 3D test problems
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(pow2_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_3D_half,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(sbrc_3D,
                         bitwise_repro_test,
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
                         bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    inner_batch_3D,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    inner_batch_3D_half,
    bitwise_repro_test,
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
    bitwise_repro_test::TestName);
