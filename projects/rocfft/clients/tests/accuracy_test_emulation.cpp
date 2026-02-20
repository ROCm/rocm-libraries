
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
#include "accuracy_tests_range.h"

const auto emulation_tokens = {
    // clang-format off
    "complex_forward_len_4_double_ip_batch_1_istride_1_CI_ostride_1_CI_idist_4_odist_4_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_4_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_4_odist_4_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_8_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_8_odist_8_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_16_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_16_odist_16_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_32_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_32_odist_32_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_64_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_64_odist_64_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_128_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_128_odist_128_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_27_double_ip_batch_1_istride_1_CI_ostride_1_CI_idist_27_odist_27_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_27_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_27_odist_27_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_27_27_double_ip_batch_1_istride_27_1_CI_ostride_27_1_CI_idist_729_odist_729_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_27_27_single_ip_batch_1_istride_27_1_CI_ostride_27_1_CI_idist_729_odist_729_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_125_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_125_odist_125_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_125_125_single_ip_batch_1_istride_125_1_CI_ostride_125_1_CI_idist_15625_odist_15625_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_121_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_121_odist_121_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_121_121_single_ip_batch_1_istride_121_1_CI_ostride_121_1_CI_idist_14641_odist_14641_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_216_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_216_odist_216_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_10000_double_ip_batch_1_istride_1_CI_ostride_1_CI_idist_10000_odist_10000_ioffset_0_0_ooffset_0_0",
    "complex_forward_len_128_50_128_single_ip_batch_1_istride_6400_128_1_CI_ostride_6400_128_1_CI_idist_819200_odist_819200_ioffset_0_0_ooffset_0_0",
    "real_forward_len_16_256_256_single_op_batch_2_istride_65536_256_1_R_ostride_33024_129_1_HI_idist",
    "real_forward_len_256_128_256_single_op_batch_1_istride_32768_256_1_R_ostride_16512_129_1_HI_idist"
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(emulation_token,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_token(emulation_prob,
                                                                   emulation_tokens)),
                         accuracy_test::TestName);

const static std::vector<size_t> emulation_range_1D
    = {2, 3, 5, 16, 17, 29, 32, 64, 75, 128, 200, 256, 288, 298};

const static std::vector<size_t> emulation_range_2D
    = {2, 3, 5, 16, 29, 17, 64, 76, 96, 112, 128, 150, 315};

const static std::vector<size_t> emulation_range_3D = {2, 3, 5, 16, 29, 17, 32, 64, 128, 256};

INSTANTIATE_TEST_SUITE_P(emulation_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(emulation_prob,
                                                             generate_lengths({emulation_range_1D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(emulation_2D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(emulation_prob,
                                                             generate_lengths({emulation_range_2D,
                                                                               emulation_range_2D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(emulation_3D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(emulation_prob,
                                                             generate_lengths({emulation_range_3D,
                                                                               emulation_range_3D,
                                                                               emulation_range_3D}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             true)),
                         accuracy_test::TestName);
