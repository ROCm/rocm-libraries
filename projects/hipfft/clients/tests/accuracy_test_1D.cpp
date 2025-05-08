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

// TODO: handle special case where length=2 for real/complex transforms.
const static std::vector<size_t> pow2_range
    = {2,       4,        8,        16,       32,        128,       256,
       512,     1024,     2048,     4096,     8192,      16384,     32768,
       65536,   131072,   262144,   524288,   1048576,   2097152,   4194304,
       8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912};
// 2^30 is 1073741824;

const static std::vector<size_t> pow2_range_half
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

const static std::vector<size_t> pow3_range = {3,
                                               9,
                                               27,
                                               81,
                                               243,
                                               729,
                                               2187,
                                               6561,
                                               19683,
                                               59049,
                                               177147,
                                               531441,
                                               1594323,
                                               4782969,
                                               14348907,
                                               43046721,
                                               129140163,
                                               387420489};

const static std::vector<size_t> pow5_range
    = {5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125, 244140625};

// radix 7, 11, 13 sizes that are either pure powers or sizes people have wanted in the wild
const static std::vector<size_t> radX_range
    = {7, 49, 84, 112, 11, 13, 52, 104, 208, 343, 2401, 16807};

const static std::vector<size_t> mix_range
    = {6,   10,  12,   15,   20,   30,   56,   120,  150,  225,  240,  300,   336,   486,
       600, 900, 1250, 1500, 1875, 2160, 2187, 2250, 2500, 3000, 4000, 12000, 24000, 72000};

const static std::vector<size_t> prime_range
    = {17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

static std::vector<size_t> small_1D_sizes()
{
    static const size_t SMALL_1D_MAX = 8192;

    // generate a list of sizes from 2 and up, skipping any sizes that are already covered
    std::vector<size_t> covered_sizes;
    std::copy(pow2_range.begin(), pow2_range.end(), std::back_inserter(covered_sizes));
    std::copy(pow3_range.begin(), pow3_range.end(), std::back_inserter(covered_sizes));
    std::copy(pow5_range.begin(), pow5_range.end(), std::back_inserter(covered_sizes));
    std::copy(radX_range.begin(), radX_range.end(), std::back_inserter(covered_sizes));
    std::copy(mix_range.begin(), mix_range.end(), std::back_inserter(covered_sizes));
    std::copy(prime_range.begin(), prime_range.end(), std::back_inserter(covered_sizes));
    std::sort(covered_sizes.begin(), covered_sizes.end());

    std::vector<size_t> output;
    for(size_t i = 2; i < SMALL_1D_MAX; ++i)
    {
        if(!std::binary_search(covered_sizes.begin(), covered_sizes.end(), i))
        {
            output.push_back(i);
        }
    }
    return output;
}

const static std::vector<std::vector<size_t>> stride_range = {{1}};

const static std::vector<size_t> batch_range_1D = {4, 2, 1};

const static std::vector<std::vector<size_t>> stride_range_for_prime
    = {{1}, {2}, {3}, {64}, {65}}; //TODO: this will be merged back to stride_range

const static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
const static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

const static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
const static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(pow2_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow2_1D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half}),
                                                             {fft_precision_half},
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow2_1D_half,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow2_range_half}),
                                                             {fft_precision_half},
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow3_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow3_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow3_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(pow5_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_pow5_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({pow5_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(radX_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({radX_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_radX_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({radX_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(prime_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_prime_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({prime_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(mix_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(DISABLED_offset_mix_1D,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(test_prob,
                                                             generate_lengths({mix_range}),
                                                             precision_range_sp_dp,
                                                             batch_range_1D,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range,
                                                             false,
                                                             false)),
                         accuracy_test::TestName);

// small 1D sizes just need to make sure our factorization isn't
// completely broken, so we just check simple C2C outplace interleaved
INSTANTIATE_TEST_SUITE_P(small_1D,
                         accuracy_test,
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
                             {fft_placement_notinplace})),
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
const static std::vector<size_t>              pow2_range_for_stride      = {4096, 8192, 524288};
const static std::vector<size_t>              pow2_range_for_stride_half = {4096, 8192};
const static std::vector<std::vector<size_t>> stride_range_for_pow2      = {{2}, {3}};
const static std::vector<size_t>              batch_range_for_stride     = {2, 1};

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_complex,
    accuracy_test,
    ::testing::ValuesIn(param_generator_complex(test_prob,
                                                generate_lengths({pow2_range_for_stride}),
                                                precision_range_sp_dp,
                                                batch_range_1D,
                                                stride_range_for_pow2,
                                                stride_range_for_pow2,
                                                ioffset_range_zero,
                                                ooffset_range_zero,
                                                place_range,
                                                false,
                                                false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real,
    accuracy_test,
    ::testing::ValuesIn(param_generator_real(test_prob,
                                             generate_lengths({pow2_range_for_stride}),
                                             precision_range_sp_dp,
                                             batch_range_1D,
                                             stride_range_for_pow2,
                                             stride_range_for_pow2,
                                             ioffset_range_zero,
                                             ooffset_range_zero,
                                             place_range,
                                             false,
                                             false)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    pow2_1D_stride_real_half,
    accuracy_test,
    ::testing::ValuesIn(param_generator_real(test_prob,
                                             generate_lengths({pow2_range_for_stride_half}),
                                             {fft_precision_half},
                                             batch_range_1D,
                                             stride_range_for_pow2,
                                             stride_range_for_pow2,
                                             ioffset_range_zero,
                                             ooffset_range_zero,
                                             place_range,
                                             false,
                                             false)),
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

    // for(auto& transform_type :
    // {fft_transform_type_complex_forward, fft_transform_type_complex_inverse})
    // {

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
                for(const auto& types : generate_types(transform_type, place_range, false))
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

const static std::vector<size_t> pow2_range_2D
    = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
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

const static std::vector<size_t> pow3_range_2D = {3, 27, 81, 243, 729, 2187, 6561};
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

const static std::vector<size_t> pow5_range_2D = {5, 25, 125, 625, 3125, 15625};
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

const static std::vector<size_t> prime_range_2D = {7, 11, 13, 17, 19, 23, 29, 263, 269, 271, 277};

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
