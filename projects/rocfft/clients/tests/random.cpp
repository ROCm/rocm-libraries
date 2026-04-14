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
#include <iostream>
#include <random>

#include "../../shared/accuracy_test.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_accuracy_test.h"
#include "../../shared/test_params.h"

class random_params
    : public ::testing::TestWithParam<
          std::tuple<int, int, fft_precision, fft_result_placement, fft_transform_type>>
{
};

// TODO: Add batch and stride

auto random_param_generator(const int                                dimension,
                            const std::vector<fft_precision>&        precisions,
                            const std::vector<fft_result_placement>& place_range,
                            const fft_transform_type                 transform_type)

{
    std::vector<fft_params> params;

    int maxlen = 0;
    switch(dimension)
    {
    case 1:
        maxlen = 1 << 15;
        break;
    case 2:
        maxlen = 1 << 10;
        break;
    case 3:
        maxlen = 1 << 6;
        break;
    default:
        throw std::runtime_error("invalid dimension for random tests");
    }

    std::mt19937 rgen(random_seed);
    // Mean value of the exponential distribution is maxlen:
    std::exponential_distribution<double> distribution(1.0 / maxlen);

    std::uniform_int_distribution<int> precision_distr(0, precisions.size() - 1);
    std::uniform_int_distribution<int> place_distr(0, place_range.size() - 1);

    while(params.size() < n_random_tests)
    {
        const auto precision = precisions[precision_distr(rgen)];
        const auto placement = place_range[place_distr(rgen)];

        fft_params param;

        param.transform_type = transform_type;
        param.precision      = precision;
        param.placement      = placement;
        for(int idim = 0; idim < dimension; ++idim)
        {
            // NB: the distribution can return 0, so add 1 to avoid this issue.
            param.length.push_back(1 + (size_t)distribution(rgen));
        }

        param.validate();
        if(param.valid(0))
        {
            bool found = false;
            for(size_t idx = 0; idx < params.size(); ++idx)
            {
                if(param.token() == params[idx].token())
                {
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                params.push_back(param);
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    random_complex_1d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        1, precision_range_sp_dp, place_range, fft_transform_type_complex_forward)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    random_complex_2d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        2, precision_range_sp_dp, place_range, fft_transform_type_complex_forward)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    random_complex_3d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        3, precision_range_sp_dp, place_range, fft_transform_type_complex_forward)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    random_real_1d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        1, precision_range_sp_dp, place_range, fft_transform_type_real_forward)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    random_real_2d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        2, precision_range_sp_dp, place_range, fft_transform_type_real_forward)),
    accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(
    random_real_3d,
    accuracy_test,
    ::testing::ValuesIn(random_param_generator(
        3, precision_range_sp_dp, place_range, fft_transform_type_real_forward)),
    accuracy_test::TestName);
