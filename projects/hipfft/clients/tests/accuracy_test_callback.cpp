// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

std::vector<std::vector<size_t>> callback_sizes = {
    // some single kernel sizes
    {4},
    {16},
    {81},
    {100},

    // L1D_TRTRT sizes
    {70},
    {77},
    {1344},

    // L1D_CC sizes
    {8192},
    {10000},

    // prime
    {23},
    {29},

    // 2D_SINGLE sizes, small and big
    {16, 8},
    {32, 32},
    {9, 81},
    {27, 81},
    {81, 27},
    {256, 9},
    {9, 256},
    {125, 32},
    {32, 125},

    // 2D_RTRT
    {20, 40},
    {81, 81},

    // 2D_RC
    {128, 64},
    {128, 256},

    // more complicated children of 2D_RTRT (L1D_TRTRT, L1D_CC, prime)
    {4, 63},
    {63, 4},
    {4, 8192},
    {8192, 4},
    {4, 23},
    {23, 4},

    // 3D_TRTRTR, with complicated children
    {63, 5, 6},
    {6, 5, 63},
    {23, 5, 6},
    {6, 5, 23},
    {70, 5, 6},
    {6, 5, 70},
    {8192, 5, 6},
    {6, 5, 8192},

    // 3D_RTRT, with complicated children
    {23, 4, 4},
    {4, 4, 23},
    {70, 4, 4},
    {4, 4, 70},
    {8192, 4, 4},
    {4, 4, 8192},

    // 3D odd lengths
    {27, 27, 27},

    // 3D_BLOCK_RC
    {64, 64, 64},
};

const static std::vector<std::vector<size_t>> stride_range = {{1}};

const static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
const static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

const static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
const static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

auto transform_types = {fft_transform_type_complex_forward, fft_transform_type_real_forward};

#ifdef __HIP__
INSTANTIATE_TEST_SUITE_P(callback,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_base(test_prob,
                                                                  transform_types,
                                                                  callback_sizes,
                                                                  precision_range_sp_dp,
                                                                  batch_range,
                                                                  generate_types,
                                                                  stride_range,
                                                                  stride_range,
                                                                  ioffset_range_zero,
                                                                  ooffset_range_zero,
                                                                  place_range,
                                                                  false,
                                                                  true)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_callback,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_base(test_prob,
                                                                  transform_types,
                                                                  callback_sizes,
                                                                  precision_range_sp_dp,
                                                                  batch_range,
                                                                  generate_types,
                                                                  stride_range,
                                                                  stride_range,
                                                                  ioffset_range,
                                                                  ooffset_range,
                                                                  place_range,
                                                                  false,
                                                                  true)),
                         accuracy_test::TestName);
#endif

// one of the obvious use cases for callbacks is to implement result
// scaling manually, so use the same sizes to test rocFFT's own
// result scaling feature.
inline auto param_generator_scaling(const std::vector<std::vector<size_t>>& v_lengths)
{
    auto params = param_generator(test_prob,
                                  callback_sizes,
                                  precision_range_sp_dp,
                                  batch_range,
                                  stride_range,
                                  stride_range,
                                  ioffset_range_zero,
                                  ooffset_range_zero,
                                  place_range,
                                  false);
    for(auto& param : params)
        param.scale_factor = 7.23;
    return params;
}

// cuFFT does not support result scaling
#ifndef _CUFFT_BACKEND
INSTANTIATE_TEST_SUITE_P(scaling,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_scaling(callback_sizes)),
                         accuracy_test::TestName);
#endif
