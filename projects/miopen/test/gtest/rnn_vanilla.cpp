/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "rnn_vanilla_common.hpp"

using GPU_RNNVanilla_FP32 = RNNVanillaCommon<float>;

TEST_P(GPU_RNNVanilla_FP32, FloatTest) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_RNNVanilla_FP32, GenCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_RNNVanilla_FP32, GenCases(true));

using GPU_RNNVanilla_FP16 = RNNVanillaCommon<half_float::half>;

TEST_P(GPU_RNNVanilla_FP16, FloatTest) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_RNNVanilla_FP16, GenCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_RNNVanilla_FP16, GenCases(true));
