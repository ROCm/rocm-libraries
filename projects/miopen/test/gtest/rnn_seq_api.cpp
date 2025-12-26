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
#include "rnn_seq_api.hpp"

using GPU_RNNSeqApi_FP32 = RNNSeqApiCommon<float>;
using GPU_RNNSeqApi_FP16 = RNNSeqApiCommon<half_float::half>;

TEST_P(GPU_RNNSeqApi_FP32, TestFloat) { run(); }

TEST_P(GPU_RNNSeqApi_FP16, TestFloat16) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_RNNSeqApi_FP32, GetValidRNNSeqCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_RNNSeqApi_FP32, GetValidRNNSeqCases(true));

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_RNNSeqApi_FP16, GetValidRNNSeqCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_RNNSeqApi_FP16, GetValidRNNSeqCases(true));

using GPU_LstmMSRnn_FP32 = RNNSeqApiCommon<float>;
using GPU_LstmMSRnn_FP16 = RNNSeqApiCommon<half_float::half>;

TEST_P(GPU_LstmMSRnn_FP32, TestFloat) { run(); }

TEST_P(GPU_LstmMSRnn_FP16, TestFloat16) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LstmMSRnn_FP32, GetValidLstmMSCases<float>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_LstmMSRnn_FP32, GetValidLstmMSCases<float>(true));

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_LstmMSRnn_FP16, GetValidLstmMSCases<half_float::half>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_LstmMSRnn_FP16, GetValidLstmMSCases<half_float::half>(true));
