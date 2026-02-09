/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "lstm.hpp"
#include <hip/hip_runtime.h>

auto GetTestCases(std::string precision)
{
    std::string flags       = "test_lstm --verbose " + precision;
    std::string commonFlags = " --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 "
                              "--hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";

    // clang-format off
    return std::vector<std::string>{
        {flags + commonFlags + " -dir-mode 0 --no-hx"},
        {flags + commonFlags + " -dir-mode 0 --no-dhy"},
        {flags + commonFlags + " -dir-mode 0 --no-hx --no-dhy"},
        {flags + commonFlags + " -dir-mode 0 --no-cx"},
        {flags + commonFlags + " -dir-mode 0 --no-hx --no-cx"},
        {flags + commonFlags + " -dir-mode 0 --no-dcy"},
        {flags + commonFlags + " -dir-mode 0 --no-cx --no-dcy"},
        {flags + commonFlags + " -dir-mode 1 --no-hx"},
        {flags + commonFlags + " -dir-mode 1 --no-dhy"},
        {flags + commonFlags + " -dir-mode 1 --no-hx --no-dhy"},
        {flags + commonFlags + " -dir-mode 1 --no-cx"},
        {flags + commonFlags + " -dir-mode 1 --no-hx --no-cx"},
        {flags + commonFlags + " -dir-mode 1 --no-dcy"},
        {flags + commonFlags + " -dir-mode 1 --no-cx --no-dcy"},
        {flags + commonFlags + " -dir-mode 0 --no-hy"},
        {flags + commonFlags + " -dir-mode 0 --no-dhx"},
        {flags + commonFlags + " -dir-mode 0 --no-hy --no-dhx"},
        {flags + commonFlags + " -dir-mode 0 --no-cy"},
        {flags + commonFlags + " -dir-mode 0 --no-hy --no-cy"},
        {flags + commonFlags + " -dir-mode 0 --no-dcx"},
        {flags + commonFlags + " -dir-mode 0 --no-cy --no-dcx"},
        {flags + commonFlags + " -dir-mode 1 --no-hy"},
        {flags + commonFlags + " -dir-mode 1 --no-dhx"},
        {flags + commonFlags + " -dir-mode 1 --no-hy --no-dhx"},
        {flags + commonFlags + " -dir-mode 1 --no-cy"},
        {flags + commonFlags + " -dir-mode 1 --no-hy --no-cy"},
        {flags + commonFlags + " -dir-mode 1 --no-dcx"},
        {flags + commonFlags + " -dir-mode 1 --no-cy --no-dcx"},
	    {flags + commonFlags + " -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"},
	    {flags + commonFlags + " -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"}
    };
    // clang-format on
}

class GPU_LSTM_extra_FP32 : LSTM_test<float>, testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(GPU_LSTM_extra_FP32, FloatTest)
{
    int device_count{0};
    if((hipGetDeviceCount(&device_count) != hipSuccess) or (device_count == 0))
    {
        //GTEST_SKIP() << "No HIP devices available for testing";
    }

    int batchSize{32};
    int seqLength{3};

    //auto [dirMode, ] = GetParam();

    this->batchSize  = batchSize;
    this->seqLength  = seqLength;
    this->batchSeq   = generate_batchSeq(batchSize, seqLength)[0];
for (auto elem : this->batchSeq) std::cout << elem << ' ';
std::cout << '\n';
    this->inVecLen   = 128;
    this->hiddenSize = 128;
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_LSTM_extra_FP32,
                         testing::Combine(testing::Values(0, 1), testing::Values(0, 1)));
