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
    std::string flags = "test_lstm --verbose " + precision;
    std::string commonFlags =
        " --num-layers 1 --in-mode 1 --bias-mode 0 -dir-mode 0 --rnn-mode 0 --flat-batch-fill";

    // clang-format off
    return std::vector<std::string>{
        {flags + " --batch-size 16 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 512 --hidden-size 512" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 1024 --hidden-size 1024" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 2048 --hidden-size 2048" + commonFlags},
        {flags + " --batch-size 16 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 32 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 64 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 128 --seq-len 25 --vector-len 4096 --hidden-size 4096" + commonFlags},
        {flags + " --batch-size 8 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 16 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 32 --seq-len 50 --vector-len 1536 --hidden-size 1536" + commonFlags},
        {flags + " --batch-size 16 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags},
        {flags + " --batch-size 32 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags},
        {flags + " --batch-size 64 --seq-len 150 --vector-len 256 --hidden-size 256" + commonFlags}
    };
    // clang-format on
}

class GPU_DeepBench_LSTM_FP32 : LSTM_test<float>, testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(GPU_DeepBench_LSTM_FP32, FloatTest)
{
    int device_count{0};
    if((hipGetDeviceCount(&device_count) != hipSuccess) or (device_count == 0))
    {
        //GTEST_SKIP() << "No HIP devices available for testing";
    }

    int batchSize{32};
    int seqLength{3};

    auto [,] = GetParam();

};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_DeepBench_LSTM_FP32,
                         testing::Combine(testing::Values(0, 1), testing::Values(0, 1)));
