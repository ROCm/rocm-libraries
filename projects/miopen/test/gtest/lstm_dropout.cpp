/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "lstm_common.hpp"
#include <gtest/gtest_common.hpp>

template <class T>
struct lstm_dropout_driver : lstm_basic_driver<T>
{
    lstm_dropout_driver() : lstm_basic_driver<T>()
    {
        this->add(this->batchSize, "batch-size", this->generate_data({17}));
        this->add(this->seqLength, "seq-len", this->generate_data({25}));
        this->add(this->inVecLen, "vector-len", this->generate_data({17}));
        this->add(this->hiddenSize, "hidden-size", this->generate_data({67}));
        this->add(this->numLayers, "num-layers", this->generate_data({3}));
        this->add(this->nohx, "no-hx", this->generate_data({false}));
        this->add(this->nodhy, "no-dhy", this->generate_data({false}));
        this->add(this->nocx, "no-cx", this->generate_data({false}));
        this->add(this->nodcy, "no-dcy", this->generate_data({false}));
        this->add(this->nohy, "no-hy", this->generate_data({false}));
        this->add(this->nodhx, "no-dhx", this->generate_data({false}));
        this->add(this->nocy, "no-cy", this->generate_data({false}));
        this->add(this->nodcx, "no-dcx", this->generate_data({false}));
        this->add(this->flatBatchFill, "flat-batch-fill", this->generate_data({false, true}));
        this->add(this->useDropout, "use-dropout", this->generate_data({1}));
        this->add(this->inputMode, "in-mode", this->generate_data({0}));
        this->add(this->biasMode, "bias-mode", this->generate_data({1}));
        this->add(this->dirMode, "dir-mode", this->generate_data(std::vector<int>{0, 1}));
        this->add(this->algoMode, "algo-mode", this->generate_data({0}));
        this->add(this->batchSeq, "batch-seq", this->generate_data(std::vector<int>{0}));
    }
};

template <typename T>
struct GPU_lstm_dropout_Test : public ::testing::Test
{
};

using GPU_lstm_dropout_FP32 = GPU_lstm_dropout_Test<float>;

TEST_F(GPU_lstm_dropout_FP32, FloatTest_lstm_dropout)
{
    testing::internal::CaptureStderr();
    test_drive<lstm_dropout_driver<float>>(0, nullptr);
    auto capture = testing::internal::GetCapturedStderr();
    std::cout << capture;
}
