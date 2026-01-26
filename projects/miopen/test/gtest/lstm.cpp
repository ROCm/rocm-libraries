/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#include <hip/hip_runtime.h>
#include <gtest/gtest_common.hpp>

template <typename T>
class GPU_lstm_Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        device_count = 0;
    }

    void Run()
    {
        //std::vector<std::string> params = GetParam();
        std::cout << "running MIOpenDriver...\n";
    }

    int device_count = 0;
};

using GPU_lstm_FP32 = GPU_lstm_Test<float>;

TEST_F(GPU_lstm_FP32, FloatTest_lstm)
{
    //testing::internal::CaptureStderr();
    Run();
    //auto capture = testing::internal::GetCapturedStderr();
    //std::cout << capture;
}
