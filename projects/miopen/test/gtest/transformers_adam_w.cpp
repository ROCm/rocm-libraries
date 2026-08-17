/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "test_parameter_name_generator.hpp"
#include "transformers_adam_w.hpp"

namespace transformers_adam_w {

/////////////////////////////////////////////////////////

struct GPU_TransformersAdamWTest_FP32 : TransformersAdamWTest<float, float>
{
};

struct GPU_TransformersAdamWTest_FP16 : TransformersAdamWTest<half_float::half, half_float::half>
{
};

struct GPU_TransformersAmpAdamWTest_FP32 : TransformersAdamWTest<float, half_float::half, true>
{
};

/////////////////////////////////////////////////////////

} // namespace transformers_adam_w
using namespace transformers_adam_w;

/////////////////////////////////////////////////////////

TEST_P(GPU_TransformersAdamWTest_FP32, TransformersAdamWFloatTest)
{
    RunTest();
    Verify();
};

TEST_P(GPU_TransformersAdamWTest_FP16, TransformersAdamWFloat16Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_TransformersAmpAdamWTest_FP32, TransformersAmpAdamWTest)
{
    RunTest();
    Verify();
};

/////////////////////////////////////////////////////////

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& tc = info.param;
        std::stringstream ss;

        ss << "input_" << GetRangeAsString(tc.input, "x") << "_weight_decay_" << tc.weight_decay
           << "_correct_bias_" << tc.correct_bias << "_use_step_tensor_" << tc.use_step_tensor
           << "_use_step_size_" << tc.use_step_size << "_test_id_" << info.index;

        std::string str(ss.str());

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

/////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_TransformersAdamWTest_FP32,
                         testing::ValuesIn(TransformersAdamWTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_TransformersAdamWTest_FP16,
                         testing::ValuesIn(TransformersAdamWTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_TransformersAmpAdamWTest_FP32,
                         testing::ValuesIn(TransformersAdamWTestConfigs()),
                         TestNameGenerator{});
