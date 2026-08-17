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
#include <sstream>
#include <string_view>

#include "adam.hpp"
#include "test_parameter_name_generator.hpp"

namespace adam {

/////////////////////////////////////////////////////////

struct GPU_Adam_FP32 : AdamTest<float, float>
{
};

struct GPU_Adam_FP16 : AdamTest<half_float::half, half_float::half>
{
};

struct GPU_AmpAdam_FP32 : AdamTest<float, half_float::half>
{
};

/////////////////////////////////////////////////////////

} // namespace adam
using namespace adam;

/////////////////////////////////////////////////////////

TEST_P(GPU_Adam_FP32, AdamFloatTestFw)
{
    RunTest(true);
    Verify();
};

TEST_P(GPU_Adam_FP16, AdamFloat16TestFw)
{
    RunTest(true);
    Verify();
};

TEST_P(GPU_AmpAdam_FP32, AmpAdamTestFw)
{
    RunTest(true);
    Verify();
};

/////////////////////////////////////////////////////////

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& tc = info.param;
        std::stringstream ss;

        ss << "input_" << GetRangeAsString(tc.input, "x") << "_lr_" << tc.lr << "_beta1_"
           << tc.beta1 << "_beta2_" << tc.beta2 << "_weight_decay_" << tc.weight_decay << "_eps_"
           << tc.eps << "_lr_" << tc.lr << "_lr_" << tc.lr << std::boolalpha << "_amsgrad_"
           << tc.amsgrad << "_maximize_" << tc.maximize << "_adamw_" << tc.adamw
           << "_use_step_tensor_" << tc.use_step_tensor << "_test_id_" << info.index;

        std::string str(ss.str());

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

/////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Adam_FP32,
                         testing::ValuesIn(AdamTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Adam_FP16,
                         testing::ValuesIn(AdamTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_AmpAdam_FP32,
                         testing::ValuesIn(AdamTestConfigs()),
                         TestNameGenerator{});
