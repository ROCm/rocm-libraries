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
#include "cat.hpp"
#include "test_parameter_name_generator.hpp"

namespace cat {

struct GPU_Cat_FP32 : CatTest<float>
{
};

struct GPU_Cat_FP16 : CatTest<half_float::half>
{
};

struct GPU_Cat_BFP16 : CatTest<bfloat16>
{
};

} // namespace cat
using namespace cat;

TEST_P(GPU_Cat_FP32, CatTestFP32Fw)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Cat_FP16, CatTestFP16Fw)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Cat_BFP16, CatTestBFP16Fw)
{
    RunTest();
    Verify();
};

///////////////////////////////////////////////////////////////////////////////////

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& tc = info.param;
        auto it        = tc.inputs.begin();
        std::stringstream ss;

        ss << "inputs_" << GetRangeAsString(*it, "x");
        ++it;

        for(; it != tc.inputs.end(); ++it)
        {
            ss << "," << GetRangeAsString(*it, "x");
        }

        ss << "_dim_" << tc.dim << "_test_id_" << info.index;

        std::string str(ss.str());

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

///////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Cat_FP32,
                         testing::ValuesIn(CatTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Cat_FP16,
                         testing::ValuesIn(CatTestConfigs()),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Cat_BFP16,
                         testing::ValuesIn(CatTestConfigs()),
                         TestNameGenerator{});
