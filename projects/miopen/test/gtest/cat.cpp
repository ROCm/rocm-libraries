// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
