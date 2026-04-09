// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "network_data.hpp"

namespace {

using TestCase = NamedContainer<std::vector<int>>;

auto GenCases(bool smoke_test)
{
    std::set<std::vector<int>> output_dims;
    std::set<std::vector<int>> dims = get_inputs();

    if(smoke_test)
    {
        output_dims.emplace(*dims.begin());
    }
    else
    {
        output_dims = std::move(dims);
    }

    return MakeNamedParameterCollectionValues<std::vector<int>>("output_dims", output_dims, "x");
}

auto GetCasesFull()
{
    static const auto cases = GenCases(false);
    return cases;
}

auto GetCasesSmoke()
{
    static const auto cases = GenCases(true);
    return cases;
}

} // namespace

template <class T>
struct conv2d_bias_test : public conv_bias_test<T>, public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();

        const std::vector<int> output_dims{GetParam()};

        auto gen_value = [](auto... is) {
            return prng::gen_A_to_B(1, miopen_type<T>{} == miopenHalf ? 5 : 17) *
                   tensor_elem_gen_checkboard_sign{}(is...);
        };

        this->output = tensor<T>{output_dims}.generate(gen_value);
    }
};

using GPU_Conv2d_Bias_FP32  = conv2d_bias_test<float>;
using GPU_Conv2d_Bias_FP16  = conv2d_bias_test<half_float::half>;
using GPU_Conv2d_Bias_BFP16 = conv2d_bias_test<bfloat16>;

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const std::vector<int> output_dims = info.param;
        std::stringstream ss;
        std::string str;

        ss << "output_dims_" << GetRangeAsString(output_dims, "x") << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

TEST_P(GPU_Conv2d_Bias_FP32, TestFloat) { run(); }
TEST_P(GPU_Conv2d_Bias_FP16, TestFloat16) { run(); }
TEST_P(GPU_Conv2d_Bias_BFP16, TestBFloat16) { run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2d_Bias_FP32, GetCasesSmoke(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2d_Bias_FP32, GetCasesFull(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2d_Bias_FP16, GetCasesSmoke(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2d_Bias_FP16, GetCasesFull(), TestNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Conv2d_Bias_BFP16, GetCasesSmoke(), TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2d_Bias_BFP16, GetCasesFull(), TestNameGenerator{});
