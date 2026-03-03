// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "conv_common_gtest.hpp"
#include "test_parameter_name_generator.hpp"

namespace {

std::vector<std::vector<int>> GetConv3dBiasTestCases()
{
    const auto input_set = get_3d_conv_input_shapes<int>();
    return {input_set.begin(), input_set.end()};
}

std::vector<std::vector<int>> GetConv3dBiasSmokeCases()
{
    const auto full_cases = GetConv3dBiasTestCases();
    if(full_cases.empty())
        return {};
    return {full_cases.front()};
}

struct TestParameterNameGenerator
{
    std::string operator()(const testing::TestParamInfo<std::vector<int>>& info) const
    {
        std::stringstream ss;
        ss << "output_" << GetRangeAsString(info.param, "x") << "_test_id_" << info.index;
        return ss.str();
    }
};

template <class T>
struct GPU_Conv3dBias : public testing::TestWithParam<std::vector<int>>
{
    void Run() const
    {
        const auto& out_dims = this->GetParam();
        std::vector<std::size_t> lens(out_dims.begin(), out_dims.end());

        auto gen_value = [](auto... is) {
            return prng::gen_A_to_B(1, miopen_type<T>{} == miopenHalf ? 5 : 17) *
                   tensor_elem_gen_checkboard_sign{}(is...);
        };

        tensor<T> output(lens);
        output.generate(gen_value);

        const auto spatial_dim = output.desc.GetNumDims() - 2;
        std::vector<std::size_t> bias_lens(2 + spatial_dim, 1);
        bias_lens[1] = output.desc.GetLengths()[1];
        tensor<T> bias(bias_lens);

        test_helpers::CompareResults(miopen::test::conv::verify_backwards_bias<T>{output, bias});
    }
};

using GPU_Conv3dBias_FP32 = GPU_Conv3dBias<float>;

} // namespace

TEST_P(GPU_Conv3dBias_FP32, TestFP32) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3dBias_FP32,
                         testing::ValuesIn(GetConv3dBiasTestCases()),
                         TestParameterNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3dBias_FP32,
                         testing::ValuesIn(GetConv3dBiasSmokeCases()),
                         TestParameterNameGenerator{});
