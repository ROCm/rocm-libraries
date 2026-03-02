// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include <array>

namespace {

struct group_conv_case
{
    std::array<std::size_t, 4> input;
    std::array<std::size_t, 4> weights;
    std::array<int, 6> pads_strides_dilations;
    int group_count;
    bool disable_backward_weights = false;
};

const std::vector<group_conv_case>& GetCuratedCases()
{
    static const std::vector<group_conv_case> cases = {
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{16, 256, 56, 56}, {512, 8, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{16, 256, 28, 28}, {512, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{16, 512, 28, 28}, {1024, 16, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{16, 1024, 14, 14}, {2048, 32, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{16, 1024, 7, 7}, {2048, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{32, 128, 56, 56}, {256, 4, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{32, 256, 56, 56}, {512, 8, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{32, 256, 28, 28}, {512, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, true},
        {{32, 512, 28, 28}, {1024, 16, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{32, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{32, 1024, 14, 14}, {2048, 32, 3, 3}, {1, 1, 2, 2, 1, 1}, 32, false},
        {{32, 1024, 7, 7}, {2048, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 32, false},
        {{4, 4, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 4, false},
        {{8, 2, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{16, 4, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 4, false},
        {{32, 2, 161, 700}, {32, 1, 5, 20}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{4, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{8, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{16, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{32, 32, 79, 341}, {32, 16, 5, 10}, {0, 0, 2, 2, 1, 1}, 2, false},
        {{16, 4, 48, 480}, {16, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 4, false},
        {{16, 16, 24, 240}, {32, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 16, false},
        {{16, 32, 12, 120}, {64, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 4, false},
        {{16, 64, 6, 60}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 4, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 3, false},
        {{8, 64, 54, 54}, {64, 8, 3, 3}, {1, 1, 1, 1, 1, 1}, 8, false},
        {{8, 128, 27, 27}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 8, false},
        {{8, 3, 224, 224}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 3, false},
        {{8, 64, 112, 112}, {128, 32, 3, 3}, {1, 1, 1, 1, 1, 1}, 2, false},
        {{16, 9, 224, 224}, {63, 3, 3, 3}, {1, 1, 1, 1, 1, 1}, 3, false},
        {{16, 64, 112, 112}, {128, 16, 3, 3}, {1, 1, 1, 1, 1, 1}, 4, true},
        {{16, 3, 224, 224}, {63, 1, 7, 7}, {3, 3, 2, 2, 1, 1}, 3, false},
        {{16, 192, 28, 28}, {32, 12, 5, 5}, {2, 2, 1, 1, 1, 1}, 16, false},
        {{16, 832, 7, 7}, {128, 52, 5, 5}, {2, 2, 1, 1, 1, 1}, 16, false},
        {{16, 192, 28, 28}, {32, 24, 1, 1}, {0, 0, 1, 1, 1, 1}, 8, false},
        {{16, 832, 7, 7}, {128, 104, 1, 1}, {0, 0, 1, 1, 1, 1}, 8, false},
        {{11, 23, 161, 700}, {46, 1, 7, 7}, {1, 1, 2, 2, 1, 1}, 23, false},
        {{8, 7, 224, 224}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 7, false},
        {{8, 7, 224, 224}, {63, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, 7, false},
        {{8, 7, 224, 224}, {63, 1, 3, 3}, {0, 0, 2, 2, 1, 1}, 7, false},
        {{8, 7, 224, 224}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 7, false},
        {{8, 7, 224, 224}, {63, 1, 3, 3}, {2, 2, 2, 2, 1, 1}, 7, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, 3, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, 3, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {0, 0, 2, 2, 1, 1}, 3, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {1, 1, 2, 2, 1, 1}, 3, false},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2, 2, 2, 1, 1}, 3, false},
    };
    return cases;
}

auto GetDataset()
{
    std::vector<miopen::test::conv::conv_test_input> cases{};

    for(const auto& c : GetCuratedCases())
    {
        miopen::test::conv::conv_test_input input{};
        input.batch_size           = c.input[0];
        input.input_channels       = c.input[1];
        input.spatial_dim_elements = {c.input[2], c.input[3]};
        input.output_channels      = c.weights[0];
        input.filter_dims          = {c.weights[2], c.weights[3]};
        input.pads_strides_dilations.assign(
            c.pads_strides_dilations.begin(), c.pads_strides_dilations.end());
        input.trans_output_pads = {0, 0};
        input.in_layout         = "NCHW";
        input.fil_layout        = "NCHW";
        input.out_layout        = "NCHW";
        input.pad_mode          = "default";
        input.deterministic     = false;
        input.tensor_vect       = 0;
        input.vector_length     = 1;
        input.output_type       = "int32";
        input.int8_vectorize    = false;
        input.groupCount        = c.group_count;
        input.do_forward        = true;
        input.do_backward_data  = true;
        input.do_backward_weights = !c.disable_backward_weights;

        if(miopen::test::conv::IsValidCtestStyleConfig(input))
            cases.push_back(input);
    }
    return cases;
}

} // namespace

template <class T>
struct conv_group_test : miopen::test::conv::conv_test_base<T>
{
};

using GPU_conv_group_FP32 = conv_group_test<float>;

TEST_P(GPU_conv_group_FP32, TestFP32) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_conv_group_FP32, ::testing::ValuesIn(GetDataset()));
