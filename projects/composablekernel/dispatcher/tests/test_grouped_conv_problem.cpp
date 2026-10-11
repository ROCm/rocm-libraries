// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvProblem using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace ck_tile::dispatcher;

namespace {
void set_signature(GroupedConvProblem& problem,
                   const std::string& layout = "nhwgc_gkyxc_nhwgk")
{
    problem.dtype_in = problem.dtype_wei = problem.dtype_out = "fp16";
    problem.layout = layout;
    problem.arch = "gfx1100";
}

GroupedConvProblemBuilder valid_builder()
{
    return GroupedConvProblemBuilder()
        .data_types("fp16", "fp16", "fp16")
        .layout("nhwgc_gkyxc_nhwgk")
        .arch("gfx1100");
}
} // namespace

void test_grouped_conv_problem_defaults()
{
    std::cout << "  test_grouped_conv_problem_defaults... ";
    GroupedConvProblem p;
    assert(p.N == 1);
    assert(p.C == 64);
    assert(p.K == 64);
    assert(p.G == 1);
    assert(p.Hi() == 28);
    assert(p.Wi() == 28);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.op == GroupedConvOp::Forward);
    assert(p.stride[0] == 1 && p.stride[1] == 1 && p.stride[2] == 1);
    assert(p.padding_left[0] == 0 && p.padding_left[1] == 0 && p.padding_left[2] == 0);
    assert(p.dilation[0] == 1 && p.dilation[1] == 1 && p.dilation[2] == 1);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_2d()
{
    std::cout << "  test_grouped_conv_problem_2d... ";
    GroupedConvProblem p(4, 64, 128, 28, 28, 3, 3);
    p.compute_output_size();
    assert(p.N == 4);
    assert(p.C == 64);
    assert(p.K == 128);
    assert(p.Hi() == 28);
    assert(p.Wi() == 28);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.Ho() == 26);
    assert(p.Wo() == 26);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_strided()
{
    std::cout << "  test_grouped_conv_problem_strided... ";
    GroupedConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 2, 2};
    p.padding_left   = {0, 1, 1};
    p.padding_right  = p.padding_left;
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.Ho() == 7);
    assert(p.Wo() == 7);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_grouped()
{
    std::cout << "  test_grouped_conv_problem_grouped... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 4;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding_left   = {0, 0, 0};
    p.padding_right  = p.padding_left;
    p.dilation       = {1, 1, 1};
    set_signature(p);
    assert(p.compute_output_size());
    assert(p.G == 4);
    assert(p.C % p.G == 0);
    assert(p.K % p.G == 0);
    assert(p.is_valid());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_depthwise()
{
    std::cout << "  test_grouped_conv_problem_depthwise... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 64;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding_left   = {0, 0, 0};
    p.padding_right  = p.padding_left;
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.is_depthwise());
    assert(p.G == p.C && p.G == p.K);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_pointwise()
{
    std::cout << "  test_grouped_conv_problem_pointwise... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 1, 1};
    p.stride         = {1, 1, 1};
    p.padding_left   = {0, 0, 0};
    p.padding_right  = p.padding_left;
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.is_pointwise());
    assert(p.Y() == 1 && p.X() == 1);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_flops()
{
    std::cout << "  test_grouped_conv_problem_flops... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding_left   = {0, 0, 0};
    p.padding_right  = p.padding_left;
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    double flops = p.get_flops();
    assert(flops > 0);
    assert(flops == 2.0 * p.N * p.K * p.Ho() * p.Wo() * (p.C / p.G) * p.Y() * p.X());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_is_valid()
{
    std::cout << "  test_grouped_conv_problem_is_valid... ";
    GroupedConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.padding_right  = p.padding_left;
    set_signature(p);
    assert(p.compute_output_size());
    assert(p.is_valid());

    p.N = 0;
    assert(!p.is_valid());
    p.N = 1;

    p.C = 0;
    assert(!p.is_valid());
    p.C = 64;

    p.K = 0;
    assert(!p.is_valid());
    p.K = 64;

    p.G = 0;
    assert(!p.is_valid());
    p.G = 1;

    p.C = 64;
    p.K = 64;
    p.G = 3;
    assert(!p.is_valid());
    p.G = 4;
    assert(p.is_valid());
    p.ndim_spatial = 1;
    assert(!p.is_valid());
    p.ndim_spatial = 2;
    assert(p.is_valid());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_builder()
{
    std::cout << "  test_grouped_conv_problem_builder... ";
    auto p = valid_builder()
                 .batch(8)
                 .channels(128, 256)
                 .groups(4)
                 .input_size(32, 32)
                 .filter_size(3, 3)
                 .stride(2, 2)
                 .padding(1, 1)
                 .dilation(1, 1)
                 .operation(GroupedConvOp::Forward)
                 .build();
    assert(p.N == 8);
    assert(p.C == 128);
    assert(p.K == 256);
    assert(p.G == 4);
    assert(p.Hi() == 32);
    assert(p.Wi() == 32);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.stride[1] == 2 && p.stride[2] == 2);
    assert(p.padding_left[1] == 1 && p.padding_left[2] == 1);
    assert(p.op == GroupedConvOp::Forward);
    assert(p.is_valid());

    bool threw = false;
    try
    {
        (void)valid_builder()
            .batch(0)
            .channels(64, 64)
            .groups(1)
            .input_size(14, 14)
            .filter_size(3, 3)
            .build();
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
    std::cout << "PASSED\n";
}


void test_runtime_signature_and_full_validation()
{
    GroupedConvProblem p;
    assert(p.dtype_in.empty() && p.dtype_wei.empty() && p.dtype_out.empty());
    assert(p.layout.empty() && p.ndim_spatial == 2);
    assert(p.arch.empty() && p.op == GroupedConvOp::Forward);
    assert(!p.is_valid());
    assert((p.padding_left == std::array<std::int64_t, 3>{0, 0, 0}));
    assert((p.padding_right == std::array<std::int64_t, 3>{0, 0, 0}));

    p.dtype_in = "bf16"; p.dtype_wei = "fp32"; p.dtype_out = "fp64";
    p.layout = "custom"; p.ndim_spatial = 3; p.arch = "gfx942";
    p.op = GroupedConvOp::BackwardData;
    assert(p.dtype_in == "bf16" && p.dtype_wei == "fp32" && p.dtype_out == "fp64");
    assert(p.layout == "custom" && p.ndim_spatial == 3 && p.arch == "gfx942");

    auto valid = valid_builder().channels(64, 128).groups(4).build();
    valid.G = -1; assert(!valid.is_valid());
    valid = valid_builder().build(); valid.C = 65; valid.G = 4;
    assert(!valid.is_valid());
    valid = valid_builder().build(); valid.K = 65; valid.G = 4;
    assert(!valid.is_valid());
}

void test_invalid_spatial_contract()
{
    auto p = valid_builder().build();
    p.input_spatial[1] = 0; assert(!p.is_valid());
    p = valid_builder().build(); p.filter_spatial[1] = 0; assert(!p.is_valid());
    p = valid_builder().build(); p.stride[1] = 0; assert(!p.is_valid());
    p = valid_builder().build(); p.dilation[1] = 0; assert(!p.is_valid());
    p = valid_builder().build(); p.padding_left[1] = -1; assert(!p.is_valid());
    p = valid_builder().build(); p.padding_right[1] = -1; assert(!p.is_valid());
    p = valid_builder().build(); ++p.output_spatial[1]; assert(!p.is_valid());
    p = valid_builder().build(); p.filter_spatial[1] = 100;
    p.output_spatial[1] = 1; assert(!p.is_valid());
}

void test_2d_depth_contract_and_asymmetric_padding()
{
    auto p = valid_builder().build();
    p.input_spatial[0] = 2; assert(!p.is_valid());
    p = valid_builder().build(); p.filter_spatial[0] = 2; assert(!p.is_valid());
    p = valid_builder().build(); p.output_spatial[0] = 2; assert(!p.is_valid());
    p = valid_builder().build(); p.stride[0] = 2; assert(!p.is_valid());
    p = valid_builder().build(); p.dilation[0] = 2; assert(!p.is_valid());
    p = valid_builder().build(); p.padding_left[0] = 1; assert(!p.is_valid());
    p = valid_builder().build(); p.padding_right[0] = 1; assert(!p.is_valid());

    auto asymmetric = valid_builder().input_size(7, 8).filter_size(3, 3)
                          .stride(2, 2).padding(0, 1, 2, 3).build();
    assert((asymmetric.padding_left == std::array<std::int64_t, 3>{0, 0, 1}));
    assert((asymmetric.padding_right == std::array<std::int64_t, 3>{0, 2, 3}));
    assert((asymmetric.output_spatial == std::array<std::int64_t, 3>{1, 4, 5}));
}

void test_3d_builder_and_grouped_flops()
{
    auto p = valid_builder().ndim(3).layout("ndhwgc_gkzyxc_ndhwgk")
                 .input_size(8, 9, 10).filter_size(3, 3, 3).stride(1, 2, 2)
                 .padding(1, 1, 2).dilation(1, 1, 1).build();
    assert(p.ndim_spatial == 3 && p.layout == "ndhwgc_gkzyxc_ndhwgk");
    assert((p.padding_left == std::array<std::int64_t, 3>{1, 1, 2}));
    assert(p.padding_left == p.padding_right);
    assert((p.output_spatial == std::array<std::int64_t, 3>{8, 5, 6}));

    auto grouped = valid_builder().batch(2).channels(64, 128).groups(4)
                       .input_size(14, 14).filter_size(3, 3).build();
    assert(grouped.get_flops() == 2.0 * 2 * 128 * 12 * 12 * 16 * 3 * 3);
}

void test_builder_setters_are_order_independent()
{
    const auto first = valid_builder().ndim(3).layout("custom_3d")
                           .input_size(8, 9, 10).filter_size(3, 3, 3).build();
    const auto second = valid_builder().input_size(8, 9, 10).filter_size(3, 3, 3)
                            .layout("custom_3d").ndim(3).build();
    assert(first.ndim_spatial == second.ndim_spatial);
    assert(first.layout == second.layout);
    assert(first.input_spatial == second.input_spatial);
    assert(first.filter_spatial == second.filter_spatial);
    assert(first.output_spatial == second.output_spatial);
}

void test_compute_output_size_contract()
{
    GroupedConvProblem p;
    const auto original = p.output_spatial;
    p.stride[1] = 0;
    assert(!p.compute_output_size());
    assert(p.output_spatial == original);

    p.stride[1] = 1;
    p.filter_spatial[1] = 100;
    assert(!p.compute_output_size());
    assert(p.output_spatial == original);

    p.filter_spatial[1] = 3;
    assert(p.compute_output_size());
    assert((p.output_spatial == std::array<std::int64_t, 3>{1, 26, 26}));
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Problem ===\n\n";
    test_grouped_conv_problem_defaults();
    test_builder_setters_are_order_independent();
    test_compute_output_size_contract();
    test_runtime_signature_and_full_validation();
    test_invalid_spatial_contract();
    test_2d_depth_contract_and_asymmetric_padding();
    test_3d_builder_and_grouped_flops();
    test_grouped_conv_problem_2d();
    test_grouped_conv_problem_strided();
    test_grouped_conv_problem_grouped();
    test_grouped_conv_problem_depthwise();
    test_grouped_conv_problem_pointwise();
    test_grouped_conv_problem_flops();
    test_grouped_conv_problem_is_valid();
    test_grouped_conv_problem_builder();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
